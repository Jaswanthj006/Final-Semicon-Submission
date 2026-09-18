#!/usr/bin/env python3
"""DriftLoc: find the 100x reference patch inside the 10x search image.

Three stages in one file. Run them in order; each stage prints its own
validation before you move on.

  stage1  PROPOSALS  Phase 2 grid: scale 8..12 (0.5 step) x rotation -5..+5 (1 deg),
                     top-20 candidate windows per pair.
                     Validates: recall@20 and ZNCC-top1 pass@3 on val (present pairs).
  stage2  VERIFIER   small CNN that picks the true window among the ZNCC
                     candidates. Hard negatives = the other ZNCC peaks.
                     Validates: end-to-end pass@5 on val vs the ZNCC baseline.
  stage3  FULL RUN   proposals + verifier + sub-pixel on a held-out split.
                     Validates: pass@5/4/2/1 per noise bucket, PR curves,
                     runtime, failure panels.

Why this design (measured on output/val, 250 pairs, before writing this):
  - global ZNCC max: pass@5 ~55% (DRAM repeats every ~10 px -> wrong cell)
  - true window inside top-20 ZNCC peaks: ~85-90%   <- proposals are enough
  - ECC re-scoring 48%, multi-map consensus 27%     <- classical ranking fails
  - when the right cell is picked, sub-pixel error is already <1 px
  So the only open problem is choosing among ~20 look-alike windows.
  That is what the verifier learns, from manifest.csv labels only.

Commands (run manually, in order):

  python train.py stage1 --data splits --split val
  python train.py stage3 --data splits --out model --split val
  python train.py localize --reference REF.png --search SEARCH.png --out model

`localize` prints "x y" for one pair (sponsor interface). Without a trained
checkpoint it falls back to plain ZNCC + sub-pixel.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

# ---------------------------------------------------------------------------
# Geometry / pipeline constants
# ---------------------------------------------------------------------------
# Phase 2 disclosed bounds (hard-coding allowed per addendum).
SCALE_MIN, SCALE_MAX, SCALE_STEP = 8.0, 12.0, 0.5
ANGLE_MIN, ANGLE_MAX, ANGLE_STEP = -5.0, 5.0, 1.0

# Phase 3 generator/spec, not fitted on any SEM split:
# cad_pipeline.SCALE_FACTOR is fixed at 10 (1 nm/px ref, 10 nm/px search);
# cad_pipeline.MAX_SEARCH_ROTATION_DEG is 10. Searching a parameter the
# generator cannot vary only adds compute.
PHASE3_SCALE = 10.0
PHASE3_SCALES = (PHASE3_SCALE,)
PHASE3_ANGLE_MIN, PHASE3_ANGLE_MAX = -10.0, 10.0
MATCH_GREY = "grey"
MATCH_ORIENT = "orientation"


def _frange(start: float, stop: float, step: float) -> tuple[float, ...]:
    n = int(round((stop - start) / step)) + 1
    return tuple(round(start + i * step, 4) for i in range(n))


# Full Phase 2 search grid: 9 scales x 11 angles = 99 ZNCC maps per pair.
SCALES = _frange(SCALE_MIN, SCALE_MAX, SCALE_STEP)
ANGLES = _frange(ANGLE_MIN, ANGLE_MAX, ANGLE_STEP)
PHASE3_ANGLES = _frange(PHASE3_ANGLE_MIN, PHASE3_ANGLE_MAX, ANGLE_STEP)
PEAKS_PER_MAP = 12                         # local maxima kept per ZNCC map
TOP_N = 50                                 # candidates kept per pair after NMS
NMS_RADIUS = 2.0                           # px between distinct candidates
POS_RADIUS = 3.0                           # candidate <= this from GT = positive
                                           # (scoring is at 2 px; 5.0 taught the
                                           # verifier that 5 px was correct, and
                                           # NMS_RADIUS=2.0 floors how tight this
                                           # can go before positives starve)
CROP = 128                                 # verifier input resolution
K_TRAIN = 16                               # positive + 15 hard negatives
SEARCH_CENTER = (500.0, 500.0)

# Coarse-to-fine proposals. Scoring all 99 (scale, angle) maps at full
# resolution is ~1.6 s/pair and dominates runtime; the reference machine is a
# 4-core CPU with a 5 s median budget. A half-resolution sweep costs a quarter
# of that and only has to *rank* the grid cells -- the survivors are then
# matched at full resolution, so candidate patches reaching the verifier are
# bit-identical to the exhaustive path.
COARSE_DIV = 4                             # 1 disables the coarse pass
COARSE_TOP_SA = 14                         # grid cells promoted to full res
COARSE_SEEDS = 3                           # cells whose grid neighbours tag along
# div=5 measurably loses localization credit (0.865 vs 0.890 on val); div=4
# matches the exhaustive sweep cell-for-cell while costing a third as much.

# Pose refinement. The grid quantises scale to 0.5 and angle to 1.0 deg, which
# caps pose credit at the middle tier no matter how good the localization is.
# Re-matching inside a window around the winner costs ~60 ms.
REFINE_SCALE_STEP = 0.05
REFINE_ANGLE_STEP = 0.1
REFINE_MARGIN_PX = 10
REFINE_MAX_SHIFT_PX = 5.0                  # further than this = a different site
# FOV-edge crops: match the overlapping rectangle instead of skipping.
# A sliver of DRAM is still periodic; require ~40% of the template.
PARTIAL_MIN_FRAC = 0.40
TIE_PROB_GAP = 0.05                        # spec tie-break: near-ties only

PASS_THRESHOLDS = (1, 2, 3, 4, 5)

# Set C rejection (tuned on splits_dram val: absent conf~0.27, present conf~0.99).
# Confidence-only gate — margin OR falsely rejects present pairs when the verifier
# splits probability across repeated DRAM peaks.
REJECT_CONF = 0.38
REJECT_MARGIN = 0.05  # logged in metrics only

# CAD->SEM: the Phase 2 verifier was trained on SEM-SEM crops and can
# override a clearly better ZNCC peak (strip-heavy references). If ZNCC's
# winner is a different site and beats the verifier's ZNCC by this margin,
# keep ZNCC. Lattice ties (close ZNCC scores) still go to the verifier.
ZNCC_OVERRIDE_MARGIN = 0.05
ZNCC_OVERRIDE_SHIFT_PX = 5.0
REJECT_ZNCC = 0.55          # found-gate when the pick came from the override

# CAD-CAD site in search pixels. Search rotation about the image centre
# moves a corner crop by ~r*θ (10 deg at 900 px ≈ 160 px). Far lattice
# aliases are hundreds of px; keep the SEM pick inside that ball.
CAD_PRIOR_RADIUS_PX = 160.0

# Stage 1 exit targets (present pairs on val) — see stage1() summary.
STAGE1_RECALL_TARGET = 0.90
STAGE1_SET_A_PASS3_TARGET = 0.70
STAGE1_SET_B_PASS3_TARGET = 0.55
STAGE1_TIME_TARGET_S = 5.0


# ---------------------------------------------------------------------------
# Stage 1: ZNCC proposal generator
# ---------------------------------------------------------------------------
def rotate_template(t: np.ndarray, ang: float) -> np.ndarray:
    if ang == 0.0:
        return t
    h, w = t.shape
    M = cv2.getRotationMatrix2D((w / 2 - 0.5, h / 2 - 0.5), ang, 1.0)
    return cv2.warpAffine(t, M, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)


def cad_silhouette(img: np.ndarray) -> np.ndarray:
    """Binary occupancy of a CAD raster. Fill/yield grey is discarded."""
    u8 = img if img.dtype == np.uint8 else np.clip(img, 0, 255).astype(np.uint8)
    if u8.size == 0 or int(u8.max()) == int(u8.min()):
        return np.zeros(u8.shape, dtype=np.float32)
    _, mask = cv2.threshold(u8, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask.astype(np.float32)


def orientation_maps(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Polarity-invariant 2θ maps: mag * (cos 2θ, sin 2θ).

    Doubling the Sobel angle maps a 180° contrast flip (bright-on-dark vs
    dark-on-bright) onto the same value, so dose/gamma/polarity of the SEM
    do not have to match the CAD silhouette.
    """
    g = np.ascontiguousarray(img, dtype=np.float32)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    two = np.arctan2(gy, gx) * 2.0
    re = np.ascontiguousarray(mag * np.cos(two))
    im = np.ascontiguousarray(mag * np.sin(two))
    return re, im


def zncc_maps(sea: np.ndarray, tmpl: np.ndarray, match: str,
              sea_ori: tuple[np.ndarray, np.ndarray] | None = None) -> np.ndarray:
    """ZNCC on greys, or on 2θ orientation maps."""
    if match != MATCH_ORIENT:
        R = cv2.matchTemplate(np.ascontiguousarray(sea),
                              np.ascontiguousarray(tmpl),
                              cv2.TM_CCOEFF_NORMED)
        return np.nan_to_num(R, nan=-1.0, posinf=-1.0, neginf=-1.0)
    sea_re, sea_im = sea_ori if sea_ori is not None else orientation_maps(sea)
    t_re, t_im = orientation_maps(tmpl)
    sea_re = np.ascontiguousarray(sea_re)
    sea_im = np.ascontiguousarray(sea_im)
    t_re = np.ascontiguousarray(t_re)
    t_im = np.ascontiguousarray(t_im)
    if (t_re.shape[0] > sea_re.shape[0] or t_re.shape[1] > sea_re.shape[1]):
        return np.zeros((1, 1), dtype=np.float32)
    R = 0.5 * (cv2.matchTemplate(sea_re, t_re, cv2.TM_CCOEFF_NORMED)
               + cv2.matchTemplate(sea_im, t_im, cv2.TM_CCOEFF_NORMED))
    return np.nan_to_num(R, nan=-1.0, posinf=-1.0, neginf=-1.0)


def peak_near(R: np.ndarray, xi: float, yi: float, radius: float):
    """Best score inside `radius` of (xi, yi) in a correlation map."""
    h, w = R.shape
    x0 = max(0, int(np.floor(xi - radius)))
    x1 = min(w, int(np.ceil(xi + radius)) + 1)
    y0 = max(0, int(np.floor(yi - radius)))
    y1 = min(h, int(np.ceil(yi + radius)) + 1)
    patch = R[y0:y1, x0:x1]
    if patch.size == 0:
        return None
    yy, xx = np.unravel_index(int(np.argmax(patch)), patch.shape)
    return int(x0 + xx), int(y0 + yy), float(patch[yy, xx])


def _clip_template_window(sea: np.ndarray, tmpl: np.ndarray,
                          exp_tl_x: float, exp_tl_y: float, margin: int,
                          min_frac: float = PARTIAL_MIN_FRAC):
    """Visible overlap of a template placed at expected top-left.

    Returns None if less than `min_frac` of the template is in-bounds.
    Otherwise (win, ptmpl, xa, ya, left, top, xi_exp, yi_exp) where a
    peak (xi, yi) in matchTemplate(win, ptmpl) is full-template centre
    (xa + xi - left + tw/2, ya + yi - top + th/2).
    """
    H, W = sea.shape[:2]
    th, tw = tmpl.shape[:2]
    etx = int(round(exp_tl_x))
    ety = int(round(exp_tl_y))
    left = max(0, -etx)
    top = max(0, -ety)
    right = max(0, etx + tw - W)
    bottom = max(0, ety + th - H)
    ph = th - top - bottom
    pw = tw - left - right
    min_h = max(8, int(round(min_frac * th)))
    min_w = max(8, int(round(min_frac * tw)))
    if ph < min_h or pw < min_w:
        return None
    ptmpl = tmpl[top:th - bottom, left:tw - right]
    px0 = etx + left - margin
    py0 = ety + top - margin
    px1 = px0 + pw + 2 * margin
    py1 = py0 + ph + 2 * margin
    xa, ya = max(0, px0), max(0, py0)
    xb, yb = min(W, px1), min(H, py1)
    if (yb - ya) < ph or (xb - xa) < pw:
        return None
    win = sea[ya:yb, xa:xb]
    xi_exp = float((etx + left) - xa)
    yi_exp = float((ety + top) - ya)
    return win, ptmpl, xa, ya, left, top, xi_exp, yi_exp


def local_peaks(R: np.ndarray, k: int, nsize: int = 5):
    """Top-k local maxima of a ZNCC map -> [(xi, yi, score)]."""
    ker = np.ones((nsize, nsize), np.uint8)
    dil = cv2.dilate(R, ker)
    ys, xs = np.where(R >= dil)
    if len(xs) == 0:
        return []
    ss = R[ys, xs]
    order = np.argsort(-ss)[:k]
    return [(int(xs[i]), int(ys[i]), float(ss[i])) for i in order]


def select_grid_cells(ref: np.ndarray, sea: np.ndarray,
                      scales: tuple[float, ...], angles: tuple[float, ...],
                      div: int | None = None, match: str = MATCH_GREY) -> set:
    """Rank (scale, angle) cells on a downsampled search image.

    Only the ranking is used, never the coordinates, so the resolution loss
    costs nothing downstream. Grid neighbours of the strongest cells are kept
    as well, since the     coarse peak can straddle two adjacent cells.
    """
    div = COARSE_DIV if div is None else div
    small = cv2.resize(sea, (sea.shape[1] // div, sea.shape[0] // div),
                       interpolation=cv2.INTER_AREA)
    sea_ori = orientation_maps(small) if match == MATCH_ORIENT else None
    scored = []
    for sc in scales:
        tw = int(round(1000.0 / sc / div))
        if tw < 8 or tw > small.shape[0]:
            continue
        base = cv2.resize(ref, (tw, tw), interpolation=cv2.INTER_AREA)
        for ang in angles:
            tmpl = rotate_template(base, ang)
            if tmpl.shape[0] > small.shape[0] or tmpl.shape[1] > small.shape[1]:
                continue
            R = zncc_maps(small, tmpl, match, sea_ori=sea_ori)
            scored.append((float(R.max()), sc, ang))
    if not scored:
        return {(sc, ang) for sc in scales for ang in angles}
    scored.sort(key=lambda t: -t[0])

    valid = {(sc, ang) for sc in scales for ang in angles}
    keep = {(sc, ang) for _, sc, ang in scored[:COARSE_TOP_SA]}
    for _, sc, ang in scored[:COARSE_SEEDS]:
        for ds in (-SCALE_STEP, 0.0, SCALE_STEP):
            for da in (-ANGLE_STEP, 0.0, ANGLE_STEP):
                keep.add((round(sc + ds, 4), round(ang + da, 4)))
    return keep & valid


def _nms_candidates(raw: list, top_n: int) -> list:
    raw.sort(key=lambda c: -c["score"])
    kept = []
    for c in raw:
        if all((c["cx"] - k["cx"]) ** 2 + (c["cy"] - k["cy"]) ** 2 > NMS_RADIUS ** 2
               for k in kept):
            kept.append(c)
        if len(kept) >= top_n:
            break
    return kept


def propose_near_prior(ref: np.ndarray, sea: np.ndarray,
                       prior_xy: tuple[float, float],
                       scales: tuple[float, ...], angles: tuple[float, ...],
                       match: str, top_n: int) -> tuple[list, list]:
    """Pose search at the CAD-CAD site: one window per (scale, angle).

    Cell identity is already fixed. The correlation peak is only allowed
    to walk REFINE_MAX_SHIFT_PX — further than that is a different lattice
    copy, which subpixel matching cannot recover from. If the 100 px
    template hangs off the PNG, match the overlapping rectangle instead
    of dropping the window.
    """
    tmpl_src = cad_silhouette(ref) if match == MATCH_ORIENT else ref
    sea_ori = orientation_maps(sea) if match == MATCH_ORIENT else None
    px, py = prior_xy
    maps, raw = [], []
    radius = REFINE_MAX_SHIFT_PX
    for sc in scales:
        tw = int(round(1000.0 / sc))
        if tw < 8 or tw > sea.shape[0]:
            continue
        base = cv2.resize(tmpl_src, (tw, tw), interpolation=cv2.INTER_AREA)
        for ang in angles:
            tmpl = rotate_template(base, ang)
            ex, ey = _rotate_point(px, py, ang)
            clipped = _clip_template_window(
                sea, tmpl, ex - tw / 2.0, ey - tw / 2.0, REFINE_MARGIN_PX)
            if clipped is None:
                continue
            win, ptmpl, xa, ya, left, top, xi_exp, yi_exp = clipped
            if sea_ori is None:
                win_ori = None
            else:
                win_ori = (sea_ori[0][ya:ya + win.shape[0], xa:xa + win.shape[1]],
                           sea_ori[1][ya:ya + win.shape[0], xa:xa + win.shape[1]])
            R = zncc_maps(win, ptmpl, match, sea_ori=win_ori)
            hit = peak_near(R, xi_exp, yi_exp, radius)
            if hit is None:
                continue
            xi, yi, s = hit
            cx = xa + xi - left + tw / 2.0
            cy = ya + yi - top + tw / 2.0
            if np.hypot(cx - ex, cy - ey) > REFINE_MAX_SHIFT_PX:
                continue
            mi = len(maps)
            maps.append({"R": R, "tw": tw, "scale": sc, "angle": ang,
                         "tmpl": tmpl, "origin": (xa, ya)})
            raw.append({"cx": cx, "cy": cy,
                        "xi": xi, "yi": yi, "score": s,
                        "scale": sc, "angle": ang, "map_idx": mi})
    return _nms_candidates(raw, top_n), maps


def propose(ref: np.ndarray, sea: np.ndarray, top_n: int = TOP_N,
            scales: tuple[float, ...] = SCALES,
            angles: tuple[float, ...] = ANGLES,
            match: str = MATCH_GREY,
            prior_xy: tuple[float, float] | None = None):
    """Multi-scale/rotation ZNCC. Returns (candidates, maps).

    candidate: dict(cx, cy, xi, yi, score, scale, angle, map_idx)
    maps[i]:   dict(R, tw, scale, angle, tmpl)  -- kept for sub-pixel + crops
    """
    if prior_xy is not None:
        return propose_near_prior(ref, sea, prior_xy, scales, angles,
                                  match, top_n)

    tmpl_src = cad_silhouette(ref) if match == MATCH_ORIENT else ref
    sea_work = sea if match != MATCH_ORIENT else np.ascontiguousarray(
        sea, dtype=np.float32)
    sea_ori = orientation_maps(sea_work) if match == MATCH_ORIENT else None

    cells = None
    if COARSE_DIV > 1 and len(scales) * len(angles) > COARSE_TOP_SA:
        cells = select_grid_cells(tmpl_src, sea_work, scales, angles,
                                  match=match)

    maps = []
    raw = []
    for sc in scales:
        tw = int(round(1000.0 / sc))
        if tw < 8 or tw > sea.shape[0]:
            continue
        if cells is not None and not any((sc, a) in cells for a in angles):
            continue
        base = cv2.resize(tmpl_src, (tw, tw), interpolation=cv2.INTER_AREA)
        for ang in angles:
            if cells is not None and (sc, ang) not in cells:
                continue
            tmpl = rotate_template(base, ang)
            if tmpl.shape[0] > sea.shape[0] or tmpl.shape[1] > sea.shape[1]:
                continue
            R = zncc_maps(sea_work, tmpl, match, sea_ori=sea_ori)
            mi = len(maps)
            maps.append({"R": R, "tw": tw, "scale": sc, "angle": ang, "tmpl": tmpl})
            for xi, yi, s in local_peaks(R, PEAKS_PER_MAP):
                raw.append({"cx": xi + tw / 2.0, "cy": yi + tw / 2.0,
                            "xi": xi, "yi": yi, "score": s,
                            "scale": sc, "angle": ang, "map_idx": mi})
    return _nms_candidates(raw, top_n), maps


def subpixel(R: np.ndarray, xi: int, yi: int):
    """1D parabolic fit around an integer ZNCC peak -> (dx, dy) in [-1, 1]."""
    dx = dy = 0.0
    if 0 < xi < R.shape[1] - 1:
        l, c, r = float(R[yi, xi - 1]), float(R[yi, xi]), float(R[yi, xi + 1])
        d = l - 2 * c + r
        if abs(d) > 1e-9:
            dx = float(np.clip(0.5 * (l - r) / d, -1, 1))
    if 0 < yi < R.shape[0] - 1:
        u, c, w = float(R[yi - 1, xi]), float(R[yi, xi]), float(R[yi + 1, xi])
        d = u - 2 * c + w
        if abs(d) > 1e-9:
            dy = float(np.clip(0.5 * (u - w) / d, -1, 1))
    return dx, dy


def refine_pose(ref: np.ndarray, sea: np.ndarray, cx: float, cy: float,
                scale0: float, angle0: float,
                match: str = MATCH_GREY,
                scale_min: float = SCALE_MIN, scale_max: float = SCALE_MAX,
                angle_min: float = ANGLE_MIN, angle_max: float = ANGLE_MAX,
                lock_scale: bool = False):
    """Sub-grid scale/angle search inside a window around the chosen candidate.

    Returns (score, x, y, scale, angle, tw) or None. Bounded to half a grid
    step either way, so it sharpens the winning pose rather than re-deciding
    which site won; a solution that walks further than REFINE_MAX_SHIFT_PX is
    a different lattice cell and gets discarded.
    """
    tmpl_src = cad_silhouette(ref) if match == MATCH_ORIENT else ref
    n_s = 0 if lock_scale else int(round((SCALE_STEP / 2.0) / REFINE_SCALE_STEP))
    n_a = int(round((ANGLE_STEP / 2.0) / REFINE_ANGLE_STEP))
    best = None
    for i in range(-n_s, n_s + 1):
        sc = round(scale0 + i * REFINE_SCALE_STEP, 4)
        if not (scale_min <= sc <= scale_max):
            continue
        tw = int(round(1000.0 / sc))
        if tw < 8 or tw > sea.shape[0]:
            continue
        base = cv2.resize(tmpl_src, (tw, tw), interpolation=cv2.INTER_AREA)
        for j in range(-n_a, n_a + 1):
            ang = round(angle0 + j * REFINE_ANGLE_STEP, 4)
            if not (angle_min <= ang <= angle_max):
                continue
            tmpl = rotate_template(base, ang)
            clipped = _clip_template_window(
                sea, tmpl, cx - tw / 2.0, cy - tw / 2.0, REFINE_MARGIN_PX)
            if clipped is None:
                continue
            win, ptmpl, xa, ya, left, top, _, _ = clipped
            win_ori = orientation_maps(win) if match == MATCH_ORIENT else None
            R = zncc_maps(win, ptmpl, match, sea_ori=win_ori)
            _, mx, _, mloc = cv2.minMaxLoc(R)
            if best is None or mx > best[0]:
                dx, dy = subpixel(R, mloc[0], mloc[1])
                best = (float(mx),
                        xa + mloc[0] + dx - left + tw / 2.0,
                        ya + mloc[1] + dy - top + tw / 2.0,
                        sc, ang, float(tw))
    if best is None or np.hypot(best[1] - cx, best[2] - cy) > REFINE_MAX_SHIFT_PX:
        return None
    return best


def refine_ecc(ref: np.ndarray, sea: np.ndarray, cx: float, cy: float,
               scale: float, angle: float):
    """Euclidean ECC on 2θ real maps, starting from the ZNCC pose.

    Translation/rotation only — no affine shear fitted to chase a label.
    Discarded if the warp walks off the current lattice site. If the
    100 px box hangs off the PNG, ECC the overlapping rectangle; skip
    ECC when that overlap is below PARTIAL_MIN_FRAC.
    """
    tw = int(round(1000.0 / scale))
    if tw < 8:
        return None
    sil = cad_silhouette(ref)
    tmpl = rotate_template(
        cv2.resize(sil, (tw, tw), interpolation=cv2.INTER_AREA), angle)
    x0 = int(round(cx - tw / 2.0))
    y0 = int(round(cy - tw / 2.0))
    H, W = sea.shape[:2]
    left = max(0, -x0)
    top = max(0, -y0)
    right = max(0, x0 + tw - W)
    bottom = max(0, y0 + tw - H)
    ph, pw = tw - top - bottom, tw - left - right
    min_side = max(8, int(round(PARTIAL_MIN_FRAC * tw)))
    if ph < min_side or pw < min_side:
        return None
    ptmpl = tmpl[top:tw - bottom, left:tw - right]
    cx0, cy0 = x0 + left, y0 + top
    crop = sea[cy0:cy0 + ph, cx0:cx0 + pw]
    t_re, _ = orientation_maps(ptmpl)
    s_re, _ = orientation_maps(crop)

    def _norm(a: np.ndarray):
        a = np.ascontiguousarray(a, dtype=np.float32)
        sd = float(a.std())
        if sd < 1e-6:
            return None
        return (a - float(a.mean())) / sd

    t_n, s_n = _norm(t_re), _norm(s_re)
    if t_n is None or s_n is None:
        return None
    warp = np.eye(2, 3, dtype=np.float32)
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 1e-4)
    try:
        try:
            cv2.findTransformECC(t_n, s_n, warp, cv2.MOTION_EUCLIDEAN,
                                 crit, None, 1)
        except TypeError:
            cv2.findTransformECC(t_n, s_n, warp, cv2.MOTION_EUCLIDEAN, crit)
    except cv2.error:
        return None
    tcx = tw / 2.0 - left
    tcy = tw / 2.0 - top
    mapped = warp @ np.array([tcx, tcy, 1.0], dtype=np.float32)
    nx, ny = float(cx0 + mapped[0]), float(cy0 + mapped[1])
    if np.hypot(nx - cx, ny - cy) > REFINE_MAX_SHIFT_PX:
        return None
    dtheta = float(np.degrees(np.arctan2(warp[1, 0], warp[0, 0])))
    return nx, ny, angle + dtheta


def candidate_patch(sea: np.ndarray, cand: dict, maps: list) -> np.ndarray:
    """2-channel verifier input: (search window, matched template), CROPxCROP."""
    m = maps[cand["map_idx"]]
    tw = m["tw"]
    ox, oy = m.get("origin", (0, 0))
    yi, xi = oy + cand["yi"], ox + cand["xi"]
    win = sea[yi:yi + tw, xi:xi + tw]
    win = cv2.resize(win, (CROP, CROP), interpolation=cv2.INTER_LINEAR)
    tmp = cv2.resize(m["tmpl"], (CROP, CROP), interpolation=cv2.INTER_LINEAR)
    return np.stack([win, tmp])  # uint8 (2, CROP, CROP)


def cand_scalars(cand: dict) -> list[float]:
    return [cand["score"], cand["scale"] / 10.0 - 1.0, cand["angle"] / 2.0]


def is_present(row: dict) -> bool:
    if "found" in row:
        return str(row["found"]).strip() in ("1", "True", "true")
    if "present" in row:
        return str(row["present"]).strip() in ("1", "True", "true")
    return float(row.get("gt_x", 0) or 0) != 0 or float(row.get("gt_y", 0) or 0) != 0


def pair_set(row: dict) -> str:
    return str(row.get("set", row.get("noise_bucket", "?")))


def should_reject(conf: float, second: float, n_cands: int,
                  rank_source: str = "verifier") -> bool:
    """True -> absent (Set C): output found=0. Present A/B almost always pass."""
    if n_cands == 0 or conf <= 0.0:
        return True
    thresh = REJECT_ZNCC if rank_source == "zncc_override" else REJECT_CONF
    return conf < thresh


def _rotate_point(x: float, y: float, angle_deg: float,
                  origin=SEARCH_CENTER) -> tuple[float, float]:
    """Where an unrotated CAD site lands after rotating the search image."""
    M = cv2.getRotationMatrix2D(origin, float(angle_deg), 1.0)
    return (M[0, 0] * x + M[0, 1] * y + M[0, 2],
            M[1, 0] * x + M[1, 1] * y + M[1, 2])


def cad_prior_index(cands: list, prior_xy: tuple[float, float],
                    radius: float = CAD_PRIOR_RADIUS_PX) -> int:
    """SEM candidate nearest the (possibly rotated) CAD-CAD site."""
    px, py = prior_xy
    best_i, best_d = 0, 1e9
    near_i, near_d = None, 1e9
    for i, c in enumerate(cands):
        ex, ey = _rotate_point(px, py, c.get("angle", 0.0))
        d = float(np.hypot(c["cx"] - ex, c["cy"] - ey))
        if d < best_d:
            best_i, best_d = i, d
        if d <= radius and d < near_d:
            near_i, near_d = i, d
    return int(near_i if near_i is not None else best_i)


def zncc_override_index(cands: list, ver_i: int) -> int | None:
    """Return the ZNCC-best index when it is a clearly stronger, different site."""
    if not cands:
        return None
    z_i = int(np.argmax([c["score"] for c in cands]))
    if z_i == ver_i:
        return None
    z, v = cands[z_i], cands[ver_i]
    if (z["score"] - v["score"]) < ZNCC_OVERRIDE_MARGIN:
        return None
    if np.hypot(z["cx"] - v["cx"], z["cy"] - v["cy"]) <= ZNCC_OVERRIDE_SHIFT_PX:
        return None
    return z_i


def apply_rejection(out: dict) -> dict:
    """Gate Set C without changing localization when found=1."""
    out = dict(out)
    margin = float(out["conf"] - out["second"])
    out["margin"] = margin
    reject = should_reject(out["conf"], out["second"], len(out.get("cands", ())),
                           out.get("rank_source", "verifier"))
    out["found"] = 0 if reject else 1
    if reject:
        out["x"] = out["y"] = 0.0
        out["angle"] = 0.0
        out["scale"] = 0.0
    return out


def rejection_metrics(results: list[dict]) -> dict:
    """F1 on found flag; breakdown for Set C absent pairs."""
    tp = fp = tn = fn = 0
    for r in results:
        gt = int(r["present"])
        pred = int(r["pred_found"])
        if pred and gt:
            tp += 1
        elif pred and not gt:
            fp += 1
        elif not pred and not gt:
            tn += 1
        else:
            fn += 1
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    absent = [r for r in results if not r["present"]]
    present = [r for r in results if r["present"]]
    c_reject = sum(1 for r in absent if not r["pred_found"]) / max(len(absent), 1)
    false_reject = sum(1 for r in present if not r["pred_found"]) / max(len(present), 1)
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "precision": prec, "recall_present": rec, "f1": f1,
            "absent_reject_acc": c_reject, "present_false_reject": false_reject,
            "n_absent": len(absent), "n_present": len(present)}


def print_rejection_summary(metrics: dict, title: str):
    print(f"\n  {title}")
    print(f"    absent correct reject: {metrics['tn']}/{metrics['n_absent']} "
          f"({metrics['absent_reject_acc']:.1%})")
    print(f"    present false reject:  {metrics['fn']}/{metrics['n_present']} "
          f"({metrics['present_false_reject']:.1%})")
    print(f"    F1(found): {metrics['f1']:.3f}  "
          f"precision={metrics['precision']:.3f}  recall(present)={metrics['recall_present']:.3f}")


# ---------------------------------------------------------------------------
# Dataset plumbing
# ---------------------------------------------------------------------------
def find_split_dir(data_root: str, name: str) -> str:
    aliases = {"val": ("val", "validation"), "train": ("train",),
               "test": ("test",), "eval": ("eval",)}
    for cand in aliases.get(name, (name,)):
        p = os.path.join(data_root, cand)
        if os.path.isfile(os.path.join(p, "manifest.csv")):
            return p
    raise FileNotFoundError(f"no split '{name}' with manifest.csv under {data_root}")


def load_manifest(split_dir: str) -> list[dict]:
    with open(os.path.join(split_dir, "manifest.csv"), newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"empty manifest in {split_dir}")
    return rows


def resolve_image(split_dir: str, p: str, kind: str) -> str:
    for cand in (p, os.path.join(split_dir, p),
                 os.path.join(split_dir, kind, os.path.basename(p))):
        if os.path.isfile(cand):
            return cand
    raise FileNotFoundError(f"cannot resolve {kind} image: {p}")


def load_pair(split_dir: str, row: dict):
    ref = cv2.imread(resolve_image(split_dir, row["reference_path"], "reference"),
                     cv2.IMREAD_GRAYSCALE)
    sea = cv2.imread(resolve_image(split_dir, row["search_path"], "search"),
                     cv2.IMREAD_GRAYSCALE)
    if ref is None or sea is None:
        raise FileNotFoundError(f"unreadable pair id={row.get('id')}")
    return ref, sea


def pass_table(errs) -> dict:
    e = np.asarray(errs, dtype=np.float64)
    out = {f"pass@{t}": float((e <= t).mean()) for t in PASS_THRESHOLDS}
    out.update(median=float(np.median(e)), mean=float(e.mean()),
               worst=float(e.max()), n=int(len(e)))
    return out


def print_bucket_table(per_bucket: dict, title: str):
    print(f"\n  {title}")
    hdr = f"  {'bucket':<14} {'n':>4} " + " ".join(f"p@{t:<2}" for t in PASS_THRESHOLDS) \
          + "  median   mean  worst"
    print(hdr)
    for b, m in per_bucket.items():
        cells = " ".join(f"{m[f'pass@{t}']:.2f}" for t in PASS_THRESHOLDS)
        print(f"  {b:<14} {m['n']:>4} {cells}  {m['median']:6.1f} {m['mean']:6.1f} {m['worst']:6.0f}")


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Stage 1 command: validate proposal recall (Phase 2 grid)
# ---------------------------------------------------------------------------
def _stage1_record(row: dict, cands: list, elapsed: float) -> dict:
    gx, gy = float(row["gt_x"]), float(row["gt_y"])
    pset = pair_set(row)
    if not cands:
        return {"set": pset, "present": is_present(row),
                "recall": False, "top1_err": float("inf"), "best_err": float("inf"),
                "n_cands": 0, "time": elapsed}
    errs = [float(np.hypot(c["cx"] - gx, c["cy"] - gy)) for c in cands]
    return {"set": pset, "present": is_present(row),
            "recall": min(errs) <= POS_RADIUS,
            "top1_err": errs[0], "best_err": min(errs),
            "n_cands": len(cands), "time": elapsed}


def _summarize_stage1(recs: list[dict]) -> dict:
    present = [r for r in recs if r["present"]]
    out = {
        "n_total": len(recs),
        "n_present": len(present),
        "n_absent_skipped_metrics": len(recs) - len(present),
        "recall_at_top_n": float(np.mean([r["recall"] for r in present])) if present else 0.0,
        "zncc_top1_pass3": float(np.mean([r["top1_err"] <= 3 for r in present])) if present else 0.0,
        "zncc_top1_pass5": float(np.mean([r["top1_err"] <= 5 for r in present])) if present else 0.0,
        "avg_proposal_time_s": float(np.mean([r["time"] for r in recs])) if recs else 0.0,
        "per_set": {},
    }
    for s in sorted({r["set"] for r in present}):
        sub = [r for r in present if r["set"] == s]
        out["per_set"][s] = {
            "n": len(sub),
            "recall_at_top_n": float(np.mean([r["recall"] for r in sub])),
            "zncc_top1_pass3": float(np.mean([r["top1_err"] <= 3 for r in sub])),
            "zncc_top1_pass5": float(np.mean([r["top1_err"] <= 5 for r in sub])),
            "median_top1_err": float(np.median([r["top1_err"] for r in sub])),
        }
    return out


def stage1(args):
    split_dir = find_split_dir(args.data, args.split)
    rows = load_manifest(split_dir)
    if args.limit:
        rows = rows[:args.limit]
    n_maps = len(SCALES) * len(ANGLES)
    print(f"stage1: Phase 2 proposal recall on {args.split} ({len(rows)} pairs)")
    print(f"  grid: {len(SCALES)} scales {SCALES[0]}..{SCALES[-1]} (step {SCALE_STEP})")
    print(f"        x {len(ANGLES)} angles {ANGLES[0]}..{ANGLES[-1]} (step {ANGLE_STEP})")
    print(f"        = {n_maps} ZNCC maps/pair, top-{TOP_N} after {NMS_RADIUS}px NMS")
    print(f"  metrics on present pairs only (Set C absent pairs skipped for GT error)")

    recs = []
    for row in tqdm(rows):
        ref, sea = load_pair(split_dir, row)
        t0 = time.time()
        cands, _ = propose(ref, sea)
        recs.append(_stage1_record(row, cands, time.time() - t0))

    summary = _summarize_stage1(recs)
    present = [r for r in recs if r["present"]]

    print(f"\n  {'set':<6} {'n':>4}  recall@{TOP_N}  zncc-top1 p@3  zncc-top1 p@5  med top1 err")
    print(f"  {'ALL':<6} {summary['n_present']:>4}  "
          f"{summary['recall_at_top_n']:13.3f}  "
          f"{summary['zncc_top1_pass3']:13.3f}  "
          f"{summary['zncc_top1_pass5']:13.3f}")
    for s, m in summary["per_set"].items():
        print(f"  {s:<6} {m['n']:>4}  "
              f"{m['recall_at_top_n']:13.3f}  "
              f"{m['zncc_top1_pass3']:13.3f}  "
              f"{m['zncc_top1_pass5']:13.3f}  {m['median_top1_err']:8.1f}")

    t_avg = summary["avg_proposal_time_s"]
    print(f"\n  avg proposal time: {t_avg:.2f}s/pair  ({n_maps} maps/pair)")
    print(f"  absent pairs in split: {summary['n_absent_skipped_metrics']} (not scored)")

    # Exit criteria for Stage 1
    set_a = summary["per_set"].get("A", {})
    set_b = summary["per_set"].get("B", {})
    checks = [
        (f"recall@{TOP_N} (present)", summary["recall_at_top_n"], STAGE1_RECALL_TARGET, ">="),
        ("Set A zncc-top1 pass@3", set_a.get("zncc_top1_pass3", 0.0), STAGE1_SET_A_PASS3_TARGET, ">="),
        ("Set B zncc-top1 pass@3", set_b.get("zncc_top1_pass3", 0.0), STAGE1_SET_B_PASS3_TARGET, ">="),
        ("proposal time (s/pair)", t_avg, STAGE1_TIME_TARGET_S, "<="),
    ]
    print("\n  Stage 1 exit criteria (present pairs, ZNCC-top1 baseline):")
    all_pass = True
    for name, val, target, op in checks:
        ok = val >= target if op == ">=" else val <= target
        mark = "PASS" if ok else "FAIL"
        if not ok:
            all_pass = False
        if op == ">=":
            print(f"    [{mark}] {name}: {val:.3f}  (need >= {target:.2f})")
        else:
            print(f"    [{mark}] {name}: {val:.3f}  (need <= {target:.2f})")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / f"stage1_metrics_{args.split}.json"
    payload = {"split": args.split, "grid": {"scales": SCALES, "angles": ANGLES},
               "summary": summary, "exit_pass": all_pass, "checks": [
                   {"name": n, "value": v, "target": t, "op": o}
                   for n, v, t, o in checks]}
    with open(metrics_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n  metrics -> {metrics_path}")
    if all_pass:
        print("  Stage 1 COMPLETE — proceed to Stage 2 (retrain verifier).")
    else:
        print("  Stage 1 not yet at target — review grid/runtime before Stage 2.")
    print("  Next: python train.py stage3 --data splits --out model --split val")
    print("        (end-to-end with existing verifier; expect gains from proposals)")


# ---------------------------------------------------------------------------
# Stage 2: dump candidates, train verifier
# ---------------------------------------------------------------------------
def dump_candidates(data_root: str, split: str, out_dir: Path,
                    limit: int | None, redump: bool):
    npy_path = out_dir / f"cands_{split}.npy"
    csv_path = out_dir / f"cands_{split}.csv"
    if npy_path.is_file() and csv_path.is_file() and not redump:
        print(f"  reusing {npy_path.name} / {csv_path.name} (use --redump to rebuild)")
        return npy_path, csv_path

    split_dir = find_split_dir(data_root, split)
    rows = load_manifest(split_dir)
    if limit:
        rows = rows[:limit]
    print(f"  dumping candidates for {split} ({len(rows)} pairs)")

    crops = np.empty((len(rows) * TOP_N, 2, CROP, CROP), dtype=np.uint8)
    meta, count = [], 0
    for pi, row in enumerate(tqdm(rows)):
        ref, sea = load_pair(split_dir, row)
        gx, gy = float(row["gt_x"]), float(row["gt_y"])
        cands, maps = propose(ref, sea)
        errs = [float(np.hypot(c["cx"] - gx, c["cy"] - gy)) for c in cands]
        pos_i = int(np.argmin(errs)) if errs and min(errs) <= POS_RADIUS else -1
        for ci, (c, err) in enumerate(zip(cands, errs)):
            crops[count] = candidate_patch(sea, c, maps)
            meta.append({
                "row": count, "pair": pi, "id": row.get("id", pi),
                "bucket": row.get("noise_bucket", "?"),
                "cx": c["cx"], "cy": c["cy"], "score": c["score"],
                "scale": c["scale"], "angle": c["angle"], "err": err,
                "is_pos": int(ci == pos_i),
                # other candidates inside POS_RADIUS are neither pos nor neg
                "is_ignore": int(ci != pos_i and err <= POS_RADIUS),
                "gt_x": gx, "gt_y": gy,
            })
            count += 1

    np.save(npy_path, crops[:count])
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(meta[0].keys()))
        w.writeheader()
        w.writerows(meta)
    n_pos = sum(m["is_pos"] for m in meta)
    print(f"  saved {count} candidates ({n_pos}/{len(rows)} pairs have a positive)")
    return npy_path, csv_path


def read_cand_csv(csv_path: Path) -> list[dict]:
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in ("row", "pair", "is_pos", "is_ignore"):
            r[k] = int(r[k])
        for k in ("cx", "cy", "score", "scale", "angle", "err", "gt_x", "gt_y"):
            r[k] = float(r[k])
    return rows


def group_pairs(meta: list[dict]) -> list[dict]:
    pairs = {}
    for m in meta:
        pairs.setdefault(m["pair"], []).append(m)
    out = []
    for pi in sorted(pairs):
        cands = pairs[pi]
        pos = next((c for c in cands if c["is_pos"]), None)
        negs = [c for c in cands if not c["is_pos"] and not c["is_ignore"]]
        out.append({"pair": pi, "cands": cands, "pos": pos, "negs": negs,
                    "bucket": cands[0]["bucket"],
                    "gt": (cands[0]["gt_x"], cands[0]["gt_y"])})
    return out


class Verifier(nn.Module):
    """Scores one (search window, template) crop: is this the true location?"""

    def __init__(self):
        super().__init__()
        def block(ci, co, stride):
            return [nn.Conv2d(ci, co, 3, stride, 1, bias=False),
                    nn.BatchNorm2d(co), nn.ReLU(inplace=True)]
        self.features = nn.Sequential(
            *block(2, 32, 2),     # 64
            *block(32, 64, 2),    # 32
            *block(64, 64, 1),
            *block(64, 96, 2),    # 16
            *block(96, 128, 2),   # 8
        )
        self.head = nn.Sequential(
            nn.Linear(128 + 3, 64), nn.ReLU(inplace=True), nn.Linear(64, 1))

    def forward(self, x, scalars):
        f = self.features(x).mean(dim=(2, 3))
        return self.head(torch.cat([f, scalars], dim=1)).squeeze(1)


def augment_channel(x: np.ndarray) -> np.ndarray:
    if random.random() < 0.7:
        x = np.clip(x * random.uniform(0.8, 1.2) + random.uniform(-0.08, 0.08), 0.0, 1.0)
    if random.random() < 0.5:
        x = np.clip(x + np.random.randn(*x.shape).astype(np.float32)
                    * random.uniform(0.005, 0.03), 0.0, 1.0)
    return x


class CandidateSet(Dataset):
    """One item = one training pair: positive crop first, then hard negatives."""

    def __init__(self, npy_path: Path, pairs: list[dict], train: bool):
        self.npy_path = npy_path
        self.pairs = [p for p in pairs if p["pos"] is not None and p["negs"]]
        self.train = train
        self._mm = None

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        if self._mm is None:
            self._mm = np.load(self.npy_path, mmap_mode="r")
        p = self.pairs[i]
        n_neg = K_TRAIN - 1
        negs = (random.sample(p["negs"], n_neg) if len(p["negs"]) >= n_neg
                else p["negs"] + random.choices(p["negs"], k=n_neg - len(p["negs"])))
        rows = [p["pos"]] + negs
        crops = np.stack([self._mm[r["row"]] for r in rows]).astype(np.float32) / 255.0
        if self.train:
            for j in range(crops.shape[0]):
                for c in range(2):
                    crops[j, c] = augment_channel(crops[j, c])
        scal = np.array([[r["score"], r["scale"] / 10.0 - 1.0, r["angle"] / 2.0]
                         for r in rows], dtype=np.float32)
        return torch.from_numpy(crops), torch.from_numpy(scal)


@torch.no_grad()
def rank_validation(model, device, npy_path: Path, pairs: list[dict]):
    """Pick argmax-scored candidate per pair; compare against GT and ZNCC top1."""
    model.eval()
    mm = np.load(npy_path, mmap_mode="r")
    v_err, z_err, buckets = [], [], []
    for p in pairs:
        cands = p["cands"]
        crops = torch.from_numpy(
            np.stack([mm[c["row"]] for c in cands]).astype(np.float32) / 255.0)
        scal = torch.tensor([[c["score"], c["scale"] / 10.0 - 1.0, c["angle"] / 2.0]
                             for c in cands], dtype=torch.float32)
        logits = model(crops.to(device), scal.to(device)).cpu().numpy()
        pick = cands[int(np.argmax(logits))]
        ztop = max(cands, key=lambda c: c["score"])
        gx, gy = p["gt"]
        v_err.append(float(np.hypot(pick["cx"] - gx, pick["cy"] - gy)))
        z_err.append(float(np.hypot(ztop["cx"] - gx, ztop["cy"] - gy)))
        buckets.append(p["bucket"])
    return np.array(v_err), np.array(z_err), buckets


def stage2(args):
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = pick_device()
    print(f"stage2: verifier training  (device: {device.type}, out: {out_dir})")

    train_npy, train_csv = dump_candidates(args.data, "train", out_dir,
                                           args.limit, args.redump)
    val_npy, val_csv = dump_candidates(args.data, "val", out_dir, None, args.redump)

    train_pairs = group_pairs(read_cand_csv(train_csv))
    val_pairs = group_pairs(read_cand_csv(val_csv))
    train_ds = CandidateSet(train_npy, train_pairs, train=True)
    usable = len(train_ds)
    print(f"  train pairs: {len(train_pairs)} total, {usable} with a positive "
          f"({usable / max(len(train_pairs), 1):.0%}) -> "
          f"~{usable * (K_TRAIN - 1)} hard negatives per epoch")

    loader = DataLoader(train_ds, batch_size=args.batch_pairs, shuffle=True,
                        num_workers=0, drop_last=True)
    model = Verifier().to(device)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"  verifier parameters: {n_par / 1e3:.0f}k")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    ckpt_path = out_dir / "verifier.pt"
    hist_path = out_dir / "history.json"
    best_p5, history, since_best = -1.0, [], 0

    for epoch in range(args.epochs):
        model.train()
        t0, tot, nstep = time.time(), 0.0, 0
        pbar = tqdm(loader, desc=f"epoch {epoch + 1}/{args.epochs}")
        for crops, scal in pbar:
            B, K = crops.shape[:2]
            logits = model(crops.view(B * K, 2, CROP, CROP).to(device),
                           scal.view(B * K, 3).to(device)).view(B, K)
            # positive is always index 0 within its own candidate set
            loss = F.cross_entropy(logits, torch.zeros(B, dtype=torch.long,
                                                       device=device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += loss.item()
            nstep += 1
            pbar.set_postfix(loss=f"{tot / nstep:.4f}")
        sched.step()

        v_err, z_err, buckets = rank_validation(model, device, val_npy, val_pairs)
        p5 = float((v_err <= 5).mean())
        zp5 = float((z_err <= 5).mean())
        print(f"  val end-to-end: verifier p@5={p5:.3f} med={np.median(v_err):.1f}px"
              f"  |  zncc-top1 p@5={zp5:.3f}  ({time.time() - t0:.0f}s)")
        history.append({"epoch": epoch, "loss": tot / max(nstep, 1),
                        "val_pass@5": p5, "zncc_pass@5": zp5,
                        "val_median": float(np.median(v_err))})
        with open(hist_path, "w") as f:
            json.dump(history, f, indent=2)

        if p5 > best_p5:
            best_p5, since_best = p5, 0
            torch.save({"model": model.state_dict(),
                        "config": {"crop": CROP, "scales": SCALES,
                                   "angles": ANGLES, "top_n": TOP_N}}, ckpt_path)
            print(f"  ** new best (val p@5 {p5:.3f}) -> {ckpt_path}")
        else:
            since_best += 1
            if since_best >= args.patience:
                print(f"  early stop: no val improvement for {args.patience} epochs")
                break

    # final per-bucket comparison with the best checkpoint
    model.load_state_dict(torch.load(ckpt_path, map_location=device)["model"])
    v_err, z_err, buckets = rank_validation(model, device, val_npy, val_pairs)
    per_v = {"ALL": pass_table(v_err)}
    per_z = {"ALL": pass_table(z_err)}
    for b in sorted(set(buckets)):
        idx = [i for i, x in enumerate(buckets) if x == b]
        per_v[b] = pass_table(v_err[idx])
        per_z[b] = pass_table(z_err[idx])
    print_bucket_table(per_z, "val, ZNCC top-1 baseline (candidate-level)")
    print_bucket_table(per_v, "val, VERIFIER pick (candidate-level)")
    print(f"\n  best checkpoint: {ckpt_path}")
    print("  PASS if the verifier beats zncc-top1 clearly (target: p@5 ~0.75+).")
    print("  Then run stage3 for the held-out split with sub-pixel refinement.")


# ---------------------------------------------------------------------------
# Stage 3: full pipeline on a held-out split
# ---------------------------------------------------------------------------
def load_verifier(out_dir: Path, device):
    ckpt_path = out_dir / "verifier.pt"
    if not ckpt_path.is_file():
        return None
    model = Verifier().to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device)["model"])
    model.eval()
    return model


@torch.no_grad()
def localize_pair(ref, sea, model, device, apply_gate: bool = True,
                  prior_xy: tuple[float, float] | None = None,
                  match: str = MATCH_GREY,
                  scales: tuple[float, ...] | None = None,
                  angles: tuple[float, ...] | None = None,
                  lock_scale: bool = False):
    """Full pipeline on one pair -> dict with prediction + diagnostics.

    apply_gate=False keeps the pose even when softmax/ZNCC would reject.
    Phase 3 uses that when CAD-CAD has already decided found=1.

    prior_xy, if given, is the CAD-CAD site in search-image pixels. The
    SEM matcher then only chooses among peaks of that cell instead of
    ranking look-alike DRAM copies across the die.

    match=orientation scores Sobel 2θ maps instead of raw greys, so the
    pose does not depend on SEM dose/gamma/polarity matching the CAD fill.
    """
    scales = SCALES if scales is None else scales
    angles = ANGLES if angles is None else angles
    angle_min = float(min(angles))
    angle_max = float(max(angles))
    scale_min = float(min(scales))
    scale_max = float(max(scales))

    t0 = time.time()
    cands, maps = propose(ref, sea, scales=scales, angles=angles,
                          match=match, prior_xy=prior_xy)
    t_prop = time.time() - t0

    def _refine(px, py, sc0, ang0):
        best = refine_pose(ref, sea, px, py, sc0, ang0, match=match,
                           scale_min=scale_min, scale_max=scale_max,
                           angle_min=angle_min, angle_max=angle_max,
                           lock_scale=lock_scale)
        if best is not None:
            _, px, py, sc0, ang0, tw = best
        else:
            tw = float(int(round(1000.0 / sc0)))
        if match == MATCH_ORIENT:
            ecc = refine_ecc(ref, sea, px, py, sc0, ang0)
            if ecc is not None:
                px, py, ang0 = ecc
        return px, py, sc0, ang0, tw

    t0 = time.time()
    # No ZNCC peaks (flat/NaN maps): still print a coordinate. Spec tie-break
    # is the search-image center when nothing else ranks.
    if not cands:
        px, py = (prior_xy if prior_xy is not None else SEARCH_CENTER)
        scale_out, angle_out, box_w = float(scales[len(scales) // 2]), 0.0, 100.0
        if prior_xy is not None:
            px, py, scale_out, angle_out, box_w = _refine(
                px, py, scale_out, angle_out)
        out = {"x": px, "y": py,
               "conf": 0.0, "second": 0.0, "scale": scale_out, "angle": angle_out,
               "box_w": box_w, "cands": cands, "t_prop": t_prop,
               "t_rank": time.time() - t0, "rank_source": "cad_prior" if prior_xy else "none"}
        if apply_gate:
            return apply_rejection(out)
        out["found"] = 1
        out["margin"] = 0.0
        return out
    if prior_xy is not None:
        # Windowed propose already sits on the CAD-CAD cell. Remaining choice
        # is which (scale, angle) matched; every candidate is near its own
        # rotated prior, so distance-to-prior cannot break the tie.
        top = int(np.argmax([c["score"] for c in cands]))
        conf = float(cands[top]["score"])
        rest = [c["score"] for i, c in enumerate(cands) if i != top]
        second = float(max(rest)) if rest else 0.0
        rank_source = "cad_prior"
    elif model is not None:
        crops = torch.from_numpy(np.stack(
            [candidate_patch(sea, c, maps) for c in cands]).astype(np.float32) / 255.0)
        scal = torch.tensor([cand_scalars(c) for c in cands], dtype=torch.float32)
        logits = model(crops.to(device), scal.to(device)).cpu().numpy()
        probs = np.exp(logits - logits.max())
        probs /= probs.sum()
        order = np.argsort(-probs)
        top = order[0]
        # spec tie-break: among near-tied candidates, closest to image center
        tied = [i for i in order if probs[i] >= probs[top] - TIE_PROB_GAP]
        if len(tied) > 1:
            top = min(tied, key=lambda i: np.hypot(cands[i]["cx"] - SEARCH_CENTER[0],
                                                   cands[i]["cy"] - SEARCH_CENTER[1]))
        conf = float(probs[top])
        second = float(probs[order[1]]) if len(order) > 1 else 0.0
        rank_source = "verifier"
        over = zncc_override_index(cands, int(top))
        if over is not None:
            # CAD template: verifier overrode a clearly better ZNCC site.
            ver_zncc = float(cands[int(top)]["score"])
            top = over
            conf = float(cands[top]["score"])
            second = ver_zncc
            rank_source = "zncc_override"
    else:  # classical fallback: ZNCC score only
        top = int(np.argmax([c["score"] for c in cands]))
        conf = float(cands[top]["score"])
        second = float(sorted((c["score"] for c in cands), reverse=True)[1]) \
            if len(cands) > 1 else 0.0
        rank_source = "zncc"
    t_rank = time.time() - t0

    c = cands[top]
    R = maps[c["map_idx"]]["R"]
    dx, dy = subpixel(R, c["xi"], c["yi"])
    tw = maps[c["map_idx"]]["tw"]
    origin = maps[c["map_idx"]].get("origin", (0, 0))
    px = origin[0] + c["xi"] + dx + tw / 2.0
    py = origin[1] + c["yi"] + dy + tw / 2.0
    scale_out, angle_out, box_w = c["scale"], c["angle"], float(tw)

    # Pose columns are zeroed on a reject, so refining those pairs buys nothing
    # unless CAD-CAD already certified presence.
    gate = REJECT_ZNCC if rank_source == "zncc_override" else REJECT_CONF
    if (not apply_gate) or conf >= gate:
        t0 = time.time()
        px, py, scale_out, angle_out, box_w = _refine(
            px, py, c["scale"], c["angle"])
        t_rank += time.time() - t0

    out = {"x": px, "y": py, "conf": conf, "second": second,
           "scale": scale_out, "angle": angle_out, "box_w": box_w,
           "cands": cands, "t_prop": t_prop, "t_rank": t_rank,
           "rank_source": rank_source}
    if apply_gate:
        return apply_rejection(out)
    out["found"] = 1
    out["margin"] = float(conf - second)
    return out


def save_failure_panel(sea, gt, pred, box_w, err, path: Path):
    img = cv2.cvtColor(sea, cv2.COLOR_GRAY2BGR)
    for (cx, cy), color in ((gt, (0, 255, 0)), (pred, (0, 0, 255))):
        h = box_w / 2.0
        cv2.rectangle(img, (int(cx - h), int(cy - h)), (int(cx + h), int(cy + h)),
                      color, 2)
    cv2.putText(img, f"err={err:.1f}px  green=GT red=pred", (12, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    cv2.imwrite(str(path), img)


def stage3(args):
    out_dir = Path(args.out)
    device = pick_device()
    model = load_verifier(out_dir, device)
    mode = "verifier" if model is not None else "ZNCC fallback (no verifier.pt!)"
    split_dir = find_split_dir(args.data, args.split)
    rows = load_manifest(split_dir)
    if args.limit:
        rows = rows[:args.limit]
    print(f"stage3: full pipeline on {args.split} ({len(rows)} pairs, "
          f"ranking: {mode}, device: {device.type})")

    results = []
    for row in tqdm(rows):
        ref, sea = load_pair(split_dir, row)
        gx, gy = float(row["gt_x"]), float(row["gt_y"])
        out = localize_pair(ref, sea, model, device)
        present = int(is_present(row))
        pred_found = int(out["found"])
        # Localization error: only meaningful when GT is present and we predict found.
        if present and pred_found:
            err = float(np.hypot(out["x"] - gx, out["y"] - gy))
        elif present and not pred_found:
            err = float("inf")
        else:
            err = 0.0 if not pred_found else float(np.hypot(out["x"] - gx, out["y"] - gy))
        best_cand = min((np.hypot(c["cx"] - gx, c["cy"] - gy) for c in out["cands"]),
                        default=float("inf"))
        results.append({
            "id": row.get("id", ""), "set": pair_set(row),
            "bucket": row.get("noise_bucket", "?"),
            "present": present, "pred_found": pred_found,
            "gt_x": gx, "gt_y": gy, "pred_x": out["x"], "pred_y": out["y"],
            "err": err, "conf": out["conf"], "margin": out["margin"],
            "scale": out["scale"], "angle": out["angle"], "box_w": out["box_w"],
            "best_cand_err": float(best_cand),
            "t_prop": out["t_prop"], "t_rank": out["t_rank"],
        })

    rej = rejection_metrics(results)
    print_rejection_summary(rej, f"{args.split}, Set C rejection (found flag)")

    # Per-set rejection on absent (Set C) only
    absent_by_set = {}
    for r in results:
        if not r["present"]:
            s = r["set"]
            absent_by_set.setdefault(s, {"n": 0, "reject": 0})
            absent_by_set[s]["n"] += 1
            absent_by_set[s]["reject"] += int(not r["pred_found"])
    if absent_by_set:
        print(f"  absent pairs by set:")
        for s in sorted(absent_by_set):
            m = absent_by_set[s]
            print(f"    {s}: reject {m['reject']}/{m['n']} "
                  f"({m['reject']/m['n']:.1%})")

    # Localization on present pairs that we accepted (pred_found=1)
    loc_results = [r for r in results if r["present"] and r["pred_found"]]
    errs = np.array([r["err"] for r in loc_results]) if loc_results else np.array([])
    buckets = [r["bucket"] for r in loc_results]
    if len(errs):
        per = {"ALL": pass_table(errs)}
        for b in sorted(set(buckets)):
            per[b] = pass_table(errs[[i for i, x in enumerate(buckets) if x == b]])
    else:
        per = {}

    # Per-set A/B on present pairs accepted (found=1); A/B coords unchanged vs no gate.
    per_set = {}
    for s in sorted({r["set"] for r in loc_results}):
        sub = [r for r in loc_results if r["set"] == s]
        per_set[s] = pass_table([r["err"] for r in sub])
    if per_set:
        print_bucket_table(per_set,
                           f"{args.split}, present + pred found=1 by set (A/B)")

    if per:
        print_bucket_table(per, f"{args.split}, present + pred found=1 (all)")

    present_results = [r for r in results if r["present"]]
    rec_present = np.mean([r["best_cand_err"] <= POS_RADIUS
                           for r in present_results]) if present_results else 0.0
    rec_all = np.mean([r["best_cand_err"] <= POS_RADIUS for r in results])
    tp = np.mean([r["t_prop"] for r in results])
    tr = np.mean([r["t_rank"] for r in results])
    print(f"\n  proposal recall@{TOP_N} (present only): {rec_present:.3f}")
    print(f"  proposal recall@{TOP_N} (all pairs):      {rec_all:.3f}")
    print(f"  time/pair: {tp + tr:.2f}s (propose {tp:.2f}s + rank {tr:.2f}s)")

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"results_{args.split}.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)

    # failure panels for the explainability slide
    fails = sorted((r for r in results if r["err"] > 5.0),
                   key=lambda r: -r["err"])[:args.failures]
    if fails:
        fdir = out_dir / f"failures_{args.split}"
        fdir.mkdir(exist_ok=True)
        by_id = {str(row.get("id", i)): row for i, row in enumerate(rows)}
        for r in fails:
            row = by_id[str(r["id"])]
            _, sea = load_pair(split_dir, row)
            save_failure_panel(sea, (r["gt_x"], r["gt_y"]),
                               (r["pred_x"], r["pred_y"]), r["box_w"], r["err"],
                               fdir / f"{r['id']}_{r['bucket']}.png")
        print(f"  {len(fails)} failure panels -> {fdir}")

    metrics = {"split": args.split, "mode": mode, "per_bucket": per,
               "per_set_present": per_set,
               "rejection": rej,
               "reject_conf": REJECT_CONF, "reject_margin": REJECT_MARGIN,
               "proposal_recall_present": float(rec_present),
               "proposal_recall_all": float(rec_all),
               "time_per_pair_s": float(tp + tr)}
    with open(out_dir / f"metrics_{args.split}.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  metrics -> {out_dir / f'metrics_{args.split}.json'}")


# ---------------------------------------------------------------------------
# Sponsor interface: one pair -> "x y"
# ---------------------------------------------------------------------------
def localize(args):
    ref = cv2.imread(args.reference, cv2.IMREAD_GRAYSCALE)
    sea = cv2.imread(args.search, cv2.IMREAD_GRAYSCALE)
    if ref is None or sea is None:
        raise SystemExit("could not read reference or search image")
    device = pick_device()
    model = None if args.plain else load_verifier(Path(args.out), device)
    out = localize_pair(ref, sea, model, device)
    print(f"{out['x']:.2f} {out['y']:.2f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("stage1", help="validate Phase 2 ZNCC proposal recall")
    p1.add_argument("--data", default="splits")
    p1.add_argument("--split", default="val")
    p1.add_argument("--out", default="model")
    p1.add_argument("--limit", type=int, default=0)

    p2 = sub.add_parser("stage2", help="dump candidates + train verifier")
    p2.add_argument("--data", default="splits")
    p2.add_argument("--out", default="model")
    p2.add_argument("--epochs", type=int, default=20)
    p2.add_argument("--batch-pairs", type=int, default=8)
    p2.add_argument("--lr", type=float, default=3e-4)
    p2.add_argument("--patience", type=int, default=5)
    p2.add_argument("--limit", type=int, default=0,
                    help="dump only the first N train pairs (smoke test)")
    p2.add_argument("--redump", action="store_true")

    p3 = sub.add_parser("stage3", help="full pipeline metrics on a split")
    p3.add_argument("--data", default="splits")
    p3.add_argument("--out", default="model")
    p3.add_argument("--split", default="eval")
    p3.add_argument("--limit", type=int, default=0)
    p3.add_argument("--failures", type=int, default=5)

    pl = sub.add_parser("localize", help="one pair -> print 'x y'")
    pl.add_argument("--reference", required=True)
    pl.add_argument("--search", required=True)
    pl.add_argument("--out", default="model")
    pl.add_argument("--plain", action="store_true",
                    help="skip the verifier, use ZNCC + sub-pixel only")

    args = ap.parse_args()
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    {"stage1": stage1, "stage2": stage2, "stage3": stage3,
     "localize": localize}[args.cmd](args)


if __name__ == "__main__":
    main()
