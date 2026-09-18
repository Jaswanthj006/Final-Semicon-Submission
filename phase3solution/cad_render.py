#!/usr/bin/env python3
"""Phase 3: turn a reference GDSII file into the grayscale raster the
Phase 2 matcher already knows how to consume.

A GDS polygon carries a layer *number*, never a brightness, so the render
has to assign one. Brightness is derived from the layer's position in the
stack via a secondary-electron yield model: a layer higher in the stack has
less material above it attenuating the SE signal, so it reads brighter.
Layers are painted bottom-to-top, so an overlap shows only the topmost
layer -- a real SEM sees the top surface, not what is buried under it.

The only acquisition effect applied is the beam PSF. The search image was
formed as blur(1 nm/px) -> 10x area-downsample, and train.propose() already
does the area-downsample when it cuts each template, so blurring once here
reproduces that chain. Noise is deliberately not added: it would corrupt
the template without making the correlation any more honest.

  python cad_render.py --gds reference/00000.gds --out /tmp/ref.png
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
import gdstk
import numpy as np

REFERENCE_SIZE_PX = 1000
PIXEL_SIZE_REF_NM = 1.0
SEARCH_SIZE_PX = 1000
PIXEL_SIZE_SEARCH_NM = 10.0
SEARCH_FOV_NM = REFERENCE_SIZE_PX * PIXEL_SIZE_SEARCH_NM  # 10 um die
# CAD-CAD is scored on a clean design raster, not the SEM. 5 nm/px keeps
# clipped window edges from dropping a true crop into the absent-DRAM band.
PIXEL_SIZE_CAD_NM = 5.0
CAD_MAX_PX = 2048

# Illustrative SE yields, matching the generator that produced the data.
BASE_YIELD = 0.20
TOP_YIELD = 0.85
BACKGROUND_YIELD = 0.12

BEAM_SPOT_NM = 5.0

# CAD-CAD presence on DRAM. Peak ZNCC alone is not enough: a decoy array
# of the same pitch can still score ~0.93. After aligning at the ZNCC
# peak, a true crop's yield rasters overlay (MAD a couple of grey
# levels from clip-edge rounding); a lookalike does not.
CAD_PRESENT_MIN = 0.80
# Kept as documentation of the old integer-peak overlay cut. found uses
# neighbourhood min-MAD vs the runner-up, not this constant.
CAD_MAD_MAX = 5.0
CAD_MAD_NEIGHBOR = 1
# Hole around the ZNCC peak so the runner-up is another cell, not the
# same overlay's shoulder. ~one DRAM period at 5 nm/px.
CAD_SECOND_PEAK_RADIUS = 10
CAD_MAD_GAP = 1.0
# Loose backstop for a textureless die, not a 5/7 overlay fit.
CAD_MAD_CAP = 15.0

# Both architectures in the Phase 3 generator draw 8 layers. A clipped
# reference can legitimately be missing its top layers (a crop with no strap
# in it), and inferring the stack depth from max(layer) would then shift the
# whole palette -- layer 5 would be painted with the topmost intensity and
# stop agreeing with the search render. Floor the inferred depth instead.
DEFAULT_NUM_LAYERS = 8

# A user unit is one pixel at 1 nm/px, which is how the generator writes
# them. Only rescale when the coordinates are orders of magnitude off (a
# file written in um or m), never for a merely sparse crop.
_SPAN_MIN_PX = 50.0
_SPAN_MAX_PX = 20000.0


def layer_yield(layer: int, num_layers: int) -> float:
    if num_layers <= 1:
        return TOP_YIELD
    return BASE_YIELD + (TOP_YIELD - BASE_YIELD) * (layer / (num_layers - 1))


def yield_to_intensity(value: float) -> int:
    return int(round(max(0.0, min(1.0, value)) * 255))


def layer_intensity(layer: int, num_layers: int) -> int:
    return yield_to_intensity(layer_yield(layer, num_layers))


def background_intensity() -> int:
    return yield_to_intensity(BACKGROUND_YIELD)


def default_palette(num_layers: int = DEFAULT_NUM_LAYERS) -> dict[int, int]:
    return {i: layer_intensity(i, num_layers) for i in range(num_layers)}


def read_polygons(gds_path) -> list:
    """Every polygon in the file's first top-level cell."""
    lib = gdstk.read_gds(str(gds_path))
    cells = lib.top_level()
    if not cells:
        raise ValueError(f"{gds_path}: no top-level cell")
    polygons = cells[0].get_polygons()
    if not polygons:
        raise ValueError(f"{gds_path}: no polygons")
    return polygons


def _unit_scale(polygons: list, size_px: int) -> float:
    """Pixels per user unit, for a file not written in nm-as-user-units."""
    pts = np.concatenate([p.points for p in polygons])
    span = float(max(np.ptp(pts[:, 0]), np.ptp(pts[:, 1])))
    if span <= 0.0 or _SPAN_MIN_PX <= span <= _SPAN_MAX_PX:
        return 1.0
    return float(10.0 ** round(math.log10(size_px / span)))


def _user_units_to_nm(polygons: list) -> float:
    """gdstk user units -> nm. Generator files already store nm."""
    pts = np.concatenate([p.points for p in polygons])
    span = float(max(np.ptp(pts[:, 0]), np.ptp(pts[:, 1]), pts.max(initial=0.0)))
    if span <= 0.0 or span >= _SPAN_MIN_PX:
        return 1.0
    # Typical um-scale layout: 1-20 um FOV.
    if span <= 50.0:
        return 1000.0
    return float(10.0 ** round(math.log10(SEARCH_FOV_NM / span)))


def apply_cd_bias(img: np.ndarray, bias_px: float) -> np.ndarray:
    """Grow (positive) or shrink (negative) drawn features.

    The search render carries fabrication distortion -- per-polygon size
    outliers and a global CD/etch bias -- while the reference GDS is the
    exact design. A max/min filter over the painted raster closes part of
    that gap without needing to know which way the fab actually went.
    """
    r = int(round(abs(bias_px)))
    if r < 1:
        return img
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    return cv2.dilate(img, kernel) if bias_px > 0 else cv2.erode(img, kernel)


def rasterize_polygons(polygons: list, size_px: int = REFERENCE_SIZE_PX,
                       num_layers: int | None = None,
                       palette: dict[int, int] | None = None) -> np.ndarray:
    """Paint layers bottom-to-top onto a size_px square at 1 nm/px."""
    inferred = max(int(p.layer) for p in polygons) + 1
    if num_layers is None:
        num_layers = max(inferred, DEFAULT_NUM_LAYERS)
    scale = _unit_scale(polygons, size_px)

    by_layer: dict[int, list] = {}
    for poly in polygons:
        pts = poly.points * scale if scale != 1.0 else poly.points
        by_layer.setdefault(int(poly.layer), []).append(
            np.round(pts).astype(np.int32))

    img = np.full((size_px, size_px), background_intensity(), dtype=np.uint8)
    for layer in sorted(by_layer):
        intensity = (palette or {}).get(layer)
        if intensity is None:
            intensity = layer_intensity(layer, num_layers)
        cv2.fillPoly(img, by_layer[layer], int(intensity))
    return img


def rasterize_fov(polygons: list, size_px: int, pixel_size_nm: float,
                  origin_nm: tuple[float, float] = (0.0, 0.0),
                  num_layers: int | None = None,
                  palette: dict[int, int] | None = None) -> np.ndarray:
    """Paint polygons onto a square FOV at `pixel_size_nm` (no beam PSF).

    Used for CAD-CAD: both the 1 um reference window and the 10 um search
    die are drawn with the same yield palette, so presence is a clean
    template match rather than a SEM correlation.
    """
    inferred = max(int(p.layer) for p in polygons) + 1
    if num_layers is None:
        num_layers = max(inferred, DEFAULT_NUM_LAYERS)
    unit = _user_units_to_nm(polygons)
    origin = np.asarray(origin_nm, dtype=np.float64)

    by_layer: dict[int, list] = {}
    for poly in polygons:
        pts_nm = np.asarray(poly.points, dtype=np.float64) * unit
        pts_px = (pts_nm - origin) / pixel_size_nm
        by_layer.setdefault(int(poly.layer), []).append(
            np.round(pts_px).astype(np.int32))

    img = np.full((size_px, size_px), background_intensity(), dtype=np.uint8)
    for layer in sorted(by_layer):
        intensity = (palette or {}).get(layer)
        if intensity is None:
            intensity = layer_intensity(layer, num_layers)
        cv2.fillPoly(img, by_layer[layer], int(intensity))
    return img


def _search_canvas(polygons: list) -> tuple[int, float]:
    """Die raster size and nm/px. Crop-sized files stay at 1 nm/px."""
    unit = _user_units_to_nm(polygons)
    pts = np.concatenate([p.points for p in polygons]) * unit
    extent = float(max(pts[:, 0].max(), pts[:, 1].max(), 1.0))
    if extent < 2.5 * REFERENCE_SIZE_PX * PIXEL_SIZE_REF_NM:
        return REFERENCE_SIZE_PX, PIXEL_SIZE_REF_NM
    size_px = int(round(max(extent, SEARCH_FOV_NM) / PIXEL_SIZE_CAD_NM))
    size_px = min(max(size_px, int(round(SEARCH_FOV_NM / PIXEL_SIZE_CAD_NM))), CAD_MAX_PX)
    return size_px, PIXEL_SIZE_CAD_NM


def _window_mad(sea: np.ndarray, tmpl: np.ndarray, xi: int, yi: int) -> float | None:
    th, tw = tmpl.shape
    if yi < 0 or xi < 0 or yi + th > sea.shape[0] or xi + tw > sea.shape[1]:
        return None
    win = sea[yi:yi + th, xi:xi + tw]
    return float(np.mean(np.abs(tmpl.astype(np.int16) - win.astype(np.int16))))


def _min_mad_neighborhood(sea: np.ndarray, tmpl: np.ndarray, xi: int, yi: int,
                          radius: int = CAD_MAD_NEIGHBOR) -> float | None:
    """Lowest overlay MAD in a (2r+1)^2 box around an integer ZNCC peak."""
    best = None
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            mad = _window_mad(sea, tmpl, xi + dx, yi + dy)
            if mad is None:
                continue
            if best is None or mad < best:
                best = mad
    return best


def cad_cad_presence(reference_gds, search_gds,
                     present_min: float = CAD_PRESENT_MIN) -> dict:
    """Is the reference cell a window of search.gds?

    Rasterizes both DRAM designs with the yield model (no SEM noise),
    locates the 1 um reference window in the 10 um die by ZNCC, then
    accepts only if that peak's overlay MAD (min over a 1 px neighbourhood)
    beats the next peak by a gap. Peak ZNCC alone would accept a same-pitch
    decoy array. Integer ZNCC can sit 1 px off a true overlay.
    """
    ref_polys = read_polygons(reference_gds)
    sea_polys = read_polygons(search_gds)
    sea_px, sea_nm = _search_canvas(sea_polys)
    tmpl_px = max(int(round(REFERENCE_SIZE_PX * PIXEL_SIZE_REF_NM / sea_nm)), 8)
    # Same nm/px as the die raster, so a true crop is an exact window.
    tmpl = rasterize_fov(ref_polys, size_px=tmpl_px, pixel_size_nm=sea_nm)
    sea_img = rasterize_fov(sea_polys, size_px=sea_px, pixel_size_nm=sea_nm)
    sea_match = sea_img

    if tmpl.shape[0] > sea_img.shape[0] or tmpl.shape[1] > sea_img.shape[1]:
        # Search file is itself a crop: one correlation of the two rasters.
        side = min(tmpl.shape[0], sea_img.shape[0])
        tmpl = cv2.resize(tmpl, (side, side), interpolation=cv2.INTER_AREA)
        sea_match = cv2.resize(sea_img, (side, side), interpolation=cv2.INTER_AREA)
    R = cv2.matchTemplate(sea_match, tmpl, cv2.TM_CCOEFF_NORMED)

    empty = {"found": 0, "score": 0.0, "mad": 0.0, "x": 0.0, "y": 0.0,
             "search_x": 0.0, "search_y": 0.0, "pixel_size_nm": sea_nm}
    if not np.isfinite(R).any() or float(np.nanstd(tmpl)) < 1e-3:
        return empty

    peak = float(np.nanmax(R))
    yi, xi = np.unravel_index(int(np.nanargmax(R)), R.shape)
    th, tw = tmpl.shape
    mad_best = _min_mad_neighborhood(sea_match, tmpl, xi, yi)
    if mad_best is None:
        return empty

    yy, xx = np.ogrid[:R.shape[0], :R.shape[1]]
    same = (xx - xi) ** 2 + (yy - yi) ** 2 <= CAD_SECOND_PEAK_RADIUS ** 2
    rest = np.where(same | ~np.isfinite(R), -np.inf, R)
    if np.isfinite(rest).any():
        y2, x2 = np.unravel_index(int(np.argmax(rest)), rest.shape)
        mad_second = _min_mad_neighborhood(sea_match, tmpl, int(x2), int(y2))
        if mad_second is None:
            mad_second = float("inf")
    else:
        mad_second = float("inf")

    cx = float(xi + tw / 2.0)
    cy = float(yi + th / 2.0)
    sx, sy = to_search_px(cx, cy, sea_nm)
    found = int(
        peak >= present_min
        and mad_best < CAD_MAD_CAP
        and mad_best < mad_second - CAD_MAD_GAP
    )
    return {
        "found": found,
        "score": peak,
        "mad": mad_best,
        "x": cx,
        "y": cy,
        "search_x": sx,
        "search_y": sy,
        "pixel_size_nm": sea_nm,
    }


def to_search_px(x: float, y: float, pixel_size_nm: float) -> tuple[float, float]:
    """CAD-raster pixels -> search-image pixels (10 nm/px)."""
    s = float(pixel_size_nm) / PIXEL_SIZE_SEARCH_NM
    return float(x) * s, float(y) * s


def beam_blur(img: np.ndarray, spot_nm: float = BEAM_SPOT_NM,
              pixel_size_nm: float = PIXEL_SIZE_REF_NM) -> np.ndarray:
    sigma = max(spot_nm / pixel_size_nm, 1e-6)
    k = max(int(2 * round(3 * sigma) + 1), 3)
    return cv2.GaussianBlur(img, (k, k), sigmaX=sigma, sigmaY=sigma)


def render_gds(gds_path, spot_nm: float = BEAM_SPOT_NM,
               size_px: int = REFERENCE_SIZE_PX, cd_bias_px: float = 0.0,
               num_layers: int | None = None,
               palette: dict[int, int] | None = None) -> np.ndarray:
    """Reference GDS -> uint8 raster ready for train.localize_pair()."""
    img = rasterize_polygons(read_polygons(gds_path), size_px=size_px,
                             num_layers=num_layers, palette=palette)
    img = apply_cd_bias(img, cd_bias_px)
    return beam_blur(img, spot_nm=spot_nm)


def is_gds_path(path) -> bool:
    return str(path).lower().endswith((".gds", ".gds2", ".gdsii"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gds", required=True, help="reference .gds to render")
    ap.add_argument("--out", required=True, help="PNG to write")
    ap.add_argument("--spot-nm", type=float, default=BEAM_SPOT_NM)
    ap.add_argument("--cd-bias-px", type=float, default=0.0)
    args = ap.parse_args()

    img = render_gds(args.gds, spot_nm=args.spot_nm, cd_bias_px=args.cd_bias_px)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), img)
    print(f"{img.shape[1]}x{img.shape[0]} -> {out}")


if __name__ == "__main__":
    main()
