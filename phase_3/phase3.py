#!/usr/bin/env python3
"""Phase 3 submission entry point.

  python phase3.py --input pairs.csv --output predictions.csv

Same output contract as Phase 2 (one row per pair_id):
  pair_id, x, y, theta, scale, found, score
When found=0, x/y/theta/scale are 0.

Blind pairs.csv columns that are actually used:
  search_path, reference_gds_path, search_gds_path
reference_sem_path and params_json_path are empty on the scored run and
are never read.

found is CAD-CAD: rasterize reference.gds and search.gds with the same
yield paint (no SEM noise) and accept only a real design-window hit.
If found=1, the CAD-CAD site is passed to the GDS->SEM matcher as a
pose prior so it refines that cell instead of a look-alike DRAM copy.
Pose is Sobel 2θ orientation ZNCC (optional ECC), not grey intensity.
If found=0, pose columns are zero. Softmax/ZNCC gates are not used for
presence. A scored row without search.gds is malformed: found=0, no
SEM-confidence fallback.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cad_render
import train
from localize import predict

PRED_FIELDS = ("pair_id", "x", "y", "theta", "scale", "found", "score")
ID_KEYS = ("pair_id", "sample_id", "id", "sample", "name")
# GDS first: on the training split a reference_sem_path also exists, and
# matching against it would exercise a code path the blind run never takes.
REF_KEYS = ("reference_gds_path", "reference_gds", "reference_path",
            "reference", "ref")
SEA_KEYS = ("search_path", "search", "search_image", "search_image_path")
SEARCH_GDS_KEYS = ("search_gds_path", "search_gds", "search_cad_path",
                   "search_cad")

# A pair we cannot read still needs a row: a missing row scores zero, and a
# raised exception loses every row after it too.
FALLBACK_ROW = {"x": "0.0000", "y": "0.0000", "theta": "0.0000",
                "scale": "0.0000", "found": "0", "score": "0.000000"}


def _first_key(row: dict, keys: tuple[str, ...]) -> str | None:
    lower = {k.lower().strip(): v for k, v in row.items() if k}
    for key in keys:
        if key in lower and str(lower[key]).strip():
            return str(lower[key]).strip()
    return None


def _resolve(raw: str, base: Path) -> Path:
    """Locate a file listed in pairs.csv.

    Paths are relative to the dataset root, which is normally the CSV's own
    directory. They may also arrive with Windows separators, which are a
    legal filename character on POSIX and so silently resolve to nothing.
    """
    for cand in (raw, raw.replace("\\", "/")):
        p = Path(cand)
        if p.is_file():
            return p
        if (base / cand).is_file():
            return base / cand
    norm = raw.replace("\\", "/")
    for parent in (base, base.parent):
        hit = parent / Path(norm).name
        if hit.is_file():
            return hit
    return base / norm


def _existing_file(raw: str | None, base: Path) -> Path | None:
    if not raw:
        return None
    path = _resolve(raw, base)
    return path if path.is_file() else None


def _sibling_search_gds(search_path: Path) -> Path | None:
    """Official splits may put the die CAD next to the SEM, not only in CSV."""
    for cand in (
        search_path.with_suffix(".gds"),
        search_path.with_suffix(".gdsii"),
        search_path.parent.parent / "search_gds" / f"{search_path.stem}.gds",
        search_path.parent / "gds" / f"{search_path.stem}.gds",
    ):
        if cand.is_file():
            return cand
    return None


def read_pairs_csv(csv_path: Path) -> list[dict]:
    base = csv_path.parent
    rows_out = []
    with open(csv_path, newline="") as f:
        for i, row in enumerate(csv.DictReader(f)):
            ref = _first_key(row, REF_KEYS)
            sea = _first_key(row, SEA_KEYS)
            if not ref or not sea:
                raise SystemExit(
                    f"{csv_path} row {i}: need a reference GDS column "
                    f"({', '.join(REF_KEYS)}) and a search path column "
                    f"({', '.join(SEA_KEYS)})"
                )
            pid = _first_key(row, ID_KEYS) or Path(ref.replace("\\", "/")).stem
            sea_path = _resolve(sea, base)
            sea_gds = _existing_file(_first_key(row, SEARCH_GDS_KEYS), base)
            if sea_gds is None:
                sea_gds = _sibling_search_gds(sea_path)
            rows_out.append({
                "pair_id": pid,
                "reference_path": _resolve(ref, base),
                "search_path": sea_path,
                "search_gds_path": sea_gds,
            })
    if not rows_out:
        raise SystemExit(f"empty CSV: {csv_path}")
    return rows_out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="pairs.csv from organizers")
    ap.add_argument("--output", required=True, help="predictions.csv to write")
    ap.add_argument("--out", default=None,
                    help="ignored: Phase 3 does not load verifier.pt")
    ap.add_argument("--spot-nm", type=float, default=cad_render.BEAM_SPOT_NM,
                    help="beam PSF sigma used when rendering the CAD reference")
    ap.add_argument("--cd-bias-px", type=float, default=0.0,
                    help="grow (+) or shrink (-) rendered features, in nm at 1 nm/px")
    ap.add_argument("--limit", type=int, default=0,
                    help="process only the first N pairs (0 = all)")
    ap.add_argument("--cad-present-min", type=float,
                    default=cad_render.CAD_PRESENT_MIN,
                    help="CAD-CAD ZNCC threshold for found=1")
    args = ap.parse_args()

    pairs = read_pairs_csv(Path(args.input))
    if args.limit:
        pairs = pairs[:args.limit]
    device = train.pick_device()

    n_cad = sum(1 for r in pairs if r["search_gds_path"] is not None)
    print(f"CAD-CAD presence on {n_cad}/{len(pairs)} pairs "
          f"(min ZNCC {args.cad_present_min:.2f})")
    if n_cad < len(pairs):
        print("ERROR: pairs without search.gds are malformed; "
              "writing found=0 (no SEM confidence fallback)")

    out_rows, failed = [], 0
    for row in pairs:
        try:
            if row["search_gds_path"] is None:
                print(f"ERROR: pair {row['pair_id']} has no search.gds; "
                      "writing found=0")
                failed += 1
                out_rows.append({"pair_id": row["pair_id"], **FALLBACK_ROW})
                continue
            cad = cad_render.cad_cad_presence(
                row["reference_path"], row["search_gds_path"],
                present_min=args.cad_present_min)
            if not cad["found"]:
                out_rows.append({
                    "pair_id": row["pair_id"],
                    "x": "0.0000", "y": "0.0000",
                    "theta": "0.0000", "scale": "0.0000",
                    "found": "0",
                    "score": f"{cad['score']:.6f}",
                })
                continue
            prior = (float(cad["search_x"]), float(cad["search_y"]))
            pred = predict(row["reference_path"], row["search_path"],
                           model=None, device=device, sample_id=row["pair_id"],
                           spot_nm=args.spot_nm, cd_bias_px=args.cd_bias_px,
                           apply_gate=False, prior_xy=prior,
                           match=train.MATCH_ORIENT,
                           scales=train.PHASE3_SCALES,
                           angles=train.PHASE3_ANGLES,
                           lock_scale=True, load_weights=False)
            pred["found"] = 1
            pred["confidence"] = float(cad["score"])
            out_rows.append({
                "pair_id": pred["sample_id"],
                "x": f"{pred['pred_x']:.4f}",
                "y": f"{pred['pred_y']:.4f}",
                "theta": f"{pred['pred_theta']:.4f}",
                "scale": f"{pred['pred_scale']:.4f}",
                "found": str(pred["found"]),
                "score": f"{pred['confidence']:.6f}",
            })
        except Exception as exc:  # one bad pair must not cost every other row
            failed += 1
            print(f"WARNING: pair {row['pair_id']} failed ({exc}); writing found=0")
            out_rows.append({"pair_id": row["pair_id"], **FALLBACK_ROW})

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(PRED_FIELDS))
        w.writeheader()
        w.writerows(out_rows)
    print(f"wrote {len(out_rows)} rows -> {out_path}"
          + (f"  ({failed} fallback rows)" if failed else ""))


if __name__ == "__main__":
    main()
