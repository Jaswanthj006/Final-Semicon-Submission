#!/usr/bin/env python3
"""Phase 2 submission entry point.

  python register.py --input pairs.csv --output predictions.csv

predictions.csv columns (one row per pair_id):
  pair_id, x, y, theta, scale, found, score
When found=0, x/y/theta/scale are 0.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from localize import default_weight_dir, load_model, predict

PRED_FIELDS = ("pair_id", "x", "y", "theta", "scale", "found", "score")
ID_KEYS = ("pair_id", "sample_id", "id", "sample", "name")
REF_KEYS = ("reference_path", "reference", "ref", "reference_image", "reference_image_path")
SEA_KEYS = ("search_path", "search", "search_image", "search_image_path")

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
    """Locate an image listed in pairs.csv.

    Paths may arrive with Windows separators, which are a legal filename
    character on POSIX and so silently resolve to nothing.
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


def read_pairs_csv(csv_path: Path) -> list[dict]:
    base = csv_path.parent
    rows_out = []
    with open(csv_path, newline="") as f:
        for i, row in enumerate(csv.DictReader(f)):
            ref = _first_key(row, REF_KEYS)
            sea = _first_key(row, SEA_KEYS)
            if not ref or not sea:
                raise SystemExit(
                    f"{csv_path} row {i}: need reference and search path columns"
                )
            pid = _first_key(row, ID_KEYS) or Path(ref.replace("\\", "/")).stem
            ref_path = _resolve(ref, base)
            sea_path = _resolve(sea, base)
            rows_out.append({
                "pair_id": pid,
                "reference_path": ref_path,
                "search_path": sea_path,
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
                    help="folder with verifier.pt (default: model/ next to this script)")
    args = ap.parse_args()

    pairs = read_pairs_csv(Path(args.input))
    # Resolve relative to this file, not the caller's cwd: a cwd-relative default
    # silently loads no verifier and downgrades every pair to plain ZNCC.
    weights_dir = Path(args.out) if args.out else default_weight_dir()
    model, device = load_model(weights_dir)
    if model is None:
        print(f"WARNING: no verifier.pt under {weights_dir} - falling back to ZNCC")

    out_rows, failed = [], 0
    for row in pairs:
        try:
            pred = predict(row["reference_path"], row["search_path"],
                           model=model, device=device, sample_id=row["pair_id"])
            out_rows.append({
                "pair_id": pred["sample_id"],
                "x": f"{pred['pred_x']:.4f}",
                "y": f"{pred['pred_y']:.4f}",
                "theta": f"{pred['pred_theta']:.4f}",
                "scale": f"{pred['pred_scale']:.4f}",
                "found": str(pred["found"]),
                "score": f"{pred['confidence']:.6f}",
            })
        except Exception as exc:  # one bad pair must not cost the other 199
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
