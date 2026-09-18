#!/usr/bin/env python3
"""Inference: one reference + one search image -> centre (x, y).

Does not train or overwrite weights.

One pair (prints x y):
  python localize.py --reference REF.png --search SEARCH.png

Batch folder (reference/ + search/, matching names):
  python localize.py --data /path/to/split

Batch CSV (paths + ids; extra columns ignored):
  python localize.py --pairs-csv list.csv

Reusable function:
  from localize import predict
  predict("REF.png", "SEARCH.png")
  -> sample_id, pred_x, pred_y, confidence, inference_time_ms
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import cv2

import train

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
CSV_FIELDS = ("sample_id", "pred_x", "pred_y", "confidence", "inference_time_ms")
ID_KEYS = ("sample_id", "id", "sample", "name")
REF_KEYS = ("reference_path", "reference", "ref", "reference_image", "reference_image_path")
SEA_KEYS = ("search_path", "search", "search_image", "search_image_path")


def default_weight_dir() -> Path:
    return Path(__file__).resolve().parent / "model"


def load_model(weights_dir: Path | None = None):
    device = train.pick_device()
    model = train.load_verifier(Path(weights_dir or default_weight_dir()), device)
    return model, device


def predict(reference_image_path, search_image_path, model=None, device=None,
            sample_id: str | None = None, weights_dir=None) -> dict:
    """Run the matcher on one pair. No training.

    Returns sample_id, pred_x, pred_y, found, confidence, inference_time_ms.
    Origin is the top-left of the search image.
    """
    if model is None or device is None:
        model, device = load_model(weights_dir)
    ref_path, sea_path = Path(reference_image_path), Path(search_image_path)
    ref = cv2.imread(str(ref_path), cv2.IMREAD_GRAYSCALE)
    sea = cv2.imread(str(sea_path), cv2.IMREAD_GRAYSCALE)
    if ref is None or sea is None:
        raise FileNotFoundError(f"could not read {ref_path} or {sea_path}")
    t0 = time.perf_counter()
    out = train.localize_pair(ref, sea, model, device)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    sid = sample_id if sample_id is not None else ref_path.stem
    return {
        "sample_id": str(sid),
        "pred_x": float(out["x"]),
        "pred_y": float(out["y"]),
        "found": int(out["found"]),
        "pred_theta": float(out["angle"]),
        "pred_scale": float(out["scale"]),
        "confidence": float(out["conf"]),
        "inference_time_ms": float(elapsed_ms),
    }


def format_xy(row: dict) -> str:
    return f"{row['pred_x']:.2f} {row['pred_y']:.2f}"


def format_csv_row(row: dict) -> str:
    return (
        f"{row['sample_id']},{row['pred_x']:.4f},{row['pred_y']:.4f},"
        f"{row['confidence']:.6f},{row['inference_time_ms']:.2f}"
    )


def find_pair_dirs(root: Path) -> tuple[Path, Path]:
    root = root.resolve()
    candidates = [root]
    for name in ("test", "eval", "train", "val", "optical", "dataset"):
        candidates.append(root / name)
        candidates.append(root / name / name)
    for base in candidates:
        ref_dir, sea_dir = base / "reference", base / "search"
        if ref_dir.is_dir() and sea_dir.is_dir():
            return ref_dir, sea_dir
    raise SystemExit(
        f"no reference/ and search/ folders under {root}\n"
        "expected: FOLDER/reference/*.png and FOLDER/search/*.png"
    )


def list_pairs(ref_dir: Path, sea_dir: Path) -> list[tuple[str, Path, Path]]:
    refs = sorted(
        p for p in ref_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )
    pairs, missing = [], []
    for ref_path in refs:
        sea_path = sea_dir / ref_path.name
        if sea_path.is_file():
            pairs.append((ref_path.stem, ref_path, sea_path))
        else:
            missing.append(ref_path.name)
    if not pairs:
        raise SystemExit(f"no matching image pairs in {ref_dir} and {sea_dir}")
    if missing:
        print(f"skipping {len(missing)} reference files with no matching search image")
    return pairs


def _first_key(row: dict, keys: tuple[str, ...]) -> str | None:
    lower = {k.lower().strip(): v for k, v in row.items() if k}
    for key in keys:
        if key in lower and str(lower[key]).strip():
            return str(lower[key]).strip()
    return None


def list_pairs_from_csv(csv_path: Path) -> list[tuple[str, Path, Path]]:
    base = csv_path.parent
    pairs = []
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"empty CSV: {csv_path}")
    for i, row in enumerate(rows):
        ref = _first_key(row, REF_KEYS)
        sea = _first_key(row, SEA_KEYS)
        if not ref or not sea:
            raise SystemExit(
                f"{csv_path} row {i}: need a reference path column "
                f"({', '.join(REF_KEYS)}) and a search path column "
                f"({', '.join(SEA_KEYS)})"
            )
        sid = _first_key(row, ID_KEYS) or Path(ref).stem
        ref_path = Path(ref) if Path(ref).is_file() else base / ref
        sea_path = Path(sea) if Path(sea).is_file() else base / sea
        pairs.append((sid, ref_path, sea_path))
    return pairs


def run_batch(pairs: list[tuple[str, Path, Path]], model, device,
              output_csv: Path | None) -> None:
    print(",".join(CSV_FIELDS))
    written = []
    for sid, ref_path, sea_path in pairs:
        row = predict(ref_path, sea_path, model=model, device=device, sample_id=sid)
        print(format_csv_row(row))
        written.append(row)
    if output_csv is not None:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(output_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(CSV_FIELDS))
            w.writeheader()
            for row in written:
                w.writerow({
                    "sample_id": row["sample_id"],
                    "pred_x": f"{row['pred_x']:.4f}",
                    "pred_y": f"{row['pred_y']:.4f}",
                    "confidence": f"{row['confidence']:.6f}",
                    "inference_time_ms": f"{row['inference_time_ms']:.2f}",
                })
        print(f"wrote {len(written)} rows -> {output_csv}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reference", help="reference PNG, or a dataset folder")
    ap.add_argument("--search", help="search PNG")
    ap.add_argument("--data", help="folder with reference/ and search/")
    ap.add_argument("--pairs-csv", help="CSV with sample id + image paths")
    ap.add_argument(
        "--output",
        help="optional predictions CSV (sample_id,pred_x,pred_y,confidence,inference_time_ms)",
    )
    ap.add_argument(
        "--out",
        default=str(default_weight_dir()),
        help="folder that contains verifier.pt (read-only)",
    )
    args = ap.parse_args()

    model, device = load_model(Path(args.out))
    output_csv = Path(args.output) if args.output else None

    if args.pairs_csv:
        run_batch(list_pairs_from_csv(Path(args.pairs_csv)), model, device, output_csv)
        return

    data_root = None
    if args.data:
        data_root = Path(args.data)
    elif args.reference and not args.search and Path(args.reference).is_dir():
        data_root = Path(args.reference)

    if data_root is not None:
        if not data_root.is_dir():
            raise SystemExit(f"not a folder: {data_root}")
        ref_dir, sea_dir = find_pair_dirs(data_root)
        run_batch(list_pairs(ref_dir, sea_dir), model, device, output_csv)
        return

    if not args.reference or not args.search:
        raise SystemExit(
            "one pair:  python localize.py --reference REF.png --search SEARCH.png\n"
            "folder:    python localize.py --data /path/to/split\n"
            "csv list:  python localize.py --pairs-csv list.csv"
        )

    row = predict(args.reference, args.search, model=model, device=device)
    print(format_xy(row))
    if output_csv is not None:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(output_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(CSV_FIELDS))
            w.writeheader()
            w.writerow({
                "sample_id": row["sample_id"],
                "pred_x": f"{row['pred_x']:.4f}",
                "pred_y": f"{row['pred_y']:.4f}",
                "confidence": f"{row['confidence']:.6f}",
                "inference_time_ms": f"{row['inference_time_ms']:.2f}",
            })
        print(f"wrote 1 row -> {output_csv}")


if __name__ == "__main__":
    main()
