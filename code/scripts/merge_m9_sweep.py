#!/usr/bin/env python3
"""Merges `results/m9_sweep_<dataset>.csv` (one per dataset, written separately by
`run_m9_sweep.py` so 3 concurrent PBS jobs never race on one shared output file) into the single
`results/m9_sweep.csv` `08_FIX_PLAN.md` §8 names. No computation -- pure concatenation.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "code" / "src"))

from p3fcl import provenance  # noqa: E402

DATASETS = ["cifar100", "cub200", "imagenet_r"]


def main() -> int:
    all_rows = []
    fieldnames = None
    for dataset in DATASETS:
        path = REPO_ROOT / "results" / f"m9_sweep_{dataset}.csv"
        if not path.exists():
            print(f"WARNING: {path} not found -- skipping", file=sys.stderr)
            continue
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = fieldnames or reader.fieldnames
            all_rows.extend(reader)

    if not all_rows:
        print("no per-dataset m9_sweep files found", file=sys.stderr)
        return 1

    out_csv = REPO_ROOT / "results" / "m9_sweep.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(all_rows)
    print(f"merged {len(all_rows)} rows from {len(DATASETS)} per-dataset files into {out_csv}")

    config = {"seed": 0, "purpose": "FX5 merge of per-dataset M9 core-sweep files into m9_sweep.csv"}
    manifest = provenance.run_manifest(config, seed=0)
    provenance.finalize(manifest, [out_csv])
    return 0


if __name__ == "__main__":
    sys.exit(main())
