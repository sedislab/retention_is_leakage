#!/usr/bin/env python3
"""Collects `runs/utility_baseline/*.json` (one per method x dataset x n_tasks x seed, written by
`run_utility_baseline.py`) into the Gate P2 deliverable `results/tab05_utility_baselines.csv`.
Tidy format per `03_RESULTS_SPEC.md`'s convention: one row per (method, dataset, n_tasks, seed).
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "code" / "src"))

from p3fcl import provenance  # noqa: E402

RUNS_DIR = REPO_ROOT / "runs" / "utility_baseline"
OUT_CSV = REPO_ROOT / "results" / "tab05_utility_baselines.csv"

FIELDS = [
    "method", "dataset", "n_tasks_requested", "n_tasks_actual", "seed",
    "final_avg_acc", "bwt", "avg_incremental_acc", "n_ledger_records",
]


def main() -> int:
    files = sorted(f for f in RUNS_DIR.glob("*.json") if not f.name.endswith(".meta.json"))
    if not files:
        print(f"no run files found under {RUNS_DIR}")
        return 1

    rows = []
    for f in files:
        rec = json.loads(f.read_text())
        rows.append({k: rec[k] for k in FIELDS})

    rows.sort(key=lambda r: (r["method"], r["dataset"], r["n_tasks_requested"], r["seed"]))

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    print(f"wrote {len(rows)} rows to {OUT_CSV}")

    expected = 126
    if len(rows) != expected:
        print(f"WARNING: expected {expected} rows (7 methods x 6 dataset/n_tasks configs x 3 seeds), "
              f"got {len(rows)} — sweep may be incomplete.")

    config = {"seed": 0, "purpose": "TAB05 utility baseline collection"}
    manifest = provenance.run_manifest(config, seed=0)
    provenance.finalize(manifest, [OUT_CSV])
    return 0 if len(rows) == expected else 2


if __name__ == "__main__":
    sys.exit(main())
