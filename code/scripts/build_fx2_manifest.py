#!/usr/bin/env python3
"""FX2: builds `build/waves/fx2_leak_main.tsv`, one row per (dataset, method, family, view) the leak
pipeline (`run_lira_pertask.py` + `build_fx2_summary.py`) still needs to run against wave V2 data.
`m0_fedavg` is excluded -- it reuses the pre-fix `shadows/` store and is handled by its own job
(see `build/jobs/fx2_summary_m0_cifar100.sh`), not this manifest.

Columns: `dataset, method, family, view, seeds` (seeds comma-joined, e.g. "0,1,2").
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

DATASETS = ["cifar100", "cub200", "imagenet_r"]
SEEDS = [0, 1, 2]
F1_METHODS = ["m1_glfc", "m2_target", "m3_fot", "m5_hybrid_replay"]
MULTIVIEW_METHODS = {"m4_proto": "F2", "m8_analytic": "F5"}
F1_FAMILY = {"m1_glfc": "F1", "m2_target": "F1", "m3_fot": "F1", "m5_hybrid_replay": "F1"}
VIEWS_MULTIVIEW = ["full", "aggregate", "global"]


def main() -> int:
    rows = []
    for dataset in DATASETS:
        for method in F1_METHODS:
            rows.append({
                "dataset": dataset, "method": method, "family": F1_FAMILY[method],
                "view": "full", "seeds": ",".join(str(s) for s in SEEDS),
            })
        for method, family in MULTIVIEW_METHODS.items():
            for view in VIEWS_MULTIVIEW:
                rows.append({
                    "dataset": dataset, "method": method, "family": family,
                    "view": view, "seeds": ",".join(str(s) for s in SEEDS),
                })

    out_path = REPO_ROOT / "build" / "waves" / "fx2_leak_main.tsv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "method", "family", "view", "seeds"], delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
