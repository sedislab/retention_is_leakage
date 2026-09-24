#!/usr/bin/env python3
"""Only M0 has a defined decoupling ratio in the final FX9 table."""

from fx9_io import ROOT, merge
from p3fcl.experiment import DATASETS

if __name__ == "__main__":
    paths = [ROOT / f"results/fx9_ratio_{ds}_m0_fedavg_full.csv" for ds in DATASETS]
    rows = merge(paths, ROOT / "results/decoupling_ratio.csv", ["dataset", "method", "view"], 3)
    for row in rows:
        print(f"FX9-9 RATIO ACCEPT {row}", flush=True)
