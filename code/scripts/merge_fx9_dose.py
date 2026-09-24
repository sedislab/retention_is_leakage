#!/usr/bin/env python3
"""Merge the 36 independent dose products, then print all acceptance means."""

from fx9_io import ROOT, merge
from p3fcl import metrics

METHODS = ["m5_hybrid_replay", "m2_target"]
LEVELS = [1, 2, 5, 10, 20, 50]
if __name__ == "__main__":
    for kind in ["accuracy", "lira"]:
        paths = [
            ROOT / f"results/dose_response_{kind}_{m}_{lv}_seed{s}.csv"
            for m in METHODS
            for lv in LEVELS
            for s in range(3)
        ]
        rows = merge(paths, ROOT / f"results/dose_response_{kind}.csv", ["method", "knob_value", "seed"], 36)
        for m in METHODS:
            for lv in LEVELS:
                rr = [r for r in rows if r["method"] == m and int(r["knob_value"]) == lv]
                fields = ["retention_bwt", "final_acc"] if kind == "accuracy" else ["tpr1"]
                print(
                    "FX9-6 ACCEPT",
                    m,
                    lv,
                    {k: metrics.seed_ci([float(r[k]) for r in rr]) for k in fields},
                    flush=True,
                )
