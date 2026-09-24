#!/usr/bin/env python3
"""FX9 V3 manifest: 576 disjoint shadow chunks, plus four ImageNet-R pilot rows."""

import csv
import json
from pathlib import Path

from p3fcl.experiment import DATASETS, gate_overrides
from p3fcl.paths import shadow_dir

ROOT = Path(__file__).resolve().parents[2]
METHODS = ["m1_glfc", "m2_target", "m4_proto", "m5_hybrid_replay"]


def main():
    rows = []
    for ds in DATASETS:
        overrides = gate_overrides(ds)
        for method in METHODS:
            for seed in range(3):
                for start in range(0, 1024, 64):
                    rows.append(
                        dict(
                            dataset=ds,
                            method=method,
                            seed=seed,
                            start=start,
                            count=64,
                            out_dir=str(shadow_dir(ds, method, seed)),
                            method_config_override=json.dumps(overrides.get(method, {}), sort_keys=True),
                        )
                    )
    assert len(rows) == 576
    pilot = [r for r in rows if r["dataset"] == "imagenet_r" and r["seed"] == 0 and r["start"] == 0]
    for name, rr in [("v3_main", rows), ("v3_pilot", pilot)]:
        path = ROOT / f"build/waves/{name}.tsv"
        with path.open("w", newline="") as f:
            w = csv.DictWriter(
                f,
                fieldnames=list(rr[0]),
                delimiter="\t",
                lineterminator="\n",
                quoting=csv.QUOTE_NONE,
                quotechar="",
            )
            w.writeheader()
            w.writerows(rr)
        print(f"{name} rows={len(rr)}", flush=True)


if __name__ == "__main__":
    main()
