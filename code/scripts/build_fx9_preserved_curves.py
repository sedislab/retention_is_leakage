#!/usr/bin/env python3
"""Bootstrap new curve CIs from preserved scored NPZs; leave old per-combo summaries intact."""

import csv
import sys
from pathlib import Path

from build_fx2_summary import build_summary
from p3fcl import provenance
from p3fcl.shadow_runner import METHOD_REGISTRY

ROOT = Path(__file__).resolve().parents[2]


def main():
    ds, method, view = sys.argv[1:4]
    assert method in ["m0_fedavg", "m3_fot", "m8_analytic"]
    seeds = list(range(5)) if method == "m0_fedavg" else [0, 1, 2]
    manifest = provenance.run_manifest(
        dict(phase="FX9-1", hypothesis="H2", dataset=ds, method=method, view=view, seeds=seeds), seed=0
    )
    curves = []
    build_summary(
        ds,
        method,
        METHOD_REGISTRY[method][1].value,
        view,
        seeds,
        curve_rows=curves,
        ablations=("trajectory",),
    )
    path = ROOT / f"results/retention_leak_{ds}_{method}_{view}.csv"
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(curves[0]))
        w.writeheader()
        w.writerows(curves)
    provenance.finalize(manifest, [path])
    print(f"preserved curve {ds}/{method}/{view} rows={len(curves)}", flush=True)


if __name__ == "__main__":
    main()
