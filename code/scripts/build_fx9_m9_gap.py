#!/usr/bin/env python3
"""Paired-seed M9 minus non-private M8 utility, from the preserved M9 sweep only."""

from fx9_io import ROOT, read, write
from p3fcl import metrics

if __name__ == "__main__":
    rows = read(ROOT / "results/m9_sweep.csv")
    baseline = {
        (r["dataset"], r["seed"]): float(r["final_avg_acc"]) for r in rows if r["method"] == "m8_analytic"
    }
    groups = {}
    for r in rows:
        if r["method"] == "m9_contractive" and r["gamma"] == "1.0" and r["n_clients"] == "10":
            groups.setdefault((r["dataset"], r["unit"], r["eps"]), []).append(r)
    out = []
    for (dataset, unit, eps), rr in sorted(groups.items()):
        gaps = [float(r["final_avg_acc"]) - baseline[(dataset, r["seed"])] for r in rr]
        mean, lo, hi = metrics.seed_ci(gaps)
        out.append(
            dict(
                dataset=dataset,
                unit=unit,
                eps=eps,
                n_seeds=len(rr),
                gap_m9_minus_m8=mean,
                gap_m9_minus_m8_ci_lo=lo,
                gap_m9_minus_m8_ci_hi=hi,
            )
        )
        if unit == "U2" and float(eps) == 1:
            print(
                f"FX9-10 H7 ACCEPT {dataset} U2 eps=1 paired_gap={mean:.12f} CI=[{lo:.12f},{hi:.12f}]",
                flush=True,
            )
    assert len(out) == 36
    write(
        ROOT / "results/fx9_m9_gap_summary.csv",
        out,
        dict(hypothesis="H7", source="m9_sweep.csv", statistic="paired seed difference M9 minus M8"),
    )
