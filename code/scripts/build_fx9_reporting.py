#!/usr/bin/env python3
"""Compute citable utility/gate means once, from independent per-combo producers."""

import json

from fx9_io import ROOT, current, read, write
from p3fcl import metrics
from p3fcl.experiment import DATASETS, METHODS


def summarize(rows, fields):
    out = {}
    for field in fields:
        mean, lo, hi = metrics.seed_ci([float(r[field]) for r in rows])
        out.update({field: mean, field + "_ci_lo": lo, field + "_ci_hi": hi})
    return out


if __name__ == "__main__":
    gates, utility = [], []
    comparisons = []
    old_gate = read(ROOT / "archive/2026-09-23b_pre_fx9/results/fx4_gate.csv")
    for ds in DATASETS:
        path = ROOT / f"results/fx9_gate_{ds}.csv"
        current(path)
        rows = read(path)
        for method in sorted({r["method"] for r in rows}):
            rr = [r for r in rows if r["method"] == method]
            gates.append(
                dict(
                    dataset=ds,
                    method=method,
                    n_seeds=3,
                    gate_pass=rr[0]["pass"],
                    gate_pass_numeric=(
                        int(rr[0]["pass"].lower() == "true")
                        if rr[0]["pass"].lower() in ["true", "false"]
                        else ""
                    ),
                    replay_weight=rr[0]["replay_weight"],
                    used_cfg_json=rr[0]["used_cfg_json"],
                    **summarize(rr, ["final_avg_acc", "last_task_acc", "bwt"]),
                )
            )
        for method in METHODS:
            rr = [
                json.loads(
                    (
                        ROOT / f'runs/fx9_utility_baseline/{method.split("_")[0].upper()}_{ds}_t10_s{s}.json'
                    ).read_text()
                )
                for s in range(5)
            ]
            assert all(r["phase"] == "FX9" for r in rr)
            utility.append(
                dict(
                    dataset=ds,
                    method=method,
                    n_seeds=5,
                    **summarize(rr, ["final_avg_acc", "bwt", "avg_incremental_acc"]),
                )
            )
    for gate in gates:
        if gate["method"] not in ["m1_glfc", "m2_target", "m5_hybrid_replay"]:
            continue
        old = [r for r in old_gate if r["dataset"] == gate["dataset"] and r["method"] == gate["method"]]
        previous = summarize(old, ["last_task_acc", "bwt"])
        comparisons.append(
            dict(
                dataset=gate["dataset"],
                method=gate["method"],
                last_before=previous["last_task_acc"],
                last_after=gate["last_task_acc"],
                bwt_before=previous["bwt"],
                bwt_after=gate["bwt"],
                gate_pass=gate["gate_pass"],
            )
        )
    write(
        ROOT / "results/fx9_gate_comparison.csv",
        comparisons,
        dict(hypothesis="H2", source="archived FX4 gate and per-dataset FX9 gate files"),
    )
    write(
        ROOT / "results/fx9_gate_summary.csv",
        gates,
        dict(hypothesis="H2", sources=[f"fx9_gate_{d}.csv" for d in DATASETS]),
    )
    write(
        ROOT / "results/fx9_utility_summary.csv",
        utility,
        dict(hypothesis="H12", source="runs/fx9_utility_baseline/<method>_<dataset>_t10_s<seed>.json"),
    )

    dose = read(ROOT / "results/fig03_dose_response.csv")
    dose_summary = []
    for method in ["m2_target", "m5_hybrid_replay"]:
        for level in [1, 2, 5, 10, 20, 50]:
            rr = [r for r in dose if r["method"] == method and int(r["knob_value"]) == level]
            assert len(rr) == 3
            stats = summarize(rr, ["tpr1", "retention_bwt", "final_acc"])
            dose_summary.append(
                dict(
                    dataset="cifar100",
                    method=method,
                    knob_value=level,
                    n_seeds=3,
                    forgetting=-stats["retention_bwt"],
                    forgetting_ci_lo=-stats["retention_bwt_ci_hi"],
                    forgetting_ci_hi=-stats["retention_bwt_ci_lo"],
                    **stats,
                )
            )
    write(
        ROOT / "results/fx9_dose_summary.csv",
        dose_summary,
        dict(hypothesis="H2", source="fig03_dose_response.csv"),
    )
