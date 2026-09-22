#!/usr/bin/env python3
"""FIG11 — secure aggregation ablation for A1 (H11, `03_RESULTS_SPEC.md`: "do not cut"). Reads
`results/secure_agg_pilot_m0.csv` (M0's secure-agg pilot) and the existing full-view
`results/a1_lira_cifar100_m0_fedavg[_seed<N>].csv` files -- no computation here, both are already
final per-seed numbers, aggregated with `metrics.seed_ci` exactly as every other headline number in
this project is. Grouped bars: TPR@1%FPR with vs. without secure aggregation, at 3 elapsed values, M0
(F1/model-delta) only -- M4/M8 (F2/F5) are annotated as a separate, structural finding (the attack has
no defined score at all under secure aggregation for these families), not a bar, since there is no
numeric value to plot for them.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "code" / "src"))

from p3fcl import metrics, plotting  # noqa: E402

ELAPSED_GRID = [0, 5, 9]
SEEDS = [0, 1, 2]


def main() -> int:
    secure_csv = REPO_ROOT / "results" / "secure_agg_pilot_m0.csv"
    if not secure_csv.exists():
        print(f"{secure_csv} does not exist yet -- nothing to plot")
        return 1
    with open(secure_csv, newline="") as f:
        secure_rows = list(csv.DictReader(f))

    secure_by_elapsed = {e: [] for e in ELAPSED_GRID}
    for r in secure_rows:
        secure_by_elapsed[int(r["elapsed"])].append(float(r["tpr1"]))

    full_by_elapsed = {e: [] for e in ELAPSED_GRID}
    for seed in SEEDS:
        suffix = f"_seed{seed}" if seed != 0 else ""
        full_csv = REPO_ROOT / "results" / f"a1_lira_cifar100_m0_fedavg{suffix}.csv"
        if not full_csv.exists():
            continue
        with open(full_csv, newline="") as f:
            full_rows = list(csv.DictReader(f))
        for e in ELAPSED_GRID:
            row = next(r for r in full_rows if r["ablation"] == "trajectory" and int(r["elapsed"]) == e)
            full_by_elapsed[e].append(float(row["tpr_at_1pct_fpr"]))

    fig, ax = plt.subplots(figsize=(plotting.column_width("single"), 4.4))
    x = range(len(ELAPSED_GRID))
    width = 0.35

    full_means, full_errs = [], [[], []]
    secure_means, secure_errs = [], [[], []]
    for e in ELAPSED_GRID:
        fm, flo, fhi = metrics.seed_ci(full_by_elapsed[e])
        sm, slo, shi = metrics.seed_ci(secure_by_elapsed[e])
        full_means.append(fm)
        full_errs[0].append(fm - flo)
        full_errs[1].append(fhi - fm)
        secure_means.append(sm)
        secure_errs[0].append(sm - slo)
        secure_errs[1].append(shi - sm)

    ax.bar([i - width / 2 for i in x], full_means, width, yerr=full_errs, capsize=3,
           color="#1b9e77", label="full ledger (no secure agg)")
    ax.bar([i + width / 2 for i in x], secure_means, width, yerr=secure_errs, capsize=3,
           color="#d95f02", label="secure agg (aggregate view only)")
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"elapsed={e}" for e in ELAPSED_GRID], fontsize=7)
    ax.set_ylabel("TPR@1%FPR", fontsize=8)
    ax.tick_params(axis="y", labelsize=7)
    ax.set_title("FIG11 — M0 (F1): secure aggregation\nprovides no detectable protection", fontsize=8)
    ax.legend(fontsize=6, loc="upper right")
    fig.text(
        0.02, 0.04,
        "M4 (F2) and M8 (F5): the attack has no defined score at all\n"
        "under secure aggregation (0/N targets scoreable, every round)\n"
        "-- a structural result, not a bar. See\n"
        "notes/2026-09-21_secure_agg_a1.md.",
        fontsize=6, va="bottom", wrap=True,
    )
    fig.tight_layout(rect=(0, 0.24, 1, 1))
    plotting.save(fig, "fig11_secure_agg", out_dir=REPO_ROOT / "figs")
    print("wrote figs/fig11_secure_agg.{pdf,png}")
    print("REMINDER: M0 arm is a 3-seed PILOT (1,024-shadow budget) -- see PAPER_BRIEFING.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
