#!/usr/bin/env python3
"""FIG10 — the M9 audit (`08_FIX_PLAN.md` §8, the empirical check of Theorem 11). Reads
`results/fig10_m9_audit.csv` (`code/scripts/run_m9_audit.py --score`'s output -- no computation
here). One panel per eps in {1, 4, inf}: the empirical log-log ROC (online LiRA against M9's
`global` state, elapsed=0, trajectory ablation) against the analytic Gaussian mechanism's DP bound
`TPR <= e^eps * FPR + delta`. eps=inf (no DP noise) is the non-private control -- it should show the
attack has real power (well above the diagonal), which is what makes the finite-eps panels passing
underneath their bound a meaningful result rather than a powerless one.
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "code" / "src"))

from p3fcl import plotting  # noqa: E402

_EPS_ORDER = ["1.0", "4.0", "inf"]
_EPS_LABEL = {"1.0": r"$\varepsilon=1$", "4.0": r"$\varepsilon=4$", "inf": r"$\varepsilon=\infty$ (non-private control)"}


def main() -> int:
    csv_path = REPO_ROOT / "results" / "fig10_m9_audit.csv"
    if not csv_path.exists():
        print(f"{csv_path} does not exist yet -- nothing to plot")
        return 1
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))

    by_eps: dict = defaultdict(list)
    for r in rows:
        by_eps[r["eps"]].append(r)

    fig, axes = plt.subplots(1, 3, figsize=(plotting.column_width("double"), 3.2), sharey=True)

    for ax, eps_key in zip(axes, _EPS_ORDER):
        eps_rows = sorted(by_eps.get(eps_key, []), key=lambda r: float(r["fpr"]))
        if not eps_rows:
            continue
        fpr = [float(r["fpr"]) for r in eps_rows]
        tpr = [float(r["tpr_empirical"]) for r in eps_rows]
        bound = [float(r["tpr_dp_bound"]) for r in eps_rows]
        n_pos, n_neg = eps_rows[0]["n_pos"], eps_rows[0]["n_neg"]

        ax.plot(fpr, tpr, color="#1b9e77", linewidth=1.3, label="empirical (A1 LiRA)")
        if eps_key != "inf":
            ax.plot(fpr, bound, color="#d95f02", linewidth=1.1, linestyle="--", label=r"DP bound $e^\varepsilon$FPR+$\delta$")
        ax.plot([1e-4, 1], [1e-4, 1], color="0.7", linewidth=0.5, linestyle=":")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(1e-4, 1)
        ax.set_ylim(1e-4, 1)
        ax.set_title(_EPS_LABEL[eps_key], fontsize=8)
        ax.set_xlabel("FPR", fontsize=7)
        ax.text(0.02, 0.98, f"n_pos={n_pos}\nn_neg={n_neg}", transform=ax.transAxes, fontsize=7, va="top", ha="left")

    axes[0].set_ylabel("TPR", fontsize=7)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=7, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.0))
    fig.subplots_adjust(left=0.08, right=0.99, top=0.85, bottom=0.30, wspace=0.15)
    plotting.save(fig, "fig10_m9_audit", out_dir=REPO_ROOT / "figs")
    print("wrote figs/fig10_m9_audit.{pdf,png}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
