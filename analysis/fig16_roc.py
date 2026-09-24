#!/usr/bin/env python3
"""FIG16 — log-log ROC curves (CLAUDE.md non-negotiable #5: "log-log ROC is mandatory in the
appendix"). Reads `results/fig16_roc.csv` (`code/scripts/build_fig16.py`'s output -- no computation
here). One panel per dataset, one line per method (elapsed=0, trajectory ablation, seed=0 --
see `build_fig16.py`'s docstring for the exact scope this curve represents).
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

_LINESTYLES = ["-", "--", ":", "-.", (0, (3, 1, 1, 1, 1, 1))]


def main() -> int:
    csv_path = REPO_ROOT / "results" / "fig16_roc.csv"
    if not csv_path.exists():
        print(f"{csv_path} does not exist yet -- nothing to plot")
        return 1
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))

    datasets = sorted({r["dataset"] for r in rows})
    methods = sorted({r["method"] for r in rows})
    by_dataset_method: dict = defaultdict(list)
    for r in rows:
        by_dataset_method[(r["dataset"], r["method"])].append(r)

    fig, axes = plt.subplots(
        1, len(datasets), figsize=(5.5, 3.2),
        squeeze=False,
    )

    for col, dataset in enumerate(datasets):
        ax = axes[0][col]
        for method in methods:
            method_rows = sorted(by_dataset_method[(dataset, method)], key=lambda r: float(r["fpr"]))
            if not method_rows:
                continue
            linestyle = _LINESTYLES[methods.index(method) % len(_LINESTYLES)]
            style = plotting.method_style(method)
            fpr = [max(float(r["fpr"]), 1e-4) for r in method_rows]
            tpr = [max(float(r["tpr"]), 1e-4) for r in method_rows]
            ax.plot(fpr, tpr, color=style["color"], linestyle=linestyle, linewidth=1.2, label=plotting.display_name(method))
        ax.plot([1e-4, 1], [1e-4, 1], color="0.7", linewidth=0.7, linestyle=":")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(1e-4, 1)
        ax.set_ylim(1e-4, 1)
        ax.set_title(dataset, fontsize=8)
        ax.set_xlabel("FPR", fontsize=7)
        ax.tick_params(labelsize=7)

    axes[0][0].set_ylabel("TPR", fontsize=7)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=7, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.32, 1, 0.96))
    plotting.save(fig, "fig16_roc", out_dir=REPO_ROOT / "figs")
    print("wrote figs/fig16_roc.{pdf,png}")
    # Paper caption: "Log-log ROC for A1 cross-task LiRA, elapsed=0, one representative seed per
    # (dataset, method). Diagonal is chance. The community-standard companion to TAB03's
    # TPR@1%/0.1%FPR summary numbers (CLAUDE.md non-negotiable #5)."
    return 0


if __name__ == "__main__":
    sys.exit(main())
