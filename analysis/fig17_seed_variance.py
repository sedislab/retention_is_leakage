#!/usr/bin/env python3
"""FIG17 — seed variance (`03_RESULTS_SPEC.md` appendix). Reads `results/fig17_seed_variance.csv`
(`scripts/build_fig17.py`'s output -- no computation here, CLAUDE.md non-negotiable #3) and draws a
strip plot of the per-seed TPR@1%FPR (elapsed=0, trajectory ablation) for every (dataset, method), so
the seed-to-seed spread behind every headline point in FIG01/TAB03 is visible directly rather than
only as a CI half-width.
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



def main() -> int:
    csv_path = REPO_ROOT / "results" / "fig17_seed_variance.csv"
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))

    datasets = sorted({r["dataset"] for r in rows})
    methods = sorted({r["method"] for r in rows})
    by_dataset_method: dict = defaultdict(list)
    for r in rows:
        by_dataset_method[(r["dataset"], r["method"])].append(float(r["value"]))

    fig, axes = plt.subplots(
        1, len(datasets), figsize=(plotting.column_width("double"), 3.2), sharey=True, squeeze=False,
    )

    for col, dataset in enumerate(datasets):
        ax = axes[0][col]
        for i, method in enumerate(methods):
            values = by_dataset_method[(dataset, method)]
            if not values:
                continue
            style = plotting.method_style(method)
            n = len(values)
            jitter_x = [i + 0.15 * ((j / max(n - 1, 1)) - 0.5) for j in range(n)]
            ax.scatter(jitter_x, values, color=style["color"], marker=style["marker"], s=18, alpha=0.85)
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels([plotting.display_name(m, short=True) for m in methods], fontsize=7, rotation=45, ha="right")
        ax.set_title(dataset, fontsize=8)
        ax.tick_params(labelsize=7)

    axes[0][0].set_ylabel("TPR@1%FPR (elapsed=0, trajectory)", fontsize=7)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    plotting.save(fig, "fig17_seed_variance", out_dir=REPO_ROOT / "figs")
    print("wrote figs/fig17_seed_variance.{pdf,png}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
