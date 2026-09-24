#!/usr/bin/env python3
"""FX9 FIG13: 25 independent class draws per feasible cell; median of per-trial mean cosine, with percentile bootstrap CI. No anisotropy panel or causal anisotropy claim."""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "code" / "src"))

from p3fcl import plotting  # noqa: E402

_REF_QUALITY_STYLE = {
    "clean_full": {"color": "#1b9e77", "marker": "o", "label": "clean, full ref pool"},
    "clean_small_pool": {"color": "#d95f02", "marker": "s", "label": "clean, small ref pool"},
    "noisy_full": {"color": "#7570b3", "marker": "^", "label": "noisy, full ref pool"},
}


def main() -> int:
    curve_csv = REPO_ROOT / "results" / "fig13_gram_inversion_summary.csv"
    with open(curve_csv, newline="") as f:
        rows = list(csv.DictReader(f))

    datasets = sorted({r["dataset"] for r in rows})
    by_dataset_rq: dict = defaultdict(list)
    for r in rows:
        by_dataset_rq[(r["dataset"], r["ref_quality"])].append(r)

    fig, axes = plt.subplots(1, len(datasets), figsize=(5.5, 2.5))

    for col, dataset in enumerate(datasets):
        ax = axes[col]
        all_n: set = set()
        for rq, style in _REF_QUALITY_STYLE.items():
            rq_rows = sorted(by_dataset_rq.get((dataset, rq), []), key=lambda r: int(r["n_per_class"]))
            if not rq_rows:
                continue
            n = [int(r["n_per_class"]) for r in rq_rows]
            all_n.update(n)
            mean_cos = [float(r["median_cos"]) for r in rq_rows]
            lo = [float(r["ci_lo"]) for r in rq_rows]
            hi = [float(r["ci_hi"]) for r in rq_rows]
            ax.plot(n, mean_cos, color=style["color"], marker=style["marker"], markersize=4,
                     label=style["label"])
            plotting.ci_band(ax, n, lo, hi, color=style["color"])
        ax.axhline(0.8, color="0.7", linewidth=0.5, linestyle="--")
        ax.set_xscale("log", base=2)
        ax.set_xticks(sorted(all_n))
        ax.set_xticklabels([str(v) for v in sorted(all_n)])
        ax.set_ylim(0, 1.05)
        ax.set_title(dataset, fontsize=8)
        ax.set_xlabel("n per class", fontsize=7)
        ax.tick_params(labelsize=7)

    axes[0].set_ylabel("median reconstruction cosine", fontsize=7)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=7, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.0))

    fig.tight_layout(rect=(0, 0.14, 1, 0.93))
    plotting.save(fig, "fig13_gram_inversion", out_dir=REPO_ROOT / "figs")
    print("wrote figs/fig13_gram_inversion.{pdf,png}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
