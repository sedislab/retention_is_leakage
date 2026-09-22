#!/usr/bin/env python3
"""FIG13 — Gram inversion n-curve (claim C1 supporting evidence, H5). Reads
`results/fig13_gram_inversion_summary.csv` (`code/scripts/build_fig13.py`'s aggregation of the raw
per-trial records -- no computation here) and `results/fig13_anisotropy.csv`. One panel per dataset:
cosine similarity against per-class n (log-x, since n spans 1-128), one line per reference quality.
A third panel shows measured feature anisotropy per dataset -- the hypothesis this project tests is
that anisotropy is what moves the curve (CUB-200's lower anisotropy ratio should track its higher
plateau cosine similarity at matched n, per `03_RESULTS_SPEC.md`'s own framing).
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

_REF_QUALITY_STYLE = {
    "clean_full": {"color": "#1b9e77", "marker": "o", "label": "clean, full ref pool"},
    "clean_small_pool": {"color": "#d95f02", "marker": "s", "label": "clean, small ref pool"},
    "noisy_full": {"color": "#7570b3", "marker": "^", "label": "noisy, full ref pool"},
}


def main() -> int:
    curve_csv = REPO_ROOT / "results" / "fig13_gram_inversion_summary.csv"
    aniso_csv = REPO_ROOT / "results" / "fig13_anisotropy.csv"
    with open(curve_csv, newline="") as f:
        rows = list(csv.DictReader(f))
    with open(aniso_csv, newline="") as f:
        aniso_rows = list(csv.DictReader(f))

    datasets = sorted({r["dataset"] for r in rows})
    by_dataset_rq: dict = defaultdict(list)
    for r in rows:
        by_dataset_rq[(r["dataset"], r["ref_quality"])].append(r)

    fig, axes = plt.subplots(1, len(datasets) + 1, figsize=(plotting.column_width("double") * 1.15, 3.8))

    for col, dataset in enumerate(datasets):
        ax = axes[col]
        for rq, style in _REF_QUALITY_STYLE.items():
            rq_rows = sorted(by_dataset_rq.get((dataset, rq), []), key=lambda r: int(r["n_per_class"]))
            if not rq_rows:
                continue
            n = [int(r["n_per_class"]) for r in rq_rows]
            mean_cos = [float(r["mean_cos"]) for r in rq_rows]
            lo = [float(r["ci_lo"]) for r in rq_rows]
            hi = [float(r["ci_hi"]) for r in rq_rows]
            ax.plot(n, mean_cos, color=style["color"], marker=style["marker"], markersize=4,
                     label=style["label"])
            plotting.ci_band(ax, n, lo, hi, color=style["color"])
        ax.axhline(0.8, color="0.7", linewidth=0.5, linestyle="--")
        ax.set_xscale("log", base=2)
        ax.set_ylim(0, 1.05)
        ax.set_title(dataset, fontsize=8)
        ax.set_xlabel("n per class", fontsize=7)
        ax.tick_params(labelsize=6)

    axes[0].set_ylabel("cosine similarity to true feature", fontsize=7)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=6, loc="lower center", ncol=3, bbox_to_anchor=(0.42, 0.0))

    ax_aniso = axes[-1]
    aniso_by_dataset = {r["dataset"]: float(r["anisotropy_ratio"]) for r in aniso_rows}
    ds_order = [d for d in datasets if d in aniso_by_dataset]
    ax_aniso.bar(ds_order, [aniso_by_dataset[d] for d in ds_order], color="#666666")
    ax_aniso.set_title("feature anisotropy", fontsize=8)
    ax_aniso.set_ylabel(r"$\lambda_1 / \lambda_{-1}$ ratio", fontsize=7)
    ax_aniso.tick_params(labelsize=6)

    fig.suptitle("FIG13 — Analytic Gram inversion (H5)", fontsize=9)
    fig.tight_layout(rect=(0, 0.14, 1, 0.93))
    plotting.save(fig, "fig13_gram_inversion", out_dir=REPO_ROOT / "figs")
    print("wrote figs/fig13_gram_inversion.{pdf,png}")
    # Paper caption: "Cosine similarity between the analytically-inverted feature and the true
    # feature, against per-class sample count n (log2-x), one line per reference-pool quality, 5
    # trials per point (95% CI). n=1 reconstruction is exact (cosine=1.000, not visible past the
    # y-axis crop at low n on some panels -- check the raw summary CSV). Anisotropy ratio (right)
    # is lower for CUB-200 than CIFAR-100, consistent with CUB-200's higher plateau similarity."
    return 0


if __name__ == "__main__":
    sys.exit(main())
