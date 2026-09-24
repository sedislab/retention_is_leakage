#!/usr/bin/env python3
"""Forgetting versus leakage, with dose labels and both intervals stored in CSV."""

import csv
import textwrap
from pathlib import Path
import matplotlib.pyplot as plt
from p3fcl import plotting

ROOT = Path(__file__).resolve().parents[1]


def main():
    with (ROOT / "results/fx9_dose_summary.csv").open() as f:
        rows = list(csv.DictReader(f))
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.8), sharey=True, sharex=True)
    for ax, method in zip(axes, ["m5_hybrid_replay", "m2_target"]):
        rr = sorted(
            [r for r in rows if r["method"] == method],
            key=lambda r: int(r["knob_value"]),
        )
        style = plotting.method_style(method)
        x = [float(r["forgetting"]) for r in rr]
        y = [float(r["tpr1"]) for r in rr]
        ax.plot(x, y, **style, markersize=4, linewidth=0.7)
        for r, xx, yy in zip(rr, x, y):
            ax.hlines(
                yy,
                float(r["forgetting_ci_lo"]),
                float(r["forgetting_ci_hi"]),
                color=style["color"],
                linewidth=0.6,
            )
            ax.vlines(
                xx,
                float(r["tpr1_ci_lo"]),
                float(r["tpr1_ci_hi"]),
                color=style["color"],
                linewidth=0.6,
            )
            ax.annotate(
                r["knob_value"],
                (xx, yy),
                xytext=(3, 3),
                textcoords="offset points",
                fontsize=7,
            )
        ax.set_title(textwrap.fill(plotting.display_name(method), 25))
        ax.set_xlabel("forgetting (−BWT)")
    axes[0].set_ylabel("TPR@1%FPR at e=6")
    fig.text(
        0.5,
        0.02,
        "numbers: samples per class · intervals: 95% across three seeds",
        ha="center",
        fontsize=7,
    )
    fig.subplots_adjust(left=0.11, right=0.98, top=0.83, bottom=0.25, wspace=0.3)
    plotting.save(fig, "fig04_semantic_vs_individual", out_dir=ROOT / "figs")


if __name__ == "__main__":
    main()
