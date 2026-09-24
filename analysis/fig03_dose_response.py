#!/usr/bin/env python3
"""Balanced replay dose response; every point and CI already stored in CSV."""

import csv
from pathlib import Path
import matplotlib.pyplot as plt
from p3fcl import plotting

ROOT = Path(__file__).resolve().parents[1]


def main():
    with (ROOT / "results/fx9_dose_summary.csv").open() as f:
        rows = list(csv.DictReader(f))
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.8))
    for method in ["m5_hybrid_replay", "m2_target"]:
        rr = sorted(
            [r for r in rows if r["method"] == method],
            key=lambda r: int(r["knob_value"]),
        )
        style = plotting.method_style(method)
        levels = [int(r["knob_value"]) for r in rr]
        for ax, field in zip(axes, ["forgetting", "tpr1"]):
            ax.plot(
                levels,
                [float(r[field]) for r in rr],
                **style,
                markersize=4,
                label=plotting.display_name(method),
            )
            plotting.ci_band(
                ax,
                levels,
                [float(r[field + "_ci_lo"]) for r in rr],
                [float(r[field + "_ci_hi"]) for r in rr],
                color=style["color"],
            )
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel("samples per class")
    axes[0].set_ylabel("forgetting (−BWT)")
    axes[1].set_ylabel("TPR@1%FPR at e=6")
    fig.legend(
        *axes[0].get_legend_handles_labels(), loc="lower center", ncol=1, frameon=False
    )
    fig.subplots_adjust(left=0.11, right=0.99, top=0.95, bottom=0.32, wspace=0.4)
    plotting.save(fig, "fig03_dose_response", out_dir=ROOT / "figs")


if __name__ == "__main__":
    main()
