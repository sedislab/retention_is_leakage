#!/usr/bin/env python3
"""FX9 FIG02 retention-at-horizon scatter; half-life numbers remain in TAB04."""

import csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from adjustText import adjust_text
from p3fcl import plotting

ROOT = Path(__file__).resolve().parents[1]


def main():
    with (ROOT / "results/retention_curves.csv").open() as f:
        rows = [r for r in csv.DictReader(f) if r["elapsed"] == "6"]
    fig, axes = plt.subplots(1, 3, figsize=(5.5, 2.4))
    for ax, ds in zip(axes, ["cifar100", "cub200", "imagenet_r"]):
        rr = [r for r in rows if r["dataset"] == ds]
        texts = []
        for leak in [r for r in rr if r["quantity"] == "leak_tpr1"]:
            acc = next(
                r
                for r in rr
                if r["quantity"] == "acc"
                and r["method"] == leak["method"]
                and r["view"] == leak["view"]
            )
            if not leak["norm"] or not acc["norm"]:
                continue
            x, y = float(acc["norm"]), float(leak["norm"])
            if not np.isfinite([x, y]).all():
                continue
            # CIs may exclude the point estimate; draw interval endpoints directly.
            style = plotting.method_style(leak["method"])
            ax.hlines(
                y,
                float(acc["norm_ci_lo"]),
                float(acc["norm_ci_hi"]),
                color=style["color"],
                linewidth=0.6,
            )
            ax.vlines(
                x,
                float(leak["norm_ci_lo"]),
                float(leak["norm_ci_hi"]),
                color=style["color"],
                linewidth=0.6,
            )
            marker = {"full": "o", "aggregate": "s", "global": "^"}[leak["view"]]
            ax.plot(
                x,
                y,
                marker=marker,
                color=style["color"],
                markersize=3,
                linestyle="none",
            )
            texts.append(ax.text(x, y, plotting.display_name(leak["method"], short=True), fontsize=7))
        ax.axhline(1, color=".5", linestyle="--", linewidth=0.6)
        ax.axline((0, 0), slope=1, color=".6", linestyle=":", linewidth=0.6)
        ax.set_title(ds)
        ax.set_xlabel("accuracy A(6)")
        if texts:
            adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="-", color="0.5", lw=0.5))
    axes[0].set_ylabel("leakage L(6)")
    fig.text(
        0.5,
        0.015,
        "○ full   □ aggregate   △ global     dashed: leakage unchanged; dotted: y = x",
        ha="center",
        fontsize=7,
    )
    fig.subplots_adjust(left=0.10, right=0.99, top=0.90, bottom=0.25, wspace=0.38)
    plotting.save(fig, "fig02_retention_at_horizon", out_dir=ROOT / "figs")
    plotting.save(fig, "fig02_halflife", out_dir=ROOT / "figs")


if __name__ == "__main__":
    main()
