#!/usr/bin/env python3
"""FX9 FIG01: normalized accuracy and TPR retention, only from retention_curves.csv."""

import csv
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from p3fcl import plotting

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ["cifar100", "cub200", "imagenet_r"]


def main():
    with (ROOT / "results/retention_curves.csv").open() as f:
        rows = list(csv.DictReader(f))
    methods = sorted({r["method"] for r in rows})
    fig, axes = plt.subplots(2, 3, figsize=(5.5, 3.4), sharex=True)
    for col, ds in enumerate(DATASETS):
        for method in methods:
            style = plotting.method_style(method)
            for ri, quantity in enumerate(["acc", "leak_tpr1"]):
                views = (
                    ["full"]
                    if ri == 0 or method not in ["m4_proto", "m8_analytic"]
                    else ["global", "full"]
                )
                for view in views:
                    rr = sorted(
                        [
                            r
                            for r in rows
                            if r["dataset"] == ds
                            and r["method"] == method
                            and r["quantity"] == quantity
                            and r["view"] == view
                        ],
                        key=lambda r: int(r["elapsed"]),
                    )
                    e = [int(r["elapsed"]) for r in rr]

                    def vals(field):
                        return [
                            float(r[field]) if r[field] else float("nan") for r in rr
                        ]

                    dashed = (
                        ri == 1
                        and method in ["m4_proto", "m8_analytic"]
                        and view == "full"
                    )
                    axes[ri, col].plot(
                        e,
                        vals("norm"),
                        color=style["color"],
                        linestyle="--" if dashed else "-",
                        linewidth=0.9,
                    )
                    axes[ri, col].fill_between(
                        e,
                        vals("norm_ci_lo"),
                        vals("norm_ci_hi"),
                        color=style["color"],
                        alpha=0.12,
                        linewidth=0,
                    )
        axes[0, col].set_title(ds)
        axes[1, col].set_xlabel("elapsed tasks")
        axes[1, col].set_xticks([0, 2, 4, 6])
    axes[0, 0].set_ylabel("accuracy A(e)")
    axes[1, 0].set_ylabel("leakage L(e)")
    handles = [
        Line2D(
            [],
            [],
            color=plotting.method_style(m)["color"],
            label=plotting.display_name(m),
        )
        for m in methods
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.03),
        ncol=2,
        fontsize=7,
        frameon=False,
        columnspacing=0.8,
        handlelength=1.4,
    )
    fig.text(
        0.5,
        0.008,
        "- - per-client release (constant by construction)",
        ha="center",
        fontsize=7,
    )
    fig.subplots_adjust(
        left=0.10, right=0.99, top=0.94, bottom=0.31, wspace=0.35, hspace=0.32
    )
    plotting.save(fig, "fig01_decoupling", out_dir=ROOT / "figs")


if __name__ == "__main__":
    main()
