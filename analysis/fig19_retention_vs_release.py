#!/usr/bin/env python3
"""M4/M8 release and state retention; all series read from retention_curves.csv."""

import csv
from pathlib import Path
import matplotlib.pyplot as plt
from p3fcl import plotting

ROOT = Path(__file__).resolve().parents[1]


def main():
    with (ROOT / "results/retention_curves.csv").open() as f:
        rows = [r for r in csv.DictReader(f) if r["dataset"] == "cifar100"]
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.5))
    for ax, method in zip(axes, ["m4_proto", "m8_analytic"]):
        for q, view, ls, color, label in [
            (
                "leak_tpr1",
                "full",
                "--",
                "#D55E00",
                "per-client release (constant by construction)",
            ),
            ("leak_tpr1", "global", "-", "#D55E00", "global state leakage"),
            ("acc", "full", ":", "#0072B2", "accuracy"),
        ]:
            rr = sorted(
                [
                    r
                    for r in rows
                    if r["method"] == method
                    and r["quantity"] == q
                    and r["view"] == view
                ],
                key=lambda r: int(r["elapsed"]),
            )
            e = [int(r["elapsed"]) for r in rr]
            def vals(k):
                return [float(r[k]) if r[k] else float("nan") for r in rr]
            ax.plot(e, vals("norm"), linestyle=ls, color=color, label=label)
            ax.fill_between(
                e,
                vals("norm_ci_lo"),
                vals("norm_ci_hi"),
                alpha=0.15,
                color=color,
                linewidth=0,
            )
        ax.set_title(plotting.display_name(method))
        ax.set_xlabel("elapsed tasks")
    axes[0].set_ylabel("normalized retention")
    fig.legend(
        *axes[0].get_legend_handles_labels(), loc="lower center", ncol=1, frameon=False
    )
    fig.subplots_adjust(left=0.11, right=0.98, top=0.88, bottom=0.33, wspace=0.3)
    plotting.save(fig, "fig19_retention_vs_release", out_dir=ROOT / "figs")


if __name__ == "__main__":
    main()
