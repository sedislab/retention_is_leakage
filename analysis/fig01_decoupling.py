#!/usr/bin/env python3
"""FIG01 — retention/leakage decoupling, the headline figure (claim C1, H2, H10). Reads
`results/fig01_decoupling.csv` (already fully aggregated by `scripts/build_fig01.py` — this script
does no computation, per CLAUDE.md non-negotiable #3) and plots it. No property-inference (A6) panel:
A6 is scoped to F2/M4 only (`notes/2026-09-16_p3_a6_h10.md`), a stated cut, not a silent one.

All 7 A1-validated methods, one column per dataset (2 rows: accuracy, leakage) — **grouping only by
method name and ignoring dataset was a real bug this script had for one revision**: with two datasets
present, the same method's rows from each dataset got sorted together by elapsed value alone and
plotted as a single zigzagging line jumping between datasets at every point. Fixed by grouping on
`(dataset, method)` and giving each dataset its own column.

Five of the seven methods (M0/M1/M2/M3/M5) share family F1 and so share one color by `plotting.py`'s
family-styling convention ("one line per artifact family") — distinguished from each other here by
linestyle, so the family identity and the individual method are both legible.
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
from p3fcl.artifacts import Family  # noqa: E402

_LINESTYLES = ["-", "--", ":", "-.", (0, (3, 1, 1, 1, 1, 1))]


def main() -> int:
    csv_path = REPO_ROOT / "results" / "fig01_decoupling.csv"
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))

    datasets = sorted({r["dataset"] for r in rows})
    by_dataset_method: dict = defaultdict(list)
    for r in rows:
        by_dataset_method[(r["dataset"], r["method"])].append(r)

    methods = sorted({r["method"] for r in rows})
    linestyle_for_method: dict = {}
    for method in methods:
        family = Family(next(r["family"] for r in rows if r["method"] == method))
        n_seen = sum(1 for m in linestyle_for_method.values() if m == family)
        linestyle_for_method[method] = (family, _LINESTYLES[n_seen % len(_LINESTYLES)])

    fig, axes = plt.subplots(
        2, len(datasets), figsize=(plotting.column_width("double") * len(datasets) / 2, 5.6),
        sharex=True, squeeze=False,
    )

    for col, dataset in enumerate(datasets):
        ax_acc, ax_leak = axes[0][col], axes[1][col]
        for method in methods:
            method_rows = sorted(by_dataset_method[(dataset, method)], key=lambda r: int(r["elapsed"]))
            if not method_rows:
                continue
            family, linestyle = linestyle_for_method[method]
            style = plotting.style_for(family)
            elapsed = [int(r["elapsed"]) for r in method_rows]

            acc = [float(r["acc_mean"]) for r in method_rows]
            acc_lo = [float(r["acc_ci_lo"]) for r in method_rows]
            acc_hi = [float(r["acc_ci_hi"]) for r in method_rows]
            ax_acc.plot(
                elapsed, acc, color=style["color"], marker=style["marker"], linestyle=linestyle,
                markersize=4, label=f"{method} ({style['label']})",
            )
            plotting.ci_band(ax_acc, elapsed, acc_lo, acc_hi, color=style["color"])

            tpr1 = [float(r["tpr1_mean"]) for r in method_rows]
            tpr1_lo = [float(r["tpr1_ci_lo"]) for r in method_rows]
            tpr1_hi = [float(r["tpr1_ci_hi"]) for r in method_rows]
            ax_leak.plot(
                elapsed, tpr1, color=style["color"], marker=style["marker"], linestyle=linestyle, markersize=4
            )
            plotting.ci_band(ax_leak, elapsed, tpr1_lo, tpr1_hi, color=style["color"])

        ax_acc.set_title(dataset, fontsize=8)
        ax_acc.axhline(1.0, color="0.7", linewidth=0.5, linestyle="--")
        ax_leak.axhline(0.01, color="0.7", linewidth=0.5, linestyle="--")
        ax_leak.set_xlabel("elapsed tasks ($T - k$)")

    axes[0][0].set_ylabel("accuracy on task $k$\n(normalised to elapsed=0)")
    axes[1][0].set_ylabel("A1 TPR@1%FPR\non task-$k$ data")

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=6, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("FIG01 — Retention/leakage decoupling", fontsize=9)
    fig.tight_layout(rect=(0, 0.10, 1, 0.95))
    plotting.save(fig, "fig01_decoupling", out_dir=REPO_ROOT / "figs")
    print("wrote figs/fig01_decoupling.{pdf,png}")
    # Paper caption (per plotting.py's convention: takeaway first, setting second, nothing else):
    # "Accuracy on task k decays with elapsed time; membership leakage decays slower or not at all,
    # for 6 of 7 methods on CIFAR-100 and most methods on CUB-200 (one real counter-example: M3 on
    # CUB-200 shows the reverse -- accuracy shows no detectable decay while leakage does). 5 seeds."
    return 0


if __name__ == "__main__":
    sys.exit(main())
