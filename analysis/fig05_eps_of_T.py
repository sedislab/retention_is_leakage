#!/usr/bin/env python3
"""FIG05 — lifelong ε(T) by unit of privacy (claim C3, H1). Reads `results/fig05_eps_of_T.csv`
(computed by `dp.accountant.account` over real ledgers in an earlier phase — no computation here,
CLAUDE.md non-negotiable #3). One panel per method, one line per unit U1-U5, log-log axes since both
T and ε span multiple orders of magnitude for the accumulating units (U1/U4) while the task-disjoint
units (U2 for M4/M8) stay exactly flat -- that contrast is the figure (`03_RESULTS_SPEC.md` FIG05).
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

_UNIT_COLOR = {
    "U1": "#1b9e77", "U2": "#d95f02", "U3": "#7570b3", "U4": "#e7298a", "U5": "#66a61e",
}
_UNIT_MARKER = {"U1": "o", "U2": "s", "U3": "^", "U4": "D", "U5": "v"}
_UNIT_LINESTYLE = {"U1": "-", "U2": "-", "U3": "--", "U4": ":", "U5": "-."}
_UNIT_LABEL = {
    "U1": "U1 example-level", "U2": "U2 task-disjoint parallel", "U3": "U3 client-level",
    "U4": "U4 individual, unbounded", "U5": "U5 individual, renewal",
}


def main() -> int:
    csv_path = REPO_ROOT / "results" / "fig05_eps_of_T.csv"
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))

    methods = sorted({r["method"] for r in rows})
    units = sorted({r["unit"] for r in rows})
    by_method_unit: dict = defaultdict(list)
    for r in rows:
        by_method_unit[(r["method"], r["unit"])].append(r)

    ncols = 4
    nrows = -(-len(methods) // ncols)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(plotting.column_width("double"), 2.2 * nrows), squeeze=False,
    )

    for i, method in enumerate(methods):
        ax = axes[i // ncols][i % ncols]
        eps_by_unit = {}
        for unit in units:
            method_rows = sorted(by_method_unit[(method, unit)], key=lambda r: int(r["T"]))
            if not method_rows:
                continue
            T = [int(r["T"]) for r in method_rows]
            eps = [float(r["eps"]) for r in method_rows]
            eps_by_unit[unit] = tuple(round(e, 6) for e in eps)
            ax.plot(
                T, eps, color=_UNIT_COLOR[unit], marker=_UNIT_MARKER[unit], markersize=4,
                linestyle=_UNIT_LINESTYLE[unit], linewidth=1.2, label=_UNIT_LABEL[unit],
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(method, fontsize=8)
        ax.tick_params(labelsize=6)

        # Several units can be numerically IDENTICAL for a given method (the accountant only routes
        # to parallel composition for unit==U2 -- see dp/accountant.py::account -- so U1/U3/U4/U5
        # always use sequential composition regardless of task-disjointness, and their curves collide
        # whenever their per-task touch counts happen to match). Distinct linestyles/markers keep every
        # line visible even when perfectly overlapping, but annotate the coincidence explicitly too --
        # a viewer should not have to guess whether a hidden line is missing or exactly superimposed.
        groups: dict = defaultdict(list)
        for unit, vals in eps_by_unit.items():
            groups[vals].append(unit)
        coincident = [g for g in groups.values() if len(g) > 1]
        if coincident:
            note = "; ".join("=".join(sorted(g)) for g in coincident)
            ax.text(0.02, 0.98, note, transform=ax.transAxes, fontsize=5, va="top", ha="left")

    for j in range(len(methods), nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    fig.supxlabel("T (observed rounds/tasks)", fontsize=8, y=0.09)
    fig.supylabel(r"$\varepsilon$", fontsize=8)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=6, loc="lower center", ncol=5, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("FIG05 — Lifelong $\\varepsilon(T)$ by unit of privacy", fontsize=9)
    fig.tight_layout(rect=(0.02, 0.12, 1, 0.94))
    plotting.save(fig, "fig05_eps_of_T", out_dir=REPO_ROOT / "figs")
    print("wrote figs/fig05_eps_of_T.{pdf,png}")
    # Paper caption: "epsilon(T) under 5 units of privacy, all 7 methods, sigma=2.0, delta=1e-5.
    # Solid lines are certified task-disjoint (parallel composition, flat); dashed lines accumulate
    # (sequential composition, diverging). M4/M8 hold U2 exactly flat at eps=2.529 from T=1 to
    # T=1000; M0 grows from eps=16.89 to eps=4165.57 over the same range under the same unit."
    return 0


if __name__ == "__main__":
    sys.exit(main())
