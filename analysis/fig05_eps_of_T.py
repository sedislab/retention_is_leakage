#!/usr/bin/env python3
"""FIG05 v2 — lifelong ε(T) by unit of privacy (claim C3, H1; FX1, `08_FIX_PLAN.md` §6). Reads
`results/fig05_eps_of_T.csv` (computed by `dp.accountant.account`/`extrapolate_lifelong` over real
post-fix ledgers -- no computation here, CLAUDE.md non-negotiable #3). 2x3 small multiples, one panel
per method (M0, M1, M5, M4, M8, M9), one line per unit U1-U4 (U5 == U1 under this project's renewal
model, not plotted separately -- noted in the CSV's `regime`/`observed` columns being identical to U1's
by construction). The observed part of each line is solid; the extrapolated part (`observed == 0`,
beyond the method's real `n_tasks`) is dashed, per FX1's "draw the observed part solid and the
extrapolated part dashed."
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

_UNIT_COLOR = {"U1": "#1b9e77", "U2": "#d95f02", "U3": "#7570b3", "U4": "#e7298a"}
_UNIT_MARKER = {"U1": "o", "U2": "s", "U3": "^", "U4": "D"}
_UNIT_LABEL = {
    "U1": "U1 example-level (== U5)", "U2": "U2 task-level", "U3": "U3 client, bounded window",
    "U4": "U4 client, unbounded",
}
_PANEL_METHODS = ["M0", "M1", "M5", "M4", "M8", "M9"]


def main() -> int:
    csv_path = REPO_ROOT / "results" / "fig05_eps_of_T.csv"
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))

    by_method_unit: dict = defaultdict(list)
    for r in rows:
        by_method_unit[(r["method"], r["unit"])].append(r)
    methods_present = {r["method"] for r in rows}

    fig, axes = plt.subplots(2, 3, figsize=(plotting.column_width("double"), 4.6), squeeze=False)

    for i, method in enumerate(_PANEL_METHODS):
        ax = axes[i // 3][i % 3]
        if method not in methods_present:
            ax.text(0.5, 0.5, f"{method}\npending FX5", ha="center", va="center", fontsize=7, transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        eps_by_unit = {}
        regime_by_unit = {}
        for unit in ("U1", "U2", "U3", "U4"):
            method_rows = sorted(by_method_unit[(method, unit)], key=lambda r: int(r["T"]))
            if not method_rows:
                continue
            observed = [r for r in method_rows if int(r["observed"]) == 1]
            extrap = [r for r in method_rows if int(r["observed"]) == 0]
            color, marker = _UNIT_COLOR[unit], _UNIT_MARKER[unit]
            if observed:
                T_obs = [int(r["T"]) for r in observed]
                eps_obs = [float(r["eps"]) for r in observed]
                ax.plot(T_obs, eps_obs, color=color, marker=marker, markersize=3, linestyle="-",
                         linewidth=1.1, label=_UNIT_LABEL[unit])
                eps_by_unit[unit] = tuple(round(e, 6) for e in eps_obs)
                regime_by_unit[unit] = observed[0]["regime"]
            if extrap:
                # connect the dashed continuation to the last observed point so the line is unbroken.
                T_ext = ([T_obs[-1]] if observed else []) + [int(r["T"]) for r in extrap]
                eps_ext = ([eps_obs[-1]] if observed else []) + [float(r["eps"]) for r in extrap]
                ax.plot(T_ext, eps_ext, color=color, linestyle="--", linewidth=1.1)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(method, fontsize=8)
        ax.tick_params(labelsize=6)

        # Several units can be numerically identical for a given method (e.g. every unit sequential-
        # composes at the same per-task touch count) -- annotate coincidences so a reader isn't left
        # wondering whether a line is missing or exactly superimposed.
        groups: dict = defaultdict(list)
        for unit, vals in eps_by_unit.items():
            groups[vals].append(unit)
        coincident = [g for g in groups.values() if len(g) > 1]
        note_lines = []
        if coincident:
            note_lines.append("; ".join("=".join(sorted(g)) for g in coincident))
        note_lines.append(", ".join(f"{u}:{regime_by_unit[u][:4]}" for u in sorted(regime_by_unit)))
        ax.text(0.02, 0.98, "\n".join(note_lines), transform=ax.transAxes, fontsize=4.5, va="top", ha="left")

    fig.supxlabel("T (tasks; solid = observed, dashed = extrapolated)", fontsize=7, y=0.11)
    fig.supylabel(r"$\varepsilon$", fontsize=8)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=6, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.02))
    fig.suptitle("FIG05 v2 — Lifelong $\\varepsilon(T)$ by unit of privacy", fontsize=9)
    fig.subplots_adjust(left=0.07, right=0.99, top=0.86, bottom=0.20, hspace=0.5, wspace=0.3)
    plotting.save(fig, "fig05_eps_of_T", out_dir=REPO_ROOT / "figs")
    print("wrote figs/fig05_eps_of_T.{pdf,png}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
