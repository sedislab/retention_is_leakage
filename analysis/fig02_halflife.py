#!/usr/bin/env python3
"""FIG02 v2 — leakage/accuracy half-life forest plot (claim C1; FX2, `08_FIX_PLAN.md` §7c/7d).
Reads `results/fig02_halflife.csv` (`code/scripts/build_fx2_summary.py`/`build_fx2_accuracy_summary.py`'s
output — no computation here, CLAUDE.md non-negotiable #3).

Definition 10's three-way `status` (not the old binary `censored` flag): `ok` gets a real point + CI;
`censored` (real signal, never crossed 50% within the observed horizon `E`) is drawn as a right-
pointing arrow at `E`, never a fabricated fitted number past it; `no_signal` (the quantity never
cleared its chance floor with confidence) is dropped from the plot entirely and reported separately
underneath, since a point at any x-position would misleadingly suggest a defined value where there is
none. Restricted to the primary `quantity=leak_tpr1` (CLAUDE.md non-negotiable #5) and the primary
`ablation=trajectory` (H3's transcript arm) plus every `quantity=acc` row, to keep the forest plot to
one row per (dataset, method, view) rather than one row per (quantity x ablation) combination.

Log-x axis (`03_RESULTS_SPEC.md`'s own "log scale where the data is multiplicative" rule) — half-life
is exactly that kind of quantity.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "code" / "src"))

from p3fcl import plotting  # noqa: E402
from p3fcl.artifacts import Family  # noqa: E402

_QUANTITY_COLOR = {"acc": "#1b9e77", "leak_tpr1": "#d95f02"}


def main() -> int:
    csv_path = REPO_ROOT / "results" / "fig02_halflife.csv"
    with open(csv_path, newline="") as f:
        all_rows = list(csv.DictReader(f))

    rows = [
        r for r in all_rows
        if (r["quantity"] == "acc") or (r["quantity"] == "leak_tpr1" and r["ablation"] == "trajectory")
    ]
    rows.sort(key=lambda r: (r["dataset"], r["method"], r["view"], r["quantity"]))

    plotted = [r for r in rows if r["status"] != "no_signal"]
    no_signal_rows = [r for r in rows if r["status"] == "no_signal"]
    labels = [f"{r['dataset']} {r['method']}/{r['view']} · {r['quantity']}" for r in plotted]

    fig, ax = plt.subplots(figsize=(plotting.column_width("double") * 0.7, 0.35 * len(plotted) + 1.5))

    horizon_E = float(plotted[0]["horizon_E"]) if plotted else 6.0
    for i, r in enumerate(plotted):
        family = Family(r["family"])
        style = plotting.style_for(family)
        color = _QUANTITY_COLOR.get(r["quantity"], style["color"])
        if r["status"] == "censored":
            ax.annotate(
                "", xy=(horizon_E * 1.6, i), xytext=(horizon_E, i),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.5),
            )
            ax.plot([horizon_E], [i], marker="|", color=color)
        else:  # ok
            halflife = float(r["halflife"])
            if r["ci_lo"] not in ("", "None") and r["ci_hi"] not in ("", "None"):
                lo, hi = float(r["ci_lo"]), float(r["ci_hi"])
                xerr = [[max(0.0, halflife - lo)], [max(0.0, hi - halflife)]]
            else:
                xerr = None
            ax.errorbar([halflife], [i], xerr=xerr, fmt=style["marker"], color=color, capsize=3)

    ax.set_xscale("log")
    ax.axvline(horizon_E, color="0.8", linewidth=0.5, linestyle="--")
    ax.set_yticks(range(len(plotted)))
    ax.set_yticklabels(labels, fontsize=6)
    ax.set_xlabel(f"half-life (elapsed tasks, log scale; horizon E={horizon_E:g})", fontsize=8)
    ax.set_title("FIG02 v2 — Accuracy and leakage half-life", fontsize=8)
    if no_signal_rows:
        ax.text(
            0.01, -0.12, f"{len(no_signal_rows)} row(s) omitted: no_signal (never cleared chance floor)",
            transform=ax.transAxes, fontsize=6, va="top",
        )
    fig.tight_layout()
    plotting.save(fig, "fig02_halflife", out_dir=REPO_ROOT / "figs")
    print("wrote figs/fig02_halflife.{pdf,png}")
    if no_signal_rows:
        print(f"omitted {len(no_signal_rows)} no_signal rows: " + ", ".join(f"{r['dataset']}/{r['method']}/{r['view']}/{r['quantity']}" for r in no_signal_rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
