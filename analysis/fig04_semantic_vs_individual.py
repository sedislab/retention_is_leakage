#!/usr/bin/env python3
"""FIG04 — semantic (M4) vs. individual (M5) retention, dose-response PILOT (claim C2, the Red
Team's objection). Reads `results/fig04_semantic_vs_individual.csv`
(`code/scripts/build_fig04.py`'s output -- no computation here). Two panels sharing a y-axis scale so
the *slopes* are directly comparable: if the individual-retention arm (M5) shows a steeper
leakage-vs-retention relationship than the semantic arm (M4), that supports the Red Team's objection
that retention strength alone isn't what drives leakage -- it is the *kind* of retention.

**PILOT, not the full P5 spec** -- 1 dataset, 3 levels, 3 seeds per arm. See
`notes/2026-09-21_p5_dose_response_pilot.md` and its M4-arm companion note before citing this as more
than a preliminary, suggestive comparison.
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

_TYPE_TITLE = {"individual": "M5 HybridReplay (individual/exemplar)", "semantic": "M4 Prototype (semantic)"}


def main() -> int:
    csv_path = REPO_ROOT / "results" / "fig04_semantic_vs_individual.csv"
    if not csv_path.exists():
        print(f"{csv_path} does not exist yet -- run scripts/build_fig04.py once both pilot arms are done")
        return 1
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) < 2:
        print(f"only {len(rows)} row(s) in {csv_path} -- nothing useful to plot yet")
        return 1

    by_type: dict = defaultdict(list)
    for r in rows:
        by_type[r["retention_type"]].append(r)

    types = [t for t in ("individual", "semantic") if t in by_type]
    fig, axes = plt.subplots(1, len(types), figsize=(plotting.column_width("double"), 3.4), sharey="row")
    if len(types) == 1:
        axes = [axes]

    for i, (ax, rtype) in enumerate(zip(axes, types)):
        type_rows = sorted(by_type[rtype], key=lambda r: float(r["knob_value"]))
        retention = [-float(r["retention_bwt_mean"]) for r in type_rows]
        ret_lo = [-float(r["retention_bwt_ci_hi"]) for r in type_rows]
        ret_hi = [-float(r["retention_bwt_ci_lo"]) for r in type_rows]
        tpr1 = [float(r["tpr1_mean"]) for r in type_rows]
        tpr1_lo = [float(r["tpr1_ci_lo"]) for r in type_rows]
        tpr1_hi = [float(r["tpr1_ci_hi"]) for r in type_rows]
        labels = [r["knob_value"] for r in type_rows]

        ax2 = ax.twinx()
        ret_err = [[r - lo for r, lo in zip(retention, ret_lo)], [hi - r for r, hi in zip(retention, ret_hi)]]
        tpr_err = [[t - lo for t, lo in zip(tpr1, tpr1_lo)], [hi - t for t, hi in zip(tpr1, tpr1_hi)]]
        ax.errorbar(range(len(type_rows)), retention, yerr=ret_err, fmt="o-", color="#1b9e77", capsize=3)
        ax2.errorbar(range(len(type_rows)), tpr1, yerr=tpr_err, fmt="s--", color="#d95f02", capsize=3)
        ax.set_xticks(range(len(type_rows)))
        ax.set_xticklabels(labels, fontsize=7)
        ax.set_xlabel(f"{type_rows[0].get('method', '')} knob value", fontsize=6)
        ax.set_title(_TYPE_TITLE.get(rtype, rtype), fontsize=7)
        ax.tick_params(labelsize=6)
        ax2.tick_params(labelsize=6)
        if i == len(types) - 1:
            ax2.set_ylabel("leakage (TPR@1%FPR)", color="#d95f02", fontsize=7)

    axes[0].set_ylabel("retention (-BWT)", color="#1b9e77", fontsize=7)
    fig.suptitle("FIG04 (PILOT) — Semantic vs. individual retention", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    plotting.save(fig, "fig04_semantic_vs_individual", out_dir=REPO_ROOT / "figs")
    print("wrote figs/fig04_semantic_vs_individual.{pdf,png}")
    print("REMINDER: this is a PILOT (1 dataset, 3 levels, 3 seeds per arm) -- see PAPER_BRIEFING.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
