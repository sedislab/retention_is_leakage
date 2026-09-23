#!/usr/bin/env python3
"""FIG04 v2 — semantic (M2) vs. individual (M5) retention, the full FX8 dose-response sweep
(claim C2, the Red Team's objection: "retention is semantic, membership is individual").
Reads `results/fig04_semantic_vs_individual.csv`
(`code/scripts/build_fig04_semantic_vs_individual.py`'s output, one row per (method, knob_value,
seed)) -- no computation here except the cross-seed aggregation (t-interval, `metrics.seed_ci`).

Two panels sharing a y-axis scale so the *slopes* are directly comparable: if the individual-retention
arm (M5) shows a steeper leakage-vs-retention relationship than the semantic arm (M2), that supports
the Red Team's objection that retention strength alone isn't what drives leakage -- it is the *kind*
of retention. If both show a comparable slope, that is the result, drawn that way and said so
(`08_FIX_PLAN.md` §10's own instruction for FIG04, echoing `03_RESULTS_SPEC.md`).

CIFAR-100 only, 6 levels, 3 seeds, 512 shadows/level -- full FX8 scope, not a pilot (unlike the M4/M5
pilot this replaces: M2 is now the semantic arm, M4's own knob was dropped as dead code under
class-incremental streams, per H6).
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "code" / "src"))

from p3fcl import metrics, plotting  # noqa: E402

_TYPE_TITLE = {"individual": "M5 HybridReplay (individual, F8)", "semantic": "M2 Gaussian replay (semantic, F6)"}


def _aggregate(rows: list[dict], value_key: str) -> dict:
    by_level: dict = defaultdict(list)
    for r in rows:
        v = r[value_key]
        if v == "":
            continue
        by_level[int(r["knob_value"])].append(float(v))
    return {level: metrics.seed_ci(vals) for level, vals in by_level.items()}


def main() -> int:
    csv_path = REPO_ROOT / "results" / "fig04_semantic_vs_individual.csv"
    if not csv_path.exists():
        print(f"{csv_path} does not exist yet -- run build_fig04_semantic_vs_individual.py first")
        return 1
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))

    by_type: dict = defaultdict(list)
    for r in rows:
        by_type[r["retention_type"]].append(r)

    types = [t for t in ("individual", "semantic") if t in by_type]
    if not types:
        print("no rows for either retention_type -- nothing to plot")
        return 1

    fig, axes = plt.subplots(1, len(types), figsize=(plotting.column_width("double"), 3.4), sharey="row")
    if len(types) == 1:
        axes = [axes]

    # Precompute every panel's leakage series first, so both twin axes can share ONE y-range --
    # otherwise each `twinx()` auto-scales independently and a reader comparing "slopes" across
    # panels (this figure's whole point, per its own docstring) would be misled by two different
    # implicit scales that happen to look similar.
    per_type: dict = {}
    tpr_lo_global, tpr_hi_global = float("inf"), float("-inf")
    for rtype in types:
        type_rows = by_type[rtype]
        bwt_agg = _aggregate(type_rows, "retention_bwt")
        tpr_agg = _aggregate(type_rows, "tpr1")
        levels = sorted(bwt_agg)
        tpr_lo = [tpr_agg[lv][1] for lv in levels]
        tpr_hi = [tpr_agg[lv][2] for lv in levels]
        per_type[rtype] = {
            "method": type_rows[0]["method"], "bwt_agg": bwt_agg, "tpr_agg": tpr_agg, "levels": levels,
        }
        tpr_lo_global = min(tpr_lo_global, min(tpr_lo))
        tpr_hi_global = max(tpr_hi_global, max(tpr_hi))
    pad = 0.08 * (tpr_hi_global - tpr_lo_global)
    shared_tpr_ylim = (tpr_lo_global - pad, tpr_hi_global + pad)

    for i, (ax, rtype) in enumerate(zip(axes, types)):
        d = per_type[rtype]
        method, bwt_agg, tpr_agg, levels = d["method"], d["bwt_agg"], d["tpr_agg"], d["levels"]

        ret = [-bwt_agg[lv][0] for lv in levels]
        ret_lo = [-bwt_agg[lv][2] for lv in levels]
        ret_hi = [-bwt_agg[lv][1] for lv in levels]
        tpr = [tpr_agg[lv][0] for lv in levels]
        tpr_lo = [tpr_agg[lv][1] for lv in levels]
        tpr_hi = [tpr_agg[lv][2] for lv in levels]

        ax2 = ax.twinx()
        ax2.set_ylim(*shared_tpr_ylim)
        ret_err = [[r - lo for r, lo in zip(ret, ret_lo)], [hi - r for r, hi in zip(ret, ret_hi)]]
        tpr_err = [[t - lo for t, lo in zip(tpr, tpr_lo)], [hi - t for t, hi in zip(tpr, tpr_hi)]]
        ax.errorbar(range(len(levels)), ret, yerr=ret_err, fmt="o-", color="#1b9e77", capsize=3)
        ax2.errorbar(range(len(levels)), tpr, yerr=tpr_err, fmt="s--", color="#d95f02", capsize=3)
        ax.set_xticks(range(len(levels)))
        ax.set_xticklabels(levels, fontsize=7)
        ax.set_xlabel(f"{method} knob value", fontsize=6)
        ax.set_title(_TYPE_TITLE.get(rtype, rtype), fontsize=7)
        ax.tick_params(labelsize=6)
        ax2.tick_params(labelsize=6)
        if i == len(types) - 1:
            ax2.set_ylabel("leakage (TPR@1%FPR, elapsed=6)", color="#d95f02", fontsize=7)

    axes[0].set_ylabel("retention (-BWT)", color="#1b9e77", fontsize=7)
    fig.suptitle("FIG04 v2 — Semantic vs. individual retention (CIFAR-100)", fontsize=9)
    fig.subplots_adjust(left=0.09, right=0.93, top=0.85, bottom=0.20, wspace=0.35)
    plotting.save(fig, "fig04_semantic_vs_individual", out_dir=REPO_ROOT / "figs")
    print("wrote figs/fig04_semantic_vs_individual.{pdf,png}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
