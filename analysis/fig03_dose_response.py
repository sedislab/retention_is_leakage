#!/usr/bin/env python3
"""FIG03 v2 — dose-response: retention strength vs. leakage (claim C2, H2; `08_FIX_PLAN.md` §10).
Reads `results/fig03_dose_response.csv` (`code/scripts/build_fig03_dose_response.py`'s output, one
row per (method, knob_value, seed)) -- no computation here except the cross-seed aggregation (t-interval
over 3 seeds, `metrics.seed_ci`, matching this project's established "raw CSV carries per-seed data,
plot script aggregates" convention).

x = knob_value (both methods share the same six levels {1,2,5,10,20,50} by construction, so plotted
directly on a shared log axis rather than needing a nontrivial per-method rescaling), left panel =
retention (-BWT), right panel = leakage (TPR@1%FPR), colour = method. CIFAR-100 only, 3 seeds, 512
shadows/level -- full FX8 scope, not a pilot.
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

_METHOD_STYLE = {
    "m5_hybrid_replay": {"color": "#666666", "marker": "*", "label": "M5 HybridReplay (individual, F8)"},
    "m2_target": {"color": "#e6ab02", "marker": "P", "label": "M2 Gaussian replay (semantic, F6)"},
}


def _aggregate(rows: list[dict], value_key: str) -> dict:
    """`{knob_value: (mean, ci_lo, ci_hi)}` across seeds for one method."""
    by_level: dict = defaultdict(list)
    for r in rows:
        v = r[value_key]
        if v == "":
            continue
        by_level[int(r["knob_value"])].append(float(v))
    return {level: metrics.seed_ci(vals) for level, vals in by_level.items()}


def main() -> int:
    csv_path = REPO_ROOT / "results" / "fig03_dose_response.csv"
    if not csv_path.exists():
        print(f"{csv_path} does not exist yet -- nothing to plot")
        return 1
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))

    by_method: dict = defaultdict(list)
    for r in rows:
        by_method[r["method"]].append(r)

    fig, (ax_ret, ax_leak) = plt.subplots(1, 2, figsize=(plotting.column_width("double"), 3.2))

    for method, style in _METHOD_STYLE.items():
        method_rows = by_method.get(method, [])
        if not method_rows:
            continue
        bwt_agg = _aggregate(method_rows, "retention_bwt")
        tpr_agg = _aggregate(method_rows, "tpr1")
        levels = sorted(bwt_agg)

        ret = [-bwt_agg[lv][0] for lv in levels]  # -BWT: higher = more retention
        ret_lo = [-bwt_agg[lv][2] for lv in levels]
        ret_hi = [-bwt_agg[lv][1] for lv in levels]
        tpr = [tpr_agg[lv][0] for lv in levels]
        tpr_lo = [tpr_agg[lv][1] for lv in levels]
        tpr_hi = [tpr_agg[lv][2] for lv in levels]

        ax_ret.plot(levels, ret, color=style["color"], marker=style["marker"], label=style["label"])
        plotting.ci_band(ax_ret, levels, ret_lo, ret_hi, color=style["color"])
        ax_leak.plot(levels, tpr, color=style["color"], marker=style["marker"])
        plotting.ci_band(ax_leak, levels, tpr_lo, tpr_hi, color=style["color"])

    for ax in (ax_ret, ax_leak):
        ax.set_xscale("log")
        ax.set_xlabel("retention-strength knob value\n(buffer_size_per_class / n_synthetic_per_class)", fontsize=6.5)
    ax_ret.set_ylabel("retention (-BWT)", fontsize=7)
    ax_leak.set_ylabel("leakage (TPR@1%FPR, pooled K, elapsed=6)", fontsize=7)
    ax_ret.set_title("retention strength", fontsize=8)
    ax_leak.set_title("leakage", fontsize=8)
    handles, labels = ax_ret.get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=6, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("FIG03 v2 — Dose-response: retention strength vs. leakage (CIFAR-100)", fontsize=9)
    fig.subplots_adjust(left=0.09, right=0.98, top=0.85, bottom=0.32, wspace=0.3)
    plotting.save(fig, "fig03_dose_response", out_dir=REPO_ROOT / "figs")
    print("wrote figs/fig03_dose_response.{pdf,png}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
