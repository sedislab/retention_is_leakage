#!/usr/bin/env python3
"""FIG11 v2 — secure aggregation ablation (FX3, `08_FIX_PLAN.md` §5; H11). Reads
`results/fx3_views_summary.csv` (`code/scripts/build_fx3_views_summary.py`'s output -- no
computation here, CLAUDE.md non-negotiable #3).

Grouped bars of TPR@1%FPR at e=0 and e=6 (fixed-k set), bars are {full (per-client), aggregate (what
secure aggregation reveals), global (running server state)}, groups are M0 and M5 (F1), M4 (F2), M8
(F5). CIFAR-100 is the main panel; CUB-200/ImageNet-R render as an appendix variant
(`fig11_secure_agg_appendix.{pdf,png}`) via the same function.

Unlike the pre-fix version this replaces, `aggregate`/`global` are no longer "undefined, 0/N
scoreable" for M4/M8 -- FX4h gave them real per-view scores. F1's `aggregate`/`global` bars are drawn
hatched, matching `full` exactly (`identical_by_construction=True`, Prop. 3: the logit-margin attack
only ever reads the global model, identical under secure aggregation once weights are exact) --
visually distinct from a real independent measurement, not silently identical-looking bars.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "code" / "src"))

from p3fcl import plotting  # noqa: E402

_GROUPS = ["m0_fedavg", "m5_hybrid_replay", "m4_proto", "m8_analytic"]
_VIEWS = ["full", "aggregate", "global"]
_VIEW_COLOR = {"full": "#1b9e77", "aggregate": "#d95f02", "global": "#7570b3"}
_ELAPSED_GRID = [0, 6]


def _plot(dataset: str, rows: list, out_stem: str, title_suffix: str) -> None:
    fig, axes = plt.subplots(1, len(_ELAPSED_GRID), figsize=(plotting.column_width("double"), 3.2), sharey=True)

    for ax, e in zip(axes, _ELAPSED_GRID, strict=True):
        x = range(len(_GROUPS))
        width = 0.25
        for vi, view in enumerate(_VIEWS):
            means, errs_lo, errs_hi, hatches = [], [], [], []
            for method in _GROUPS:
                match = [r for r in rows if r["dataset"] == dataset and r["method"] == method and r["view"] == view and int(r["elapsed"]) == e]
                if not match:
                    means.append(0.0)
                    errs_lo.append(0.0)
                    errs_hi.append(0.0)
                    hatches.append("")
                    continue
                r = match[0]
                m = float(r["tpr1"])
                lo = float(r["tpr1_ci_lo"]) if r["tpr1_ci_lo"] not in ("", "None") else m
                hi = float(r["tpr1_ci_hi"]) if r["tpr1_ci_hi"] not in ("", "None") else m
                means.append(m)
                errs_lo.append(max(0.0, m - lo))
                errs_hi.append(max(0.0, hi - m))
                hatches.append("///" if r["identical_by_construction"] == "True" else "")
            offset = (vi - 1) * width
            bars = ax.bar(
                [i + offset for i in x], means, width, yerr=[errs_lo, errs_hi], capsize=2,
                color=_VIEW_COLOR[view], label=view if e == _ELAPSED_GRID[0] else None,
            )
            for bar, hatch in zip(bars, hatches, strict=True):
                if hatch:
                    bar.set_hatch(hatch)
                    bar.set_alpha(0.6)

        ax.axhline(0.01, color="0.7", linewidth=0.7, linestyle="--")
        ax.set_xticks(list(x))
        ax.set_xticklabels(_GROUPS, fontsize=6, rotation=20, ha="right")
        ax.set_title(f"elapsed={e}", fontsize=8)
        ax.tick_params(axis="y", labelsize=7)

    axes[0].set_ylabel("TPR@1%FPR", fontsize=8)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=6, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.06))
    fig.suptitle(f"FIG11 v2 — Secure aggregation views{title_suffix}", fontsize=9)
    fig.text(0.5, 0.01, "hatched bars: F1 aggregate/global identical to full by construction (Prop. 3)", fontsize=5, ha="center")
    fig.tight_layout(rect=(0, 0.22, 1, 0.92))
    plotting.save(fig, out_stem, out_dir=REPO_ROOT / "figs")
    print(f"wrote figs/{out_stem}.{{pdf,png}}")


def main() -> int:
    csv_path = REPO_ROOT / "results" / "fx3_views_summary.csv"
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))

    _plot("cifar100", rows, "fig11_secure_agg", " (CIFAR-100)")

    appendix_rows = [r for r in rows if r["dataset"] in ("cub200", "imagenet_r")]
    if appendix_rows:
        for dataset in ("cub200", "imagenet_r"):
            if any(r["dataset"] == dataset for r in appendix_rows):
                _plot(dataset, rows, f"fig11_secure_agg_appendix_{dataset}", f" ({dataset}, appendix)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
