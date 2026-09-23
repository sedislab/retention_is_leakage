#!/usr/bin/env python3
"""FIG01 v2 — retention/leakage decoupling, the headline figure (claim C1, H2, H10; FX2,
`08_FIX_PLAN.md`). Reads `results/fig01_decoupling.csv` (aggregated by
`code/scripts/build_fig01_decoupling.py` — this script does no computation, CLAUDE.md non-negotiable
#3) and plots it. No property-inference (A6) panel: A6 is scoped to F2/M4 only
(`notes/2026-09-16_p3_a6_h10.md`), a stated cut, not a silent one.

One column per dataset (2 rows: accuracy, leakage), grouped on `(dataset, method, view)` — **grouping
only by `(dataset, method)` was a real bug this script had for one revision, twice over**: first
(pre-fix) with only one dataset column, the same method's rows from two datasets got zigzagged
together; post-FX4h, M4/M8 additionally have up to 3 SCORING VIEWS (`full`/`aggregate`/`global`) per
(dataset, method), each its own real, distinct leakage curve -- grouping without `view` would zigzag
those together the same way. Every distinct `(method, view)` pair gets its own linestyle within a
family's shared color.

Both panels plot RAW values (real held-out test accuracy; real TPR@1%FPR), not the `acc_norm`/
`leak_norm` columns `build_fig01_decoupling.py` also writes (per `03_RESULTS_SPEC.md`'s FIG01 schema
request) -- those are a separate, display-only normalization-to-elapsed=0 view of the same data, and
this figure has no normalized-value confidence interval to plot alongside them, so it plots the raw
series instead.
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
    by_dataset_key: dict = defaultdict(list)
    for r in rows:
        by_dataset_key[(r["dataset"], r["method"], r["view"])].append(r)

    keys = sorted({(r["method"], r["view"]) for r in rows})
    linestyle_for_key: dict = {}
    for method, view in keys:
        family = Family(next(r["family"] for r in rows if r["method"] == method and r["view"] == view))
        n_seen = sum(1 for f, _ in linestyle_for_key.values() if f == family)
        linestyle_for_key[(method, view)] = (family, _LINESTYLES[n_seen % len(_LINESTYLES)])

    fig, axes = plt.subplots(
        2, len(datasets), figsize=(plotting.column_width("double") * len(datasets) / 2, 5.6),
        sharex=True, squeeze=False,
    )

    for col, dataset in enumerate(datasets):
        ax_acc, ax_leak = axes[0][col], axes[1][col]
        for method, view in keys:
            key_rows = sorted(by_dataset_key[(dataset, method, view)], key=lambda r: int(r["elapsed"]))
            if not key_rows:
                continue
            family, linestyle = linestyle_for_key[(method, view)]
            style = plotting.style_for(family)
            elapsed = [int(r["elapsed"]) for r in key_rows]
            view_suffix = f"/{view}" if view != "full" else ""

            acc = [float(r["acc_mean"]) for r in key_rows]
            acc_lo = [float(r["acc_ci_lo"]) for r in key_rows]
            acc_hi = [float(r["acc_ci_hi"]) for r in key_rows]
            ax_acc.plot(
                elapsed, acc, color=style["color"], marker=style["marker"], linestyle=linestyle,
                markersize=4, label=f"{method}{view_suffix} ({style['label']})",
            )
            plotting.ci_band(ax_acc, elapsed, acc_lo, acc_hi, color=style["color"])

            tpr1 = [float(r["tpr1_mean"]) for r in key_rows]
            tpr1_lo = [float(r["tpr1_ci_lo"]) for r in key_rows]
            tpr1_hi = [float(r["tpr1_ci_hi"]) for r in key_rows]
            ax_leak.plot(
                elapsed, tpr1, color=style["color"], marker=style["marker"], linestyle=linestyle, markersize=4
            )
            plotting.ci_band(ax_leak, elapsed, tpr1_lo, tpr1_hi, color=style["color"])

        ax_acc.set_title(dataset, fontsize=8)
        ax_leak.axhline(0.01, color="0.7", linewidth=0.5, linestyle="--")
        ax_leak.set_xlabel("elapsed tasks ($T - k$)")

    axes[0][0].set_ylabel("real held-out test accuracy\non task $k$")
    axes[1][0].set_ylabel("A1 TPR@1%FPR\non task-$k$ data")

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=5, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("FIG01 v2 — Retention/leakage decoupling", fontsize=9)
    fig.tight_layout(rect=(0, 0.16, 1, 0.95))
    plotting.save(fig, "fig01_decoupling", out_dir=REPO_ROOT / "figs")
    print("wrote figs/fig01_decoupling.{pdf,png}")
    # Paper caption (per plotting.py's convention: takeaway first, setting second, nothing else) --
    # STALE, pre-fix wording removed 2026-09-23 (the "6 of 7 methods" / M3-CUB-200-reversal narrative
    # was built on an invalid half-life extrapolation, see agents/OPEN_QUESTIONS.md's H2 entry and
    # notes/2026-09-23_fx2_decoupling_ratio_post_fix.md). Write the real caption from
    # results/decoupling_ratio.csv once FX2/FX3 finish producing the full post-fix numbers, not from
    # memory of the pre-fix story.
    return 0


if __name__ == "__main__":
    sys.exit(main())
