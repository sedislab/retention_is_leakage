#!/usr/bin/env python3
"""FIG18 — Camelyon17 natural federation vs. Dirichlet (`00_BUILD_PLAN.md`'s "never cut" list).
Reads `results/fig18_natural_federation.csv` (`code/scripts/run_fig18_report.py`'s output -- no
computation here). Answers "is the decoupling effect an artifact of synthetic client splits?" by
comparing the real, non-Dirichlet hospital partition against the same data under a synthetic
Dirichlet split.

**PILOT scope**: M0 only, 3 seeds, reduced 1,024-shadow budget, ~5,000-image class+hospital-
stratified subsample of the full 455,954-patch Camelyon17-WILDS release. See
`notes/2026-09-21_fig18_camelyon17.md` for the full scope and the CodaLab-download workaround this
required.
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

_PARTITION_STYLE = {
    "natural": {"color": "#1b9e77", "marker": "o", "label": "natural (real hospitals, no Dirichlet)"},
    "dirichlet": {"color": "#d95f02", "marker": "s", "label": "Dirichlet-subpartitioned (n=10 clients)"},
}


def main() -> int:
    csv_path = REPO_ROOT / "results" / "fig18_natural_federation.csv"
    if not csv_path.exists():
        print(f"{csv_path} does not exist yet -- nothing to plot")
        return 1
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))

    by_partition: dict = defaultdict(list)
    for r in rows:
        by_partition[r["partition"]].append(r)

    fig, (ax_acc, ax_leak) = plt.subplots(2, 1, figsize=(plotting.column_width("single"), 5.2), sharex=True)

    for partition, style in _PARTITION_STYLE.items():
        part_rows = sorted(by_partition.get(partition, []), key=lambda r: int(r["elapsed"]))
        if not part_rows:
            continue
        elapsed = [int(r["elapsed"]) for r in part_rows]
        acc = [float(r["acc"]) for r in part_rows]
        tpr1 = [float(r["tpr1"]) for r in part_rows]
        tpr_lo = [float(r["ci_lo"]) for r in part_rows]
        tpr_hi = [float(r["ci_hi"]) for r in part_rows]
        ax_acc.plot(elapsed, acc, color=style["color"], marker=style["marker"], label=style["label"])
        ax_leak.plot(elapsed, tpr1, color=style["color"], marker=style["marker"])
        plotting.ci_band(ax_leak, elapsed, tpr_lo, tpr_hi, color=style["color"])

    ax_acc.axhline(1.0, color="0.7", linewidth=0.5, linestyle="--")
    ax_acc.set_ylabel("accuracy on task $k$\n(normalised to elapsed=0)", fontsize=7)
    ax_acc.set_title("FIG18 (PILOT) — Camelyon17: natural vs.\nDirichlet client partition, M0", fontsize=8)
    ax_acc.legend(fontsize=6, loc="lower left")
    ax_leak.set_ylabel("A1 TPR@1%FPR", fontsize=7)
    ax_leak.set_xlabel("elapsed tasks ($T - k$), 5 hospitals")
    ax_leak.axhline(0.01, color="0.7", linewidth=0.5, linestyle="--")
    fig.tight_layout()
    plotting.save(fig, "fig18_natural_federation", out_dir=REPO_ROOT / "figs")
    print("wrote figs/fig18_natural_federation.{pdf,png}")
    print("REMINDER: M0-only PILOT (1,024-shadow budget, ~5k-image subsample) -- see PAPER_BRIEFING.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
