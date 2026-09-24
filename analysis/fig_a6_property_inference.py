#!/usr/bin/env python3
"""A6 property inference over time (H10) — the "middle panel" `03_RESULTS_SPEC.md`'s FIG01 spec
originally called for, dropped from the actual FIG01 grid because A6 is scoped to F2/M4 only (a
stated cut, not a silent one -- see `analysis/fig01_decoupling.py`'s docstring and
`notes/2026-09-16_p3_a6_h10.md`) while FIG01 covers all 7 methods x 3 datasets. Standalone here
instead of forced into FIG01's grid at the wrong scope.

Reads `results/a6_property_inference.csv` (M4, CIFAR-100, 10 tasks/10 clients -- no computation
here). The spec's own framing: "the story is visual: the top panel decays, the lower two do not, for
append-only families" -- this is the clean step-function version of that story: chance before the
target task, perfect immediately after, never decaying.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "code" / "src"))

from p3fcl import plotting  # noqa: E402


def main() -> int:
    csv_path = REPO_ROOT / "results" / "a6_property_inference.csv"
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    rows.sort(key=lambda r: int(r["elapsed"]))

    elapsed = [int(r["elapsed"]) for r in rows]
    bal_acc = [float(r["balanced_accuracy"]) for r in rows]

    fig, ax = plt.subplots(figsize=(plotting.column_width("single"), 3.0))
    ax.plot(elapsed, bal_acc, color="#a6761d", marker="X", markersize=5, drawstyle="steps-post")
    ax.axhline(0.5, color="0.7", linewidth=0.7, linestyle="--", label="chance")
    ax.axvline(0, color="0.4", linewidth=0.7, linestyle=":", label="task onset")
    ax.set_ylim(0.4, 1.05)
    ax.set_xlabel("elapsed tasks ($T - k$)")
    ax.set_ylabel("A6 balanced accuracy\n(property: client c held task-k class y)")
    ax.set_title("CIFAR-100 · " + plotting.display_name("m4_proto", short=True), fontsize=7)
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    plotting.save(fig, "fig_a6_property_inference", out_dir=REPO_ROOT / "figs")
    print("wrote figs/fig_a6_property_inference.{pdf,png}")
    # Paper caption: "A6 property-inference balanced accuracy (did client c hold task-k's class y?)
    # against elapsed tasks: chance before the class is introduced, perfect and never decaying after
    # -- convergent evidence with A1's decoupling finding from a much simpler attack. M4, CIFAR-100,
    # scoped to this method/dataset only (F2+F7's COUNTS join-key attack)."
    return 0


if __name__ == "__main__":
    sys.exit(main())
