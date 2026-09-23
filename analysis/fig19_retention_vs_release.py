#!/usr/bin/env python3
"""FIG19 (new) — retention versus release (FX3, `08_FIX_PLAN.md` §5). Reads
`results/fig01_decoupling.csv` (`code/scripts/build_fig01_decoupling.py`'s output -- no computation
here, CLAUDE.md non-negotiable #3).

For M4 and M8: normalised leakage against elapsed under `full` (dashed; flat by construction -- the
per-client release is a one-shot statistic that never gets re-touched) versus `global` (solid; the
running server state actually measured over time). Normalised test accuracy is overlaid as a third
line. Separates "does the transcript still contain what was released" (full, trivially yes forever)
from "does the current server state still reveal it" (global, the real retention question) --
`08_FIX_PLAN.md` §5's explicit framing.

CIFAR-100 is the main panel (M4, M8 side by side); other datasets are not duplicated here (FIG11 v2
already carries the appendix-dataset comparison for the same `full` vs `aggregate`/`global` question).
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "code" / "src"))

from p3fcl import plotting  # noqa: E402

_METHODS = [("m4_proto", "M4 (prototype, F2)"), ("m8_analytic", "M8 (exact Gram, F5)")]
_DATASET = "cifar100"


def main() -> int:
    csv_path = REPO_ROOT / "results" / "fig01_decoupling.csv"
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))

    fig, axes = plt.subplots(1, len(_METHODS), figsize=(plotting.column_width("double"), 3.0), sharey=True)

    for ax, (method, label) in zip(axes, _METHODS, strict=True):
        for view, style in (("full", "--"), ("global", "-")):
            method_rows = sorted(
                (r for r in rows if r["dataset"] == _DATASET and r["method"] == method and r["view"] == view),
                key=lambda r: int(r["elapsed"]),
            )
            if not method_rows:
                continue
            elapsed = [int(r["elapsed"]) for r in method_rows]
            leak_norm = [float(r["leak_norm"]) if r["leak_norm"] not in ("", "None") else float("nan") for r in method_rows]
            ax.plot(elapsed, leak_norm, linestyle=style, marker="o", markersize=3, color="#d95f02",
                     label=f"leakage ({view})")

        acc_rows = sorted(
            (r for r in rows if r["dataset"] == _DATASET and r["method"] == method and r["view"] == "full"),
            key=lambda r: int(r["elapsed"]),
        )
        if acc_rows:
            elapsed = [int(r["elapsed"]) for r in acc_rows]
            acc_norm = [float(r["acc_norm"]) if r["acc_norm"] not in ("", "None") else float("nan") for r in acc_rows]
            ax.plot(elapsed, acc_norm, linestyle=":", marker="s", markersize=3, color="#1b9e77", label="accuracy")

        ax.axhline(1.0, color="0.85", linewidth=0.5)
        ax.set_title(label, fontsize=8)
        ax.set_xlabel("elapsed tasks ($T-k$)", fontsize=7)
        ax.tick_params(labelsize=7)

    axes[0].set_ylabel("normalised to elapsed=0", fontsize=8)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=6, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("FIG19 — Retention vs. release: transcript persistence vs. state retention (CIFAR-100)", fontsize=9)
    fig.tight_layout(rect=(0, 0.16, 1, 0.92))
    plotting.save(fig, "fig19_retention_vs_release", out_dir=REPO_ROOT / "figs")
    print("wrote figs/fig19_retention_vs_release.{pdf,png}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
