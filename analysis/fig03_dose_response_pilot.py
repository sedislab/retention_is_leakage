#!/usr/bin/env python3
"""FIG03 (PILOT) — retention strength (-BWT) vs. leakage (A1 TPR@1%FPR), M5/CIFAR-100, 3 levels of
`buffer_size_per_class`, 3 seeds. Reads `results/fig03_dose_response_pilot.csv`
(`scripts/run_dose_response_pilot.py`'s output) — no computation here. **This is explicitly a small,
preliminary pilot, not the full P5 dose-response sweep** — see
`notes/2026-09-21_p5_dose_response_pilot.md` and `paper/PAPER_BRIEFING.md`'s claim-C2 section for the
honest scope this must be presented with.
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
    csv_path = REPO_ROOT / "results" / "fig03_dose_response_pilot.csv"
    if not csv_path.exists():
        print(f"{csv_path} does not exist yet -- nothing to plot")
        return 1
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) < 2:
        print(f"only {len(rows)} row(s) in {csv_path} -- pilot may still be running, nothing useful to plot")
        return 1

    rows.sort(key=lambda r: float(r["knob_value"]))
    retention = [-float(r["retention_bwt_mean"]) for r in rows]  # -BWT: higher = more retention
    ret_lo = [-float(r["retention_bwt_ci_hi"]) for r in rows]
    ret_hi = [-float(r["retention_bwt_ci_lo"]) for r in rows]
    tpr1 = [float(r["tpr1_mean"]) for r in rows]
    tpr1_lo = [float(r["tpr1_ci_lo"]) for r in rows]
    tpr1_hi = [float(r["tpr1_ci_hi"]) for r in rows]
    labels = [r["knob_value"] for r in rows]

    fig, ax = plt.subplots(figsize=(plotting.column_width("single"), 3.4))
    ret_err = [[r - lo for r, lo in zip(retention, ret_lo)], [hi - r for r, hi in zip(retention, ret_hi)]]
    tpr_err = [[t - lo for t, lo in zip(tpr1, tpr1_lo)], [hi - t for t, hi in zip(tpr1, tpr1_hi)]]

    ax2 = ax.twinx()
    ax.errorbar(range(len(rows)), retention, yerr=ret_err, fmt="o-", color="#1b9e77", capsize=3, label="retention (-BWT)")
    ax2.errorbar(
        range(len(rows)), tpr1, yerr=tpr_err, fmt="s--", color="#d95f02", capsize=3,
        label=f"leakage (TPR@1%FPR, elapsed={rows[0]['elapsed']})",
    )
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels([f"buffer={v}" for v in labels])
    ax.set_ylabel("retention (-BWT)", color="#1b9e77")
    ax2.set_ylabel("leakage (TPR@1%FPR)", color="#d95f02")
    ax.set_xlabel(f"{rows[0]['knob_name']} (M5 HybridReplay, CIFAR-100, n_seeds={rows[0]['n_seeds']})")
    ax.set_title("FIG03 (PILOT) — Dose-response: retention vs. leakage", fontsize=8)
    fig.tight_layout()
    plotting.save(fig, "fig03_dose_response_pilot", out_dir=REPO_ROOT / "figs")
    print("wrote figs/fig03_dose_response_pilot.{pdf,png}")
    print("REMINDER: this is a 1-method, 1-dataset, 3-level, 3-seed PILOT -- see PAPER_BRIEFING.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
