#!/usr/bin/env python3
"""FIG02 — leakage half-life forest plot (claim C1). Reads `results/fig02_halflife.csv`
(`scripts/build_fig02.py`'s output — no computation here, CLAUDE.md non-negotiable #3). A censored
row (`fit_ok=False`, no decay detected over the observed horizon) is drawn as a right-pointing arrow
at its horizon, per `00_BUILD_PLAN.md`'s explicit instruction, never as a fabricated fitted number.

Log-x axis (`03_RESULTS_SPEC.md`'s own "log scale where the data is multiplicative" rule) — half-life
is exactly that kind of quantity, and the seed-bootstrap CI on `tau = -1/slope` can genuinely blow up
when some resampled seed subset's fitted decay is very shallow (M4/leak's real 95% CI upper bound is
~4300 elapsed tasks against a horizon of 9 -- a real consequence of the reciprocal-slope transform
near-zero, not a bug); a linear axis makes every other estimate unreadable next to it.

Row labels include the dataset (`<dataset> <method> / <quantity>`) — with more than one dataset,
`method / quantity` alone is ambiguous (the same method appears once per dataset).
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


def main() -> int:
    csv_path = REPO_ROOT / "results" / "fig02_halflife.csv"
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))

    rows.sort(key=lambda r: (r["dataset"], r["method"], r["quantity"]))
    labels = [f"{r['dataset']} {r['method']} / {r['quantity']}" for r in rows]
    fig, ax = plt.subplots(figsize=(plotting.column_width("double") * 0.7, 0.5 * len(rows) + 1.0))

    for i, r in enumerate(rows):
        family = Family(r["family"])
        style = plotting.style_for(family)
        censored = r["censored"] in ("True", "true", "1")
        horizon = float(r["horizon_tasks"])
        if censored:
            ax.annotate(
                "", xy=(horizon * 1.6, i), xytext=(horizon, i),
                arrowprops=dict(arrowstyle="-|>", color=style["color"], lw=1.5),
            )
            ax.plot([horizon], [i], marker="|", color=style["color"])
        else:
            halflife = float(r["halflife"])
            lo, hi = float(r["ci_lo"]), float(r["ci_hi"])
            xerr = [[max(0.0, halflife - lo)], [max(0.0, hi - halflife)]]
            ax.errorbar([halflife], [i], xerr=xerr, fmt=style["marker"], color=style["color"], capsize=3)

    ax.set_xscale("log")
    ax.axvline(float(rows[0]["horizon_tasks"]), color="0.8", linewidth=0.5, linestyle="--")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("half-life (elapsed tasks, log scale)", fontsize=8)
    ax.set_title("FIG02 — Leakage half-life by family", fontsize=8)
    fig.tight_layout()
    plotting.save(fig, "fig02_halflife", out_dir=REPO_ROOT / "figs")
    print("wrote figs/fig02_halflife.{pdf,png}")
    # Paper caption: "Half-life estimates per family, 95% CI over seeds; an arrow marks a censored
    # fit (no decay detected over the observed horizon, lower bound shown, per 00_BUILD_PLAN.md's
    # explicit instruction not to force an exponential fit to a flat curve). CIFAR-100, 5 seeds."
    return 0


if __name__ == "__main__":
    sys.exit(main())
