#!/usr/bin/env python3
"""FIG08 v2 — privacy-utility Pareto (claim C4, H7; `08_FIX_PLAN.md` §8's restated scope). Reads
`results/fig08_pareto.csv` (built by `code/scripts/build_fig08_pareto.py` from `m9_sweep.csv` -- no
computation here, CLAUDE.md non-negotiable #3). One panel per dataset: M9 final average accuracy
against eps (log-x) for U1 and U2 at gamma=1.0, with horizontal M0/M8 non-private reference lines.

Reduced scope note (matches the CSV builder's own docstring): no DP-SGD-FedAvg/linear-probe baselines
(P2 stretch, not built) and only T=10 (the core sweep's fixed horizon; T=50 needs the 50-task stretch
run). `eps=inf` (the non-private M9 point, also present in the main grid) is plotted at the right edge
of the log-x axis via a fixed large sentinel value, annotated, not a log(inf) plotting hack.
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "code" / "src"))

from p3fcl import plotting  # noqa: E402

_INF_SENTINEL = 32.0  # plotted position for eps=inf on the log-x axis, clearly right of eps=8
_UNIT_COLOR = {"U1": "#1b9e77", "U2": "#d95f02"}
_UNIT_MARKER = {"U1": "o", "U2": "s"}
_REF_STYLE = {"m0_fedavg": ("#999999", "--"), "m8_analytic": ("#333333", ":")}


def main() -> int:
    csv_path = REPO_ROOT / "results" / "fig08_pareto.csv"
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))

    datasets = sorted({r["dataset"] for r in rows})
    fig, axes = plt.subplots(1, len(datasets), figsize=(plotting.column_width("double"), 2.6), squeeze=False)
    axes = axes[0]

    for ax, dataset in zip(axes, datasets, strict=True):
        ds_rows = [r for r in rows if r["dataset"] == dataset]

        for unit in ("U1", "U2"):
            m9_rows = [r for r in ds_rows if r["method"] == f"m9_contractive_{unit}"]
            by_eps: dict = defaultdict(list)
            for r in m9_rows:
                eps = float(r["eps_target"])
                by_eps[eps].append(r)
            if not by_eps:
                continue
            eps_sorted = sorted(by_eps)
            x = [(_INF_SENTINEL if np.isinf(e) else e) for e in eps_sorted]
            y = [float(by_eps[e][0]["final_acc_mean"]) for e in eps_sorted]
            plotting.ci_band(ax,x,[float(by_eps[e][0]["ci_lo"]) for e in eps_sorted],[float(by_eps[e][0]["ci_hi"]) for e in eps_sorted],color=_UNIT_COLOR[unit])
            ax.plot(x, y, color=_UNIT_COLOR[unit], marker=_UNIT_MARKER[unit], markersize=4,
                     linewidth=1.2, label=f"{plotting.display_name('m9_contractive', short=True)} ({unit})")

        for method in ("m0_fedavg", "m8_analytic"):
            ref_rows = [r for r in ds_rows if r["method"] == method]
            if not ref_rows:
                continue
            val = float(ref_rows[0]["final_acc_mean"])
            color, style = _REF_STYLE[method]
            ax.axhline(val, color=color, linestyle=style, linewidth=1.0, label=plotting.display_name(method) + " (non-private)")

        ax.set_xscale("log")
        ax.set_xlabel(r"$\varepsilon$", fontsize=8)
        ax.set_title(dataset, fontsize=8)
        ax.tick_params(labelsize=7)
        ax.axvline(_INF_SENTINEL * 0.6, color="#cccccc", linewidth=0.5, linestyle=":")
        ax.text(_INF_SENTINEL, 0.02, "inf", fontsize=7, ha="center", transform=ax.get_xaxis_transform())

    axes[0].set_ylabel("final average accuracy", fontsize=8)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=7, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0.02, 0.22, 1, 0.92))
    plotting.save(fig, "fig08_pareto", out_dir=REPO_ROOT / "figs")
    print("wrote figs/fig08_pareto.{pdf,png}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
