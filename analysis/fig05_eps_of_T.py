#!/usr/bin/env python3
"""Observed and extrapolated FX1 composition, including actual-noise M9 ledgers."""

import csv
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from p3fcl import plotting

ROOT = Path(__file__).resolve().parents[1]
COLORS = {"U1": "#1b9e77", "U2": "#d95f02", "U3": "#7570b3", "U4": "#e7298a"}
MARKERS = {"U1": "o", "U2": "s", "U3": "^", "U4": "D"}


def main():
    with (ROOT / "results/fig05_eps_of_T.csv").open() as f:
        rows = list(csv.DictReader(f))
    fig, axes = plt.subplots(2, 3, figsize=(5.5, 4.3))
    for ax, method in zip(axes.flat, ["M0", "M1", "M5", "M4", "M8", "M9"]):
        rr = [r for r in rows if r["method"] == method]
        assert rr, method
        for eps0 in ["1.0", "4.0"] if method == "M9" else [""]:
            for unit in COLORS:
                series = sorted(
                    [r for r in rr if r["unit"] == unit and r.get("eps0", "") == eps0],
                    key=lambda r: int(r["T"]),
                )
                assert series, (method, unit, eps0)
                observed = [r for r in series if r["observed"] == "1"]
                extra = [r for r in series if r["observed"] == "0"]
                color = COLORS[unit]
                ax.plot(
                    [int(r["T"]) for r in observed],
                    [float(r["eps"]) for r in observed],
                    color=color,
                    linestyle="-",
                    linewidth=0.8,
                    marker=MARKERS[unit],
                    markersize=2.5,
                    markerfacecolor="white" if eps0 == "4.0" else color,
                    markevery=max(1, len(observed) // 5),
                )
                if extra:
                    extra = observed[-1:] + extra
                    ax.plot(
                        [int(r["T"]) for r in extra],
                        [float(r["eps"]) for r in extra],
                        color=color,
                        linestyle="--",
                        linewidth=0.8,
                    )
        signatures = {}
        for unit in COLORS:
            signature = tuple(
                (r["T"], r.get("eps0", ""), r["eps"])
                for r in sorted(
                    [r for r in rr if r["unit"] == unit],
                    key=lambda r: (r.get("eps0", ""), int(r["T"])),
                )
            )
            signatures.setdefault(signature, []).append(unit)
        coincidences = [
            "=".join(group) for group in signatures.values() if len(group) > 1
        ]
        if coincidences:
            ax.text(
                0.03,
                0.97,
                "; ".join(coincidences),
                transform=ax.transAxes,
                va="top",
                fontsize=7,
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(plotting.display_name(method, short=True))
        ax.set_xticks([1, 10, 100, 1000])
        ax.set_xticklabels(["1", "10", "100", "1000"])
        if method == "M9":
            ax.text(
                0.03,
                0.83,
                "ε₀=1: filled\nε₀=4: open",
                va="top",
                transform=ax.transAxes,
                fontsize=7,
            )
    for ax in axes[:, 0]:
        ax.set_ylabel("accountant ε")
    for ax in axes[1]:
        ax.set_xlabel("tasks T")
    handles = [
        Line2D([], [], color=COLORS[u], marker=MARKERS[u], markersize=3, label=label)
        for u, label in zip(
            COLORS,
            [
                "U1 example (=U5)",
                "U2 client-task",
                "U3 client window",
                "U4 client history",
            ],
        )
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.025),
        ncol=2,
        frameon=False,
    )
    fig.text(
        0.5,
        0.007,
        "solid: observed; dashed: extrapolated · M9 group-unit values are conditional on sensitivity",
        ha="center",
        fontsize=7,
    )
    fig.subplots_adjust(
        left=0.11, right=0.96, top=0.94, bottom=0.22, wspace=0.48, hspace=0.46
    )
    plotting.save(fig, "fig05_eps_of_T", out_dir=ROOT / "figs")


if __name__ == "__main__":
    main()
