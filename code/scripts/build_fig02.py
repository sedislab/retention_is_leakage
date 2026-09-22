#!/usr/bin/env python3
"""Assembles `results/fig02_halflife.csv` (also TAB04's source — same data, tabular presentation) —
accuracy and leakage half-life estimates per `(dataset, method, family)`, with a bootstrap CI over
seeds and the `fit_ok`/`censored` flags `00_BUILD_PLAN.md`'s P4 protocol requires: "fitting a
half-life to a flat curve is a reporting error... the honest answer may be 'no decay detected... lower
bound on half-life > H'." Pure post-processing (reads `accuracy_matrix_<dataset>.csv` and
`a1_lira_<dataset>_<method>[_seed<N>].csv`, writes one CSV); same `DATASETS`/all-7-methods scope as
`build_fig01.py`.
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "code" / "src"))

from p3fcl import metrics, provenance  # noqa: E402
from p3fcl import rng as rng_mod  # noqa: E402
from p3fcl.artifacts import Family  # noqa: E402

DATASETS = ["cifar100", "cub200", "imagenet_r"]
SEEDS = [0, 1, 2, 3, 4]
METHOD_FAMILY = {
    "m0_fedavg": Family.MODEL_DELTA,
    "m1_glfc": Family.MODEL_DELTA,
    "m2_target": Family.MODEL_DELTA,
    "m3_fot": Family.MODEL_DELTA,
    "m4_proto": Family.PROTOTYPE,
    "m5_hybrid_replay": Family.MODEL_DELTA,
    "m8_analytic": Family.GRAM,
}
N_BOOTSTRAP = 2000


def _read_csv_rows(path: Path) -> list:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _accuracy_normalized_by_elapsed(acc_rows: list, method: str, seed: int) -> dict:
    by_k: dict = defaultdict(dict)
    for r in acc_rows:
        if r["method"] != method or int(r["seed"]) != seed:
            continue
        by_k[int(r["task_k"])][int(r["elapsed"])] = float(r["acc"])
    by_elapsed: dict = defaultdict(list)
    for k, elapsed_map in by_k.items():
        base = elapsed_map.get(0)
        if not base:
            continue
        for elapsed, acc in elapsed_map.items():
            by_elapsed[elapsed].append(acc / base)
    return {e: float(np.mean(v)) for e, v in by_elapsed.items()}


def _per_seed_curves(dataset: str, method: str, acc_rows: list) -> tuple:
    """`(acc_curves, tpr1_curves)`, each a list (one per available seed) of `{elapsed: value}` dicts."""
    acc_curves, tpr1_curves = [], []
    for seed in SEEDS:
        suffix = f"_seed{seed}" if seed != 0 else ""
        lira_path = REPO_ROOT / "results" / f"a1_lira_{dataset}_{method}{suffix}.csv"
        if not lira_path.exists():
            continue
        acc_curves.append(_accuracy_normalized_by_elapsed(acc_rows, method, seed))
        tpr1_curves.append(
            {
                int(row["elapsed"]): float(row["tpr_at_1pct_fpr"])
                for row in _read_csv_rows(lira_path)
                if row["ablation"] == "trajectory"
            }
        )
    return acc_curves, tpr1_curves


def _mean_curve(curves: list, elapsed_grid: list) -> np.ndarray:
    return np.array([np.mean([c[e] for c in curves if e in c]) for e in elapsed_grid])


def _bootstrap_halflife(curves: list, elapsed_grid: list, boot_idx: np.ndarray) -> tuple:
    """Point estimate from the real mean curve; CI from resampling *which seeds* contribute (not the
    already-reduced mean), which is the correct source of variability to bootstrap over here.
    `boot_idx`: `(N_BOOTSTRAP, n_seeds)` resample indices, shared across acc/leak so a paired
    decoupling-ratio bootstrap (same resample used for both quantities each draw) is possible."""
    point_curve = _mean_curve(curves, elapsed_grid)
    point_fit = metrics.fit_exponential_halflife(elapsed_grid, point_curve)
    if not point_fit["fit_ok"]:
        return point_fit["halflife"], float("nan"), float("nan"), point_fit["r2"], False, np.full(len(boot_idx), np.nan)

    boots = np.full(len(boot_idx), np.nan)
    for i, idx in enumerate(boot_idx):
        resample = [curves[j] for j in idx]
        curve = _mean_curve(resample, elapsed_grid)
        fit = metrics.fit_exponential_halflife(elapsed_grid, curve)
        if fit["fit_ok"] and np.isfinite(fit["halflife"]):
            boots[i] = fit["halflife"]
    finite = boots[~np.isnan(boots)]
    if len(finite) < 10:
        return point_fit["halflife"], float("nan"), float("nan"), point_fit["r2"], True, boots
    lo, hi = float(np.percentile(finite, 2.5)), float(np.percentile(finite, 97.5))
    return point_fit["halflife"], lo, hi, point_fit["r2"], True, boots


def main() -> int:
    out_rows = []
    ratio_rows = []
    for dataset in DATASETS:
        acc_matrix_path = REPO_ROOT / "results" / f"accuracy_matrix_{dataset}.csv"
        if not acc_matrix_path.exists():
            print(f"(skipping dataset {dataset}: {acc_matrix_path.name} not found)")
            continue
        acc_rows = _read_csv_rows(acc_matrix_path)

        for method, family in METHOD_FAMILY.items():
            acc_curves, tpr1_curves = _per_seed_curves(dataset, method, acc_rows)
            n_seeds = len(acc_curves)
            print(f"{dataset}/{method}: {n_seeds} seeds available")
            if n_seeds < 3:
                print(f"  skipping {dataset}/{method}: fewer than 3 seeds available")
                continue
            horizon = max(max(c) for c in acc_curves)
            elapsed_grid = list(range(0, horizon + 1))

            r = rng_mod.seeded(f"build_fig02.bootstrap::{dataset}", 0)
            boot_idx = r.integers(0, n_seeds, size=(N_BOOTSTRAP, n_seeds))

            boots_by_quantity = {}
            for quantity, curves in (("acc", acc_curves), ("leak", tpr1_curves)):
                halflife, lo, hi, r2, fit_ok, boots = _bootstrap_halflife(curves, elapsed_grid, boot_idx)
                censored = not fit_ok
                boots_by_quantity[quantity] = boots
                out_rows.append({
                    "dataset": dataset, "method": method, "family": family.value, "quantity": quantity,
                    "halflife": halflife, "ci_lo": lo, "ci_hi": hi, "r2": r2,
                    "fit_ok": fit_ok, "horizon_tasks": horizon, "censored": censored, "n_seeds": n_seeds,
                })
                status = "no decay detected (lower bound only)" if censored else f"halflife={halflife:.2f}"
                print(f"  [{quantity}] {status} (R2={r2:.3f})" if not np.isnan(r2) else f"  [{quantity}] {status}")

            # Decoupling ratio h_leak / h_acc: a paired bootstrap (same seed-resample draw feeds both
            # quantities each iteration), per 00_BUILD_PLAN.md's P4 protocol. Four cases, all reported
            # explicitly rather than forcing a number: both finite (real ratio + CI); leak censored,
            # acc finite (ratio -> infinity, maximal decoupling); **acc censored, leak finite** (ratio
            # -> 0 -- the reverse pattern: individual-level leakage genuinely fades while the retention
            # mechanism keeps accuracy essentially flat -- a real, different finding, not an error,
            # first seen on cub200/m3_fot); both censored (ratio undefined -- neither curve decays, so
            # there is nothing to compare).
            acc_row = next(
                r for r in out_rows if r["dataset"] == dataset and r["method"] == method and r["quantity"] == "acc"
            )
            leak_row = next(
                r for r in out_rows if r["dataset"] == dataset and r["method"] == method and r["quantity"] == "leak"
            )
            if acc_row["censored"] and leak_row["censored"]:
                ratio_rows.append({
                    "dataset": dataset, "method": method, "family": family.value,
                    "ratio_mean": float("nan"), "ratio_ci_lo": float("nan"), "ratio_ci_hi": float("nan"),
                    "censored": True, "n_seeds": n_seeds,
                })
                print("  [ratio] undefined -- neither accuracy nor leakage decays over the observed horizon")
            elif leak_row["censored"]:
                ratio_rows.append({
                    "dataset": dataset, "method": method, "family": family.value,
                    "ratio_mean": float("inf"), "ratio_ci_lo": float("inf"), "ratio_ci_hi": float("inf"),
                    "censored": True, "n_seeds": n_seeds,
                })
                print("  [ratio] undefined/infinite -- leakage shows no decay against a finite accuracy halflife")
            elif acc_row["censored"]:
                ratio_rows.append({
                    "dataset": dataset, "method": method, "family": family.value,
                    "ratio_mean": 0.0, "ratio_ci_lo": 0.0, "ratio_ci_hi": float("nan"),
                    "censored": True, "n_seeds": n_seeds,
                })
                print("  [ratio] ~0 -- leakage decays against an accuracy curve that shows no decay at all")
            else:
                valid = ~np.isnan(boots_by_quantity["acc"]) & ~np.isnan(boots_by_quantity["leak"])
                ratio_boots = boots_by_quantity["leak"][valid] / boots_by_quantity["acc"][valid]
                ratio_mean = leak_row["halflife"] / acc_row["halflife"]
                ratio_lo, ratio_hi = float(np.percentile(ratio_boots, 2.5)), float(np.percentile(ratio_boots, 97.5))
                ratio_rows.append({
                    "dataset": dataset, "method": method, "family": family.value,
                    "ratio_mean": ratio_mean, "ratio_ci_lo": ratio_lo, "ratio_ci_hi": ratio_hi,
                    "censored": False, "n_seeds": n_seeds,
                })
                print(f"  [ratio] h_leak/h_acc = {ratio_mean:.2f} [{ratio_lo:.2f}, {ratio_hi:.2f}]")

    out_csv = REPO_ROOT / "results" / "fig02_halflife.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)
    print(f"wrote {len(out_rows)} rows to {out_csv}")

    ratio_csv = REPO_ROOT / "results" / "decoupling_ratio.csv"
    with open(ratio_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(ratio_rows[0].keys()))
        w.writeheader()
        w.writerows(ratio_rows)
    print(f"wrote {len(ratio_rows)} rows to {ratio_csv}")

    config = {"seed": 0, "purpose": "P4 FIG02/TAB04 half-life + decoupling ratio (claim C1)", "datasets": DATASETS}
    manifest = provenance.run_manifest(config, seed=0)
    provenance.finalize(manifest, [out_csv, ratio_csv])
    return 0


if __name__ == "__main__":
    sys.exit(main())
