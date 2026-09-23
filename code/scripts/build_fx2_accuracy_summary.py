#!/usr/bin/env python3
"""FX2 (`08_FIX_PLAN.md` §7b/7c/7d), accuracy side: mirrors `build_fx2_summary.py`'s leak-side
pipeline for the accuracy half of the decoupling ratio. Reads `results/accuracy_matrix_<dataset>.csv`
(already computed, CLAUDE.md non-negotiable #3) -- no attack/shadow data needed.

Normalized accuracy: `A(e) = mean_k[acc_k(k+e) - a0(k+e)] / mean_k[acc_k(k) - a0(k)]`, `k` over
`K_SET = {0, 1, 2, 3}`. Floor `a0(t) = 1/|classes seen after task t|` for class-incremental streams
(cifar100/cub200/imagenet_r, `n_classes` split evenly over `n_tasks=10`). Camelyon17 needs the
majority-class floor instead of the class-incremental one -- not wired here, matches FIG18's own
separate Wave V3 redesign track (`08_FIX_PLAN.md` re: the confounded Camelyon17 comparison).

Hierarchical bootstrap per §7c: resample seeds (outer, >= 3 available), then resample K (inner, the 4
origin tasks) WITH replacement -- the direct accuracy-side analogue of resampling targets on the leak
side (`halflife.hierarchical_bootstrap` is generic over what the "inner unit" means). Two separate
bootstrap passes are needed, not one: (1) the DENOMINATOR's own raw (un-normalized) value, whose CI
feeds the `has_signal` guard -- a normalized curve's `e=0` is trivially always 1.0 by construction, so
its own CI would be meaningless for that guard; (2) the full normalized curve, whose per-replicate
half-life gives the half-life's CI.

Output: appends `quantity="acc"` rows to `results/fig02_halflife.csv`.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "code" / "src"))

from p3fcl import provenance  # noqa: E402
from p3fcl.halflife import (  # noqa: E402
    acc_floor_class_incremental,
    half_life,
    has_signal,
    hierarchical_bootstrap,
    no_signal_result,
)

K_SET = [0, 1, 2, 3]
E_MAX = 6
N_REPLICATES = 2000
BOOTSTRAP_SEED = 0

N_TASKS = 10
N_CLASSES = {"cifar100": 100, "cub200": 200, "imagenet_r": 200}


def _load_acc_matrix(dataset: str, method: str) -> dict:
    """`{seed: {(task_k, elapsed): acc}}`."""
    path = REPO_ROOT / "results" / f"accuracy_matrix_{dataset}.csv"
    out: dict = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if row["method"] != method:
                continue
            seed = int(row["seed"])
            acc = float(row["acc"]) if row["acc"] != "" else float("nan")
            out.setdefault(seed, {})[(int(row["task_k"]), int(row["elapsed"]))] = acc
    return out


def _floor_at_task(dataset: str, t: int) -> float:
    classes_per_task = N_CLASSES[dataset] // N_TASKS
    return acc_floor_class_incremental((t + 1) * classes_per_task)


def _build_seed_diffs(dataset: str, method: str, seeds: list) -> list:
    """One `{k: {e: acc_k(k+e) - a0(k+e)}}` dict per seed, keeping only `k`s in `K_SET` that have a
    defined diff for every `e` in `0..E_MAX` (so every seed's inner resampling draws from a uniform
    population -- a `k` missing some `e` would silently bias whichever replicates happen to pick it)."""
    acc_by_seed = _load_acc_matrix(dataset, method)
    missing = [s for s in seeds if s not in acc_by_seed]
    if missing:
        raise ValueError(f"{dataset}/{method}: accuracy_matrix has no rows for seeds {missing}")

    seed_diffs = []
    for s in seeds:
        per_k = {}
        for k in K_SET:
            diffs = {}
            for e in range(E_MAX + 1):
                T = k + e
                if T >= N_TASKS:
                    continue
                acc = acc_by_seed[s].get((k, e))
                if acc is None or np.isnan(acc):
                    continue
                diffs[e] = acc - _floor_at_task(dataset, T)
            if len(diffs) == E_MAX + 1:
                per_k[k] = diffs
        seed_diffs.append(per_k)
    return seed_diffs


def _resample_k(per_k: dict, rng) -> dict:
    """Resamples the k in `per_k` with replacement, re-keyed by draw position (not `k`) so two draws
    of the same `k` don't collide in the returned dict."""
    ks = list(per_k.keys())
    idx = rng.integers(0, len(ks), size=len(ks))
    return {pos: per_k[ks[j]] for pos, j in enumerate(idx)}


def _pooled_diff_at_e(resampled_seeds: list, e: int):
    vals = [diffs[e] for per_k in resampled_seeds for diffs in per_k.values() if e in diffs]
    return float(np.mean(vals)) if vals else None


def _replicate_denominator(resampled_seeds: list):
    return _pooled_diff_at_e(resampled_seeds, 0)


def _replicate_normalized_curve(resampled_seeds: list):
    diffs_by_e = {e: _pooled_diff_at_e(resampled_seeds, e) for e in range(E_MAX + 1)}
    base = diffs_by_e[0]
    if base is None or base <= 0:
        return None
    return {e: (v / base if v is not None else None) for e, v in diffs_by_e.items()}


def build_accuracy_halflife(dataset: str, method: str, family: str, view: str, seeds: list) -> list:
    seed_diffs = _build_seed_diffs(dataset, method, seeds)

    point_diffs_by_e = {e: _pooled_diff_at_e(seed_diffs, e) for e in range(E_MAX + 1)}
    base_value = point_diffs_by_e[0]

    denom_boot = hierarchical_bootstrap(seed_diffs, _resample_k, _replicate_denominator, n_replicates=N_REPLICATES, seed=BOOTSTRAP_SEED)
    base_ci_lo = denom_boot["ci_lo"]

    if base_value is None or base_ci_lo is None or denom_boot["censored"] or not has_signal(base_value, base_ci_lo, 0.0, 0.02):
        hl, ci_lo, ci_hi = no_signal_result(), None, None
    elif any(point_diffs_by_e[e] is None for e in range(E_MAX + 1)):
        hl, ci_lo, ci_hi = no_signal_result(), None, None
    else:
        normalized = {e: point_diffs_by_e[e] / base_value for e in range(E_MAX + 1)}
        hl = half_life(normalized, E=E_MAX)

        curve_boot = hierarchical_bootstrap(seed_diffs, _resample_k, _replicate_normalized_curve, n_replicates=N_REPLICATES, seed=BOOTSTRAP_SEED)
        hl_vals = [
            half_life(r, E=E_MAX)["halflife"] for r in curve_boot["replicates"]
            if r is not None and all(r[e] is not None for e in range(E_MAX + 1))
        ]
        n_censored = N_REPLICATES - len(hl_vals)
        if n_censored > N_REPLICATES / 2 or not hl_vals:
            ci_lo, ci_hi = None, None
        else:
            ci_lo, ci_hi = (float(v) for v in np.percentile(hl_vals, [2.5, 97.5]))

    return [{
        "dataset": dataset, "method": method, "family": family, "view": view, "quantity": "acc",
        "ablation": "n/a", "halflife": hl["halflife"], "halflife_int": hl["halflife_int"],
        "status": hl["status"], "ci_lo": ci_lo, "ci_hi": ci_hi, "horizon_E": E_MAX,
        "base_value": base_value, "base_ci_lo": base_ci_lo, "floor": "per-task a0(t)=1/classes_seen",
        "halflife_expfit": "", "r2": "",
    }]


def main() -> int:
    if len(sys.argv) < 4:
        print("usage: build_fx2_accuracy_summary.py <dataset> <method> <family> [view] [seeds...]", file=sys.stderr)
        return 2
    dataset, method, family = sys.argv[1], sys.argv[2], sys.argv[3]
    view = sys.argv[4] if len(sys.argv) > 4 else "full"
    seeds = [int(s) for s in sys.argv[5:]] if len(sys.argv) > 5 else [0, 1, 2, 3, 4]

    rows = build_accuracy_halflife(dataset, method, family, view, seeds)

    out_csv = REPO_ROOT / "results" / "fig02_halflife.csv"
    existing = []
    if out_csv.exists():
        with open(out_csv, newline="") as f:
            existing = list(csv.DictReader(f))
    key_cols = ["dataset", "method", "family", "view", "quantity", "ablation"]
    keys_new = {tuple(str(r[c]) for c in key_cols) for r in rows}
    existing = [r for r in existing if tuple(str(r[c]) for c in key_cols) not in keys_new]
    all_rows = existing + rows
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)
    print(f"wrote {len(rows)} accuracy row(s) to {out_csv}")

    config = {
        "seed": 0, "purpose": "FX2 accuracy-side half-life (claim C1)",
        "dataset": dataset, "method": method, "family": family, "view": view, "seeds": seeds,
    }
    manifest = provenance.run_manifest(config, seed=0)
    provenance.finalize(manifest, [out_csv])
    return 0


if __name__ == "__main__":
    sys.exit(main())
