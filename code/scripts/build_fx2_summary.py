#!/usr/bin/env python3
"""FX2 (`08_FIX_PLAN.md` §7b/7c/7d), leak side: the aggregation layer over `run_lira_pertask.py`'s
per-seed CSV+npz outputs. Reads only already-computed files (CLAUDE.md non-negotiable #3, no
recomputation of the attack itself) -- the npz sidecars' per-K-target raw eval-shadow score/label
arrays are what the hierarchical bootstrap resamples.

For a given (dataset, method, view, ablation): pools all available seeds' K-set targets at each
elapsed e in 0..6 for the point estimate (auc/tpr1/tpr01), applies the §7b chance-floor `no_signal`
guard to TPR@1%FPR's e=0 value using its own bootstrap CI, and (if it has signal) computes the
half-life and its bootstrap CI per §7c (resample seeds outer, targets inner, 2000 replicates).

Outputs:
- `results/a1_lira_fixedk_summary.csv`: point estimate + bootstrap CI per (dataset, method, family,
  view, ablation, elapsed) for auc/tpr1/tpr01.
- `results/fig02_halflife.csv` (leak rows only -- `quantity in {leak_tpr1, leak_auc}`; accuracy rows
  are a separate, not-yet-built script since they need `accuracy_matrix_<dataset>.csv`, a different
  input entirely).

PBS only if run over real shadow-scale data (the per-e bootstrap alone is 2000 replicates x 7 e-values
x n_seeds; still fast in absolute terms, but this project's login-node policy is about avoiding
in-process runs on real data on principle, not about a specific timing threshold).
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "code" / "src"))

from p3fcl import metrics  # noqa: E402
from p3fcl.halflife import (  # noqa: E402
    half_life,
    has_signal,
    hierarchical_bootstrap,
    leak_floor,
    no_signal_result,
    normalize,
)

N_REPLICATES = 2000
BOOTSTRAP_SEED = 0
E_MAX = 6


def _load_seed_npz(dataset: str, method: str, seed: int, view: str) -> dict:
    path = REPO_ROOT / "results" / f"a1_lira_pertask_{dataset}_{method}_seed{seed}_{view}.npz"
    with np.load(path) as z:
        return {
            "k_of_target": z["k_of_target"],
            "surf_trajectory": z["surf_trajectory"],
            "surf_last_round": z["surf_last_round"],
            "in_out": z["in_out"],
        }


def _seed_arrays(seed_npz: dict, ablation: str) -> tuple:
    """`(scores, labels)` for one seed: `scores[e]` is a dense `(n_shadows, n_K_targets)` array (NaN
    where `e` is out of range for that target's own `k`, i.e. `k+e` not in `[0, n_rounds)`), `labels`
    is `(n_shadows, n_K_targets)`. Dense/vectorised on purpose -- an earlier per-target Python-dict
    version (one dict per target, resampled with a Python-level list comprehension) was correct but
    made the hierarchical bootstrap's ~10^5 replicate-level calls dominated by pure Python loop
    overhead rather than the underlying numpy math; this representation lets `_resample_targets`/
    `_pool_at_e` be plain fancy-indexing and concatenation instead."""
    surf = seed_npz["surf_trajectory"] if ablation == "trajectory" else seed_npz["surf_last_round"]
    k_of_target = seed_npz["k_of_target"]
    n_shadows, n_targets, n_rounds = surf.shape
    scores = {}
    for e in range(E_MAX + 1):
        T = k_of_target + e
        valid = (T >= 0) & (T < n_rounds)
        col = np.full((n_shadows, n_targets), np.nan)
        valid_idx = np.where(valid)[0]
        if len(valid_idx):
            col[:, valid_idx] = surf[:, valid_idx, T[valid_idx]]
        scores[e] = col
    return scores, seed_npz["in_out"]


def _resample_targets(seed_data: tuple, rng) -> tuple:
    scores, labels = seed_data
    n_targets = labels.shape[1]
    idx = rng.integers(0, n_targets, size=n_targets)
    return {e: arr[:, idx] for e, arr in scores.items()}, labels[:, idx]


def _pool_at_e(resampled_seed_data: list, e: int) -> tuple:
    scores_list, labels_list = [], []
    for scores, labels in resampled_seed_data:
        col = scores[e]
        keep = ~np.isnan(col)
        if keep.any():
            scores_list.append(col[keep])
            labels_list.append(labels[keep])
    if not scores_list:
        return None, None
    return np.concatenate(scores_list), np.concatenate(labels_list).astype(int)


def _point_report_at_e(all_seed_records: list, e: int) -> dict | None:
    scores, labels = _pool_at_e(all_seed_records, e)
    if scores is None:
        return None
    return metrics.membership_report(scores, labels)


def _replicate_all_metrics(resampled_seed_data: list) -> dict:
    """One bootstrap replicate's full `{"auc"|"tpr1"|"tpr01": {e: value_or_None}}` -- computed once
    per replicate and reused for both the per-e CIs and the half-life CI, instead of a separate
    `hierarchical_bootstrap` call (and separate resampling) per (e, metric) combination. Measured
    speedup: ~2.6s for a single 2000-replicate `hierarchical_bootstrap` call at even a small synthetic
    scale means the naive "23 separate calls per ablation" design took several minutes per
    (dataset, method, view) -- infeasible across dozens of combinations. This consolidation does the
    outer/inner resampling exactly once per replicate."""
    out = {"auc": {}, "tpr1": {}, "tpr01": {}}
    for e in range(E_MAX + 1):
        scores, labels = _pool_at_e(resampled_seed_data, e)
        if scores is None:
            out["auc"][e] = out["tpr1"][e] = out["tpr01"][e] = None
            continue
        for key, val in (
            ("auc", metrics.roc_auc(scores, labels)),
            ("tpr1", metrics.tpr_at_fpr(scores, labels, 0.01)),
            ("tpr01", metrics.tpr_at_fpr(scores, labels, 0.001)),
        ):
            out[key][e] = None if np.isnan(val) else float(val)
    return out


def build_summary(dataset: str, method: str, family: str, view: str, seeds: list) -> tuple:
    """Returns `(fixedk_summary_rows, halflife_rows)` for every ablation."""
    npz_by_seed = {s: _load_seed_npz(dataset, method, s, view) for s in seeds}

    fixedk_rows, halflife_rows = [], []
    for ablation in ("trajectory", "last_round"):
        all_seed_records = [_seed_arrays(npz_by_seed[s], ablation) for s in seeds]

        point_by_e = {e: _point_report_at_e(all_seed_records, e) for e in range(E_MAX + 1)}

        # ONE hierarchical bootstrap pass for this ablation: every replicate computes every (metric,
        # e) value at once (`_replicate_all_metrics`), reused below for both the per-e CIs and the
        # half-life CI.
        boot = hierarchical_bootstrap(
            all_seed_records, _resample_targets, _replicate_all_metrics,
            n_replicates=N_REPLICATES, seed=BOOTSTRAP_SEED,
        )
        replicates = boot["replicates"]  # list of `_replicate_all_metrics` dicts, one per replicate

        def _ci_for(metric_key: str, e: int) -> dict:
            vals = np.array([r[metric_key][e] for r in replicates if r[metric_key][e] is not None], dtype=float)
            n_censored = len(replicates) - len(vals)
            if n_censored > len(replicates) / 2 or len(vals) == 0:
                return {"ci_lo": None, "ci_hi": None}
            lo, hi = np.percentile(vals, [2.5, 97.5])
            return {"ci_lo": float(lo), "ci_hi": float(hi)}

        for e, report in point_by_e.items():
            if report is None:
                continue
            ci_auc, ci_tpr1, ci_tpr01 = _ci_for("auc", e), _ci_for("tpr1", e), _ci_for("tpr01", e)
            fixedk_rows.append({
                "dataset": dataset, "method": method, "family": family, "view": view,
                "ablation": ablation, "elapsed": e,
                "auc": report["auc"], "auc_ci_lo": ci_auc["ci_lo"], "auc_ci_hi": ci_auc["ci_hi"],
                "tpr1": report["tpr_at_1pct_fpr"], "tpr1_ci_lo": ci_tpr1["ci_lo"], "tpr1_ci_hi": ci_tpr1["ci_hi"],
                "tpr01": report["tpr_at_0.1pct_fpr"], "tpr01_ci_lo": ci_tpr01["ci_lo"], "tpr01_ci_hi": ci_tpr01["ci_hi"],
                "n_seeds": len(seeds), "n_shadows": "n/a (per-seed, see a1_lira_pertask_*)",
            })

        # Half-life, primary quantity = TPR@1%FPR (CLAUDE.md non-negotiable #5: TPR@1%FPR/0.1%FPR are
        # primary, never AUC alone) -- also report leak_auc as a secondary half-life for completeness.
        for quantity, metric_key, report_key in (
            ("leak_tpr1", "tpr1", "tpr_at_1pct_fpr"), ("leak_auc", "auc", "auc"),
        ):
            if point_by_e.get(0) is None:
                continue
            base_value = point_by_e[0][report_key]
            base_ci = _ci_for(metric_key, 0)
            floor = leak_floor(metric_key)
            if base_ci["ci_lo"] is None or not has_signal(base_value, base_ci["ci_lo"], floor, 0.005):
                hl = no_signal_result()
                hl_ci_lo = hl_ci_hi = None
            else:
                values_by_e = {e: point_by_e[e][report_key] for e in range(E_MAX + 1) if point_by_e.get(e) is not None}
                if len(values_by_e) < E_MAX + 1:
                    hl = no_signal_result()  # can't compute a defined half-life without every e present
                    hl_ci_lo = hl_ci_hi = None
                else:
                    hl = half_life(normalize(values_by_e, floor), E=E_MAX)

                    hl_replicates = []
                    for r in replicates:
                        vals = r[metric_key]
                        if any(vals[e] is None for e in range(E_MAX + 1)):
                            continue
                        try:
                            norm = normalize(vals, floor)
                        except ValueError:
                            continue
                        hl_replicates.append(half_life(norm, E=E_MAX)["halflife"])
                    n_censored = len(replicates) - len(hl_replicates)
                    if n_censored > len(replicates) / 2 or not hl_replicates:
                        hl_ci_lo = hl_ci_hi = None
                    else:
                        hl_ci_lo, hl_ci_hi = (float(v) for v in np.percentile(hl_replicates, [2.5, 97.5]))

            halflife_rows.append({
                "dataset": dataset, "method": method, "family": family, "view": view,
                "quantity": quantity, "ablation": ablation,
                "halflife": hl["halflife"], "halflife_int": hl["halflife_int"], "status": hl["status"],
                "ci_lo": hl_ci_lo, "ci_hi": hl_ci_hi, "horizon_E": E_MAX,
                "base_value": base_value, "base_ci_lo": base_ci["ci_lo"], "floor": floor,
                "halflife_expfit": "", "r2": "",
            })

    return fixedk_rows, halflife_rows


def main() -> int:
    if len(sys.argv) < 4:
        print("usage: build_fx2_summary.py <dataset> <method> <family> [view] [seeds...]", file=sys.stderr)
        return 2
    dataset, method, family = sys.argv[1], sys.argv[2], sys.argv[3]
    view = sys.argv[4] if len(sys.argv) > 4 else "full"
    seeds = [int(s) for s in sys.argv[5:]] if len(sys.argv) > 5 else [0, 1, 2, 3, 4]

    fixedk_rows, halflife_rows = build_summary(dataset, method, family, view, seeds)

    # Per-combo output files, not the shared `a1_lira_fixedk_summary.csv`/`fig02_halflife.csv`:
    # `code/scripts/pbs/fx2_leak_wave.pbs` runs up to 30 of these concurrently, and
    # read-modify-write on one shared CSV across concurrent PBS array tasks would race (lost
    # updates or a corrupted file, no file locking here). `merge_fx2_summary.py` combines every
    # per-combo file (plus whatever already exists in the shared files, e.g. from a standalone
    # m0_fedavg run) into the final shared CSVs as a separate, non-concurrent step.
    suffix = f"{dataset}_{method}_{view}"
    fixedk_csv = REPO_ROOT / "results" / f"a1_lira_fixedk_summary_{suffix}.csv"
    with open(fixedk_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(fixedk_rows[0].keys()))
        w.writeheader()
        w.writerows(fixedk_rows)

    halflife_csv = REPO_ROOT / "results" / f"fig02_halflife_{suffix}.csv"
    with open(halflife_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(halflife_rows[0].keys()))
        w.writeheader()
        w.writerows(halflife_rows)

    print(f"wrote {len(fixedk_rows)} rows to {fixedk_csv}, {len(halflife_rows)} rows to {halflife_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
