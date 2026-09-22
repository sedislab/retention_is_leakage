#!/usr/bin/env python3
"""A4 deliverable (H4's deciding test): client-arrival onset inference via CUSUM change-point
detection on the aggregate (secure-agg-visible) F1/F2/F5 signal, real CIFAR-100 federations.

**Why the stream config differs from every other P3 attack's `n_tasks=10, n_clients=10, beta=0.5`**:
verified empirically before running anything that this configuration gives *zero* post-round-0
client-arrival events (every client already has a nonzero Dirichlet share of every task) -- see
`attacks/onset_inference.py`'s module docstring. `n_clients=50, beta=0.02` gives ~10-15 genuine
late-arrival events per seed, a real, non-trivial signal to detect.

Calibration/evaluation split by SEED (CLAUDE.md non-negotiable #6): CUSUM's threshold, and the common
flag budget given to both baselines for a fair comparison, are chosen on the calibration seeds only;
reported precision/recall are pooled over the held-out evaluation seeds with exact Clopper-Pearson
intervals (CLAUDE.md non-negotiable #4).
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "code" / "src"))

from p3fcl import features, metrics, provenance, sim, streams  # noqa: E402
from p3fcl import rng as rng_mod  # noqa: E402
from p3fcl.artifacts import Family  # noqa: E402
from p3fcl.attacks.onset_inference import (  # noqa: E402
    aggregate_signal_series,
    cusum_changepoints,
    norm_spike_baseline,
    precision_at_tolerance,
    random_baseline,
    recall_at_tolerance,
    true_arrival_rounds,
)
from p3fcl.methods.m0_fedavg import FedAvgSequential  # noqa: E402
from p3fcl.methods.m4_proto import PrototypeFCL  # noqa: E402
from p3fcl.methods.m8_analytic import AnalyticFCL  # noqa: E402

BACKBONE = "vit_base_patch16_224.augreg_in21k"
DATASET = "cifar100"
N_TASKS = 10
N_CLIENTS = 50
BETA = 0.02
N_CALIB_SEEDS = 20
N_EVAL_SEEDS = 20
TOLERANCE = 2
CUSUM_THRESHOLD_GRID = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]

METHODS = {
    "m0_fedavg": (FedAvgSequential, Family.MODEL_DELTA, {"local_epochs": 30, "lr": 0.5}),
    "m4_proto": (PrototypeFCL, Family.PROTOTYPE, {"prototype_momentum": 0.0}),
    "m8_analytic": (AnalyticFCL, Family.GRAM, {"ridge_lambda": 1.0}),
}


def _run_one_seed(method_cls, method_cfg, family, X, y, n_classes, feature_dim, seed):
    stream = streams.build_stream(
        y, np.arange(len(y)), n_tasks=N_TASKS, n_clients=N_CLIENTS, beta=BETA, seed=seed
    )
    method = method_cls({"n_classes": n_classes, "feature_dim": feature_dim, **method_cfg})
    result = sim.run(method, X, y, stream, seed=seed)
    norm_series, _ = aggregate_signal_series(result["ledger"], N_TASKS, family)
    true_rounds = true_arrival_rounds(stream)
    return norm_series, true_rounds


def _pool_and_score(flags_by_seed, true_by_seed, tolerance):
    hp = hf = hr = ht = 0
    for flags, true_rounds in zip(flags_by_seed, true_by_seed):
        h, n = precision_at_tolerance(flags, true_rounds, tolerance)
        hp += h
        hf += n
        h, n = recall_at_tolerance(flags, true_rounds, tolerance)
        hr += h
        ht += n
    precision = hp / hf if hf else float("nan")
    recall = hr / ht if ht else float("nan")
    p_lo, p_hi = metrics.clopper_pearson(hp, hf) if hf else (float("nan"), float("nan"))
    r_lo, r_hi = metrics.clopper_pearson(hr, ht) if ht else (float("nan"), float("nan"))
    return {
        "precision": precision, "precision_ci_lo": p_lo, "precision_ci_hi": p_hi, "n_flagged": hf,
        "recall": recall, "recall_ci_lo": r_lo, "recall_ci_hi": r_hi, "n_true": ht,
    }


def main() -> int:
    cache = features.load_cache(REPO_ROOT / "features", DATASET, BACKBONE, "train")
    X, y = cache["features"], cache["labels"]
    n_classes = int(y.max()) + 1
    feature_dim = X.shape[1]

    rows = []
    for method_name, (method_cls, family, method_cfg) in METHODS.items():
        print(f"=== {method_name} ===")
        calib_series, calib_true = [], []
        for s in range(N_CALIB_SEEDS):
            series, true_rounds = _run_one_seed(method_cls, method_cfg, family, X, y, n_classes, feature_dim, s)
            calib_series.append(series)
            calib_true.append(true_rounds)
        n_calib_events = sum(len(t) for t in calib_true)
        print(f"  calibration: {N_CALIB_SEEDS} seeds, {n_calib_events} true arrival events")

        best_threshold, best_f1, best_k = None, -1.0, 1
        calib_std = float(np.mean([np.std(s) for s in calib_series])) or 1.0
        for thr_mult in CUSUM_THRESHOLD_GRID:
            threshold = thr_mult * calib_std
            flags_by_seed = [cusum_changepoints(s, threshold) for s in calib_series]
            score = _pool_and_score(flags_by_seed, calib_true, TOLERANCE)
            p, r = score["precision"], score["recall"]
            f1 = 2 * p * r / (p + r) if (p + r) and not np.isnan(p) and not np.isnan(r) and (p + r) > 0 else 0.0
            if f1 > best_f1:
                best_f1, best_threshold = f1, threshold
                best_k = max(1, round(np.mean([len(f) for f in flags_by_seed])))
        print(f"  calibrated CUSUM threshold={best_threshold:.3f} (F1={best_f1:.3f}), flag budget k={best_k}")

        eval_series, eval_true = [], []
        for s in range(N_CALIB_SEEDS, N_CALIB_SEEDS + N_EVAL_SEEDS):
            series, true_rounds = _run_one_seed(method_cls, method_cfg, family, X, y, n_classes, feature_dim, s)
            eval_series.append(series)
            eval_true.append(true_rounds)
        n_eval_events = sum(len(t) for t in eval_true)
        print(f"  evaluation: {N_EVAL_SEEDS} seeds, {n_eval_events} true arrival events")

        r_rng = rng_mod.seeded(f"run_onset_inference::random_baseline::{method_name}", 0)
        detectors = {
            "cusum": [cusum_changepoints(s, best_threshold) for s in eval_series],
            "norm_spike": [norm_spike_baseline(s, best_k) for s in eval_series],
            "random": [random_baseline(N_TASKS, best_k, r_rng) for _ in eval_series],
        }
        for detector_name, flags_by_seed in detectors.items():
            score = _pool_and_score(flags_by_seed, eval_true, TOLERANCE)
            rows.append({"method": method_name, "detector": detector_name, "flag_budget": best_k, **score})
            print(
                f"  [{detector_name}] precision={score['precision']:.3f} "
                f"[{score['precision_ci_lo']:.3f},{score['precision_ci_hi']:.3f}] "
                f"recall={score['recall']:.3f} n_flagged={score['n_flagged']} n_true={score['n_true']}"
            )

    out_csv = REPO_ROOT / "results" / "a4_onset_inference.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows to {out_csv}")

    config = {
        "seed": 0, "purpose": "A4 onset inference via CUSUM change-point detection (H4)",
        "dataset": DATASET, "n_tasks": N_TASKS, "n_clients": N_CLIENTS, "beta": BETA,
        "n_calib_seeds": N_CALIB_SEEDS, "n_eval_seeds": N_EVAL_SEEDS, "tolerance": TOLERANCE,
    }
    manifest = provenance.run_manifest(config, seed=0)
    provenance.finalize(manifest, [out_csv])
    return 0


if __name__ == "__main__":
    sys.exit(main())
