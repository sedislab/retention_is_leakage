#!/usr/bin/env python3
"""FX4f (08_FIX_PLAN.md §4f): shows the one-task-lag fix (R2) in one table. For M1/M2/M5 on CUB-200,
seed 0: each task k's own accuracy right after training on it (`acc_matrix[k,k]`, the diagonal --
exactly where R2's lag manifested, since a lagged method predicts almost nothing correctly on the
task it JUST finished) -- pre-fix (from the archived accuracy matrix, itself train-set accuracy) next
to post-fix (freshly computed, held-out test-set accuracy, per FX4a).

PBS only (08_FIX_PLAN.md §0.2 compute policy) -- run via `build/jobs/fx4f_lag_diagnostic.sh`.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "code" / "src"))

from p3fcl import features, provenance, sim, streams  # noqa: E402
from p3fcl.methods.m1_glfc import GLFC  # noqa: E402
from p3fcl.methods.m2_target import TARGET  # noqa: E402
from p3fcl.methods.m5_hybrid_replay import HybridReplay  # noqa: E402

BACKBONE = "vit_base_patch16_224.augreg_in21k"
DATASET = "cub200"
SEED = 0
N_TASKS = 10
N_CLIENTS = 10
BETA = 0.5
ARCHIVE_CSV = REPO_ROOT / "archive" / "2026-09-23_pre_fix" / "stale" / "results" / "accuracy_matrix_cub200.csv"

METHODS = {
    "m1_glfc": (GLFC, {"local_epochs": 30, "lr": 0.5, "exemplar_budget": 10, "distillation_weight": 1.0, "temperature": 2.0}),
    "m2_target": (TARGET, {"local_epochs": 30, "lr": 0.5, "replay_ratio": 1.0, "n_synthetic_per_class": 20}),
    "m5_hybrid_replay": (HybridReplay, {"local_epochs": 30, "lr": 0.5, "buffer_size_per_class": 10}),
}


def _pre_fix_diagonal(method_name: str) -> dict:
    with open(ARCHIVE_CSV, newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["method"] == method_name and r["seed"] == str(SEED)]
    out = {}
    for r in rows:
        if int(r["task_k"]) == int(r["task_T"]):
            out[int(r["task_k"])] = float(r["acc"])
    return out


def main() -> int:
    cache = features.load_cache(REPO_ROOT / "features", DATASET, BACKBONE, "train")
    X, y = cache["features"], cache["labels"]
    n_classes = int(y.max()) + 1
    feature_dim = X.shape[1]
    test_cache = features.load_cache(REPO_ROOT / "features", DATASET, BACKBONE, "test")
    X_test, y_test = test_cache["features"], test_cache["labels"]
    idx_test = np.arange(len(y_test))

    rows_out = []
    for method_name, (method_cls, cfg) in METHODS.items():
        pre_fix = _pre_fix_diagonal(method_name)

        stream = streams.build_stream(y, np.arange(len(y)), n_tasks=N_TASKS, n_clients=N_CLIENTS, beta=BETA, seed=SEED)
        eval_id_lists = streams.task_eval_sets(y_test, idx_test, n_tasks=N_TASKS, seed=SEED)
        eval_sets = [(X_test[ids], y_test[ids]) for ids in eval_id_lists]
        method = method_cls({"n_classes": n_classes, "feature_dim": feature_dim, **cfg})
        result = sim.run(method, X, y, stream, seed=SEED, eval_sets=eval_sets)
        acc_matrix = result["acc_matrix"]
        acc_matrix_train = result["acc_matrix_train"]

        for k in range(N_TASKS):
            rows_out.append({
                "dataset": DATASET, "method": method_name, "seed": SEED, "task_k": k,
                "pre_fix_acc_diagonal_train": pre_fix.get(k, ""),
                "post_fix_acc_diagonal_test": float(acc_matrix[k, k]) if not np.isnan(acc_matrix[k, k]) else "",
                "post_fix_acc_diagonal_train": float(acc_matrix_train[k, k]) if not np.isnan(acc_matrix_train[k, k]) else "",
            })
        print(f"{method_name}: pre-fix diag={[round(pre_fix.get(k, float('nan')), 3) for k in range(N_TASKS)]}")
        print(f"{method_name}: post-fix test diag={[round(float(acc_matrix[k, k]), 3) if not np.isnan(acc_matrix[k, k]) else None for k in range(N_TASKS)]}")

    out_csv = REPO_ROOT / "results" / "fx4_lag_diagnostic.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        w.writeheader()
        w.writerows(rows_out)
    print(f"wrote {len(rows_out)} rows to {out_csv}")

    config = {"seed": SEED, "purpose": "FX4f lag diagnostic: pre- vs post-fix per-task diagonal accuracy", "dataset": DATASET}
    manifest = provenance.run_manifest(config, seed=SEED)
    provenance.finalize(manifest, [out_csv])
    return 0


if __name__ == "__main__":
    sys.exit(main())
