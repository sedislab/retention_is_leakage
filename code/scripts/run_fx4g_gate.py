#!/usr/bin/env python3
"""FX4g (08_FIX_PLAN.md §4g): automatic gate, simulation only (no shadows). For
datasets={cifar100,cub200,imagenet_r} x seeds={0,1,2} x methods={M0,M1,M2,M3,M5}, real (held-out
test-split) accuracy. Pass condition per (dataset, method != M0), using means over seeds:
  - BWT <= +0.02
  - last-task accuracy >= M0's last-task accuracy - 0.15
  - final average accuracy >= M0's final average accuracy

If a method fails, runs one tuning pass on a validation split carved from train (10%, stratified,
fixed seed) -- grid: lr in {0.1, 0.5}, local_epochs in {5, 30}, distillation_weight in {0.5, 1.0}
(M1 only; ignored for M2/M3/M5), selected by validation final average accuracy subject to BWT <= 0.02.
Reruns the gate with the selected config. If still failing, keeps the best config, marks GATE FAILED,
and continues (never stops the phase).

PBS only. Writes `results/fx4_gate.csv` and, if any grid search ran, `results/fx4_gate_grid.csv`.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "code" / "src"))

from p3fcl import features, provenance, sim, streams  # noqa: E402
from p3fcl import rng as rng_mod  # noqa: E402
from p3fcl.methods.m0_fedavg import FedAvgSequential  # noqa: E402
from p3fcl.methods.m1_glfc import GLFC  # noqa: E402
from p3fcl.methods.m2_target import TARGET  # noqa: E402
from p3fcl.methods.m3_fot import FOT  # noqa: E402
from p3fcl.methods.m5_hybrid_replay import HybridReplay  # noqa: E402

BACKBONE = "vit_base_patch16_224.augreg_in21k"
DATASETS = ["cifar100", "cub200", "imagenet_r"]
SEEDS = [0, 1, 2]
N_TASKS = 10
N_CLIENTS = 10
BETA = 0.5

DEFAULT_CFG = {
    "m0_fedavg": (FedAvgSequential, {"local_epochs": 30, "lr": 0.5}),
    "m1_glfc": (GLFC, {"local_epochs": 30, "lr": 0.5, "exemplar_budget": 10, "distillation_weight": 1.0, "temperature": 2.0}),
    "m2_target": (TARGET, {"local_epochs": 30, "lr": 0.5, "replay_ratio": 1.0, "n_synthetic_per_class": 20}),
    "m3_fot": (FOT, {"local_epochs": 30, "lr": 0.5, "subspace_rank": 8, "projection_strength": 1.0}),
    "m5_hybrid_replay": (HybridReplay, {"local_epochs": 30, "lr": 0.5, "buffer_size_per_class": 10}),
}
GATED_METHODS = ["m1_glfc", "m2_target", "m3_fot", "m5_hybrid_replay"]

GRID_LR = [0.1, 0.5]
GRID_EPOCHS = [5, 30]
GRID_DISTILL = [0.5, 1.0]  # only meaningful for M1


def _run_once(method_name, cfg, dataset, seed, X, y, X_test, y_test, n_classes, feature_dim):
    stream = streams.build_stream(y, np.arange(len(y)), n_tasks=N_TASKS, n_clients=N_CLIENTS, beta=BETA, seed=seed)
    eval_id_lists = streams.task_eval_sets(y_test, np.arange(len(y_test)), n_tasks=N_TASKS, seed=seed)
    eval_sets = [(X_test[ids], y_test[ids]) for ids in eval_id_lists]
    method_cls = DEFAULT_CFG[method_name][0] if method_name in DEFAULT_CFG else None
    method = method_cls({"n_classes": n_classes, "feature_dim": feature_dim, **cfg})
    result = sim.run(method, X, y, stream, seed=seed, eval_sets=eval_sets)
    T = len(stream)
    acc_matrix = result["acc_matrix"]
    last_task_acc = float(acc_matrix[T - 1, T - 1]) if not np.isnan(acc_matrix[T - 1, T - 1]) else float("nan")
    return {
        "final_avg_acc": result["final_avg_acc"], "last_task_acc": last_task_acc,
        "bwt": result["bwt"], "aia": result["avg_incremental_acc"],
    }


def _stratified_val_split(y_train, frac=0.1, seed=0):
    r = rng_mod.seeded("fx4g_gate::val_split", seed)
    val_idx = []
    for c in sorted(set(y_train.tolist())):
        class_idx = np.where(y_train == c)[0]
        k = max(1, int(round(len(class_idx) * frac)))
        chosen = class_idx[r.permutation(len(class_idx))[:k]]
        val_idx.extend(chosen.tolist())
    val_idx = np.array(sorted(val_idx))
    train_idx = np.array(sorted(set(range(len(y_train))) - set(val_idx.tolist())))
    return train_idx, val_idx


def main() -> int:
    gate_rows = []
    grid_rows = []

    for dataset in DATASETS:
        cache = features.load_cache(REPO_ROOT / "features", dataset, BACKBONE, "train")
        X, y = cache["features"], cache["labels"]
        n_classes = int(y.max()) + 1
        feature_dim = X.shape[1]
        test_cache = features.load_cache(REPO_ROOT / "features", dataset, BACKBONE, "test")
        X_test, y_test = test_cache["features"], test_cache["labels"]

        m0_by_seed = {}
        for seed in SEEDS:
            m0_by_seed[seed] = _run_once("m0_fedavg", DEFAULT_CFG["m0_fedavg"][1], dataset, seed, X, y, X_test, y_test, n_classes, feature_dim)
            gate_rows.append({
                "dataset": dataset, "method": "m0_fedavg", "seed": seed, "config_id": "default",
                "final_avg_acc": m0_by_seed[seed]["final_avg_acc"], "last_task_acc": m0_by_seed[seed]["last_task_acc"],
                "bwt": m0_by_seed[seed]["bwt"], "aia": m0_by_seed[seed]["aia"],
                "m0_final_avg_acc": m0_by_seed[seed]["final_avg_acc"], "m0_last_task_acc": m0_by_seed[seed]["last_task_acc"],
                "pass": "n/a (baseline)", "used_cfg_json": json.dumps(DEFAULT_CFG["m0_fedavg"][1], sort_keys=True),
            })
        m0_mean_final = float(np.mean([v["final_avg_acc"] for v in m0_by_seed.values()]))
        m0_mean_last = float(np.mean([v["last_task_acc"] for v in m0_by_seed.values()]))

        for method_name in GATED_METHODS:
            per_seed = {}
            for seed in SEEDS:
                per_seed[seed] = _run_once(method_name, DEFAULT_CFG[method_name][1], dataset, seed, X, y, X_test, y_test, n_classes, feature_dim)
            mean_final = float(np.mean([v["final_avg_acc"] for v in per_seed.values()]))
            mean_last = float(np.mean([v["last_task_acc"] for v in per_seed.values()]))
            mean_bwt = float(np.mean([v["bwt"] for v in per_seed.values()]))
            passed = (mean_bwt <= 0.02) and (mean_last >= m0_mean_last - 0.15) and (mean_final >= m0_mean_final)
            config_id = "default"
            used_cfg = DEFAULT_CFG[method_name][1]

            if not passed and method_name == "m1_glfc":
                # One tuning pass on a train-carved validation split (never test).
                train_idx, val_idx = _stratified_val_split(y, frac=0.1, seed=0)
                X_tr, y_tr = X[train_idx], y[train_idx]
                X_val, y_val = X[val_idx], y[val_idx]
                best = None
                for lr in GRID_LR:
                    for epochs in GRID_EPOCHS:
                        for dw in GRID_DISTILL:
                            grid_cfg = {**DEFAULT_CFG[method_name][1], "lr": lr, "local_epochs": epochs, "distillation_weight": dw}
                            # Validation-only run: train on the 90% train-carve, eval on the 10% val-carve.
                            stream_v = streams.build_stream(y_tr, np.arange(len(y_tr)), n_tasks=N_TASKS, n_clients=N_CLIENTS, beta=BETA, seed=0)
                            eval_id_lists_v = streams.task_eval_sets(y_val, np.arange(len(y_val)), n_tasks=N_TASKS, seed=0)
                            eval_sets_v = [(X_val[ids], y_val[ids]) for ids in eval_id_lists_v]
                            method_v = DEFAULT_CFG[method_name][0]({"n_classes": n_classes, "feature_dim": feature_dim, **grid_cfg})
                            result_v = sim.run(method_v, X_tr, y_tr, stream_v, seed=0, eval_sets=eval_sets_v)
                            grid_rows.append({
                                "dataset": dataset, "method": method_name, "lr": lr, "local_epochs": epochs,
                                "distillation_weight": dw, "val_final_avg_acc": result_v["final_avg_acc"],
                                "val_bwt": result_v["bwt"],
                            })
                            if result_v["bwt"] <= 0.02:
                                if best is None or result_v["final_avg_acc"] > best[1]:
                                    best = (grid_cfg, result_v["final_avg_acc"])
                if best is not None:
                    config_id = "tuned"
                    used_cfg = best[0]
                    per_seed = {}
                    for seed in SEEDS:
                        per_seed[seed] = _run_once(method_name, best[0], dataset, seed, X, y, X_test, y_test, n_classes, feature_dim)
                    mean_final = float(np.mean([v["final_avg_acc"] for v in per_seed.values()]))
                    mean_last = float(np.mean([v["last_task_acc"] for v in per_seed.values()]))
                    mean_bwt = float(np.mean([v["bwt"] for v in per_seed.values()]))
                    passed = (mean_bwt <= 0.02) and (mean_last >= m0_mean_last - 0.15) and (mean_final >= m0_mean_final)

            # `used_cfg_json` is the actual method_config_override this (dataset, method) needs
            # downstream (FX4i's wave V2 manifest reads it directly instead of re-deriving "tuned"
            # from `fx4_gate_grid.csv`'s selection rule).
            used_cfg_json = json.dumps(used_cfg, sort_keys=True)
            for seed in SEEDS:
                gate_rows.append({
                    "dataset": dataset, "method": method_name, "seed": seed, "config_id": config_id,
                    "final_avg_acc": per_seed[seed]["final_avg_acc"], "last_task_acc": per_seed[seed]["last_task_acc"],
                    "bwt": per_seed[seed]["bwt"], "aia": per_seed[seed]["aia"],
                    "m0_final_avg_acc": m0_mean_final, "m0_last_task_acc": m0_mean_last,
                    "pass": passed, "used_cfg_json": used_cfg_json,
                })
            status = "PASS" if passed else "GATE FAILED"
            print(f"{dataset}/{method_name} ({config_id}): {status} "
                  f"(final_avg_acc={mean_final:.4f} vs m0={m0_mean_final:.4f}, "
                  f"last_task_acc={mean_last:.4f} vs m0-0.15={m0_mean_last - 0.15:.4f}, bwt={mean_bwt:.4f})")

    out_csv = REPO_ROOT / "results" / "fx4_gate.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(gate_rows[0].keys()))
        w.writeheader()
        w.writerows(gate_rows)
    print(f"wrote {len(gate_rows)} rows to {out_csv}")

    if grid_rows:
        grid_csv = REPO_ROOT / "results" / "fx4_gate_grid.csv"
        with open(grid_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(grid_rows[0].keys()))
            w.writeheader()
            w.writerows(grid_rows)
        print(f"wrote {len(grid_rows)} rows to {grid_csv}")

    config = {"seed": 0, "purpose": "FX4g automatic gate (simulation only)", "datasets": DATASETS, "seeds": SEEDS}
    manifest = provenance.run_manifest(config, seed=0)
    outputs = [out_csv] + ([REPO_ROOT / "results" / "fx4_gate_grid.csv"] if grid_rows else [])
    provenance.finalize(manifest, outputs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
