#!/usr/bin/env python3
"""P4 deliverable: the real (full-population, non-shadow) accuracy matrix per (method, seed) on the
same stream A1's shadows perturb -- the accuracy half of FIG01's decoupling comparison. Cheap (one
real federation per method x seed, not thousands of shadows); still goes through qsub per cluster
policy since it is real training on real data, however small.

Usage: `run_accuracy_matrix.py [dataset]` (default `cifar100`) -- `n_classes`/`feature_dim` are read
from the cache, so this runs unchanged against any dataset with a real feature cache under
`features/` (confirmed: `cifar100`, `cub200`, `imagenet_r`, all `vit_base_patch16_224.augreg_in21k`;
Camelyon17 has its own script, `run_accuracy_matrix_camelyon17.py`, since it needs a domain-
incremental stream and two client-partition arms rather than this script's class-incremental one).
`n_tasks=10` needs the class count to divide evenly (200 for CUB-200/ImageNet-R, 100 for CIFAR-100 --
both divide by 10).

FX4a (08_FIX_PLAN.md R1): accuracy is now measured on the held-out `test` split via `eval_sets`
(`streams.task_eval_sets`, the same task partition as the training stream), not on the training ids
themselves. `acc` is the primary (test) column; `acc_train` is kept alongside as a diagnostic, never
reported as the headline number.

FX4g (08_FIX_PLAN.md §4g): "Then regenerate utility results" -- this now reads `results/fx4_gate.csv`
and, for any (dataset, method) the gate tuned (`config_id == "tuned"`), uses that row's
`used_cfg_json` in place of the TAB05 headline default, so the headline numbers reflect whichever
config the gate actually accepted rather than silently reporting a config the gate itself flagged as
failing. Absent `fx4_gate.csv`, every method uses its untuned default, same as before FX4g.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "code" / "src"))

from p3fcl import features, provenance, sim, streams  # noqa: E402
from p3fcl.experiment import METHODS, attacked_method_config  # noqa: E402
from p3fcl.shadow_runner import METHOD_REGISTRY  # noqa: E402

BACKBONE = "vit_base_patch16_224.augreg_in21k"
DATASET = sys.argv[1] if len(sys.argv) > 1 else "cifar100"
N_TASKS = 10
N_CLIENTS = 10
BETA = 0.5
SEEDS = [0, 1, 2, 3, 4]


def main() -> int:
    cache = features.load_cache(REPO_ROOT / "features", DATASET, BACKBONE, "train")
    X, y = cache["features"], cache["labels"]
    n_classes = int(y.max()) + 1
    feature_dim = X.shape[1]

    test_cache = features.load_cache(REPO_ROOT / "features", DATASET, BACKBONE, "test")
    X_test, y_test = test_cache["features"], test_cache["labels"]
    idx_test = np.arange(len(y_test))

    rows = []
    selected = [sys.argv[2]] if len(sys.argv) > 2 else METHODS
    manifest = provenance.run_manifest(dict(phase="FX9-4", hypothesis="H2", dataset=DATASET, methods=selected, seeds=SEEDS), seed=0)
    for method_name in selected:
        method_cls = METHOD_REGISTRY[method_name][0]
        method_cfg = attacked_method_config(method_name, n_classes, feature_dim, DATASET)
        for seed in SEEDS:
            stream = streams.build_stream(
                y, np.arange(len(y)), n_tasks=N_TASKS, n_clients=N_CLIENTS, beta=BETA, seed=seed
            )
            eval_id_lists = streams.task_eval_sets(y_test, idx_test, n_tasks=N_TASKS, seed=seed)
            eval_sets = [(X_test[ids], y_test[ids]) for ids in eval_id_lists]

            method = method_cls({"n_classes": n_classes, "feature_dim": feature_dim, **method_cfg})
            result = sim.run(method, X, y, stream, seed=seed, eval_sets=eval_sets)
            acc_matrix = result["acc_matrix"]
            acc_matrix_train = result["acc_matrix_train"]
            for k in range(N_TASKS):
                for T in range(k, N_TASKS):
                    acc = acc_matrix[T, k]
                    acc_train = acc_matrix_train[T, k]
                    if np.isnan(acc) and np.isnan(acc_train):
                        continue
                    rows.append({
                        "dataset": DATASET, "method": method_name, "seed": seed, "task_k": k,
                        "task_T": T, "elapsed": T - k,
                        "acc": float(acc) if not np.isnan(acc) else "",
                        "acc_train": float(acc_train) if not np.isnan(acc_train) else "",
                    })
            print(f"{method_name} seed={seed}: final_avg_acc={result['final_avg_acc']:.4f} "
                  f"bwt={result['bwt']:.4f} (train: {result['final_avg_acc_train']:.4f} / "
                  f"{result['bwt_train']:.4f})")

    suffix = "_" + selected[0] if len(selected) == 1 else ""
    out_csv = REPO_ROOT / "results" / f"accuracy_matrix_{DATASET}{suffix}.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows to {out_csv}")

    provenance.finalize(manifest, [out_csv])
    return 0


if __name__ == "__main__":
    sys.exit(main())
