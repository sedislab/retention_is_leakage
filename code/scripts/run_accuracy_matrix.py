#!/usr/bin/env python3
"""P4 deliverable: the real (full-population, non-shadow) accuracy matrix per (method, seed) on the
same stream A1's shadows perturb -- the accuracy half of FIG01's decoupling comparison. Cheap (one
real federation per method x seed, not thousands of shadows); still goes through qsub per Kodiak
policy since it is real training on real data, however small.

Usage: `run_accuracy_matrix.py [dataset]` (default `cifar100`) -- `n_classes`/`feature_dim` are read
from the cache, so this runs unchanged against any dataset with a real feature cache under
`features/` (confirmed: `cifar100`, `cub200`, `imagenet_r`, `camelyon17`, all
`vit_base_patch16_224.augreg_in21k`). `n_tasks=10` needs the class count to divide evenly (200 for
CUB-200/ImageNet-R, 100 for CIFAR-100 -- both divide by 10).
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "code" / "src"))

from p3fcl import features, provenance, sim, streams  # noqa: E402
from p3fcl.methods.m0_fedavg import FedAvgSequential  # noqa: E402
from p3fcl.methods.m1_glfc import GLFC  # noqa: E402
from p3fcl.methods.m2_target import TARGET  # noqa: E402
from p3fcl.methods.m3_fot import FOT  # noqa: E402
from p3fcl.methods.m4_proto import PrototypeFCL  # noqa: E402
from p3fcl.methods.m5_hybrid_replay import HybridReplay  # noqa: E402
from p3fcl.methods.m8_analytic import AnalyticFCL  # noqa: E402

BACKBONE = "vit_base_patch16_224.augreg_in21k"
DATASET = sys.argv[1] if len(sys.argv) > 1 else "cifar100"
N_TASKS = 10
N_CLIENTS = 10
BETA = 0.5
SEEDS = [0, 1, 2, 3, 4]

# Same TAB05 headline configs as run_utility_baseline.py / shadow_runner._method_config -- not an
# attack-specific weakening.
METHODS = {
    "m0_fedavg": (FedAvgSequential, {"local_epochs": 30, "lr": 0.5}),
    "m1_glfc": (GLFC, {"local_epochs": 30, "lr": 0.5, "exemplar_budget": 10, "distillation_weight": 1.0, "temperature": 2.0}),
    "m2_target": (TARGET, {"local_epochs": 30, "lr": 0.5, "replay_ratio": 1.0, "n_synthetic_per_class": 20}),
    "m3_fot": (FOT, {"local_epochs": 30, "lr": 0.5, "subspace_rank": 8, "projection_strength": 1.0}),
    "m4_proto": (PrototypeFCL, {"prototype_momentum": 0.0}),
    "m5_hybrid_replay": (HybridReplay, {"local_epochs": 30, "lr": 0.5, "buffer_size_per_class": 10}),
    "m8_analytic": (AnalyticFCL, {"ridge_lambda": 1.0}),
}


def main() -> int:
    cache = features.load_cache(REPO_ROOT / "features", DATASET, BACKBONE, "train")
    X, y = cache["features"], cache["labels"]
    n_classes = int(y.max()) + 1
    feature_dim = X.shape[1]

    rows = []
    for method_name, (method_cls, method_cfg) in METHODS.items():
        for seed in SEEDS:
            stream = streams.build_stream(
                y, np.arange(len(y)), n_tasks=N_TASKS, n_clients=N_CLIENTS, beta=BETA, seed=seed
            )
            method = method_cls({"n_classes": n_classes, "feature_dim": feature_dim, **method_cfg})
            result = sim.run(method, X, y, stream, seed=seed)
            acc_matrix = result["acc_matrix"]
            for k in range(N_TASKS):
                for T in range(k, N_TASKS):
                    acc = acc_matrix[T, k]
                    if np.isnan(acc):
                        continue
                    rows.append(
                        {"dataset": DATASET, "method": method_name, "seed": seed, "task_k": k,
                         "task_T": T, "elapsed": T - k, "acc": float(acc)}
                    )
            print(f"{method_name} seed={seed}: final_avg_acc={result['final_avg_acc']:.4f} bwt={result['bwt']:.4f}")

    out_csv = REPO_ROOT / "results" / f"accuracy_matrix_{DATASET}.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows to {out_csv}")

    config = {"seed": 0, "purpose": "P4 real accuracy matrix per method x seed (FIG01)", "dataset": DATASET}
    manifest = provenance.run_manifest(config, seed=0)
    provenance.finalize(manifest, [out_csv])
    return 0


if __name__ == "__main__":
    sys.exit(main())
