#!/usr/bin/env python3
"""FIG18 — Camelyon17 natural federation vs. Dirichlet (`00_BUILD_PLAN.md`'s "never cut" list).
Camelyon17 is binary-label (tumor/normal), so it cannot use the class-incremental split every other
dataset in this project uses (that needs >=2 disjoint class groups per task) -- it is domain-
incremental instead: one task per real hospital (`streams.build_stream(..., domain_field=hospital)`),
5 tasks total, same label space throughout. Two client-partition arms, both via the *same* already-
generic `build_stream`, no new stream code: **natural** (`n_clients=1` -- the hospital itself is the
sole client for its task, no Dirichlet anywhere) and **dirichlet** (`n_clients=10`, matching every
other dataset's client count, for direct comparability).

Real (non-shadow) accuracy matrix per (partition, method, seed) -- the accuracy half of FIG18,
mirroring `run_accuracy_matrix.py`'s role for FIG01. Scope: M0 only for this pilot (the essential
floor baseline; the question FIG18 asks -- does the client-partition scheme change the decoupling
finding -- is testable with one method), CIFAR-100/CUB-200/ImageNet-R's usual 5 seeds reduced to 3
here, pilot-scale like this session's other post-deadline additions.
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
from p3fcl.methods.m0_fedavg import FedAvgSequential  # noqa: E402

BACKBONE = "vit_base_patch16_224.augreg_in21k"
DATASET = "camelyon17"
SEEDS = [0, 1, 2]
PARTITIONS = {"natural": 1, "dirichlet": 10}
BETA = 0.5
METHOD_CFG = {"local_epochs": 30, "lr": 0.5}


def main() -> int:
    cache = features.load_cache(REPO_ROOT / "features", DATASET, BACKBONE, "train")
    X, y = cache["features"], cache["labels"]
    n_classes = int(y.max()) + 1
    feature_dim = X.shape[1]

    index = json.loads((REPO_ROOT / "datasets" / DATASET / "index.json").read_text())
    id_to_domain = {s["id"]: s["domain"] for s in index["samples"]}
    domain_field = np.array([id_to_domain[int(i)] for i in cache["ids"]])
    n_tasks = len(set(domain_field.tolist()))
    print(f"{n_tasks} domains (hospitals) present in the train split: {sorted(set(domain_field.tolist()))}")

    rows = []
    for partition, n_clients in PARTITIONS.items():
        for seed in SEEDS:
            stream = streams.build_stream(
                y, np.arange(len(y)), n_tasks=n_tasks, n_clients=n_clients, beta=BETA, seed=seed,
                domain_field=domain_field,
            )
            method = FedAvgSequential({"n_classes": n_classes, "feature_dim": feature_dim, **METHOD_CFG})
            result = sim.run(method, X, y, stream, seed=seed)
            acc_matrix = result["acc_matrix"]
            n_stream_tasks = len(stream)
            for k in range(n_stream_tasks):
                for T in range(k, n_stream_tasks):
                    acc = acc_matrix[T, k]
                    if np.isnan(acc):
                        continue
                    rows.append({
                        "dataset": DATASET, "partition": partition, "method": "m0_fedavg", "seed": seed,
                        "task_k": k, "task_T": T, "elapsed": T - k, "acc": float(acc),
                    })
            print(f"partition={partition} seed={seed}: final_avg_acc={result['final_avg_acc']:.4f} "
                  f"bwt={result['bwt']:.4f}")

    out_csv = REPO_ROOT / "results" / "accuracy_matrix_camelyon17.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows to {out_csv}")

    config = {
        "seed": 0, "purpose": "FIG18 Camelyon17 real accuracy matrix, natural vs dirichlet partition",
        "dataset": DATASET, "partitions": PARTITIONS, "seeds": SEEDS,
    }
    manifest = provenance.run_manifest(config, seed=0)
    provenance.finalize(manifest, [out_csv])
    return 0


if __name__ == "__main__":
    sys.exit(main())
