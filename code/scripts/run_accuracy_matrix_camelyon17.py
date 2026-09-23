#!/usr/bin/env python3
"""FIG18 — Camelyon17 natural federation vs. Dirichlet, matched-client redesign
(`08_FIX_PLAN.md`'s H13 fix, R8: the original pilot's "natural" arm had 1 client per task and its
"dirichlet" arm had 10 -- two variables changed at once, partition STRATEGY and client COUNT, so an
observed difference could not be attributed to either alone; `agents/OPEN_QUESTIONS.md`'s H13 entry
correctly flags this as OPEN/confounded, not REFUTED). Camelyon17 is binary-label (tumor/normal), so
it cannot use the class-incremental split every other dataset in this project uses -- it is domain-
incremental instead: one task per real hospital (`streams.build_stream(..., domain_field=hospital)`),
5 tasks total, same label space throughout.

Both client-partition arms now use `n_clients=5` (matching the hospital count), varying only HOW a
hospital's task-data splits into those 5 clients:
  - **natural**: `client_field="slide"` -- groups by physical microscopy slide (a genuine, indivisible
    unit within a hospital; `p3fcl.get_data.prepare_camelyon17()` records it per sample from
    `Camelyon17Dataset.metadata_fields`), distributed across 5 buckets by a seeded shuffle + round-
    robin (`streams.natural_chunked_partition`) -- real structure, not random, but still admits
    seed-to-seed variance (unlike the original pilot's `n_clients=1` arm, which had none).
  - **dirichlet**: the existing per-class Dirichlet(beta) split, same `n_clients=5`, ignoring slide
    structure entirely.

This isolates "real structure vs. random split" from "how many clients," which is the actual
comparison H13 needs. Real (non-shadow) accuracy matrix per (partition, method, seed) -- the accuracy
half of FIG18, mirroring `run_accuracy_matrix.py`'s role for FIG01, including FX4a's real test-split
`eval_sets` fix (this script's predecessor used `acc_matrix_train` only, explicitly marked
"do not cite"). Scope: M0 only (the essential floor baseline; the question FIG18 asks -- does the
client-partition scheme change the decoupling finding -- is testable with one method), 3 seeds,
matching every other post-deadline Camelyon17 addition's pilot scope.
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
N_CLIENTS = 5  # matched across both arms -- the number of real hospitals
PARTITIONS = ["natural", "dirichlet"]
BETA = 0.5
METHOD_CFG = {"local_epochs": 30, "lr": 0.5}


def _field_array(index: dict, field_name: str, ids: np.ndarray) -> np.ndarray:
    id_to_val = {s["id"]: s[field_name] for s in index["samples"]}
    return np.array([id_to_val[int(i)] for i in ids])


def main() -> int:
    cache = features.load_cache(REPO_ROOT / "features", DATASET, BACKBONE, "train")
    X, y = cache["features"], cache["labels"]
    n_classes = int(y.max()) + 1
    feature_dim = X.shape[1]

    test_cache = features.load_cache(REPO_ROOT / "features", DATASET, BACKBONE, "test")
    X_test, y_test = test_cache["features"], test_cache["labels"]

    index = json.loads((REPO_ROOT / "datasets" / DATASET / "index.json").read_text())
    domain_field = _field_array(index, "domain", cache["ids"])
    slide_field = _field_array(index, "slide", cache["ids"])
    domain_field_test = _field_array(index, "domain", test_cache["ids"])
    n_tasks = len(set(domain_field.tolist()))
    print(f"{n_tasks} domains (hospitals) present in the train split: {sorted(set(domain_field.tolist()))}")

    rows = []
    for partition in PARTITIONS:
        client_field = slide_field if partition == "natural" else None
        for seed in SEEDS:
            stream = streams.build_stream(
                y, np.arange(len(y)), n_tasks=n_tasks, n_clients=N_CLIENTS, beta=BETA, seed=seed,
                domain_field=domain_field, client_field=client_field,
            )
            eval_id_lists = streams.task_eval_sets(
                y_test, np.arange(len(y_test)), n_tasks=n_tasks, seed=seed, domain_field_eval=domain_field_test,
            )
            eval_sets = [(X_test[ids], y_test[ids]) for ids in eval_id_lists]

            method = FedAvgSequential({"n_classes": n_classes, "feature_dim": feature_dim, **METHOD_CFG})
            result = sim.run(method, X, y, stream, seed=seed, eval_sets=eval_sets)
            acc_matrix = result["acc_matrix"]
            acc_matrix_train = result["acc_matrix_train"]
            n_stream_tasks = len(stream)
            for k in range(n_stream_tasks):
                for T in range(k, n_stream_tasks):
                    acc = acc_matrix[T, k]
                    acc_train = acc_matrix_train[T, k]
                    if np.isnan(acc) and np.isnan(acc_train):
                        continue
                    rows.append({
                        "dataset": DATASET, "partition": partition, "method": "m0_fedavg", "seed": seed,
                        "n_clients": N_CLIENTS, "task_k": k, "task_T": T, "elapsed": T - k,
                        "acc": float(acc) if not np.isnan(acc) else "",
                        "acc_train": float(acc_train) if not np.isnan(acc_train) else "",
                    })
            print(f"partition={partition} seed={seed}: final_avg_acc={result['final_avg_acc']:.4f} "
                  f"bwt={result['bwt']:.4f} (train: {result['final_avg_acc_train']:.4f} / "
                  f"{result['bwt_train']:.4f})")

    out_csv = REPO_ROOT / "results" / "accuracy_matrix_camelyon17.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows to {out_csv}")

    config = {
        "seed": 0, "purpose": "FIG18 v2 Camelyon17 real accuracy matrix, matched-5-client natural vs dirichlet",
        "dataset": DATASET, "partitions": PARTITIONS, "n_clients": N_CLIENTS, "seeds": SEEDS,
    }
    manifest = provenance.run_manifest(config, seed=0)
    provenance.finalize(manifest, [out_csv])
    return 0


if __name__ == "__main__":
    sys.exit(main())
