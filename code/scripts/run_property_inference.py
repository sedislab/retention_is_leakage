#!/usr/bin/env python3
"""A6 deliverable (H10's deciding test, carries claim C1): property-inference balanced accuracy vs
elapsed = T - k, on real cached features. F2+F7 (M4) only -- see `attacks/property_inference.py`'s
module docstring for the scoping rationale. Reports the real curve, whatever shape it has; per
`00_BUILD_PLAN.md`, "no decay detected" is a legitimate, strongly-stated result here, not a failure
to find one.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "code" / "src"))

from p3fcl import features, provenance, sim, streams  # noqa: E402
from p3fcl import rng as rng_mod  # noqa: E402
from p3fcl.attacks.property_inference import balanced_accuracy, infer_held_class  # noqa: E402
from p3fcl.methods.m4_proto import PrototypeFCL  # noqa: E402

BACKBONE = "vit_base_patch16_224.augreg_in21k"
DATASET = "cifar100"
N_TASKS = 10
N_CLIENTS = 10
SEED = 0
ELAPSED_GRID = list(range(-2, 11))  # T - k, from "before the task happened" to "9 tasks later"
N_NEGATIVE_SAMPLES_PER_ELAPSED = 300


def main() -> int:
    cache = features.load_cache(REPO_ROOT / "features", DATASET, BACKBONE, "train")
    X, y = cache["features"], cache["labels"]
    idx = np.arange(len(y))
    n_classes = int(y.max()) + 1

    stream = streams.build_stream(y, idx, n_tasks=N_TASKS, n_clients=N_CLIENTS, beta=0.5, seed=SEED)
    method = PrototypeFCL({"n_classes": n_classes, "feature_dim": X.shape[1], "prototype_momentum": 0.0})
    result = sim.run(method, X, y, stream, seed=SEED)
    ledger = result["ledger"]

    positives = []  # (client, class_id, k) that really happened
    for t, shards in enumerate(stream):
        for shard in shards:
            classes_here = sorted({int(y[i]) for i in shard.ids})
            for c in classes_here:
                positives.append((shard.client, c, t))
    print(f"{len(positives)} true (client, class, task) positives")

    all_clients = sorted({p[0] for p in positives})
    r = rng_mod.seeded("run_property_inference::negatives", SEED)

    rows = []
    for elapsed in ELAPSED_GRID:
        y_true, y_pred = [], []
        for client, class_id, k in positives:
            T = k + elapsed
            if T < 0:
                continue
            y_true.append(True)
            y_pred.append(infer_held_class(ledger, client, class_id, T))

        # negatives: (client, class) pairs that never actually co-occurred, evaluated at the same T
        # relative offset from a randomly matched k, so the "T" distribution matches the positives'.
        neg_drawn = 0
        attempts = 0
        positive_set = {(c, cl) for c, cl, _ in positives}
        while neg_drawn < N_NEGATIVE_SAMPLES_PER_ELAPSED and attempts < N_NEGATIVE_SAMPLES_PER_ELAPSED * 20:
            attempts += 1
            client = all_clients[r.integers(0, len(all_clients))]
            class_id = int(r.integers(0, n_classes))
            k = int(r.integers(0, N_TASKS))
            if (client, class_id) in positive_set:
                continue
            T = k + elapsed
            if T < 0:
                continue
            y_true.append(False)
            y_pred.append(infer_held_class(ledger, client, class_id, T))
            neg_drawn += 1

        y_true_arr = np.array(y_true)
        y_pred_arr = np.array(y_pred)
        bal_acc = balanced_accuracy(y_true_arr, y_pred_arr)
        rows.append({
            "elapsed": elapsed, "balanced_accuracy": bal_acc,
            "n_pos": int(np.sum(y_true_arr)), "n_neg": int(np.sum(~y_true_arr)),
        })
        print(f"elapsed={elapsed}: balanced_accuracy={bal_acc:.4f} (n_pos={rows[-1]['n_pos']}, n_neg={rows[-1]['n_neg']})")

    out_csv = REPO_ROOT / "results" / "a6_property_inference.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows to {out_csv}")

    config = {"seed": SEED, "purpose": "A6 property inference balanced accuracy vs elapsed (H10)", "dataset": DATASET}
    manifest = provenance.run_manifest(config, seed=SEED)
    provenance.finalize(manifest, [out_csv])
    return 0


if __name__ == "__main__":
    sys.exit(main())
