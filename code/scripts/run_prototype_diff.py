#!/usr/bin/env python3
"""A3 deliverable: leakage as a function of increment size, on real cached features
(`04_METHODS_AND_ATTACKS.md`: "the quantity that matters is the increment size, and leakage as a
function of that increment is the result"). Runs `methods.m4_proto.PrototypeFCL` (momentum=0,
per-round-mean convention -- the exact-sum case) at several client counts to vary the realised
per-(client, class, round) shard size, then measures how well the exactly-recovered sum/mean
*approximates an individual sample* within that shard: `best_match_cosine` is the cosine similarity
between the recovered direction and the closest true member of the shard it came from. At
increment=1 this is trivially 1.0 (the "sum" IS that one sample); the question is how fast it decays
as the shard grows. Also runs the F7 (counts) ablation on/off to show the scaled-vs-direction-only
distinction (`attacks/prototype_diff.py`'s module docstring explains why cosine similarity alone
cannot show the ablation's effect).
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
from p3fcl.attacks.prototype_diff import PrototypeDifferenceAttack  # noqa: E402
from p3fcl.methods.m4_proto import PrototypeFCL  # noqa: E402

BACKBONE = "vit_base_patch16_224.augreg_in21k"
DATASET = "cifar100"
N_TASKS = 10
CLIENT_COUNTS = [2, 5, 10, 20, 50]  # more clients -> smaller average per-client-per-class shard
SEED = 0


def main() -> int:
    cache = features.load_cache(REPO_ROOT / "features", DATASET, BACKBONE, "train")
    X, y = cache["features"], cache["labels"]
    idx = np.arange(len(y))
    n_classes = int(y.max()) + 1

    rows = []
    attack = PrototypeDifferenceAttack({})

    for n_clients in CLIENT_COUNTS:
        stream = streams.build_stream(y, idx, n_tasks=N_TASKS, n_clients=n_clients, beta=0.5, seed=SEED)
        method = PrototypeFCL({"n_classes": n_classes, "feature_dim": X.shape[1], "prototype_momentum": 0.0})
        result = sim.run(method, X, y, stream, seed=SEED)
        ledger = result["ledger"]

        clients_seen = sorted({r.client for r in ledger})
        for client in clients_seen:
            for class_id in range(n_classes):
                recs_with = attack.run(ledger, client=client, class_id=class_id, momentum=0.0, use_counts=True)
                recs_without = attack.run(ledger, client=client, class_id=class_id, momentum=0.0, use_counts=False)
                for rec_w, rec_wo in zip(recs_with, recs_without):
                    if rec_w["increment"] is None or rec_w["increment"] == 0:
                        continue
                    client_task_ids = [
                        i for shard in stream[rec_w["task"]] if shard.client == client for i in shard.ids
                    ]
                    true_class_ids = [i for i in client_task_ids if y[i] == class_id]
                    if not true_class_ids:
                        continue
                    true_members = X[true_class_ids]

                    direction = rec_w["direction"]
                    sims = true_members @ direction / (
                        np.linalg.norm(true_members, axis=1) * np.linalg.norm(direction) + 1e-12
                    )
                    best_match_cosine = float(np.max(sims))

                    # Control: is this recovered direction unusually close to ITS OWN contributors,
                    # or just to "any same-class sample" (tight clustering alone would inflate
                    # best_match_cosine with zero individual-level signal)? Compare against an
                    # equally-sized random same-class pool drawn from outside this shard.
                    other_class_ids = np.setdiff1d(np.where(y == class_id)[0], true_class_ids)
                    r = rng_mod.seeded(f"a3_baseline::{n_clients}::{client}::{class_id}::{rec_w['round']}", SEED)
                    if len(other_class_ids) > 0:
                        pool_size = min(len(true_class_ids), len(other_class_ids))
                        baseline_ids = other_class_ids[r.permutation(len(other_class_ids))[:pool_size]]
                        baseline_members = X[baseline_ids]
                        baseline_sims = baseline_members @ direction / (
                            np.linalg.norm(baseline_members, axis=1) * np.linalg.norm(direction) + 1e-12
                        )
                        baseline_best_match_cosine = float(np.max(baseline_sims))
                    else:
                        baseline_best_match_cosine = float("nan")

                    true_sum = true_members.sum(axis=0)
                    l2_err_with_counts = float(np.linalg.norm(rec_w["recovered_sum"] - true_sum))
                    # "without counts" has no recovered_sum (direction only) -- the point of the ablation.
                    l2_err_without_counts_using_mean_as_sum = float(
                        np.linalg.norm(rec_wo["direction"] - true_sum)
                    )

                    rows.append({
                        "n_clients": n_clients, "client": client, "class": class_id,
                        "task": rec_w["task"], "round": rec_w["round"],
                        "increment": rec_w["increment"],
                        "best_match_cosine": best_match_cosine,
                        "baseline_best_match_cosine": baseline_best_match_cosine,
                        "attribution_premium": best_match_cosine - baseline_best_match_cosine,
                        "l2_err_with_counts": l2_err_with_counts,
                        "l2_err_without_counts": l2_err_without_counts_using_mean_as_sum,
                    })
        print(f"n_clients={n_clients}: {len(rows)} rows so far")

    out_csv = REPO_ROOT / "results" / "a3_prototype_diff.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows to {out_csv}")

    config = {"seed": SEED, "purpose": "A3 prototype-difference leakage vs increment size", "dataset": DATASET}
    manifest = provenance.run_manifest(config, seed=SEED)
    provenance.finalize(manifest, [out_csv])
    return 0


if __name__ == "__main__":
    sys.exit(main())
