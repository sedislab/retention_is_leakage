#!/usr/bin/env python3
"""H11's highest-priority remaining item (`agents/OPEN_QUESTIONS.md`: "A1 still has no secure-agg
variant... Blast radius: High for reviewer reception... Prioritize it"): does A1 (cross-task LiRA)
survive if the adversary can only observe `Ledger.aggregate_view()` (the secure-aggregation view --
per-round SUM over clients, `client=-1`) instead of per-client records?

This is a **structural/mechanistic check on real federations**, not a shadow-scale statistical test --
it answers a *definedness* question first (can the attack even compute a score at all from the
aggregate?), which for the PROTOTYPE/GRAM families turns out to be answerable with a single real
federation, no calibration needed:

`shadow_runner._score_target`'s PROTOTYPE/GRAM branch filters
`[r for r in ledger if r.client == target["client"] and ...]` -- a specific client's own record.
`aggregate_view()` sets `client=-1` on every record it returns (by construction: it is A SUM over
clients, so it does not, and cannot, retain any client's individual identity). `target["client"]` is
always a real client index (never -1), so this filter can **never** match once the ledger is restricted
to the aggregate view -- the score is provably NaN for every target, every round. This is a fact about
the code's control flow, verified below on real federations for M4 (PROTOTYPE) and M8 (GRAM), not an
assumption.

For MODEL_DELTA (M0 here, the cleanest F1 case), the question is different and genuinely empirical:
`_reconstruct_running_w` computes a *weighted* average of per-client deltas
(`sum(n_touched_i * delta_i) / sum(n_touched_i)`), which is mathematically identical to what a
real secure-aggregation deployment gives if clients pre-weight their update by their (non-sensitive,
typically public) sample count before the secure sum -- the standard Bonawitz-et-al.-style
SecAgg+FedAvg design. `aggregate_view()` as currently implemented does an **unweighted** sum, which is
the *more conservative* (more-protective-of-privacy, i.e. harder-for-the-adversary) reading: no
per-client weights survive at all, only a plain sum and a participant *count*
(`meta["aggregated_over_clients"]`). This script checks how much the resulting model reconstruction
(and hence the attack's score) degrades under that harder assumption, on a real federation, before
deciding whether a full shadow-scale pilot (to get a calibrated TPR@1%FPR) is worth running.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "code" / "src"))

from p3fcl import features, streams  # noqa: E402
from p3fcl.artifacts import Family  # noqa: E402
from p3fcl.methods.m0_fedavg import FedAvgSequential  # noqa: E402
from p3fcl.methods.m4_proto import PrototypeFCL  # noqa: E402
from p3fcl.methods.m8_analytic import AnalyticFCL  # noqa: E402
from p3fcl.sim import run as sim_run  # noqa: E402

DATASET = "cifar100"
BACKBONE = "vit_base_patch16_224.augreg_in21k"
N_TASKS, N_CLIENTS, BETA, SEED = 10, 10, 0.5, 0


def _reconstruct_running_w_secure_agg(agg_records, n_rounds, feature_dim, n_classes):
    """Same shape/contract as `shadow_runner._reconstruct_running_w`, but from the aggregate view:
    unweighted mean of the summed delta over the number of *distinct* clients that contributed that
    round (all per-client weight information is gone in this view -- see module docstring)."""
    w = np.zeros((feature_dim, n_classes))
    w_by_round = []
    for r in range(n_rounds):
        recs = [rec for rec in agg_records if rec.family == Family.MODEL_DELTA and rec.round == r]
        if recs:
            rec = recs[0]
            n_clients = len(rec.meta.get("aggregated_over_clients", [])) or 1
            w = w + np.asarray(rec.payload) / n_clients
        w_by_round.append(w.copy())
    return w_by_round


def check_prototype_or_gram_undefined(method_name: str, method, family: Family, X, y, stream) -> dict:
    result = sim_run(method, X, y, stream, seed=SEED)
    ledger_records = result["ledger"].records
    agg_records = result["ledger"].aggregate_view()

    # A representative target: the first id in task 0's first client shard.
    target_client_shard = stream[0][0]
    target_id = target_client_shard.ids[0]
    target_client = target_client_shard.client
    target = {"client": target_client, "task": 0, "target_id": target_id}

    from p3fcl.shadow_runner import _score_target  # noqa: E402

    n_rounds = len(stream)
    full_score = _score_target(ledger_records, family, target, X, y, n_rounds)
    agg_score = _score_target(agg_records, family, target, X, y, n_rounds)
    return {
        "method": method_name,
        "score_definable_from_full_ledger": bool(not np.all(np.isnan(full_score))),
        "score_definable_from_aggregate": bool(not np.all(np.isnan(agg_score))),
    }


def check_model_delta_degradation(X, y, stream, n_classes, feature_dim) -> dict:
    cfg = {"n_classes": n_classes, "feature_dim": feature_dim, "local_epochs": 30, "lr": 0.5}
    result = sim_run(FedAvgSequential(cfg), X, y, stream, seed=SEED)
    ledger_records = result["ledger"].records
    agg_records = result["ledger"].aggregate_view()
    n_rounds = len(stream)

    # Reuse shadow_runner's own reconstruction for the "full ledger" (per-client) baseline.
    sys.path.insert(0, str(REPO_ROOT / "code" / "src"))
    from p3fcl.shadow_runner import _reconstruct_running_w  # noqa: E402

    w_full = _reconstruct_running_w(ledger_records, n_rounds, feature_dim, n_classes)
    w_secure = _reconstruct_running_w_secure_agg(agg_records, n_rounds, feature_dim, n_classes)

    rel_errs = []
    for r in range(n_rounds):
        num = np.linalg.norm(w_full[r] - w_secure[r])
        den = np.linalg.norm(w_full[r]) + 1e-12
        rel_errs.append(float(num / den))

    # Does the reconstructed model still separate classes at all (sanity: not garbage)?
    task9_ids = np.array(stream[9][0].ids[:5])
    logits_full = X[task9_ids] @ w_full[-1]
    logits_secure = X[task9_ids] @ w_secure[-1]
    acc_full = float(np.mean(np.argmax(logits_full, axis=1) == y[task9_ids]))
    acc_secure = float(np.mean(np.argmax(logits_secure, axis=1) == y[task9_ids]))

    return {
        "method": "m0_fedavg", "mean_relative_error_w": float(np.mean(rel_errs)),
        "final_relative_error_w": rel_errs[-1], "sample_acc_full_recon": acc_full,
        "sample_acc_secure_agg_recon": acc_secure,
    }


def main() -> int:
    cache = features.load_cache(REPO_ROOT / "features", DATASET, BACKBONE, "train")
    X, y = cache["features"], cache["labels"]
    n_classes = int(y.max()) + 1
    feature_dim = X.shape[1]
    stream = streams.build_stream(y, np.arange(len(y)), n_tasks=N_TASKS, n_clients=N_CLIENTS, beta=BETA, seed=SEED)

    print("=== structural check: PROTOTYPE (M4) and GRAM (M8) under aggregate_view() ===")
    m4_cfg = {"n_classes": n_classes, "feature_dim": feature_dim, "prototype_momentum": 0.0}
    r1 = check_prototype_or_gram_undefined("m4_proto", PrototypeFCL(m4_cfg), Family.PROTOTYPE, X, y, stream)
    print(r1)
    m8_cfg = {"n_classes": n_classes, "feature_dim": feature_dim, "ridge_lambda": 1.0}
    r2 = check_prototype_or_gram_undefined("m8_analytic", AnalyticFCL(m8_cfg), Family.GRAM, X, y, stream)
    print(r2)

    print("=== empirical check: MODEL_DELTA (M0) reconstruction under aggregate_view() ===")
    r3 = check_model_delta_degradation(X, y, stream, n_classes, feature_dim)
    print(r3)
    return 0


if __name__ == "__main__":
    sys.exit(main())
