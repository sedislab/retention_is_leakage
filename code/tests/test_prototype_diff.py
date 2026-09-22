from __future__ import annotations

import numpy as np
from p3fcl import sim
from p3fcl.attacks.prototype_diff import PrototypeDifferenceAttack, recover_direction_and_sum
from p3fcl.methods.m4_proto import PrototypeFCL
from p3fcl.streams import build_stream


def _synthetic(n_classes=6, d=8, per_class=40, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((n_classes, d)) * 3
    X, y = [], []
    for c in range(n_classes):
        X.append(centers[c] + rng.standard_normal((per_class, d)) * 0.3)
        y.append(np.full(per_class, c))
    return np.concatenate(X), np.concatenate(y)


def test_recover_direction_and_sum_momentum_zero_is_exact():
    mean_t = np.array([1.0, 2.0, 3.0])
    direction, recovered_sum, increment = recover_direction_and_sum(mean_t, count_t=5, momentum=0.0)
    np.testing.assert_allclose(direction, mean_t)
    np.testing.assert_allclose(recovered_sum, mean_t * 5)
    assert increment == 5


def test_recover_direction_and_sum_without_counts_gives_direction_only():
    mean_t = np.array([1.0, 2.0, 3.0])
    direction, recovered_sum, increment = recover_direction_and_sum(mean_t, count_t=None, momentum=0.0)
    np.testing.assert_allclose(direction, mean_t)
    assert recovered_sum is None and increment is None


def test_recover_direction_and_sum_inverts_momentum_blend():
    true_new_mean = np.array([2.0, 0.0])
    old_mean = np.array([0.0, 0.0])
    momentum = 0.5
    blended = momentum * old_mean + (1 - momentum) * true_new_mean
    direction, recovered_sum, increment = recover_direction_and_sum(
        blended, count_t=8, mean_prev=old_mean, count_prev=3, momentum=momentum
    )
    np.testing.assert_allclose(direction, true_new_mean)
    assert increment == 5  # 8 - 3
    np.testing.assert_allclose(recovered_sum, true_new_mean * 5)


def test_attack_on_real_m4_ledger_momentum_zero_exact_sum_reconstruction():
    # per-round-mean convention (momentum=0): mean * count must equal the EXACT true shard sum.
    X, y = _synthetic(n_classes=4, per_class=20)
    idx = np.arange(len(y))
    stream = build_stream(y, idx, n_tasks=4, n_clients=1, beta=1.0, seed=0)
    method = PrototypeFCL({"n_classes": 4, "feature_dim": X.shape[1], "prototype_momentum": 0.0})
    result = sim.run(method, X, y, stream, seed=0)
    ledger = result["ledger"]

    attack = PrototypeDifferenceAttack({})
    for class_id in range(4):
        recs = attack.run(ledger, client=0, class_id=class_id, momentum=0.0, use_counts=True)
        assert len(recs) >= 1
        for rec in recs:
            # ground truth: the shard's true feature sum for this class at this task
            task_ids = [s for shard in stream[rec["task"]] for s in shard.ids]
            true_class_ids = [i for i in task_ids if y[i] == class_id]
            true_sum = X[true_class_ids].sum(axis=0)
            np.testing.assert_allclose(rec["recovered_sum"], true_sum, atol=1e-6)


def test_attack_increment_one_recovers_exact_single_sample():
    # a shard of exactly one sample of a class -> recovered sum IS that sample, exactly.
    X, y = _synthetic(n_classes=2, per_class=1)  # 1 sample per class
    idx = np.arange(len(y))
    stream = build_stream(y, idx, n_tasks=2, n_clients=1, beta=1.0, seed=0)
    method = PrototypeFCL({"n_classes": 2, "feature_dim": X.shape[1], "prototype_momentum": 0.0})
    result = sim.run(method, X, y, stream, seed=0)
    attack = PrototypeDifferenceAttack({})
    for class_id in range(2):
        recs = attack.run(result["ledger"], client=0, class_id=class_id, momentum=0.0)
        assert len(recs) == 1
        assert recs[0]["increment"] == 1
        true_sample = X[y == class_id][0]
        np.testing.assert_allclose(recs[0]["recovered_sum"], true_sample, atol=1e-6)


def test_without_counts_direction_still_exact_but_no_scaled_sum():
    X, y = _synthetic(n_classes=3, per_class=15)
    idx = np.arange(len(y))
    stream = build_stream(y, idx, n_tasks=3, n_clients=1, beta=1.0, seed=0)
    method = PrototypeFCL({"n_classes": 3, "feature_dim": X.shape[1], "prototype_momentum": 0.0})
    result = sim.run(method, X, y, stream, seed=0)
    attack = PrototypeDifferenceAttack({})
    for class_id in range(3):
        recs = attack.run(result["ledger"], client=0, class_id=class_id, momentum=0.0, use_counts=False)
        for rec in recs:
            assert rec["recovered_sum"] is None
            assert rec["increment"] is None
            assert rec["direction"] is not None
