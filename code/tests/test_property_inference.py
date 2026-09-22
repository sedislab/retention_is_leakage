from __future__ import annotations

import numpy as np
import pytest
from p3fcl import sim
from p3fcl.attacks.property_inference import (
    ClientAttributionAttack,
    PropertyInferenceAttack,
    balanced_accuracy,
    infer_client_class_histogram,
    infer_held_class,
    marginal_frequency_baseline,
)
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


def _real_ledger(n_classes=6, n_tasks=3, n_clients=2):
    X, y = _synthetic(n_classes=n_classes)
    idx = np.arange(len(y))
    stream = build_stream(y, idx, n_tasks=n_tasks, n_clients=n_clients, beta=1.0, seed=0)
    method = PrototypeFCL({"n_classes": n_classes, "feature_dim": X.shape[1], "prototype_momentum": 0.0})
    result = sim.run(method, X, y, stream, seed=0)
    return result["ledger"], stream, X, y


def test_balanced_accuracy_basic():
    y_true = np.array([1, 1, 1, 0, 0, 0])
    y_pred = np.array([1, 1, 0, 0, 0, 1])
    assert balanced_accuracy(y_true, y_pred) == pytest.approx(2 / 3)


def test_infer_held_class_true_before_and_after_the_task():
    ledger, stream, X, y = _real_ledger()
    # find a (client, task, class) triple that actually happened
    rec = next(r for r in ledger if r.family.value == "F7")
    client, task = rec.client, rec.task
    held_classes = [c for c, n in enumerate(rec.payload["counts"]) if n > 0]
    assert held_classes
    class_id = held_classes[0]

    # before the task's round: no evidence yet
    assert infer_held_class(ledger, client, class_id, T=task - 1) is False
    # at and after the task's round: evidence persists (F2/F7, single-pass, class-incremental)
    assert infer_held_class(ledger, client, class_id, T=task) is True
    assert infer_held_class(ledger, client, class_id, T=task + 100) is True


def test_infer_client_class_histogram_matches_ground_truth():
    ledger, stream, X, y = _real_ledger()
    rec = next(r for r in ledger if r.family.value == "F7")
    hist = infer_client_class_histogram(ledger, rec.client, T=rec.round)
    true_ids = [i for shard in stream[rec.task] if shard.client == rec.client for i in shard.ids]
    true_hist = {}
    for i in true_ids:
        true_hist[int(y[i])] = true_hist.get(int(y[i]), 0) + 1
    assert hist == true_hist


def test_property_inference_attack_wraps_infer_held_class():
    ledger, stream, X, y = _real_ledger()
    rec = next(r for r in ledger if r.family.value == "F7")
    class_id = next(c for c, n in enumerate(rec.payload["counts"]) if n > 0)
    attack = PropertyInferenceAttack({})
    assert attack.run(ledger, client=rec.client, class_id=class_id, T=rec.round) is True


def test_marginal_frequency_baseline_respects_base_rate():
    rng = np.random.default_rng(0)
    y_true = np.array([True] * 20 + [False] * 80)
    preds = marginal_frequency_baseline(y_true, rng)
    # should be close to the 0.2 base rate, not 0.5
    assert 0.05 < preds.mean() < 0.4


def test_client_attribution_finds_the_true_contributing_client():
    ledger, stream, X, y = _real_ledger(n_clients=3)
    rec = next(r for r in ledger if r.family.value == "F2")
    class_id = next(int(k) for k in rec.payload)
    query = rec.payload[str(class_id)]  # the exact released prototype -- must attribute to `rec.client`
    attack = ClientAttributionAttack({})
    attributed = attack.run(ledger, query_feature=query, class_id=class_id, task=rec.task)
    assert attributed == rec.client


def test_client_attribution_fails_on_aggregate_view():
    # A5 does not survive secure aggregation: the aggregate view has no per-client records to match.
    ledger, stream, X, y = _real_ledger(n_clients=3)
    agg = ledger.aggregate_view()
    # the aggregate record's client field is -1 (Ledger.aggregate_view's documented convention)
    agg_clients = {r.client for r in agg if r.family.value == "F2"}
    assert agg_clients == {-1}
