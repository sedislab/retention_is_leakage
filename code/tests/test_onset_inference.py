from __future__ import annotations

import numpy as np
import pytest
from p3fcl import sim, streams
from p3fcl.artifacts import Family
from p3fcl.attacks.onset_inference import (
    OnsetInferenceAttack,
    aggregate_signal_series,
    cusum_changepoints,
    norm_spike_baseline,
    precision_at_tolerance,
    random_baseline,
    recall_at_tolerance,
    true_arrival_rounds,
)
from p3fcl.methods.m0_fedavg import FedAvgSequential
from p3fcl.methods.m8_analytic import AnalyticFCL


def _synthetic(n_classes=10, d=6, per_class=60, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((n_classes, d)) * 3
    X, y = [], []
    for c in range(n_classes):
        X.append(centers[c] + rng.standard_normal((per_class, d)) * 0.3)
        y.append(np.full(per_class, c))
    return np.concatenate(X), np.concatenate(y)


def test_true_arrival_rounds_skips_round_zero_by_default():
    shard = streams.Shard
    stream = [
        [shard(client=0, task=0, ids=(1, 2)), shard(client=1, task=0, ids=(3, 4))],
        [shard(client=0, task=1, ids=(5, 6))],  # client 1 silent -- not a new arrival
        [shard(client=0, task=2, ids=(7,)), shard(client=2, task=2, ids=(8,))],  # client 2 is new
    ]
    assert true_arrival_rounds(stream) == [2]
    assert true_arrival_rounds(stream, skip_round_zero=False) == [0, 2]


def test_aggregate_signal_series_matches_hand_computed_sum():
    X, y = _synthetic()
    idx = np.arange(len(y))
    stream = streams.build_stream(y, idx, n_tasks=2, n_clients=2, beta=1.0, seed=0)
    method = FedAvgSequential({"n_classes": 10, "feature_dim": 6, "local_epochs": 3, "lr": 0.1})
    result = sim.run(method, X, y, stream, seed=0)
    ledger = result["ledger"]

    norm_series, count_series = aggregate_signal_series(ledger, n_rounds=2, family=Family.MODEL_DELTA)
    for r in range(2):
        recs = [rec for rec in ledger if rec.family == Family.MODEL_DELTA and rec.round == r]
        expected_norm = np.linalg.norm(sum(np.asarray(rec.payload) for rec in recs))
        assert norm_series[r] == pytest.approx(expected_norm)
        assert count_series[r] == len(recs)


def test_aggregate_signal_series_handles_dict_payloads():
    """F5 (Gram) releases a dict payload ({"R": ..., "Q": ...}) -- the norm should be over every value
    concatenated, not crash on np.asarray(dict)."""
    X, y = _synthetic()
    idx = np.arange(len(y))
    stream = streams.build_stream(y, idx, n_tasks=2, n_clients=2, beta=1.0, seed=0)
    method = AnalyticFCL({"n_classes": 10, "feature_dim": 6, "ridge_lambda": 1.0})
    result = sim.run(method, X, y, stream, seed=0)
    ledger = result["ledger"]

    norm_series, count_series = aggregate_signal_series(ledger, n_rounds=2, family=Family.GRAM)
    assert np.all(norm_series > 0)
    assert np.all(count_series == 2)


def test_cusum_flags_a_real_step_change_and_not_a_flat_series():
    flat = np.ones(20) * 5.0 + np.random.default_rng(0).standard_normal(20) * 0.01
    assert cusum_changepoints(flat, threshold=1.0) == []

    stepped = np.concatenate([np.ones(10) * 1.0, np.ones(10) * 20.0])
    flags = cusum_changepoints(stepped, threshold=5.0)
    assert flags, "expected the step change to be flagged"
    assert any(8 <= f <= 12 for f in flags)


def test_norm_spike_baseline_picks_the_highest_values():
    series = np.array([1.0, 9.0, 2.0, 8.0, 0.5])
    assert norm_spike_baseline(series, n_flags=2) == [1, 3]


def test_random_baseline_is_deterministic_given_a_seeded_generator():
    rng1 = np.random.default_rng(0)
    rng2 = np.random.default_rng(0)
    assert random_baseline(10, 3, rng1) == random_baseline(10, 3, rng2)


def test_precision_and_recall_at_tolerance():
    flagged = [0, 5, 9]
    true_rounds = [1, 9]
    hits, n_flagged = precision_at_tolerance(flagged, true_rounds, tolerance=2)
    assert (hits, n_flagged) == (2, 3)  # 0 is within 2 of 1; 9 hits 9; 5 hits neither
    hits, n_true = recall_at_tolerance(flagged, true_rounds, tolerance=2)
    assert (hits, n_true) == (2, 2)  # both true rounds covered


def test_onset_attack_end_to_end_detects_a_real_late_arrival():
    """A skewed stream where one client is silent for the first few tasks, then arrives -- the whole
    pipeline (real ledger -> aggregate signal -> CUSUM) should flag a round near the true arrival."""
    X, y = _synthetic(n_classes=10, per_class=80, seed=1)
    idx = np.arange(len(y))
    # A hand-built stream: 5 tasks, client 1 only appears from task 3 onward.
    stream = []
    for t in range(5):
        class_ids = idx[(y >= t * 2) & (y < t * 2 + 2)]
        shards = [streams.Shard(client=0, task=t, ids=tuple(int(i) for i in class_ids))]
        if t >= 3:
            extra = idx[(y >= (t + 1) * 2 % 10) & (y < ((t + 1) * 2 % 10) + 2)]
            shards.append(streams.Shard(client=1, task=t, ids=tuple(int(i) for i in extra[:20])))
        stream.append(shards)

    true_rounds = true_arrival_rounds(stream)
    assert true_rounds == [3]

    method = FedAvgSequential({"n_classes": 10, "feature_dim": 6, "local_epochs": 5, "lr": 0.3})
    result = sim.run(method, X, y, stream, seed=0)
    ledger = result["ledger"]

    attack = OnsetInferenceAttack({})
    flags = attack.run(ledger, n_rounds=5, threshold=0.5)
    hits, n_flagged = precision_at_tolerance(flags, true_rounds, tolerance=2)
    assert n_flagged > 0
    assert hits > 0, f"expected at least one flagged round near {true_rounds}, got {flags}"
