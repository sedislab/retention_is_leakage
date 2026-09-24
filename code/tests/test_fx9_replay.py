from types import SimpleNamespace

import numpy as np
import pytest
from p3fcl.methods.m0_fedavg import FedAvgSequential
from p3fcl.methods.m1_glfc import GLFC
from p3fcl.methods.m2_target import TARGET
from p3fcl.methods.m5_hybrid_replay import HybridReplay


@pytest.mark.parametrize("cls", [GLFC, TARGET, HybridReplay])
def test_empty_replay_one_round_matches_m0(cls):
    rng = np.random.default_rng(51)
    X, y = rng.normal(size=(12, 4)), np.arange(12) % 3
    cfg = dict(n_classes=3, feature_dim=4, local_epochs=3, lr=0.1)
    base, replay = FedAvgSequential(cfg), cls(cfg)
    shards = [
        SimpleNamespace(client=0, ids=list(range(5))),
        SimpleNamespace(client=1, ids=list(range(5, 12))),
    ]
    base.fit_task(0, X, y, np.arange(12), shards, rng)
    replay.fit_task(0, X, y, np.arange(12), shards, rng)
    np.testing.assert_allclose(base.W, replay.W, atol=1e-10, rtol=0)


@pytest.mark.parametrize("cls", [GLFC, TARGET, HybridReplay])
@pytest.mark.parametrize("duplicate_current", [False, True])
def test_balanced_ce_invariant_to_either_sample_count(cls, duplicate_current):
    rng = np.random.default_rng(52)
    W = rng.normal(size=(4, 3))
    X, y = rng.normal(size=(5, 4)), np.arange(5) % 3
    R, z = rng.normal(size=(13, 4)), np.arange(13) % 3
    # The specified M1 KD_mean(current union buffer) remains unchanged; isolate CE here.
    method = cls(dict(n_classes=3, feature_dim=4, local_epochs=1, distillation_weight=0, replay_weight=2))
    extra = [W.copy(), [0, 1]] if cls is GLFC else []
    before = method._local_train(W, X, y, R, z, *extra)
    if duplicate_current:
        X, y = np.tile(X, (2, 1)), np.tile(y, 2)
    else:
        R, z = np.tile(R, (2, 1)), np.tile(z, 2)
    after = method._local_train(W, X, y, R, z, *extra)
    np.testing.assert_allclose(before, after, atol=1e-10, rtol=0)


def test_m5_no_f8_and_replay_ids_still_touched():
    X, y = np.eye(4), np.arange(4)
    method = HybridReplay(dict(n_classes=4, feature_dim=4, buffer_size_per_class=1))
    rng = np.random.default_rng(53)
    records = method.fit_task(0, X, y, np.arange(4), [SimpleNamespace(client=0, ids=[0, 1])], rng)
    records += method.fit_task(1, X, y, np.arange(4), [SimpleNamespace(client=0, ids=[2, 3])], rng)
    assert {r.family.value for r in records} == {"F1"}
    assert records[-1].touched == frozenset(range(4))
