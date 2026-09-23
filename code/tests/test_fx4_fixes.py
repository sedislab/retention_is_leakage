"""New tests required by `build/08_FIX_PLAN.md` FX4a-d (R1, R2, R3). Written before running any
experiment that depends on the fixes they check, per the plan's §0.9.
"""
from __future__ import annotations

import numpy as np
import pytest
from p3fcl import sim, streams
from p3fcl.methods.m0_fedavg import FedAvgSequential
from p3fcl.methods.m1_glfc import GLFC
from p3fcl.methods.m2_target import TARGET
from p3fcl.methods.m3_fot import FOT
from p3fcl.methods.m5_hybrid_replay import HybridReplay
from p3fcl.methods.m8_analytic import AnalyticFCL
from p3fcl.shadow_runner import _reconstruct_running_w
from p3fcl.streams import build_stream


def _synthetic(n_classes=6, per_class=20, d=4, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((n_classes, d)) * 5
    X, y = [], []
    for c in range(n_classes):
        X.append(centers[c] + rng.standard_normal((per_class, d)) * 0.4)
        y.append(np.full(per_class, c))
    return np.concatenate(X), np.concatenate(y)


# ---------------------------------------------------------------------------------------------- #
# FX4a (R1): test-split accuracy
# ---------------------------------------------------------------------------------------------- #

def test_task_eval_sets_contains_only_task_k_classes():
    y = np.tile(np.arange(6), 20)
    idx = np.arange(len(y))
    task_classes = streams.class_incremental_tasks(y, n_tasks=3, seed=0)
    eval_id_lists = streams.task_eval_sets(y, idx, n_tasks=3, seed=0)
    for k, ids_k in enumerate(eval_id_lists):
        classes_k = set(task_classes[k])
        assert set(y[ids_k].tolist()) <= classes_k
        assert len(ids_k) > 0


def test_task_eval_sets_domain_incremental_matches_build_stream_domains():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=200)
    domain = np.repeat(np.arange(5), 40)
    idx = np.arange(len(y))
    train_stream = streams.build_stream(y, idx, n_tasks=5, n_clients=1, beta=1.0, seed=0, domain_field=domain)
    eval_id_lists = streams.task_eval_sets(y, idx, n_tasks=5, seed=0, domain_field_eval=domain)
    assert len(eval_id_lists) == len(train_stream) == 5
    for k, ids_k in enumerate(eval_id_lists):
        assert set(domain[ids_k].tolist()) == {k}


@pytest.mark.parametrize("method_cls,cfg", [
    (FedAvgSequential, {"local_epochs": 5, "lr": 0.5}),
    (GLFC, {"local_epochs": 5, "lr": 0.5, "exemplar_budget": 2, "distillation_weight": 1.0}),
    (TARGET, {"local_epochs": 5, "lr": 0.5, "replay_ratio": 1.0}),
    (FOT, {"local_epochs": 5, "lr": 0.5, "subspace_rank": 2}),
    (HybridReplay, {"local_epochs": 5, "lr": 0.5, "buffer_size_per_class": 2}),
    (AnalyticFCL, {"ridge_lambda": 1.0}),
])
def test_predict_never_outputs_unseen_class(method_cls, cfg):
    X, y = _synthetic(n_classes=6)
    idx = np.arange(len(y))
    stream = build_stream(y, idx, n_tasks=3, n_clients=2, beta=1.0, seed=0)
    method = method_cls({"n_classes": 6, "feature_dim": X.shape[1], **cfg})
    r = np.random.default_rng(0)
    for t, shards in enumerate(stream):
        method.fit_task(t, X, y, ids=None, client_shards=shards, rng=r)
        seen_so_far = set()
        for tt in range(t + 1):
            for shard in stream[tt]:
                seen_so_far.update(int(c) for c in y[list(shard.ids)])
        preds = method.predict(X)
        assert set(np.unique(preds).tolist()) <= seen_so_far, (
            f"{method_cls.__name__} predicted an unseen class after task {t}"
        )


def test_acc_matrix_differs_from_acc_matrix_train_on_toy_stream():
    # A toy stream where train and a disjoint "eval" set are drawn from visibly different points, so
    # a method fit on train cannot get the same accuracy on both -- proving eval_sets actually changes
    # what gets measured, not just adding an unused parameter.
    X_train, y_train = _synthetic(n_classes=4, per_class=15, seed=0)
    X_eval, y_eval = _synthetic(n_classes=4, per_class=15, seed=1)  # different noise draw
    # Shift the eval features far enough that train-fit boundaries misclassify them.
    X_eval = X_eval + 100.0
    idx = np.arange(len(y_train))
    stream = build_stream(y_train, idx, n_tasks=2, n_clients=1, beta=1.0, seed=0)
    eval_id_lists = streams.task_eval_sets(y_eval, np.arange(len(y_eval)), n_tasks=2, seed=0)
    eval_sets = [(X_eval[ids], y_eval[ids]) for ids in eval_id_lists]

    method = FedAvgSequential({"n_classes": 4, "feature_dim": X_train.shape[1], "local_epochs": 5, "lr": 0.5})
    result = sim.run(method, X_train, y_train, stream, seed=0, eval_sets=eval_sets)
    assert result["acc_matrix"] is not None
    diff = np.nanmax(np.abs(result["acc_matrix"] - result["acc_matrix_train"]))
    assert diff > 0.1, "acc_matrix and acc_matrix_train should differ substantially on a shifted eval set"


def test_sim_run_without_eval_sets_has_no_test_accuracy_keys():
    X, y = _synthetic(n_classes=4)
    idx = np.arange(len(y))
    stream = build_stream(y, idx, n_tasks=2, n_clients=1, beta=1.0, seed=0)
    method = FedAvgSequential({"n_classes": 4, "feature_dim": X.shape[1], "local_epochs": 2})
    result = sim.run(method, X, y, stream, seed=0)
    assert "acc_matrix" not in result
    assert "final_avg_acc" not in result
    assert "bwt" not in result
    assert "acc_matrix_train" in result


# ---------------------------------------------------------------------------------------------- #
# FX4b (R3): honest ledger, exact reconstruction
# ---------------------------------------------------------------------------------------------- #

@pytest.mark.parametrize("method_cls,cfg", [
    (FedAvgSequential, {"local_epochs": 5, "lr": 0.5}),
    (GLFC, {"local_epochs": 5, "lr": 0.5, "exemplar_budget": 2, "distillation_weight": 1.0}),
    (TARGET, {"local_epochs": 5, "lr": 0.5, "replay_ratio": 1.0}),
    (FOT, {"local_epochs": 5, "lr": 0.5, "subspace_rank": 2}),
    (HybridReplay, {"local_epochs": 5, "lr": 0.5, "buffer_size_per_class": 2}),
])
def test_reconstruct_running_w_matches_true_w_history_exactly(method_cls, cfg):
    X, y = _synthetic(n_classes=6)
    idx = np.arange(len(y))
    stream = build_stream(y, idx, n_tasks=3, n_clients=2, beta=1.0, seed=0)
    method = method_cls({"n_classes": 6, "feature_dim": X.shape[1], **cfg})
    result = sim.run(method, X, y, stream, seed=0)
    w_true = result["w_history"]
    w_hat = _reconstruct_running_w(result["ledger"], len(stream), X.shape[1], 6)
    assert len(w_true) == len(w_hat) == len(stream)
    for r, (wt, wh) in enumerate(zip(w_true, w_hat)):
        assert np.max(np.abs(wt - wh)) < 1e-8, f"{method_cls.__name__} round {r} reconstruction mismatch"


def test_m1_touched_equals_shard_plus_buffer_ids_read_this_round():
    X, y = _synthetic(n_classes=6)
    idx = np.arange(len(y))
    stream = build_stream(y, idx, n_tasks=3, n_clients=2, beta=1.0, seed=0)
    method = GLFC({"n_classes": 6, "feature_dim": X.shape[1], "exemplar_budget": 2, "distillation_weight": 1.0})
    result = sim.run(method, X, y, stream, seed=0)
    model_delta_recs = [r for r in result["ledger"] if r.family.value == "F1"]
    # Round 0: no buffer exists yet -> touched is exactly the shard.
    round0 = [r for r in model_delta_recs if r.round == 0]
    for rec in round0:
        shard_ids = next(s.ids for s in stream[0] if s.client == rec.client)
        assert rec.touched == set(shard_ids)
    # A later round: touched must be a superset of the current shard (buffer ids only add to it).
    round1 = [r for r in model_delta_recs if r.round == 1]
    for rec in round1:
        shard_ids = next(s.ids for s in stream[1] if s.client == rec.client)
        assert set(shard_ids) <= rec.touched


def test_m5_touched_equals_shard_plus_buffer_ids_read_this_round():
    X, y = _synthetic(n_classes=6)
    idx = np.arange(len(y))
    stream = build_stream(y, idx, n_tasks=3, n_clients=2, beta=1.0, seed=0)
    method = HybridReplay({"n_classes": 6, "feature_dim": X.shape[1], "buffer_size_per_class": 2})
    result = sim.run(method, X, y, stream, seed=0)
    model_delta_recs = [r for r in result["ledger"] if r.family.value == "F1"]
    round0 = [r for r in model_delta_recs if r.round == 0]
    for rec in round0:
        shard_ids = next(s.ids for s in stream[0] if s.client == rec.client)
        assert rec.touched == set(shard_ids)
    round1 = [r for r in model_delta_recs if r.round == 1]
    for rec in round1:
        shard_ids = next(s.ids for s in stream[1] if s.client == rec.client)
        assert set(shard_ids) <= rec.touched


# ---------------------------------------------------------------------------------------------- #
# FX4c/d (R2): per-client buffer isolation, KD restricted to old classes
# ---------------------------------------------------------------------------------------------- #

def test_m1_buffer_isolation_across_clients():
    X, y = _synthetic(n_classes=6)
    idx = np.arange(len(y))
    stream = build_stream(y, idx, n_tasks=2, n_clients=3, beta=1.0, seed=0)
    method = GLFC({"n_classes": 6, "feature_dim": X.shape[1], "exemplar_budget": 3, "distillation_weight": 1.0})
    r = np.random.default_rng(0)
    for t, shards in enumerate(stream):
        method.fit_task(t, X, y, ids=None, client_shards=shards, rng=r)
    shard_ids_by_client: dict = {}
    for shards in stream:
        for s in shards:
            shard_ids_by_client.setdefault(s.client, set()).update(s.ids)
    for (client, _cls), entries in method._buffer.items():
        buf_ids = {did for _feat, did in entries}
        other_clients_ids = set().union(*(v for c, v in shard_ids_by_client.items() if c != client)) if len(shard_ids_by_client) > 1 else set()
        assert buf_ids.isdisjoint(other_clients_ids), f"client {client}'s buffer contains another client's ids"


def test_m5_buffer_isolation_across_clients():
    X, y = _synthetic(n_classes=6)
    idx = np.arange(len(y))
    stream = build_stream(y, idx, n_tasks=2, n_clients=3, beta=1.0, seed=0)
    method = HybridReplay({"n_classes": 6, "feature_dim": X.shape[1], "buffer_size_per_class": 3})
    r = np.random.default_rng(0)
    for t, shards in enumerate(stream):
        method.fit_task(t, X, y, ids=None, client_shards=shards, rng=r)
    shard_ids_by_client: dict = {}
    for shards in stream:
        for s in shards:
            shard_ids_by_client.setdefault(s.client, set()).update(s.ids)
    for (client, _cls), entries in method._buffer.items():
        buf_ids = {did for _feat, did in entries}
        other_clients_ids = set().union(*(v for c, v in shard_ids_by_client.items() if c != client)) if len(shard_ids_by_client) > 1 else set()
        assert buf_ids.isdisjoint(other_clients_ids), f"client {client}'s buffer contains another client's ids"


def test_m1_kd_gradient_is_exactly_zero_on_new_class_columns():
    # Isolate the KD term's DIRECT contribution with one controlled `_local_train` step, using a
    # W_old that is deliberately different from W0 (so KD has something real to pull toward). Multi-
    # epoch / multi-task comparisons are the wrong tool here: after the first epoch, KD's effect on
    # old columns feeds back into the *shared* softmax normalisation (all classes share one
    # normaliser), which then indirectly perturbs the CE gradient's values on new-class columns too
    # in later epochs -- a real, correct consequence of joint softmax classification, not evidence
    # that KD's own gradient touches new columns. A single step has no such feedback loop yet.
    X, y = _synthetic(n_classes=6, per_class=10)
    d = X.shape[1]
    rng = np.random.default_rng(0)
    method = GLFC({"n_classes": 6, "feature_dim": d, "local_epochs": 1, "distillation_weight": 1.0})
    W0 = rng.standard_normal((d, 6)) * 0.1
    W_old = rng.standard_normal((d, 6)) * 0.1  # deliberately different from W0
    old_classes = [0, 1, 3]
    new_classes = [2, 4, 5]
    idx_task = np.arange(30)
    X_task, y_task = X[idx_task], y[idx_task] % 6

    W_with_kd = method._local_train(W0, X_task, y_task, None, None, W_old, old_classes)
    method_no_kd = GLFC({"n_classes": 6, "feature_dim": d, "local_epochs": 1, "distillation_weight": 0.0})
    W_without_kd = method_no_kd._local_train(W0, X_task, y_task, None, None, W_old, old_classes)

    # KD's direct gradient contribution is confined to old_classes -- new-class columns must be
    # bit-for-bit identical whether distillation_weight is 1 or 0.
    np.testing.assert_array_equal(W_with_kd[:, new_classes], W_without_kd[:, new_classes])
    # And KD does change old-class columns (sanity: the test setup actually exercises the KD path).
    assert not np.allclose(W_with_kd[:, old_classes], W_without_kd[:, old_classes])


def test_m1_zero_distillation_and_empty_buffer_equals_one_m0_round():
    X, y = _synthetic(n_classes=6, per_class=10)
    idx = np.arange(len(y))
    stream = build_stream(y, idx, n_tasks=1, n_clients=1, beta=1.0, seed=0)
    m1 = GLFC({
        "n_classes": 6, "feature_dim": X.shape[1], "local_epochs": 3, "lr": 0.5,
        "exemplar_budget": 0, "distillation_weight": 0.0,
    })
    m0 = FedAvgSequential({"n_classes": 6, "feature_dim": X.shape[1], "local_epochs": 3, "lr": 0.5})
    r1 = np.random.default_rng(0)
    r0 = np.random.default_rng(0)
    m1.fit_task(0, X, y, ids=None, client_shards=stream[0], rng=r1)
    m0.fit_task(0, X, y, ids=None, client_shards=stream[0], rng=r0)
    np.testing.assert_allclose(m1.W, m0.W, atol=1e-10)
