from __future__ import annotations

import numpy as np
from p3fcl import sim
from p3fcl.dp.accountant import check_disjointness
from p3fcl.methods.m0_fedavg import FedAvgSequential
from p3fcl.methods.m1_glfc import GLFC
from p3fcl.methods.m2_target import TARGET
from p3fcl.methods.m3_fot import FOT
from p3fcl.methods.m4_proto import PrototypeFCL
from p3fcl.methods.m5_hybrid_replay import HybridReplay
from p3fcl.methods.m8_analytic import AnalyticFCL
from p3fcl.streams import build_stream


def _synthetic(n_classes=6, d=8, per_class=40, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((n_classes, d)) * 3
    X, y = [], []
    for c in range(n_classes):
        X.append(centers[c] + rng.standard_normal((per_class, d)) * 0.3)
        y.append(np.full(per_class, c))
    X = np.concatenate(X)
    y = np.concatenate(y)
    return X, y


def test_m0_fedavg_sequential_end_to_end_on_synthetic_features():
    X, y = _synthetic()
    idx = np.arange(len(y))
    stream = build_stream(y, idx, n_tasks=3, n_clients=2, beta=1.0, seed=0)
    method = FedAvgSequential({"n_classes": 6, "feature_dim": X.shape[1], "local_epochs": 20, "lr": 0.5})
    result = sim.run(method, X, y, stream, seed=0)
    assert result["final_avg_acc_train"] > 0.3  # learns the current task; no anti-forgetting mechanism at all
    families = {r.family.value for r in result["ledger"]}
    assert families == {"F1"}


def test_m0_fedavg_sequential_single_local_epoch_is_task_disjoint():
    # exactly one local SGD step per round -> no within-task multi-pass, no cross-task carry-over.
    X, y = _synthetic()
    idx = np.arange(len(y))
    stream = build_stream(y, idx, n_tasks=3, n_clients=2, beta=1.0, seed=0)
    method = FedAvgSequential({"n_classes": 6, "feature_dim": X.shape[1], "local_epochs": 1})
    result = sim.run(method, X, y, stream, seed=0)
    report = check_disjointness(result["ledger"])
    assert report.task_disjoint, report.violations


def test_m0_fedavg_sequential_multi_epoch_violates_v5b():
    # >1 local SGD epochs over the same shard within one round IS sequential composition within the
    # task (real: more gradient steps over the same data leak more about it) -> V5b, correctly.
    X, y = _synthetic()
    idx = np.arange(len(y))
    stream = build_stream(y, idx, n_tasks=3, n_clients=2, beta=1.0, seed=0)
    method = FedAvgSequential({"n_classes": 6, "feature_dim": X.shape[1], "local_epochs": 5})
    result = sim.run(method, X, y, stream, seed=0)
    report = check_disjointness(result["ledger"])
    assert not report.task_disjoint
    assert "V5b" in report.violations


def test_m0_fedavg_sequential_touched_is_exactly_the_task_shard():
    X, y = _synthetic()
    idx = np.arange(len(y))
    stream = build_stream(y, idx, n_tasks=3, n_clients=2, beta=1.0, seed=0)
    method = FedAvgSequential({"n_classes": 6, "feature_dim": X.shape[1], "local_epochs": 3})
    result = sim.run(method, X, y, stream, seed=0)
    for rec in result["ledger"]:
        shard = next(s for shards in stream for s in shards if s.client == rec.client and s.task == rec.task)
        assert rec.touched == frozenset(shard.ids)
        assert rec.passes_over_data == 3


def test_m3_fot_end_to_end_on_synthetic_features():
    X, y = _synthetic()
    idx = np.arange(len(y))
    stream = build_stream(y, idx, n_tasks=3, n_clients=2, beta=1.0, seed=0)
    method = FOT({
        "n_classes": 6, "feature_dim": X.shape[1], "local_epochs": 10, "lr": 0.5,
        "subspace_rank": 2, "projection_strength": 1.0,
    })
    result = sim.run(method, X, y, stream, seed=0)
    assert result["final_avg_acc_train"] > 0.3
    families = {r.family.value for r in result["ledger"]}
    # FX4b (08_FIX_PLAN.md R3): the subspace U is now its own honest release (family GRAM, F5),
    # separate from the MODEL_DELTA record -- not folded silently into F1's touched.
    assert families == {"F1", "F5"}


def test_m3_fot_is_task_disjoint_after_fx4b():
    # FX4b fix (08_FIX_PLAN.md R3): the pre-fix version inflated every later MODEL_DELTA's `touched`
    # with every id that had ever fed the subspace U, on the theory that using U in the gradient
    # projection is a fresh read of that old raw data. That was the actual bug -- using a previously
    # *released* statistic (the subspace, now its own GRAM record at the task that built it) is
    # post-processing (Lemma 8), not a new read. Each release (MODEL_DELTA per client-round, GRAM per
    # task) now touches only that task's own ids, so the whole ledger is task-disjoint -- the same
    # status as M8, and for the same underlying reason (single-pass release of task-local statistics).
    X, y = _synthetic()
    idx = np.arange(len(y))
    stream = build_stream(y, idx, n_tasks=3, n_clients=2, beta=1.0, seed=0)
    method = FOT({"n_classes": 6, "feature_dim": X.shape[1], "local_epochs": 1, "subspace_rank": 2})
    result = sim.run(method, X, y, stream, seed=0)
    report = check_disjointness(result["ledger"])
    assert report.task_disjoint, report.violations


def test_m3_fot_touched_is_exactly_current_shard():
    # FX4b test (ii): touched in later rounds is a SUBSET of (here: exactly) that client's current
    # shard -- no accumulation from earlier tasks' data or from the subspace, regardless of
    # projection_strength (which only affects the gradient, never what counts as touched).
    X, y = _synthetic()
    idx = np.arange(len(y))
    stream = build_stream(y, idx, n_tasks=2, n_clients=1, beta=1.0, seed=0)
    method = FOT({"n_classes": 6, "feature_dim": X.shape[1], "local_epochs": 1, "projection_strength": 0.0})
    result = sim.run(method, X, y, stream, seed=0)
    recs_task1_model_delta = [r for r in result["ledger"] if r.task == 1 and r.family.value == "F1"]
    task1_shard_ids = {s.client: set(s.ids) for s in stream[1]}
    task0_ids = {i for shard in stream[0] for i in shard.ids}
    for rec in recs_task1_model_delta:
        assert rec.touched == task1_shard_ids[rec.client]
        assert rec.touched.isdisjoint(task0_ids)


def test_m5_hybrid_replay_end_to_end_on_synthetic_features():
    X, y = _synthetic()
    idx = np.arange(len(y))
    stream = build_stream(y, idx, n_tasks=3, n_clients=2, beta=1.0, seed=0)
    method = HybridReplay({
        "n_classes": 6, "feature_dim": X.shape[1], "local_epochs": 10, "lr": 0.5,
        "buffer_size_per_class": 4,
    })
    result = sim.run(method, X, y, stream, seed=0)
    assert result["final_avg_acc_train"] > 0.3
    families = {r.family.value for r in result["ledger"]}
    assert families == {"F1"}  # FX9: exemplars stay private on their own client.


def test_m5_hybrid_replay_is_not_task_disjoint_once_buffer_is_reused():
    # this is the whole point of the 2026-09-15 fix: F8 must NOT falsely certify as disjoint (H6).
    X, y = _synthetic()
    idx = np.arange(len(y))
    stream = build_stream(y, idx, n_tasks=3, n_clients=2, beta=1.0, seed=0)
    method = HybridReplay({"n_classes": 6, "feature_dim": X.shape[1], "buffer_size_per_class": 4})
    result = sim.run(method, X, y, stream, seed=0)
    report = check_disjointness(result["ledger"])
    assert not report.task_disjoint
    assert "V2/V3" in report.violations


def test_m5_buffer_is_private_and_contains_exact_features_with_per_class_cap():
    X, y = _synthetic()
    stream = build_stream(y, np.arange(len(y)), n_tasks=3, n_clients=2, beta=1.0, seed=0)
    method = HybridReplay({"n_classes": 6, "feature_dim": X.shape[1], "buffer_size_per_class": 3})
    result = sim.run(method, X, y, stream, seed=0)
    assert not any(r.family.value == "F8" for r in result["ledger"])
    for entries in method._buffer.values():
        assert len(entries) <= 3
        for feature, did in entries:
            np.testing.assert_array_equal(feature, X[did])


def test_m1_glfc_end_to_end_on_synthetic_features():
    X, y = _synthetic()
    idx = np.arange(len(y))
    stream = build_stream(y, idx, n_tasks=3, n_clients=2, beta=1.0, seed=0)
    method = GLFC({
        "n_classes": 6, "feature_dim": X.shape[1], "local_epochs": 10, "lr": 0.5,
        "exemplar_budget": 4, "distillation_weight": 1.0,
    })
    result = sim.run(method, X, y, stream, seed=0)
    assert result["final_avg_acc_train"] > 0.3
    families = {r.family.value for r in result["ledger"]}
    # FX4b (08_FIX_PLAN.md §4b, "Local buffers are not releases"): M1's exemplar buffer is private
    # local client state, read only by that same client's own next local-training step -- nothing
    # about it is ever transmitted, so it is no longer an F8 ledger record (see the dedicated test
    # below). Only F1 (model delta) and F7 (per-class counts) are real releases.
    assert families == {"F1", "F7"}


def test_m1_glfc_no_replay_no_distillation_is_task_disjoint():
    # exemplar_budget=0 AND distillation_weight=0 -> nothing at all carried forward, equivalent to
    # M0 at local_epochs=1 -> disjoint.
    X, y = _synthetic()
    idx = np.arange(len(y))
    stream = build_stream(y, idx, n_tasks=2, n_clients=1, beta=1.0, seed=0)
    method = GLFC({
        "n_classes": 6, "feature_dim": X.shape[1], "local_epochs": 1,
        "exemplar_budget": 0, "distillation_weight": 0.0,
    })
    result = sim.run(method, X, y, stream, seed=0)
    report = check_disjointness(result["ledger"])
    assert report.task_disjoint, report.violations


def test_m1_glfc_replay_alone_violates_disjointness_even_without_distillation():
    # exemplar replay in the cross-entropy term reuses old-task ids regardless of the KD weight —
    # distillation is not the only channel that breaks disjointness here, and the checker must not
    # be fooled by distillation_weight=0 into a false "safe" certification.
    X, y = _synthetic()
    idx = np.arange(len(y))
    stream = build_stream(y, idx, n_tasks=2, n_clients=1, beta=1.0, seed=0)
    method = GLFC({
        "n_classes": 6, "feature_dim": X.shape[1], "local_epochs": 1,
        "exemplar_budget": 3, "distillation_weight": 0.0,
    })
    result = sim.run(method, X, y, stream, seed=0)
    report = check_disjointness(result["ledger"])
    assert not report.task_disjoint
    assert "V2/V3" in report.violations


def test_m1_glfc_with_distillation_violates_disjointness():
    X, y = _synthetic()
    idx = np.arange(len(y))
    stream = build_stream(y, idx, n_tasks=3, n_clients=2, beta=1.0, seed=0)
    method = GLFC({
        "n_classes": 6, "feature_dim": X.shape[1], "local_epochs": 1,
        "exemplar_budget": 3, "distillation_weight": 1.0,
    })
    result = sim.run(method, X, y, stream, seed=0)
    report = check_disjointness(result["ledger"])
    assert not report.task_disjoint
    assert "V2/V3" in report.violations


def test_m1_glfc_exemplar_buffer_is_never_released():
    # FX4b (08_FIX_PLAN.md R3, "Local buffers are not releases"): M1's per-client exemplar buffer is
    # private local state -- it feeds that same client's own next local-training step and nothing
    # else, so it must never appear in the ledger at all, regardless of exemplar_budget.
    X, y = _synthetic()
    idx = np.arange(len(y))
    stream = build_stream(y, idx, n_tasks=2, n_clients=1, beta=1.0, seed=0)
    method = GLFC({"n_classes": 6, "feature_dim": X.shape[1], "exemplar_budget": 3})
    result = sim.run(method, X, y, stream, seed=0)
    exemplar_recs = [r for r in result["ledger"] if r.family.value == "F8"]
    assert exemplar_recs == []
    assert method._buffer  # the buffer itself still exists and is nonempty -- it's just not released


def test_m2_target_end_to_end_on_synthetic_features():
    X, y = _synthetic()
    idx = np.arange(len(y))
    stream = build_stream(y, idx, n_tasks=3, n_clients=2, beta=1.0, seed=0)
    method = TARGET({
        "n_classes": 6, "feature_dim": X.shape[1], "local_epochs": 10, "lr": 0.5,
        "replay_ratio": 1.0, "n_synthetic_per_class": 15,
    })
    result = sim.run(method, X, y, stream, seed=0)
    assert result["final_avg_acc_train"] > 0.3
    families = {r.family.value for r in result["ledger"]}
    assert families == {"F1", "F6"}


def test_m2_target_zero_replay_ratio_is_task_disjoint():
    X, y = _synthetic()
    idx = np.arange(len(y))
    stream = build_stream(y, idx, n_tasks=2, n_clients=1, beta=1.0, seed=0)
    method = TARGET({"n_classes": 6, "feature_dim": X.shape[1], "local_epochs": 1, "replay_ratio": 0.0})
    result = sim.run(method, X, y, stream, seed=0)
    report = check_disjointness(result["ledger"])
    assert report.task_disjoint, report.violations


def test_m2_target_with_replay_is_task_disjoint_after_fx4b():
    # FX4b fix (08_FIX_PLAN.md R3): the pre-fix version re-added the *original* generator-fitting ids
    # to every later round's model-delta touched for as long as a generator kept getting sampled from
    # -- treating "sample from an already-broadcast Gaussian" as a fresh read of the raw data that
    # fit it. That was the bug: replay sampling from an already-*released* statistic is
    # post-processing (Lemma 8), not a new read. Each client's F1 and F6 releases now touch only that
    # client's own current shard, so the whole ledger is task-disjoint -- a real, correct consequence
    # of the fix, not a weakening of the disjointness checker.
    X, y = _synthetic()
    idx = np.arange(len(y))
    stream = build_stream(y, idx, n_tasks=3, n_clients=2, beta=1.0, seed=0)
    method = TARGET({"n_classes": 6, "feature_dim": X.shape[1], "local_epochs": 1, "replay_ratio": 1.0})
    result = sim.run(method, X, y, stream, seed=0)
    report = check_disjointness(result["ledger"])
    assert report.task_disjoint, report.violations


def test_m2_target_generative_payload_is_moments_not_raw_features():
    X, y = _synthetic()
    idx = np.arange(len(y))
    stream = build_stream(y, idx, n_tasks=1, n_clients=1, beta=1.0, seed=0)
    method = TARGET({"n_classes": 6, "feature_dim": X.shape[1]})
    result = sim.run(method, X, y, stream, seed=0)
    gen_recs = [r for r in result["ledger"] if r.family.value == "F6"]
    assert gen_recs
    for rec in gen_recs:
        for key, val in rec.payload.items():
            assert val.shape == (X.shape[1],)  # mean/var vectors, not a per-sample raw feature matrix


def test_m8_analytic_fcl_end_to_end_on_synthetic_features():
    X, y = _synthetic()
    idx = np.arange(len(y))
    stream = build_stream(y, idx, n_tasks=3, n_clients=2, beta=1.0, seed=0)
    method = AnalyticFCL({"n_classes": 6, "feature_dim": X.shape[1], "ridge_lambda": 1.0})
    result = sim.run(method, X, y, stream, seed=0)
    assert result["final_avg_acc_train"] > 0.5  # well-separated synthetic clusters, should be easy
    assert len(result["ledger"]) > 0
    report = check_disjointness(result["ledger"])
    assert report.task_disjoint, report.violations


def test_m4_prototype_half_end_to_end_on_synthetic_features():
    X, y = _synthetic()
    idx = np.arange(len(y))
    stream = build_stream(y, idx, n_tasks=3, n_clients=2, beta=1.0, seed=0)
    method = PrototypeFCL({"n_classes": 6, "feature_dim": X.shape[1], "prototype_momentum": 0.0})
    result = sim.run(method, X, y, stream, seed=0)
    assert result["final_avg_acc_train"] > 0.5
    families = {r.family.value for r in result["ledger"]}
    assert {"F2", "F7"} <= families


def test_m4_prototype_half_class_incremental_is_task_disjoint_even_with_momentum():
    # classes never repeat across tasks in class-incremental streams, so momentum carry-over never
    # actually fires -> still task-disjoint (matches H6: "F2 single-pass admits task-disjointness").
    X, y = _synthetic()
    idx = np.arange(len(y))
    stream = build_stream(y, idx, n_tasks=3, n_clients=2, beta=1.0, seed=0)
    method = PrototypeFCL({"n_classes": 6, "feature_dim": X.shape[1], "prototype_momentum": 0.9})
    result = sim.run(method, X, y, stream, seed=0)
    report = check_disjointness(result["ledger"])
    assert report.task_disjoint, report.violations


def test_m4_server_momentum_is_postprocessing_not_a_raw_data_reread():
    # FX9 local releases are fresh class means; server momentum only reads released statistics.
    rng = np.random.default_rng(0)
    n_classes, d = 2, 6
    X, y, domain = [], [], []
    for dom in range(3):
        for c in range(n_classes):
            n = 20
            X.append(rng.standard_normal((n, d)) + c * 3)
            y.append(np.full(n, c))
            domain.append(np.full(n, dom))
    X = np.concatenate(X)
    y = np.concatenate(y)
    domain = np.concatenate(domain)
    idx = np.arange(len(y))
    stream = build_stream(y, idx, n_tasks=3, n_clients=1, beta=1.0, seed=0, domain_field=domain)
    method = PrototypeFCL({"n_classes": n_classes, "feature_dim": d, "prototype_momentum": 0.5})
    result = sim.run(method, X, y, stream, seed=0)
    report = check_disjointness(result["ledger"])
    assert report.task_disjoint, report.violations


def test_m4_prototype_half_release_counts_can_be_disabled():
    X, y = _synthetic()
    idx = np.arange(len(y))
    stream = build_stream(y, idx, n_tasks=3, n_clients=2, beta=1.0, seed=0)
    method = PrototypeFCL({"n_classes": 6, "feature_dim": X.shape[1], "release_counts": False})
    result = sim.run(method, X, y, stream, seed=0)
    families = {r.family.value for r in result["ledger"]}
    assert "F7" not in families
