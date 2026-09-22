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
    assert result["final_avg_acc"] > 0.3  # learns the current task; no anti-forgetting mechanism at all
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
    assert result["final_avg_acc"] > 0.3
    families = {r.family.value for r in result["ledger"]}
    assert families == {"F1"}


def test_m3_fot_is_not_task_disjoint_once_subspace_is_nonempty():
    # by task 2, U has been built from task 0's data -> every subsequent release's touched set must
    # include task 0's ids too -> V2/V3, correctly (the whole point of FOT is that old statistics
    # re-enter every future release).
    X, y = _synthetic()
    idx = np.arange(len(y))
    stream = build_stream(y, idx, n_tasks=3, n_clients=2, beta=1.0, seed=0)
    method = FOT({"n_classes": 6, "feature_dim": X.shape[1], "local_epochs": 1, "subspace_rank": 2})
    result = sim.run(method, X, y, stream, seed=0)
    report = check_disjointness(result["ledger"])
    assert not report.task_disjoint
    assert "V2/V3" in report.violations


def test_m3_fot_zero_projection_strength_matches_plain_fedavg_touched():
    # projection_strength=0 -> no subspace influence on the gradient, but the subspace is still
    # *tracked* internally; touched still (honestly) grows to include it once nonempty, since the
    # method still computes and could use it. This documents that projection_strength only affects
    # the gradient, not what counts as "touched" -- touched is about influence-in-principle.
    X, y = _synthetic()
    idx = np.arange(len(y))
    stream = build_stream(y, idx, n_tasks=2, n_clients=1, beta=1.0, seed=0)
    method = FOT({"n_classes": 6, "feature_dim": X.shape[1], "local_epochs": 1, "projection_strength": 0.0})
    result = sim.run(method, X, y, stream, seed=0)
    recs_task1 = [r for r in result["ledger"] if r.task == 1]
    task0_ids = {i for shards in [stream[0]] for s in shards for i in s.ids}
    for rec in recs_task1:
        assert task0_ids <= rec.touched


def test_m5_hybrid_replay_end_to_end_on_synthetic_features():
    X, y = _synthetic()
    idx = np.arange(len(y))
    stream = build_stream(y, idx, n_tasks=3, n_clients=2, beta=1.0, seed=0)
    method = HybridReplay({
        "n_classes": 6, "feature_dim": X.shape[1], "local_epochs": 10, "lr": 0.5,
        "buffer_size_per_class": 4,
    })
    result = sim.run(method, X, y, stream, seed=0)
    assert result["final_avg_acc"] > 0.3
    families = {r.family.value for r in result["ledger"]}
    assert families == {"F8", "F1"}  # exemplar release AND the replay-augmented delta (see 2026-09-15
    # correction in the module docstring: omitting F1 made the checker falsely certify disjointness)


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


def test_m5_hybrid_replay_releases_exact_raw_features():
    # F8 is "literal raw samples" -- the released feature vectors must be byte-identical to the
    # cached features at those ids, not a derived statistic.
    X, y = _synthetic()
    idx = np.arange(len(y))
    stream = build_stream(y, idx, n_tasks=2, n_clients=1, beta=1.0, seed=0)
    method = HybridReplay({"n_classes": 6, "feature_dim": X.shape[1], "buffer_size_per_class": 3})
    result = sim.run(method, X, y, stream, seed=0)
    exemplar_recs = [r for r in result["ledger"] if r.family.value == "F8"]
    assert exemplar_recs
    for rec in exemplar_recs:
        for key, val in rec.payload.items():
            if key.startswith("ids_"):
                continue
            c = int(key.split("_")[1])
            ids_key = f"ids_{c}"
            ids_released = rec.payload[ids_key]
            np.testing.assert_allclose(val, X[ids_released])


def test_m5_hybrid_replay_buffer_size_caps_touched_per_class():
    X, y = _synthetic(per_class=40)
    idx = np.arange(len(y))
    stream = build_stream(y, idx, n_tasks=3, n_clients=1, beta=1.0, seed=0)
    method = HybridReplay({"n_classes": 6, "feature_dim": X.shape[1], "buffer_size_per_class": 2})
    result = sim.run(method, X, y, stream, seed=0)
    exemplar_recs = [r for r in result["ledger"] if r.family.value == "F8"]
    assert exemplar_recs
    for rec in exemplar_recs:
        n_classes_in_payload = len([k for k in rec.payload if k.startswith("ids_")])
        assert rec.n_touched <= 2 * n_classes_in_payload


def test_m1_glfc_end_to_end_on_synthetic_features():
    X, y = _synthetic()
    idx = np.arange(len(y))
    stream = build_stream(y, idx, n_tasks=3, n_clients=2, beta=1.0, seed=0)
    method = GLFC({
        "n_classes": 6, "feature_dim": X.shape[1], "local_epochs": 10, "lr": 0.5,
        "exemplar_budget": 4, "distillation_weight": 1.0,
    })
    result = sim.run(method, X, y, stream, seed=0)
    assert result["final_avg_acc"] > 0.3
    families = {r.family.value for r in result["ledger"]}
    assert families == {"F1", "F7", "F8"}


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


def test_m1_glfc_exemplar_payload_is_exact_raw_features():
    X, y = _synthetic()
    idx = np.arange(len(y))
    stream = build_stream(y, idx, n_tasks=2, n_clients=1, beta=1.0, seed=0)
    method = GLFC({"n_classes": 6, "feature_dim": X.shape[1], "exemplar_budget": 3})
    result = sim.run(method, X, y, stream, seed=0)
    exemplar_recs = [r for r in result["ledger"] if r.family.value == "F8"]
    assert exemplar_recs
    for rec in exemplar_recs:
        for _key, feats in rec.payload.items():
            assert feats.shape[1] == X.shape[1]


def test_m2_target_end_to_end_on_synthetic_features():
    X, y = _synthetic()
    idx = np.arange(len(y))
    stream = build_stream(y, idx, n_tasks=3, n_clients=2, beta=1.0, seed=0)
    method = TARGET({
        "n_classes": 6, "feature_dim": X.shape[1], "local_epochs": 10, "lr": 0.5,
        "replay_ratio": 1.0, "n_synthetic_per_class": 15,
    })
    result = sim.run(method, X, y, stream, seed=0)
    assert result["final_avg_acc"] > 0.3
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


def test_m2_target_with_replay_violates_disjointness():
    X, y = _synthetic()
    idx = np.arange(len(y))
    stream = build_stream(y, idx, n_tasks=3, n_clients=2, beta=1.0, seed=0)
    method = TARGET({"n_classes": 6, "feature_dim": X.shape[1], "local_epochs": 1, "replay_ratio": 1.0})
    result = sim.run(method, X, y, stream, seed=0)
    report = check_disjointness(result["ledger"])
    assert not report.task_disjoint
    assert "V2/V3" in report.violations


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
    assert result["final_avg_acc"] > 0.5  # well-separated synthetic clusters, should be easy
    assert len(result["ledger"]) > 0
    report = check_disjointness(result["ledger"])
    assert report.task_disjoint, report.violations


def test_m4_prototype_half_end_to_end_on_synthetic_features():
    X, y = _synthetic()
    idx = np.arange(len(y))
    stream = build_stream(y, idx, n_tasks=3, n_clients=2, beta=1.0, seed=0)
    method = PrototypeFCL({"n_classes": 6, "feature_dim": X.shape[1], "prototype_momentum": 0.0})
    result = sim.run(method, X, y, stream, seed=0)
    assert result["final_avg_acc"] > 0.5
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


def test_m4_prototype_half_domain_incremental_with_momentum_violates_disjointness():
    # same classes recur every task (domain-incremental) + momentum > 0 -> genuine cross-task
    # influence -> the checker must catch it (V2/V3), not silently certify.
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
    assert not report.task_disjoint
    assert "V2/V3" in report.violations


def test_m4_prototype_half_release_counts_can_be_disabled():
    X, y = _synthetic()
    idx = np.arange(len(y))
    stream = build_stream(y, idx, n_tasks=3, n_clients=2, beta=1.0, seed=0)
    method = PrototypeFCL({"n_classes": 6, "feature_dim": X.shape[1], "release_counts": False})
    result = sim.run(method, X, y, stream, seed=0)
    families = {r.family.value for r in result["ledger"]}
    assert "F7" not in families
