from __future__ import annotations

import numpy as np
import pytest
from p3fcl.streams import (
    Shard,
    build_stream,
    class_incremental_tasks,
    dirichlet_partition,
    domain_incremental_tasks,
    natural_chunked_partition,
    natural_partition,
)


def test_class_incremental_tasks_disjoint_and_covers_all_classes():
    labels = np.repeat(np.arange(10), 5)
    tasks = class_incremental_tasks(labels, n_tasks=5, seed=0)
    assert len(tasks) == 5
    flat = [c for t in tasks for c in t]
    assert sorted(flat) == list(range(10))
    assert len(set(flat)) == 10


def test_class_incremental_tasks_rejects_indivisible_split():
    labels = np.arange(10)
    with pytest.raises(ValueError):
        class_incremental_tasks(labels, n_tasks=3, seed=0)


def test_class_incremental_tasks_deterministic():
    labels = np.repeat(np.arange(10), 5)
    a = class_incremental_tasks(labels, n_tasks=5, seed=1)
    b = class_incremental_tasks(labels, n_tasks=5, seed=1)
    assert a == b


def test_dirichlet_partition_covers_all_ids_disjointly():
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 4, size=100)
    idx = np.arange(100)
    parts = dirichlet_partition(labels, idx, n_clients=5, beta=0.5, seed=0)
    all_ids = sorted(i for p in parts for i in p)
    assert all_ids == list(range(100))


def test_dirichlet_partition_beta_inf_is_roughly_uniform():
    labels = np.zeros(1000, dtype=int)
    idx = np.arange(1000)
    parts = dirichlet_partition(labels, idx, n_clients=4, beta=float("inf"), seed=0)
    sizes = [len(p) for p in parts]
    assert max(sizes) - min(sizes) <= 4  # near-exact quartering


def test_natural_partition_groups_by_domain():
    domain = np.array(["a", "a", "b", "b", "b"])
    idx = np.arange(5)
    parts = natural_partition(domain, idx)
    assert set(parts.keys()) == {"a", "b"}
    assert list(parts["a"]) == [0, 1]
    assert list(parts["b"]) == [2, 3, 4]


def test_domain_incremental_tasks_order_respected():
    domain = np.array([0, 1, 0, 1, 2])
    idx = np.arange(5)
    tasks = domain_incremental_tasks(domain, idx, task_order=[1, 0, 2])
    assert tasks[0] == [1, 3]
    assert tasks[1] == [0, 2]
    assert tasks[2] == [4]


def test_natural_chunked_partition_never_splits_a_group_across_clients():
    """`08_FIX_PLAN.md`'s H13 fix: a group (e.g. one Camelyon17 slide) is an indivisible physical
    unit -- every id sharing a field value must land in exactly one client's bucket."""
    field = np.array(["s1", "s1", "s1", "s2", "s2", "s3", "s4", "s4", "s4", "s4"])
    idx = np.arange(10)
    parts = natural_chunked_partition(field, idx, n_clients=3, seed=0)
    assert len(parts) == 3
    all_ids = sorted(i for p in parts for i in p)
    assert all_ids == list(range(10))
    # every group's ids all land in the SAME client
    for group_val in ("s1", "s2", "s3", "s4"):
        group_ids = set(np.where(field == group_val)[0].tolist())
        landed_in = [ci for ci, p in enumerate(parts) if group_ids & set(p.tolist())]
        assert len(landed_in) == 1, f"group {group_val} split across clients: {landed_in}"


def test_natural_chunked_partition_covers_all_ids_disjointly():
    rng = np.random.default_rng(0)
    field = rng.integers(0, 20, size=200)  # 20 groups, uneven sizes
    idx = np.arange(200)
    parts = natural_chunked_partition(field, idx, n_clients=5, seed=1)
    all_ids = sorted(i for p in parts for i in p)
    assert all_ids == list(range(200))


def test_natural_chunked_partition_is_deterministic_given_a_seed():
    field = np.array([f"s{i % 7}" for i in range(50)])
    idx = np.arange(50)
    a = natural_chunked_partition(field, idx, n_clients=4, seed=3)
    b = natural_chunked_partition(field, idx, n_clients=4, seed=3)
    for pa, pb in zip(a, b):
        assert list(pa) == list(pb)


def test_natural_chunked_partition_varies_with_seed():
    """Real seed-to-seed variance is the whole point of the seeded shuffle -- a purely sorted-order
    assignment would make every seed give byte-identical client assignments, defeating the >=3-seed
    variance requirement this redesign is specifically meant to support (unlike the original FIG18
    pilot's `n_clients=1` natural arm, which had none)."""
    field = np.array([f"s{i % 11}" for i in range(80)])
    idx = np.arange(80)
    a = natural_chunked_partition(field, idx, n_clients=4, seed=0)
    b = natural_chunked_partition(field, idx, n_clients=4, seed=1)
    assert any(list(pa) != list(pb) for pa, pb in zip(a, b))


def test_build_stream_with_client_field_never_splits_a_group_and_matches_n_clients():
    """End to end through `build_stream` itself: `client_field` must reach the within-task client
    split, giving exactly `n_clients` buckets per task with slide-like groups intact."""
    labels = np.array([0] * 30 + [1] * 30)  # 2 classes, one per task (n_tasks=2)
    idx = np.arange(60)
    slide = np.array([f"slide{i % 10}" for i in range(60)])  # 10 slides, 6 ids each
    stream = build_stream(labels, idx, n_tasks=2, n_clients=5, beta=0.5, seed=0, client_field=slide)
    assert len(stream) == 2
    for shards in stream:
        assert len(shards) <= 5  # at most n_clients per task (fewer if a client got 0 ids)
        for shard in shards:
            slide_vals = {slide[i] for i in shard.ids}
            for sv in slide_vals:
                all_sv_ids = set(np.where(slide == sv)[0].tolist())
                # every id from this slide that appears in THIS task must all be in THIS shard
                task_ids = {i for s in shards for i in s.ids}
                sv_in_task = all_sv_ids & task_ids
                assert sv_in_task.issubset(set(shard.ids))


def test_build_stream_ids_are_stable_and_disjoint_across_tasks():
    labels = np.repeat(np.arange(10), 20)
    idx = np.arange(len(labels))
    stream = build_stream(labels, idx, n_tasks=5, n_clients=3, beta=0.5, seed=0)
    assert len(stream) == 5
    seen = set()
    for shards in stream:
        for shard in shards:
            assert isinstance(shard, Shard)
            assert not (seen & set(shard.ids)), "ids must not repeat across tasks (class-incremental)"
            seen.update(shard.ids)
    assert seen == set(idx.tolist())
