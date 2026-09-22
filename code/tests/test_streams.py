from __future__ import annotations

import numpy as np
import pytest
from p3fcl.streams import (
    Shard,
    build_stream,
    class_incremental_tasks,
    dirichlet_partition,
    domain_incremental_tasks,
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
