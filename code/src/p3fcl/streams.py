"""Task streams and client partitions. `ids` in every `Shard` are stable global datum ids — the
ledger's `touched` refers to these, so they must survive subsetting and shuffling unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import rng as rng_mod


@dataclass(frozen=True)
class Shard:
    client: int
    task: int
    ids: tuple


def class_incremental_tasks(labels, n_tasks: int, seed: int) -> list:
    """Shuffle the class list with `seed`, split into `n_tasks` disjoint groups. Returns a list of
    class-id lists, one per task."""
    labels = np.asarray(labels)
    classes = sorted({int(c) for c in labels})
    if len(classes) % n_tasks != 0:
        raise ValueError(f"{len(classes)} classes not evenly divisible into {n_tasks} tasks")
    r = rng_mod.seeded("streams.class_incremental_tasks", seed)
    order = r.permutation(len(classes))
    shuffled = [classes[i] for i in order]
    per_task = len(shuffled) // n_tasks
    return [shuffled[t * per_task : (t + 1) * per_task] for t in range(n_tasks)]


def dirichlet_partition(labels, idx, n_clients: int, beta: float, seed: int) -> list:
    """Standard non-IID partition of `idx` (dataset indices, e.g. one task's ids) across `n_clients`
    via a per-class Dirichlet(beta) split. `beta == inf` gives a uniform (IID) split."""
    idx = np.asarray(idx)
    labels = np.asarray(labels)
    r = rng_mod.seeded("streams.dirichlet_partition", seed)
    client_ids: list = [[] for _ in range(n_clients)]
    if len(idx) == 0:
        return [np.array([], dtype=int) for _ in range(n_clients)]
    sub_labels = labels[idx]
    for c in sorted(set(sub_labels.tolist())):
        class_idx = idx[sub_labels == c]
        class_idx = class_idx[r.permutation(len(class_idx))]
        if np.isinf(beta):
            props = np.full(n_clients, 1.0 / n_clients)
        else:
            props = r.dirichlet(np.full(n_clients, beta))
        splits = (np.cumsum(props) * len(class_idx)).astype(int)[:-1]
        parts = np.split(class_idx, splits)
        for cl, part in enumerate(parts):
            client_ids[cl].extend(int(i) for i in part)
    return [np.array(sorted(ids), dtype=int) for ids in client_ids]


def natural_partition(domain_field, idx) -> dict:
    """Client = real domain value (e.g. Camelyon17 hospital id). Returns `{domain_value: ids}`."""
    idx = np.asarray(idx)
    domain_field = np.asarray(domain_field)
    out: dict = {}
    for i in idx:
        d = domain_field[i]
        key = d.item() if hasattr(d, "item") else d
        out.setdefault(key, []).append(int(i))
    return {k: np.array(sorted(v), dtype=int) for k, v in out.items()}


def domain_incremental_tasks(domain_field, idx, task_order) -> list:
    """One task per domain value, in `task_order`. Returns a list of id-lists, one per task."""
    idx = np.asarray(idx)
    domain_field = np.asarray(domain_field)
    tasks = []
    for dv in task_order:
        mask = domain_field[idx] == dv
        tasks.append(idx[mask].tolist())
    return tasks


def build_stream(labels, idx, n_tasks: int, n_clients: int, beta: float, seed: int, domain_field=None) -> list:
    """Class-incremental by default. If `domain_field` is given, one task per domain value instead
    (`n_tasks` is then ignored). Returns `stream[t] = list[Shard]`, one Shard per client that
    participates in task `t` (clients with zero assigned ids in a task are omitted, not zero-padded)."""
    idx = np.asarray(idx)
    labels = np.asarray(labels)
    if domain_field is not None:
        domain_field = np.asarray(domain_field)
        domains = sorted(set(domain_field[idx].tolist()))
        task_id_lists = domain_incremental_tasks(domain_field, idx, domains)
    else:
        task_classes = class_incremental_tasks(labels, n_tasks, seed)
        task_id_lists = []
        for classes in task_classes:
            mask = np.isin(labels[idx], classes)
            task_id_lists.append(idx[mask].tolist())

    stream: list = []
    for t, ids_t in enumerate(task_id_lists):
        client_ids = dirichlet_partition(labels, np.array(ids_t, dtype=int), n_clients, beta, seed=seed + 1000 * t)
        shards = [
            Shard(client=c, task=t, ids=tuple(sorted(int(i) for i in ci)))
            for c, ci in enumerate(client_ids)
            if len(ci) > 0
        ]
        stream.append(shards)
    return stream
