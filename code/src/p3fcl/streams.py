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


def natural_chunked_partition(field, idx, n_clients: int, seed: int) -> list:
    """`08_FIX_PLAN.md`'s H13 fix ("matched-N-client redesign"): groups `idx` by `field`'s distinct
    values (e.g. a Camelyon17 slide id) -- a genuine, indivisible physical unit that must never split
    across clients, unlike `dirichlet_partition`'s per-class random draw which ignores any such
    structure entirely -- then distributes those WHOLE groups across `n_clients` buckets via a seeded
    shuffle of the group order followed by round-robin assignment (not a per-class Dirichlet draw, and
    not a fixed sorted-order assignment either: the shuffle gives genuine seed-to-seed stream variance,
    which CLAUDE.md non-negotiable #4's >=3-seed requirement needs; round-robin, rather than contiguous
    chunking, gives better client-size balance on average without ever rebalancing WITHIN a group).
    Client sizes are whatever falls out of the real group-size distribution -- no attempt to equalize
    them, since imposing balance would defeat the point of testing a real distribution against
    Dirichlet's synthetic one. Returns one sorted id array per client, `0..n_clients-1`."""
    idx = np.asarray(idx)
    field = np.asarray(field)
    groups: dict = {}
    for i in idx:
        v = field[i]
        key = v.item() if hasattr(v, "item") else v
        groups.setdefault(key, []).append(int(i))
    group_keys = sorted(groups.keys())
    r = rng_mod.seeded("streams.natural_chunked_partition", seed)
    order = r.permutation(len(group_keys))
    buckets: list = [[] for _ in range(n_clients)]
    for j, gi in enumerate(order):
        buckets[j % n_clients].extend(groups[group_keys[gi]])
    return [np.array(sorted(b), dtype=int) for b in buckets]


def domain_incremental_tasks(domain_field, idx, task_order) -> list:
    """One task per domain value, in `task_order`. Returns a list of id-lists, one per task."""
    idx = np.asarray(idx)
    domain_field = np.asarray(domain_field)
    tasks = []
    for dv in task_order:
        mask = domain_field[idx] == dv
        tasks.append(idx[mask].tolist())
    return tasks


def task_eval_sets(y_eval, idx_eval, n_tasks: int, seed: int, domain_field_eval=None) -> list:
    """FX4a (08_FIX_PLAN.md R1): eval-split ids per task, using the exact same task-to-class (or
    task-to-domain) partition `build_stream` used for the *training* stream -- `class_incremental_tasks`
    depends only on the class universe + `n_tasks` + `seed` (not on which ids are passed), so calling
    it again here with the eval split's own labels reproduces the identical partition as long as the
    eval split covers the same classes. Domain-incremental: recomputes the domain list from the eval
    split's own domain field (assumes -- and callers should confirm -- the eval split covers the same
    domains as train; a domain absent from eval simply gets an empty task, handled by callers)."""
    idx_eval = np.asarray(idx_eval)
    y_eval = np.asarray(y_eval)
    if domain_field_eval is not None:
        domain_field_eval = np.asarray(domain_field_eval)
        domains = sorted(set(domain_field_eval[idx_eval].tolist()))
        return domain_incremental_tasks(domain_field_eval, idx_eval, domains)
    task_classes = class_incremental_tasks(y_eval, n_tasks, seed)
    out = []
    for classes in task_classes:
        mask = np.isin(y_eval[idx_eval], classes)
        out.append(idx_eval[mask].tolist())
    return out


def build_stream(
    labels, idx, n_tasks: int, n_clients: int, beta: float, seed: int, domain_field=None, client_field=None,
) -> list:
    """Class-incremental by default. If `domain_field` is given, one task per domain value instead
    (`n_tasks` is then ignored). If `client_field` is given, each task's within-task client split uses
    `natural_chunked_partition` (grouping by that field's real values, e.g. a Camelyon17 slide id, into
    `n_clients` buckets) instead of `dirichlet_partition`'s random per-class draw -- `08_FIX_PLAN.md`'s
    H13 "matched-N-client redesign": both partition strategies can now share the same `n_clients`, so a
    real-vs-synthetic client-structure comparison no longer also confounds client COUNT (`beta` is
    ignored when `client_field` is given, since there is no Dirichlet draw to concentrate). Returns
    `stream[t] = list[Shard]`, one Shard per client that participates in task `t` (clients with zero
    assigned ids in a task are omitted, not zero-padded)."""
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

    if client_field is not None:
        client_field = np.asarray(client_field)

    stream: list = []
    for t, ids_t in enumerate(task_id_lists):
        ids_t_arr = np.array(ids_t, dtype=int)
        if client_field is not None:
            client_ids = natural_chunked_partition(client_field, ids_t_arr, n_clients, seed=seed + 1000 * t)
        else:
            client_ids = dirichlet_partition(labels, ids_t_arr, n_clients, beta, seed=seed + 1000 * t)
        shards = [
            Shard(client=c, task=t, ids=tuple(sorted(int(i) for i in ci)))
            for c, ci in enumerate(client_ids)
            if len(ci) > 0
        ]
        stream.append(shards)
    return stream
