"""R3 (`rebuttal/R3_PROTOCOL.md`): size-matched pseudo-clients on ref-train, the size-matched clip
norm C(n, p), and per-run U2 diagnostics computed with M9's own helpers."""
from __future__ import annotations

import numpy as np

from .. import rng as rng_mod
from ..methods.m9_contractive import clip_pair, client_stats, project_and_cap

N_PSEUDO_PER_TASK = 50
PERCENTILE = 95


def mean_client_size(n_train: int, n_tasks: int, n_clients: int) -> int:
    return max(1, int(round(n_train / (n_tasks * n_clients))))


def pseudo_client_ids(y_ref_train: np.ndarray, task_classes: list, m: int, dataset: str, n: int,
                      n_per_task: int = N_PSEUDO_PER_TASK, beta: float = 0.5) -> list:
    """`n_per_task` pseudo-clients per task, each `m` ids drawn with replacement from ref-train:
    Dirichlet(beta) class proportions over the task's classes, Multinomial(m) class counts."""
    r = rng_mod.seeded(f"rb_r3::pseudo::{dataset}::{n}", 0)
    by_class = {int(c): np.flatnonzero(y_ref_train == c) for c in np.unique(y_ref_train)}
    out = []
    for classes in task_classes:
        classes = [int(c) for c in classes if int(c) in by_class]
        for _ in range(n_per_task):
            props = r.dirichlet(np.full(len(classes), beta))
            counts = r.multinomial(m, props)
            ids = [r.choice(by_class[c], size=k, replace=True) for c, k in zip(classes, counts) if k > 0]
            out.append(np.concatenate(ids))
    return out


def pair_norms(X: np.ndarray, y: np.ndarray, id_lists: list, P: np.ndarray, n_classes: int, B: float = 1.0) -> np.ndarray:
    norms = []
    for ids in id_lists:
        G, H = client_stats(project_and_cap(X[ids], P, B), y[ids], n_classes)
        norms.append(float(np.sqrt(np.sum(G**2) + np.sum(H**2))))
    return np.array(norms)


def size_matched_clip(X_ref_train, y_ref_train, pseudo_ids, P, n_classes) -> float:
    return float(np.percentile(pair_norms(X_ref_train, y_ref_train, pseudo_ids, P, n_classes), PERCENTILE))


def u2_diagnostics(X, y, stream: list, P: np.ndarray, n_classes: int, clip_C, clip: bool, B: float = 1.0) -> dict:
    """`frac_clipped` over every (task, client) pair (None when `clip` is False) and `agg0_frob`, the
    Frobenius norm of task 0's noiseless aggregate after clipping. Mirrors `ContractiveDPAnalytic.fit_task`."""
    n_clip = n_total = 0
    agg0 = None
    for t, shards in enumerate(stream):
        if t > 0 and not clip:
            break  # only task 0's aggregate is needed without clipping
        G_sum = np.zeros((P.shape[1], P.shape[1]))
        H_sum = np.zeros((P.shape[1], n_classes))
        for shard in shards:
            idx = np.array(shard.ids, dtype=int)
            Gc, Hc = client_stats(project_and_cap(X[idx], P, B), y[idx], n_classes)
            if clip:
                norm = float(np.sqrt(np.sum(Gc**2) + np.sum(Hc**2)))
                n_clip += int(norm > clip_C)
                n_total += 1
                Gc, Hc = clip_pair(Gc, Hc, clip_C)
            if t == 0:
                G_sum += Gc
                H_sum += Hc
        if t == 0:
            agg0 = float(np.sqrt(np.sum(G_sum**2) + np.sum(H_sum**2)))
    return {"frac_clipped": (n_clip / n_total) if clip and n_total else None, "agg0_frob": agg0}


# ---------------------------------------------------------------- R5 amendment 1: McMahan weight cap

CAP_SEEDS = list(range(1000, 1020))  # namespace disjoint from the run seeds 0-2


def public_client_sizes(y_ref_train: np.ndarray, n_train: int, n_tasks: int, n_clients: int, beta: float,
                        seeds=CAP_SEEDS) -> np.ndarray:
    """Client sizes of size-matched pseudo-federations: each class gets round(n_train * its ref-train
    frequency) pseudo-examples, the task classes come from `class_incremental_tasks` on the ref-train
    labels, and the paper's `dirichlet_partition` (beta) splits each task across `n_clients`. Only the
    public dataset size and the ref-train labels are used."""
    from .. import streams
    classes, ref_counts = np.unique(y_ref_train, return_counts=True)
    counts = dict(zip(classes.tolist(), np.maximum(1, np.round(n_train * ref_counts / ref_counts.sum())).astype(int).tolist()))
    sizes = []
    for s in seeds:
        for t, task_cls in enumerate(streams.class_incremental_tasks(y_ref_train, n_tasks, s)):
            labels = np.concatenate([np.full(counts[int(c)], int(c)) for c in task_cls])
            parts = streams.dirichlet_partition(labels, np.arange(len(labels)), n_clients, beta, seed=s + 1000 * t)
            sizes.extend(len(p) for p in parts if len(p) > 0)
    return np.array(sizes)


def public_weight_cap(y_ref_train, n_train, n_tasks, n_clients, beta, pct=PERCENTILE) -> float:
    return float(np.percentile(public_client_sizes(y_ref_train, n_train, n_tasks, n_clients, beta), pct))
