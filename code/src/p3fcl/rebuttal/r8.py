"""R8 (`rebuttal/R8_PROTOCOL.md`): the size-matched public selection federation and the edge flag."""
from __future__ import annotations

import numpy as np

from .. import rng as rng_mod
from .. import streams


def selection_federation(X_ref_train: np.ndarray, y_ref_train: np.ndarray, y_train: np.ndarray, dataset: str,
                         n_tasks: int = 10, seed: int = 0) -> tuple:
    """`(X_sel, y_sel, counts)`: for each task (class sets from `class_incremental_tasks(.., seed)`), draw
    N_t = (# training labels in the task) examples with replacement from ref-train, keeping ref-train's
    within-task class composition. Only training label counts (public metadata) are read from train."""
    r = rng_mod.seeded(f"rb_r8::selection_federation::{dataset}", seed)
    ids, counts = [], {}
    for task_cls in streams.class_incremental_tasks(y_ref_train, n_tasks, seed):
        task_cls = [int(c) for c in task_cls]
        n_t = int(np.isin(y_train, task_cls).sum())
        ref_c = np.array([np.sum(y_ref_train == c) for c in task_cls], dtype=float)
        target = np.round(n_t * ref_c / ref_c.sum()).astype(int)
        for c, k in zip(task_cls, target):
            pool = np.flatnonzero(y_ref_train == c)
            if k > 0 and len(pool):
                ids.append(r.choice(pool, size=int(k), replace=True))
                counts[c] = int(k)
    ids = np.concatenate(ids)
    return X_ref_train[ids], y_ref_train[ids], counts


def on_edge(point: dict, grids: dict) -> list:
    """Axes (e.g. `lr_min`, `lambda_max`) on which `point` sits at the grid's min or max."""
    out = []
    for axis, values in grids.items():
        if point[axis] == min(values):
            out.append(f"{axis}_min")
        if point[axis] == max(values):
            out.append(f"{axis}_max")
    return out
