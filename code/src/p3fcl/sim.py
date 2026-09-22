"""The FCL simulation loop. The accuracy matrix and the ledger are the two outputs everything
downstream consumes — nothing else reads the method's internals.
"""
from __future__ import annotations

import numpy as np

from . import metrics as metrics_mod
from . import rng as rng_mod
from .artifacts import Ledger


def run(method, X, y, stream, seed: int) -> dict:
    """`stream`: `list[list[Shard]]` as produced by `streams.build_stream`.

    Returns `acc_matrix[T,T]` (`acc_matrix[t, k]` = accuracy on task k after finishing task t,
    `nan` for `k > t`), `final_avg_acc`, `bwt`, `avg_incremental_acc`, `ledger`.
    """
    T = len(stream)
    ledger = Ledger()
    acc_matrix = np.full((T, T), np.nan)
    r = rng_mod.seeded(f"sim.run::{getattr(method.spec, 'name', 'method')}", seed)

    task_eval_ids = [sorted({i for shard in stream[t] for i in shard.ids}) for t in range(T)]

    for t in range(T):
        recs = method.fit_task(t, X, y, ids=None, client_shards=stream[t], rng=r)
        ledger.extend(recs)
        for k in range(t + 1):
            ids_k = task_eval_ids[k]
            if not ids_k:
                continue
            y_true = y[ids_k]
            y_pred = method.predict(X[ids_k])
            acc_matrix[t, k] = metrics_mod.accuracy(y_true, y_pred)

    return {
        "acc_matrix": acc_matrix,
        "final_avg_acc": float(np.nanmean(acc_matrix[T - 1, :])) if T > 0 else float("nan"),
        "bwt": metrics_mod.backward_transfer(acc_matrix),
        "avg_incremental_acc": metrics_mod.average_incremental_accuracy(acc_matrix),
        "ledger": ledger,
    }
