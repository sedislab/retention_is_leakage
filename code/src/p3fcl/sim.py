"""The FCL simulation loop. The accuracy matrix and the ledger are the two outputs everything
downstream consumes — nothing else reads the method's internals.
"""
from __future__ import annotations

import numpy as np

from . import metrics as metrics_mod
from . import rng as rng_mod
from .artifacts import Ledger


def run(method, X, y, stream, seed: int, eval_sets=None) -> dict:
    """`stream`: `list[list[Shard]]` as produced by `streams.build_stream`.

    `eval_sets`: FX4a (08_FIX_PLAN.md R1) -- optional `list[(X_eval_k, y_eval_k)]`, one held-out
    (test-split) pair per task, built by the caller via `streams.task_eval_sets` and the eval feature
    cache. When given, `acc_matrix[t, k]` (the primary, reported accuracy) is computed on
    `eval_sets[k]` instead of task k's own training ids -- the pre-fix behavior measured accuracy on
    the same examples the method trained on, which is not a meaningful accuracy number at all. The
    old train-id evaluation is *always* still computed and returned as `acc_matrix_train`, so BWT/AIA
    on the training data remain available as a diagnostic, never as the headline number.

    Returns `acc_matrix[T,T]` (`nan` for `k > t`, or wherever `eval_sets[k]` is empty),
    `acc_matrix_train`, `final_avg_acc`/`bwt`/`avg_incremental_acc` (test, only present when
    `eval_sets` is given) plus their `_train` counterparts (always present), `ledger`.
    """
    T = len(stream)
    ledger = Ledger()
    acc_matrix_train = np.full((T, T), np.nan)
    acc_matrix = np.full((T, T), np.nan) if eval_sets is not None else None
    r = rng_mod.seeded(f"sim.run::{getattr(method.spec, 'name', 'method')}", seed)

    task_eval_ids = [sorted({i for shard in stream[t] for i in shard.ids}) for t in range(T)]
    # FX4b test (i) (08_FIX_PLAN.md §4b): a validation-only channel, never used by any real attack --
    # the true post-round global weight for F1 (model-delta) methods, so `_reconstruct_running_w`'s
    # ledger-only reconstruction can be checked against ground truth. Duck-typed (`hasattr(method,
    # "W")`) rather than an interface change, since non-F1 methods (M2/M4/M8) have no single running
    # weight matrix in this sense.
    w_history = []

    for t in range(T):
        recs = method.fit_task(t, X, y, ids=None, client_shards=stream[t], rng=r)
        ledger.extend(recs)
        if hasattr(method, "W"):
            w_history.append(method.W.copy())
        for k in range(t + 1):
            ids_k = task_eval_ids[k]
            if ids_k:
                y_true = y[ids_k]
                y_pred = method.predict(X[ids_k])
                acc_matrix_train[t, k] = metrics_mod.accuracy(y_true, y_pred)
            if eval_sets is not None:
                X_eval_k, y_eval_k = eval_sets[k]
                if len(y_eval_k) > 0:
                    y_pred_eval = method.predict(X_eval_k)
                    acc_matrix[t, k] = metrics_mod.accuracy(y_eval_k, y_pred_eval)

    result = {
        "acc_matrix_train": acc_matrix_train,
        "final_avg_acc_train": float(np.nanmean(acc_matrix_train[T - 1, :])) if T > 0 else float("nan"),
        "bwt_train": metrics_mod.backward_transfer(acc_matrix_train),
        "avg_incremental_acc_train": metrics_mod.average_incremental_accuracy(acc_matrix_train),
        "ledger": ledger,
        "w_history": w_history,
    }
    if eval_sets is not None:
        result.update({
            "acc_matrix": acc_matrix,
            "final_avg_acc": float(np.nanmean(acc_matrix[T - 1, :])) if T > 0 else float("nan"),
            "bwt": metrics_mod.backward_transfer(acc_matrix),
            "avg_incremental_acc": metrics_mod.average_incremental_accuracy(acc_matrix),
        })
    return result
