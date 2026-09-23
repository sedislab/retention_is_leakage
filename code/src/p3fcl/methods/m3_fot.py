"""M3 — FOT (ICLR'24), the orthogonality-regularisation family (family F1 + F5 subspace). Maintains a
running orthonormal subspace `U` spanning the dominant feature directions of every task seen so far;
each task's local gradient is projected to remove its component along `U` before the weight update,
so new-task learning does not overwrite directions the old tasks relied on. `projection_strength`
(0 = plain FedAvg, 1 = full orthogonal projection) is the retention knob for FIG03/FIG04.

**FX4b fix, 08_FIX_PLAN.md R3**: the subspace `U` is itself now an honest, separate release (family
GRAM, `client=-1`, one record per task, payload = this task's feature covariance, `touched` = exactly
this task's ids) — computing it from task data is the one genuine read of raw data; every later
round's use of `U` to project gradients is post-processing of that already-released statistic, per
Lemma 8 (08_FIX_PLAN.md §4b), and adds nothing further to any record's `touched`. The MODEL_DELTA
record's `touched` is therefore exactly the current round's shard, no longer inflated by
`_subspace_touched` accumulated across every prior task.
"""
from __future__ import annotations

import numpy as np

from ..artifacts import ArtifactRecord, Family
from .base import FCLMethod, MethodSpec


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class FOT(FCLMethod):
    spec = MethodSpec(
        name="M3_fot",
        families=(Family.MODEL_DELTA, Family.GRAM),
        cacheable=True,
        retention_type="individual",
        retention_knob_name="projection_strength",
    )

    def __init__(self, config: dict):
        super().__init__(config)
        self.n_classes = int(config["n_classes"])
        self.d = int(config["feature_dim"])
        self.local_epochs = int(config.get("local_epochs", 1))
        self.lr = float(config.get("lr", 0.5))
        self.subspace_rank = int(config.get("subspace_rank", 4))
        self.projection_strength = float(config.get("projection_strength", 1.0))
        self.W = np.zeros((self.d, self.n_classes))
        self.U = np.zeros((self.d, 0))
        self._round = 0

    def rounds_per_task(self) -> int:
        return 1

    def _project(self, grad: np.ndarray) -> np.ndarray:
        if self.U.shape[1] == 0 or self.projection_strength == 0.0:
            return grad
        proj = self.U @ (self.U.T @ grad)
        return grad - self.projection_strength * proj

    def _local_train(self, W0: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        W = W0.copy()
        n = len(y)
        Y = np.zeros((n, self.n_classes))
        Y[np.arange(n), y] = 1.0
        for _ in range(self.local_epochs):
            P = _softmax(X @ W)
            grad = self._project(X.T @ (P - Y) / n)
            W = W - self.lr * grad
        return W

    def fit_task(self, task_idx, X, y, ids, client_shards, rng) -> list:
        records = []
        deltas, weights = [], []
        task_ids: set = set()
        for shard in client_shards:
            idx = np.array(shard.ids, dtype=int)
            self.update_classes_seen(y[idx])
            task_ids.update(int(i) for i in idx)
            Wc = self._local_train(self.W, X[idx], y[idx])
            delta = Wc - self.W
            deltas.append(delta)
            weight = len(idx)
            weights.append(weight)
            touched = frozenset(int(i) for i in idx)
            records.append(
                ArtifactRecord(
                    round=self._round,
                    task=task_idx,
                    client=shard.client,
                    family=Family.MODEL_DELTA,
                    payload=delta,
                    touched=touched,
                    n_touched=len(touched),
                    passes_over_data=self.local_epochs,
                    meta={
                        "local_epochs": self.local_epochs,
                        "lr": self.lr,
                        "projection_strength": self.projection_strength,
                        "subspace_rank": int(self.U.shape[1]),
                        "agg_weight": float(weight),
                    },
                )
            )
        weights = np.array(weights, dtype=float)
        if weights.sum() > 0:
            avg_delta = sum(w * d for w, d in zip(weights, deltas)) / weights.sum()
            self.W = self.W + avg_delta

        # The subspace update reads this task's raw features -- a genuine, one-time read, honestly
        # released as its own record (family GRAM, server-side/client=-1) rather than silently folded
        # into every future MODEL_DELTA's touched set.
        Xt = X[sorted(task_ids)]
        Xc = Xt - Xt.mean(axis=0, keepdims=True)
        cov = Xc.T @ Xc
        records.append(
            ArtifactRecord(
                round=self._round,
                task=task_idx,
                client=-1,
                family=Family.GRAM,
                payload={"R": cov},
                touched=frozenset(task_ids),
                n_touched=len(task_ids),
                passes_over_data=1,
                meta={"subspace_rank_added": self.subspace_rank},
            )
        )
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = np.argsort(eigvals)[::-1]
        k = min(self.subspace_rank, self.d)
        new_dirs = eigvecs[:, order[:k]]
        combined = np.concatenate([self.U, new_dirs], axis=1) if self.U.shape[1] else new_dirs
        self.U, _ = np.linalg.qr(combined)
        self._round += 1
        return records

    def predict(self, X) -> np.ndarray:
        return np.argmax(self.mask_unseen_logits(X @ self.W), axis=1)
