"""M3 — FOT (ICLR'24), the orthogonality-regularisation family (family F1 + subspace). Maintains a
running orthonormal subspace `U` spanning the dominant feature directions of every task seen so far;
each task's local gradient is projected to remove its component along `U` before the weight update,
so new-task learning does not overwrite directions the old tasks relied on. `projection_strength`
(0 = plain FedAvg, 1 = full orthogonal projection) is the retention knob for FIG03/FIG04.

Cacheable (head-only, frozen backbone), but **not** task-disjoint: `U` is a running statistic built
from every prior task's features, and every subsequent release is a function of it — the model-delta
payload literally depends on data from earlier tasks through the projection. `touched` says so
honestly (CLAUDE.md non-negotiable #2): each release's touched set is this task's shard **union**
every id that ever contributed to `U`. This is the "V3 anti-forgetting regularisers... old data is
gone but old statistics re-enter the release" case named in RESEARCH_PLAN.md §4.3.
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
        families=(Family.MODEL_DELTA,),
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
        self._subspace_touched: frozenset = frozenset()
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
            task_ids.update(int(i) for i in idx)
            Wc = self._local_train(self.W, X[idx], y[idx])
            delta = Wc - self.W
            deltas.append(delta)
            weights.append(len(idx))
            touched = frozenset(int(i) for i in idx) | self._subspace_touched
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
                    },
                )
            )
        weights = np.array(weights, dtype=float)
        if weights.sum() > 0:
            avg_delta = sum(w * d for w, d in zip(weights, deltas)) / weights.sum()
            self.W = self.W + avg_delta

        # Grow the subspace with this task's dominant feature directions (bounded at rank d by QR).
        Xt = X[sorted(task_ids)]
        Xc = Xt - Xt.mean(axis=0, keepdims=True)
        cov = Xc.T @ Xc
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = np.argsort(eigvals)[::-1]
        k = min(self.subspace_rank, self.d)
        new_dirs = eigvecs[:, order[:k]]
        combined = np.concatenate([self.U, new_dirs], axis=1) if self.U.shape[1] else new_dirs
        self.U, _ = np.linalg.qr(combined)
        self._subspace_touched = self._subspace_touched | frozenset(task_ids)
        self._round += 1
        return records

    def predict(self, X) -> np.ndarray:
        return np.argmax(X @ self.W, axis=1)
