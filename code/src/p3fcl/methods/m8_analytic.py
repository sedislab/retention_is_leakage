"""M8 — Analytic FCL (family F5): a closed-form ridge classifier over frozen features.

Each client emits one `(R_c, Q_c)` release per task: `R_c = X_c^T X_c + lambda*I` (exact Gram) and
`Q_c = X_c^T Y_c` (cross-correlation with one-hot labels), computed from that task's local data only.
Single pass, one release per client-task -> task-disjoint by construction (this is what makes T1-fwd
apply and what the H6/H5-exact-at-n=1 claim tests check). The server keeps a running `(R, Q)` summed
over every task and client seen so far and predicts by closed-form ridge regression `W = R^-1 Q` —
the retention *is* the running sum, but each release itself never re-touches old data.
"""
from __future__ import annotations

import numpy as np

from ..artifacts import ArtifactRecord, Family
from .base import FCLMethod, MethodSpec


class AnalyticFCL(FCLMethod):
    spec = MethodSpec(
        name="M8_analytic_fcl",
        families=(Family.GRAM,),
        cacheable=True,
        retention_type="individual",
        retention_knob_name="ridge_lambda",
    )

    def __init__(self, config: dict):
        super().__init__(config)
        self.ridge_lambda = float(config.get("ridge_lambda", 1.0))
        self.n_classes = int(config["n_classes"])
        self.d = int(config["feature_dim"])
        self._R = None
        self._Q = None
        self._round = 0

    def rounds_per_task(self) -> int:
        return 1

    def _one_hot(self, y) -> np.ndarray:
        Y = np.zeros((len(y), self.n_classes))
        Y[np.arange(len(y)), y] = 1.0
        return Y

    def fit_task(self, task_idx, X, y, ids, client_shards, rng) -> list:
        if self._R is None:
            self._R = np.zeros((self.d, self.d))
            self._Q = np.zeros((self.d, self.n_classes))
        records = []
        for shard in client_shards:
            idx = np.array(shard.ids, dtype=int)
            Xc, yc = X[idx], y[idx]
            Rc = Xc.T @ Xc + self.ridge_lambda * np.eye(self.d)
            Qc = Xc.T @ self._one_hot(yc)
            self._R += Rc
            self._Q += Qc
            records.append(
                ArtifactRecord(
                    round=self._round,
                    task=task_idx,
                    client=shard.client,
                    family=Family.GRAM,
                    payload={"R": Rc, "Q": Qc},
                    touched=frozenset(int(i) for i in idx),
                    n_touched=len(idx),
                    passes_over_data=1,
                    meta={"ridge_lambda": self.ridge_lambda},
                )
            )
        self._round += 1
        return records

    def predict(self, X) -> np.ndarray:
        if self._R is None:
            return np.zeros(len(X), dtype=int)
        W = np.linalg.solve(self._R, self._Q)
        return np.argmax(X @ W, axis=1)
