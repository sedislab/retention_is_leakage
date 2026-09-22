"""M1 — GLFC (CVPR'22), the classic FCIL reference. Families F1 (model delta), F7 (per-class counts,
for GLFC's class-imbalance "compensation"), F8 (a small exemplar buffer feeding both replay and
distillation). The original paper fine-tunes the whole backbone; in this harness it is restricted to
the frozen-feature head (`cacheable=True` reflects the "partial (feature space)" entry in
`04_METHODS_AND_ATTACKS.md` — the reimplementation is feature-space-only, not the original's).

Retention mechanism: local training combines cross-entropy on task data + replayed exemplars with a
temperature-scaled knowledge-distillation term pulling the new head's predictions toward the
*previous round's* head's predictions. That previous head was itself shaped by every task before it,
so distillation is exactly the "old statistics re-enter the release" channel (RESEARCH_PLAN.md §4.3,
V3) — `touched` honestly accumulates every id that has ever fed the model once
`distillation_weight > 0`, not just the current round's.
"""
from __future__ import annotations

import numpy as np

from ..artifacts import ArtifactRecord, Family
from .base import FCLMethod, MethodSpec


def _softmax(z: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = z / temperature
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class GLFC(FCLMethod):
    spec = MethodSpec(
        name="M1_glfc",
        families=(Family.MODEL_DELTA, Family.COUNTS, Family.EXEMPLAR),
        cacheable=True,
        retention_type="individual",
        retention_knob_name="distillation_weight",
    )

    def __init__(self, config: dict):
        super().__init__(config)
        self.n_classes = int(config["n_classes"])
        self.d = int(config["feature_dim"])
        self.local_epochs = int(config.get("local_epochs", 1))
        self.lr = float(config.get("lr", 0.5))
        self.exemplar_budget = int(config.get("exemplar_budget", 5))
        self.distillation_weight = float(config.get("distillation_weight", 1.0))
        self.temperature = float(config.get("temperature", 2.0))
        self.W = np.zeros((self.d, self.n_classes))
        self._buffer: dict = {}
        self._all_touched_ever: frozenset = frozenset()
        self._round = 0

    def rounds_per_task(self) -> int:
        return 1

    def _buffer_arrays(self):
        xs, ys, ids = [], [], []
        for c, entries in self._buffer.items():
            for feat, did in entries:
                xs.append(feat)
                ys.append(c)
                ids.append(did)
        if not xs:
            return None, None, []
        return np.array(xs), np.array(ys), ids

    def _local_train(self, W0, X_task, y_task, X_buf, y_buf, W_old) -> np.ndarray:
        W = W0.copy()
        X_all = np.concatenate([X_task, X_buf], axis=0) if X_buf is not None else X_task
        y_all = np.concatenate([y_task, y_buf], axis=0) if y_buf is not None else y_task
        n = len(y_all)
        Y = np.zeros((n, self.n_classes))
        Y[np.arange(n), y_all] = 1.0
        for _ in range(self.local_epochs):
            logits = X_all @ W
            grad = X_all.T @ (_softmax(logits) - Y) / n
            if W_old is not None and self.distillation_weight > 0:
                soft_old = _softmax(X_all @ W_old, self.temperature)
                soft_new = _softmax(logits, self.temperature)
                grad_kd = X_all.T @ (soft_new - soft_old) / n
                grad = grad + self.distillation_weight * grad_kd
            W = W - self.lr * grad
        return W

    def fit_task(self, task_idx, X, y, ids, client_shards, rng) -> list:
        records = []
        deltas, weights = [], []
        X_buf, y_buf, buf_ids = self._buffer_arrays()
        W_old = self.W.copy() if self._round > 0 else None
        distilling = W_old is not None and self.distillation_weight > 0
        task_touched: set = set()

        for shard in client_shards:
            idx = np.array(shard.ids, dtype=int)
            task_touched.update(int(i) for i in idx)
            Wc = self._local_train(self.W, X[idx], y[idx], X_buf, y_buf, W_old)
            delta = Wc - self.W
            deltas.append(delta)
            weights.append(len(idx))

            classes_here = sorted({int(c) for c in y[idx]})
            counts = np.zeros(self.n_classes, dtype=int)
            exemplar_touched: set = set()
            for c in classes_here:
                class_idx = idx[y[idx] == c]
                counts[c] = len(class_idx)
                k = min(self.exemplar_budget, len(class_idx))
                chosen = class_idx[rng.permutation(len(class_idx))[:k]]
                self._buffer[c] = [(X[i].copy(), int(i)) for i in chosen]
                exemplar_touched.update(int(i) for i in chosen)

            touched_delta = frozenset(int(i) for i in idx) | frozenset(buf_ids)
            if distilling:
                touched_delta = touched_delta | self._all_touched_ever
            records.append(
                ArtifactRecord(
                    round=self._round, task=task_idx, client=shard.client, family=Family.MODEL_DELTA,
                    payload=delta, touched=touched_delta, n_touched=len(touched_delta),
                    passes_over_data=self.local_epochs,
                    meta={"distillation_weight": self.distillation_weight, "temperature": self.temperature},
                )
            )
            records.append(
                ArtifactRecord(
                    round=self._round, task=task_idx, client=shard.client, family=Family.COUNTS,
                    payload={"counts": counts}, touched=frozenset(int(i) for i in idx),
                    n_touched=len(idx), passes_over_data=1, meta={},
                )
            )
            records.append(
                ArtifactRecord(
                    round=self._round, task=task_idx, client=shard.client, family=Family.EXEMPLAR,
                    payload={f"feat_{c}": np.array([f for f, _d in self._buffer[c]]) for c in classes_here},
                    touched=frozenset(exemplar_touched), n_touched=len(exemplar_touched),
                    passes_over_data=1, meta={"exemplar_budget": self.exemplar_budget},
                )
            )

        weights = np.array(weights, dtype=float)
        if weights.sum() > 0:
            avg_delta = sum(w * d for w, d in zip(weights, deltas)) / weights.sum()
            self.W = self.W + avg_delta
        self._all_touched_ever = self._all_touched_ever | frozenset(task_touched) | frozenset(buf_ids)
        self._round += 1
        return records

    def predict(self, X) -> np.ndarray:
        return np.argmax(X @ self.W, axis=1)
