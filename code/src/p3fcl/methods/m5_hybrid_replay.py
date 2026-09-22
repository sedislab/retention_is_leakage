"""M5 — Hybrid Replay FCIL (ICLR'25), families F8 (raw exemplar buffer) + F1 (the model delta the
replay-augmented local training produces). F8 is the "raw-exemplar upper bound on leakage"
(RESEARCH_PLAN.md §2.2 — "literal raw samples... trivially critical"): each client keeps a per-class
buffer of literal cached feature vectors and releases whichever exemplars it just added, every round.

**Correction, 2026-09-15**: an earlier version of this method released *only* the F8 artifact, on the
theory that F1 leakage was already covered by M0/M1/M3 and mixing signals would dilute FIG04. That
was wrong: local training already retrains on task-data + buffered exemplars every round (see
`_local_train`), so the resulting delta genuinely *is* a function of old-task data via replay — and
suppressing its release made `dp.accountant.check_disjointness` certify M5 as task-disjoint, directly
contradicting H6 (F8 is hypothesized to NOT admit task-disjointness) and CLAUDE.md non-negotiable #2
("under-reporting `touched` silently converts a false privacy claim into a 'proved' one"). The delta
is released again now, with `touched` honestly including every exemplar id currently in the buffer
that this round's training drew on — not just the ids newly added to it.
"""
from __future__ import annotations

import numpy as np

from ..artifacts import ArtifactRecord, Family
from .base import FCLMethod, MethodSpec


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class HybridReplay(FCLMethod):
    spec = MethodSpec(
        name="M5_hybrid_replay",
        families=(Family.EXEMPLAR, Family.MODEL_DELTA),
        cacheable=True,
        retention_type="individual",
        retention_knob_name="buffer_size_per_class",
    )

    def __init__(self, config: dict):
        super().__init__(config)
        self.n_classes = int(config["n_classes"])
        self.d = int(config["feature_dim"])
        self.local_epochs = int(config.get("local_epochs", 1))
        self.lr = float(config.get("lr", 0.5))
        self.buffer_size_per_class = int(config.get("buffer_size_per_class", 5))
        self.W = np.zeros((self.d, self.n_classes))
        self._buffer: dict = {}  # class -> list[(feature_vector, datum_id)]
        self._round = 0

    def rounds_per_task(self) -> int:
        return 1

    def _buffer_arrays_and_ids(self):
        xs, ys, ids = [], [], []
        for c, entries in self._buffer.items():
            for feat, did in entries:
                xs.append(feat)
                ys.append(c)
                ids.append(did)
        if not xs:
            return None, None, frozenset()
        return np.array(xs), np.array(ys), frozenset(ids)

    def _local_train(self, W0: np.ndarray, X_task: np.ndarray, y_task: np.ndarray, X_buf, y_buf) -> np.ndarray:
        W = W0.copy()
        X_all = np.concatenate([X_task, X_buf], axis=0) if X_buf is not None else X_task
        y_all = np.concatenate([y_task, y_buf], axis=0) if y_buf is not None else y_task
        n = len(y_all)
        Y = np.zeros((n, self.n_classes))
        Y[np.arange(n), y_all] = 1.0
        for _ in range(self.local_epochs):
            P = _softmax(X_all @ W)
            grad = X_all.T @ (P - Y) / n
            W = W - self.lr * grad
        return W

    def fit_task(self, task_idx, X, y, ids, client_shards, rng) -> list:
        records = []
        deltas, weights = [], []
        X_buf, y_buf, buf_ids = self._buffer_arrays_and_ids()

        for shard in client_shards:
            idx = np.array(shard.ids, dtype=int)
            Wc = self._local_train(self.W, X[idx], y[idx], X_buf, y_buf)
            delta = Wc - self.W
            deltas.append(delta)
            weights.append(len(idx))

            classes_here = sorted({int(c) for c in y[idx]})
            payload: dict = {}
            new_touched: set = set()
            for c in classes_here:
                class_idx = idx[y[idx] == c]
                k = min(self.buffer_size_per_class, len(class_idx))
                chosen = class_idx[rng.permutation(len(class_idx))[:k]]
                self._buffer[c] = [(X[i].copy(), int(i)) for i in chosen]
                payload[f"feat_{c}"] = X[chosen].copy()
                payload[f"ids_{c}"] = chosen.astype(np.int64)
                new_touched.update(int(i) for i in chosen)

            records.append(
                ArtifactRecord(
                    round=self._round,
                    task=task_idx,
                    client=shard.client,
                    family=Family.EXEMPLAR,
                    payload=payload,
                    touched=frozenset(new_touched),
                    n_touched=len(new_touched),
                    passes_over_data=1,
                    meta={"buffer_size_per_class": self.buffer_size_per_class},
                )
            )

            delta_touched = frozenset(int(i) for i in idx) | buf_ids
            records.append(
                ArtifactRecord(
                    round=self._round,
                    task=task_idx,
                    client=shard.client,
                    family=Family.MODEL_DELTA,
                    payload=delta,
                    touched=delta_touched,
                    n_touched=len(delta_touched),
                    passes_over_data=self.local_epochs,
                    meta={"local_epochs": self.local_epochs, "lr": self.lr},
                )
            )

        weights = np.array(weights, dtype=float)
        if weights.sum() > 0:
            avg_delta = sum(w * d for w, d in zip(weights, deltas)) / weights.sum()
            self.W = self.W + avg_delta
        self._round += 1
        return records

    def predict(self, X) -> np.ndarray:
        return np.argmax(X @ self.W, axis=1)
