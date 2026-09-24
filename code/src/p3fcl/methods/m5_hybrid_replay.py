"""M5 — exemplar replay in frozen feature space (Hybrid Replay-style, ICLR 2025).

The per-client buffer is private local state. Only the model delta (F1) is released;
its touched set includes every buffered example read that round. Local training
uses CE_mean(current) + replay_weight * CE_mean(buffer), with default weight 1.
"""
from __future__ import annotations

import numpy as np

from ..artifacts import ArtifactRecord, Family
from .base import FCLMethod, MethodSpec
from .replay import balanced_gradient


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class HybridReplay(FCLMethod):
    spec = MethodSpec(
        name="M5_hybrid_replay",
        families=(Family.MODEL_DELTA,),
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
        self.replay_weight = float(config.get("replay_weight", 1.0))
        self.buffer_size_per_class = int(config.get("buffer_size_per_class", 5))
        self.W = np.zeros((self.d, self.n_classes))
        self._buffer: dict = {}  # (client, class) -> list[(feature_vector, datum_id)]
        self._round = 0

    def rounds_per_task(self) -> int:
        return 1

    def _client_buffer_arrays_and_ids(self, client: int):
        xs, ys, ids = [], [], []
        for (cl, c), entries in self._buffer.items():
            if cl != client:
                continue
            for feat, did in entries:
                xs.append(feat)
                ys.append(c)
                ids.append(did)
        if not xs:
            return None, None, frozenset()
        return np.array(xs), np.array(ys), frozenset(ids)

    def _local_train(self, W0: np.ndarray, X_task: np.ndarray, y_task: np.ndarray, X_buf, y_buf) -> np.ndarray:
        W = W0.copy()
        for _ in range(self.local_epochs):
            grad = balanced_gradient(W, X_task, y_task, X_buf, y_buf, self.replay_weight)
            W = W - self.lr * grad
        return W

    def fit_task(self, task_idx, X, y, ids, client_shards, rng) -> list:
        records = []
        deltas, weights = [], []

        for shard in client_shards:
            idx = np.array(shard.ids, dtype=int)
            self.update_classes_seen(y[idx])
            X_buf, y_buf, buf_ids = self._client_buffer_arrays_and_ids(shard.client)
            Wc = self._local_train(self.W, X[idx], y[idx], X_buf, y_buf)
            delta = Wc - self.W
            deltas.append(delta)
            weight = len(idx)
            weights.append(weight)

            classes_here = sorted({int(c) for c in y[idx]})
            for c in classes_here:
                class_idx = idx[y[idx] == c]
                k = min(self.buffer_size_per_class, len(class_idx))
                chosen = class_idx[rng.permutation(len(class_idx))[:k]]
                self._buffer[(shard.client, c)] = [(X[i].copy(), int(i)) for i in chosen]

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
                    meta={"local_epochs": self.local_epochs, "lr": self.lr, "replay_weight": self.replay_weight, "agg_weight": float(weight)},
                )
            )

        weights = np.array(weights, dtype=float)
        if weights.sum() > 0:
            avg_delta = sum(w * d for w, d in zip(weights, deltas)) / weights.sum()
            self.W = self.W + avg_delta
        self._round += 1
        return records

    def predict(self, X) -> np.ndarray:
        return np.argmax(self.mask_unseen_logits(X @ self.W), axis=1)
