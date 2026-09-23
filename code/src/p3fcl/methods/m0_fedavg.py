"""M0 — FedAvg + sequential fine-tuning (family F1): the retention *lower bound* and the attack
calibration point (04_METHODS_AND_ATTACKS.md §1). No replay, no distillation, no regulariser — each
task's clients locally fine-tune a shared linear softmax head by plain SGD for `local_epochs` steps,
and the server FedAvg-aggregates the deltas. Nothing carries old-task data forward except whatever
survives in the shared weight itself, which is exactly the "does forgetting protect privacy for
free" baseline the whole project is contrasting every other method against.
"""
from __future__ import annotations

import numpy as np

from ..artifacts import ArtifactRecord, Family
from .base import FCLMethod, MethodSpec


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class FedAvgSequential(FCLMethod):
    spec = MethodSpec(
        name="M0_fedavg_sequential",
        families=(Family.MODEL_DELTA,),
        cacheable=True,
        retention_type="individual",
        retention_knob_name="local_epochs",
    )

    def __init__(self, config: dict):
        super().__init__(config)
        self.n_classes = int(config["n_classes"])
        self.d = int(config["feature_dim"])
        self.local_epochs = int(config.get("local_epochs", 1))
        self.lr = float(config.get("lr", 0.5))
        self.W = np.zeros((self.d, self.n_classes))
        self._round = 0

    def rounds_per_task(self) -> int:
        return 1

    def _local_train(self, W0: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        W = W0.copy()
        n = len(y)
        Y = np.zeros((n, self.n_classes))
        Y[np.arange(n), y] = 1.0
        for _ in range(self.local_epochs):
            P = _softmax(X @ W)
            grad = X.T @ (P - Y) / n
            W = W - self.lr * grad
        return W

    def fit_task(self, task_idx, X, y, ids, client_shards, rng) -> list:
        records = []
        deltas, weights = [], []
        for shard in client_shards:
            idx = np.array(shard.ids, dtype=int)
            self.update_classes_seen(y[idx])
            Wc = self._local_train(self.W, X[idx], y[idx])
            delta = Wc - self.W
            deltas.append(delta)
            weights.append(len(idx))
        weights = np.array(weights, dtype=float)
        total_weight = float(weights.sum())
        for shard, delta, w in zip(client_shards, deltas, weights):
            idx = np.array(shard.ids, dtype=int)
            records.append(
                ArtifactRecord(
                    round=self._round,
                    task=task_idx,
                    client=shard.client,
                    family=Family.MODEL_DELTA,
                    payload=delta,
                    touched=frozenset(int(i) for i in idx),
                    n_touched=len(idx),
                    passes_over_data=self.local_epochs,
                    # FX4b (08_FIX_PLAN.md R3): the exact weight the server gave this client in this
                    # round's FedAvg -- for M0 this is exactly the raw shard size (no buffer, no
                    # distillation), identical to `n_touched`, but recorded explicitly so
                    # `_reconstruct_running_w` never has to assume `n_touched` IS the true weight for
                    # methods where the two diverge (M1/M2/M3/M5).
                    meta={"local_epochs": self.local_epochs, "lr": self.lr, "agg_weight": float(w)},
                )
            )
        if total_weight > 0:
            avg_delta = sum(w * d for w, d in zip(weights, deltas)) / total_weight
            self.W = self.W + avg_delta
        self._round += 1
        return records

    def predict(self, X) -> np.ndarray:
        return np.argmax(self.mask_unseen_logits(X @ self.W), axis=1)
