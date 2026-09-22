"""M2 — TARGET (ICCV'23), family F1 (model delta) + F6 (generative replay). "Expected worst case"
(04_METHODS_AND_ATTACKS.md): the fitted generator's parameters are themselves the leak
(RESEARCH_PLAN.md §2.2 — "a generative model fitted to client data... Critical, the model IS the
leak"). In this feature-space harness the generator is the cheapest thing that is still honestly a
*fitted generative model of the client's data* rather than literal stored samples: a per-class
diagonal Gaussian (mean, variance) over the frozen ViT features, released as family F6 and sampled
from in later tasks to synthesize replay data for local training. A per-class Gaussian is, not
coincidentally, close kin to F5's Gram statistic (both are second-moment sufficient statistics) —
that kinship is exactly why TARGET is the paper's "expected worst case" rather than a strawman.

Retention knob `replay_ratio` controls how much synthetic old-class data gets mixed into each
round's local training. Model-delta releases honestly carry forward every id that ever fit a
generator that gets sampled from, since the released delta is a function of that synthetic data,
which is itself a function of the real data that fit the generator.
"""
from __future__ import annotations

import numpy as np

from ..artifacts import ArtifactRecord, Family
from .base import FCLMethod, MethodSpec


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class TARGET(FCLMethod):
    spec = MethodSpec(
        name="M2_target",
        families=(Family.MODEL_DELTA, Family.GENERATIVE),
        cacheable=True,
        retention_type="individual",
        retention_knob_name="replay_ratio",
    )

    def __init__(self, config: dict):
        super().__init__(config)
        self.n_classes = int(config["n_classes"])
        self.d = int(config["feature_dim"])
        self.local_epochs = int(config.get("local_epochs", 1))
        self.lr = float(config.get("lr", 0.5))
        self.replay_ratio = float(config.get("replay_ratio", 1.0))
        self.n_synthetic_per_class = int(config.get("n_synthetic_per_class", 20))
        self.W = np.zeros((self.d, self.n_classes))
        self._generators: dict = {}  # class -> (mean, var)
        self._generator_touched: dict = {}  # class -> frozenset of ids that fit this generator
        self._round = 0

    def rounds_per_task(self) -> int:
        return 1

    def _sample_replay(self, rng):
        if not self._generators or self.replay_ratio <= 0:
            return None, None, frozenset()
        xs, ys, touched = [], [], set()
        n_per = max(1, int(round(self.n_synthetic_per_class * self.replay_ratio)))
        for c, (mean, var) in self._generators.items():
            samples = mean + rng.standard_normal((n_per, self.d)) * np.sqrt(var)
            xs.append(samples)
            ys.append(np.full(n_per, c))
            touched.update(self._generator_touched.get(c, frozenset()))
        return np.concatenate(xs), np.concatenate(ys), frozenset(touched)

    def _local_train(self, W0, X_task, y_task, X_syn, y_syn) -> np.ndarray:
        W = W0.copy()
        X_all = np.concatenate([X_task, X_syn], axis=0) if X_syn is not None else X_task
        y_all = np.concatenate([y_task, y_syn], axis=0) if y_syn is not None else y_task
        n = len(y_all)
        Y = np.zeros((n, self.n_classes))
        Y[np.arange(n), y_all] = 1.0
        for _ in range(self.local_epochs):
            grad = X_all.T @ (_softmax(X_all @ W) - Y) / n
            W = W - self.lr * grad
        return W

    def fit_task(self, task_idx, X, y, ids, client_shards, rng) -> list:
        records = []
        deltas, weights = [], []
        X_syn, y_syn, syn_touched = self._sample_replay(rng)

        for shard in client_shards:
            idx = np.array(shard.ids, dtype=int)
            Wc = self._local_train(self.W, X[idx], y[idx], X_syn, y_syn)
            delta = Wc - self.W
            deltas.append(delta)
            weights.append(len(idx))
            touched_delta = frozenset(int(i) for i in idx) | syn_touched
            records.append(
                ArtifactRecord(
                    round=self._round, task=task_idx, client=shard.client, family=Family.MODEL_DELTA,
                    payload=delta, touched=touched_delta, n_touched=len(touched_delta),
                    passes_over_data=self.local_epochs,
                    meta={"replay_ratio": self.replay_ratio},
                )
            )

            classes_here = sorted({int(c) for c in y[idx]})
            gen_payload: dict = {}
            gen_touched: set = set()
            for c in classes_here:
                class_idx = idx[y[idx] == c]
                feats = X[class_idx]
                mean = feats.mean(axis=0)
                var = feats.var(axis=0) + 1e-6
                self._generators[c] = (mean, var)
                self._generator_touched[c] = frozenset(int(i) for i in class_idx)
                gen_payload[f"mean_{c}"] = mean
                gen_payload[f"var_{c}"] = var
                gen_touched.update(int(i) for i in class_idx)
            records.append(
                ArtifactRecord(
                    round=self._round, task=task_idx, client=shard.client, family=Family.GENERATIVE,
                    payload=gen_payload, touched=frozenset(gen_touched), n_touched=len(gen_touched),
                    passes_over_data=1, meta={"n_synthetic_per_class": self.n_synthetic_per_class},
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
