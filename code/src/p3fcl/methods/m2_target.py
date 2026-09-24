"""M2 — Gaussian feature replay (TARGET-style), family F1 (model delta) + F6 (generative replay).
`display_name` states plainly what this is: a per-class diagonal Gaussian fitted on frozen ViT
features, **not** TARGET (ICCV'23)'s actual data-free generator — `spec.name` stays `m2_target` so
every path/config built around that string is unaffected (FX4e, 08_FIX_PLAN.md).

Each client fits its own per-class (mean, diagonal variance) on its own current shard and releases
them once, at task k, as an F6 record whose `touched` is exactly that shard — a genuine, one-time
read of raw data, honestly reported. The server then aggregates every client's per-class statistics
(count-weighted) into the broadcast generator used for replay sampling in later rounds. Sampling from
that already-released, broadcast Gaussian is post-processing (Lemma 8, 08_FIX_PLAN.md §4b) and adds
nothing further to any record's `touched` — unlike the pre-fix version, which re-added the *original*
generator-fitting ids to every later round's model-delta `touched` for as long as a generator kept
getting sampled from.

FX9 local CE gives current and replay sets separate means, with replay_weight=1.
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


class TARGET(FCLMethod):
    spec = MethodSpec(
        name="M2_target",
        families=(Family.MODEL_DELTA, Family.GENERATIVE),
        cacheable=True,
        # FX8 (08_FIX_PLAN.md §10) fix: was "individual", found while wiring FIG04 v2's
        # semantic-vs-individual split (`base.py`: "FIG04's split") -- this method's whole retention
        # mechanism is a per-CLASS distributional summary (mean + diagonal variance), the same kind of
        # class-level aggregate as M4's prototype (also "semantic"), not a per-example or raw-exemplar
        # mechanism like M5/M8/M9's ("individual"). `08_FIX_PLAN.md` §10 itself already calls this
        # "M2 (semantic retention)" explicitly when specifying the FX8 dose-response sweep -- the old
        # "individual" tag predates that framing and was never exercised by any FIG04 pipeline before
        # (the pre-fix pilot's FIG04 only ever used M4/M5, never M2), so this was a latent
        # misclassification, not a locked decision being revisited.
        retention_type="semantic",
        retention_knob_name="replay_ratio",
        display_name="Gaussian feature replay (TARGET-style)",
    )

    def __init__(self, config: dict):
        super().__init__(config)
        self.n_classes = int(config["n_classes"])
        self.d = int(config["feature_dim"])
        self.local_epochs = int(config.get("local_epochs", 1))
        self.lr = float(config.get("lr", 0.5))
        self.replay_weight = float(config.get("replay_weight", 1.0))
        self.replay_ratio = float(config.get("replay_ratio", 1.0))
        self.n_synthetic_per_class = int(config.get("n_synthetic_per_class", 20))
        self.W = np.zeros((self.d, self.n_classes))
        self._generators: dict = {}  # class -> (mean, var) -- the server-broadcast, aggregated state
        self._round = 0

    def rounds_per_task(self) -> int:
        return 1

    def _sample_replay(self, rng):
        """Sampling from the already-broadcast generator is post-processing -- no `touched` result,
        because nothing here reads raw data (Lemma 8)."""
        if not self._generators or self.replay_ratio <= 0:
            return None, None
        xs, ys = [], []
        n_per = max(1, int(round(self.n_synthetic_per_class * self.replay_ratio)))
        for c, (mean, var) in self._generators.items():
            samples = mean + rng.standard_normal((n_per, self.d)) * np.sqrt(var)
            xs.append(samples)
            ys.append(np.full(n_per, c))
        return np.concatenate(xs), np.concatenate(ys)

    def _local_train(self, W0, X_task, y_task, X_syn, y_syn) -> np.ndarray:
        W = W0.copy()
        for _ in range(self.local_epochs):
            grad = balanced_gradient(W, X_task, y_task, X_syn, y_syn, self.replay_weight)
            W = W - self.lr * grad
        return W

    def fit_task(self, task_idx, X, y, ids, client_shards, rng) -> list:
        records = []
        deltas, weights = [], []
        X_syn, y_syn = self._sample_replay(rng)

        # (class -> list of (mean, var, count)) from every client this task, combined into the new
        # server-broadcast generator AFTER every client has released its own honest per-shard stats.
        per_class_client_stats: dict = {}

        for shard in client_shards:
            idx = np.array(shard.ids, dtype=int)
            self.update_classes_seen(y[idx])
            Wc = self._local_train(self.W, X[idx], y[idx], X_syn, y_syn)
            delta = Wc - self.W
            deltas.append(delta)
            weight = len(idx)
            weights.append(weight)
            touched_delta = frozenset(int(i) for i in idx)
            records.append(
                ArtifactRecord(
                    round=self._round, task=task_idx, client=shard.client, family=Family.MODEL_DELTA,
                    payload=delta, touched=touched_delta, n_touched=len(touched_delta),
                    passes_over_data=self.local_epochs,
                    meta={"replay_ratio": self.replay_ratio, "replay_weight": self.replay_weight, "agg_weight": float(weight)},
                )
            )

            classes_here = sorted({int(c) for c in y[idx]})
            gen_payload: dict = {}
            for c in classes_here:
                class_idx = idx[y[idx] == c]
                feats = X[class_idx]
                mean = feats.mean(axis=0)
                var = feats.var(axis=0) + 1e-6
                per_class_client_stats.setdefault(c, []).append((mean, var, len(class_idx)))
                gen_payload[f"mean_{c}"] = mean
                gen_payload[f"var_{c}"] = var
            records.append(
                ArtifactRecord(
                    round=self._round, task=task_idx, client=shard.client, family=Family.GENERATIVE,
                    payload=gen_payload, touched=frozenset(int(i) for i in idx),
                    n_touched=len(idx),
                    passes_over_data=1, meta={"n_synthetic_per_class": self.n_synthetic_per_class},
                )
            )

        weights = np.array(weights, dtype=float)
        if weights.sum() > 0:
            avg_delta = sum(w * d for w, d in zip(weights, deltas)) / weights.sum()
            self.W = self.W + avg_delta

        # Server aggregation: count-weighted combination of every client's per-class statistics into
        # the broadcast generator used for replay sampling from now on.
        for c, stats in per_class_client_stats.items():
            counts = np.array([n for _, _, n in stats], dtype=float)
            means = np.stack([m for m, _, _ in stats])
            varss = np.stack([v for _, v, _ in stats])
            total = counts.sum()
            agg_mean = (counts[:, None] * means).sum(axis=0) / total
            agg_second_moment = (counts[:, None] * (varss + means**2)).sum(axis=0) / total
            agg_var = np.maximum(agg_second_moment - agg_mean**2, 1e-6)
            self._generators[c] = (agg_mean, agg_var)

        self._round += 1
        return records

    def predict(self, X) -> np.ndarray:
        return np.argmax(self.mask_unseen_logits(X @ self.W), axis=1)
