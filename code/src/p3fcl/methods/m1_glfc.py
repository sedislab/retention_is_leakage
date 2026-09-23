"""M1 — GLFC (CVPR'22), the classic FCIL reference. Families F1 (model delta), F7 (per-class counts,
for GLFC's class-imbalance "compensation"). The original paper fine-tunes the whole backbone; in this
harness it is restricted to the frozen-feature head (`cacheable=True` reflects the "partial (feature
space)" entry in `04_METHODS_AND_ATTACKS.md` — the reimplementation is feature-space-only, not the
original's).

Retention mechanism: local training combines cross-entropy on task data + a per-client, per-class
exemplar buffer with a temperature-scaled knowledge-distillation term, restricted to classes already
seen before this task, pulling the new head's old-class predictions toward the *previous round's*
broadcast head's old-class predictions.

FX4b/c fixes (08_FIX_PLAN.md R2, R3), replacing the pre-fix version:
- **No F8 (EXEMPLAR) ledger record.** The exemplar buffer is private local client state, read only by
  that same client's own next local-training step — nothing about it is ever transmitted, so per
  Lemma 8 (08_FIX_PLAN.md §4b) it is not a release and does not belong in the ledger. `spec.families`
  no longer lists EXEMPLAR.
- **Per-client buffer** (`self._buffer[(client, cls)]`), never shared across clients — R2's identified
  cause of the one-task lag was one *global* buffer silently mixing clients' exemplars together.
- **KD restricted to classes already seen before this task** — R2's other identified cause: KD over
  *all* classes (including brand-new ones, whose `W_old` columns are still all-zero) pulls the new
  head's predictions on its own new classes toward the "nothing here" old prediction, actively
  suppressing them. The distillation softmax is now computed over the old-class columns only, on both
  sides, so its gradient is exactly zero on new-class columns.
- **`touched` no longer accumulates `_all_touched_ever`.** Reading a broadcast model (`W_old`) is
  post-processing, not a fresh read of raw data (Lemma 8) — `touched` for the MODEL_DELTA record is
  exactly this round's shard plus whatever buffer ids fed this round's local training.
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
        families=(Family.MODEL_DELTA, Family.COUNTS),
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
        self._buffer: dict = {}  # (client, class) -> list[(feature, id)]
        self._round = 0

    def rounds_per_task(self) -> int:
        return 1

    def _client_buffer_arrays(self, client: int):
        xs, ys, ids = [], [], []
        for (cl, c), entries in self._buffer.items():
            if cl != client:
                continue
            for feat, did in entries:
                xs.append(feat)
                ys.append(c)
                ids.append(did)
        if not xs:
            return None, None, []
        return np.array(xs), np.array(ys), ids

    def _local_train(self, W0, X_task, y_task, X_buf, y_buf, W_old, old_classes: list) -> np.ndarray:
        W = W0.copy()
        X_all = np.concatenate([X_task, X_buf], axis=0) if X_buf is not None else X_task
        y_all = np.concatenate([y_task, y_buf], axis=0) if y_buf is not None else y_task
        n = len(y_all)
        Y = np.zeros((n, self.n_classes))
        Y[np.arange(n), y_all] = 1.0
        old_cols = np.array(old_classes, dtype=int) if old_classes else None
        for _ in range(self.local_epochs):
            logits = X_all @ W
            grad = X_all.T @ (_softmax(logits) - Y) / n
            if W_old is not None and self.distillation_weight > 0 and old_cols is not None and len(old_cols) > 0:
                # KD restricted to old-class columns on BOTH sides: a sub-softmax over just those
                # columns, not the full-class softmax sliced afterward -- so its gradient touches
                # only the old-class columns of W (exactly zero on new-class columns).
                soft_old = _softmax((X_all @ W_old)[:, old_cols], self.temperature)
                soft_new = _softmax(logits[:, old_cols], self.temperature)
                grad_kd_old = X_all.T @ (soft_new - soft_old) / n
                grad[:, old_cols] = grad[:, old_cols] + self.distillation_weight * grad_kd_old
            W = W - self.lr * grad
        return W

    def fit_task(self, task_idx, X, y, ids, client_shards, rng) -> list:
        records = []
        deltas, weights = [], []
        old_classes = sorted(self._classes_seen)  # snapshot BEFORE this task's classes are added
        W_old = self.W.copy() if self._round > 0 else None

        for shard in client_shards:
            idx = np.array(shard.ids, dtype=int)
            self.update_classes_seen(y[idx])
            X_buf, y_buf, buf_ids = self._client_buffer_arrays(shard.client)
            Wc = self._local_train(self.W, X[idx], y[idx], X_buf, y_buf, W_old, old_classes)
            delta = Wc - self.W
            deltas.append(delta)
            weight = len(idx)
            weights.append(weight)

            touched_delta = frozenset(int(i) for i in idx) | frozenset(buf_ids)
            records.append(
                ArtifactRecord(
                    round=self._round, task=task_idx, client=shard.client, family=Family.MODEL_DELTA,
                    payload=delta, touched=touched_delta, n_touched=len(touched_delta),
                    passes_over_data=self.local_epochs,
                    meta={
                        "distillation_weight": self.distillation_weight, "temperature": self.temperature,
                        "agg_weight": float(weight),
                    },
                )
            )

            classes_here = sorted({int(c) for c in y[idx]})
            counts = np.zeros(self.n_classes, dtype=int)
            for c in classes_here:
                class_idx = idx[y[idx] == c]
                counts[c] = len(class_idx)
            records.append(
                ArtifactRecord(
                    round=self._round, task=task_idx, client=shard.client, family=Family.COUNTS,
                    payload={"counts": counts}, touched=frozenset(int(i) for i in idx),
                    n_touched=len(idx), passes_over_data=1, meta={},
                )
            )

            # Fill this client's per-class buffer from its OWN current shard only, at the end of its
            # task (FX4c) -- deterministic given the simulation's own seeded `rng` stream.
            for c in classes_here:
                class_idx = idx[y[idx] == c]
                k = min(self.exemplar_budget, len(class_idx))
                chosen = class_idx[rng.permutation(len(class_idx))[:k]]
                self._buffer[(shard.client, c)] = [(X[i].copy(), int(i)) for i in chosen]

        weights = np.array(weights, dtype=float)
        if weights.sum() > 0:
            avg_delta = sum(w * d for w, d in zip(weights, deltas)) / weights.sum()
            self.W = self.W + avg_delta
        self._round += 1
        return records

    def predict(self, X) -> np.ndarray:
        return np.argmax(self.mask_unseen_logits(X @ self.W), axis=1)
