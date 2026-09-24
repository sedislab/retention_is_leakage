"""M4's prototype half (PILoRA, ECCV'24) — releases per-client, per-class, per-task prototypes
(family F2) and their counts (family F7). No LoRA component (that needs backprop through the ViT and
is P2/GPU work); this is the cacheable CPU half that exercises F2+F7 end to end and is one of the two
required methods for the P5 semantic-vs-individual dose-response split
(`retention_type = "semantic"`; counts can be switched off via `release_counts=False`).

FX9: clients release their local means and counts. The server first count-weights
all clients' prototypes within a round, then blends that aggregate with its prior
bank once per class. Momentum is post-processing of released statistics; it does
not read old raw examples or change the local releases' touched sets.
"""
from __future__ import annotations

import numpy as np

from ..artifacts import ArtifactRecord, Family
from .base import FCLMethod, MethodSpec


class PrototypeFCL(FCLMethod):
    spec = MethodSpec(
        name="M4_prototype_half",
        families=(Family.PROTOTYPE, Family.COUNTS),
        cacheable=True,
        retention_type="semantic",
        retention_knob_name="prototype_momentum",
    )

    def __init__(self, config: dict):
        super().__init__(config)
        self.momentum = float(config.get("prototype_momentum", 0.0))
        self.n_classes = int(config["n_classes"])
        self.d = int(config["feature_dim"])
        self.release_counts = bool(config.get("release_counts", True))
        self._prototypes = np.full((self.n_classes, self.d), np.nan)
        self._round = 0

    def rounds_per_task(self) -> int:
        return 1

    def fit_task(self, task_idx, X, y, ids, client_shards, rng) -> list:
        records = []
        sums = np.zeros_like(self._prototypes)
        weights = np.zeros(self.n_classes)
        for shard in client_shards:
            idx = np.array(shard.ids, dtype=int)
            Xc, yc = X[idx], y[idx]
            classes_here = sorted({int(c) for c in yc})
            proto_payload: dict = {}
            counts = np.zeros(self.n_classes, dtype=int)
            client_touched: set = set()
            for c in classes_here:
                mask = yc == c
                mean_c = Xc[mask].mean(axis=0)
                new_ids = frozenset(int(i) for i in idx[mask])
                proto_payload[str(c)] = mean_c.copy()
                counts[c] = int(mask.sum())
                sums[c] += counts[c] * mean_c
                weights[c] += counts[c]
                client_touched |= new_ids
            records.append(
                ArtifactRecord(
                    round=self._round,
                    task=task_idx,
                    client=shard.client,
                    family=Family.PROTOTYPE,
                    payload=proto_payload,
                    touched=frozenset(client_touched),
                    n_touched=len(client_touched),
                    passes_over_data=1,
                    meta={"momentum": 0.0, "server_momentum": self.momentum},
                )
            )
            if self.release_counts:
                records.append(
                    ArtifactRecord(
                        round=self._round,
                        task=task_idx,
                        client=shard.client,
                        family=Family.COUNTS,
                        payload={"counts": counts},
                        touched=frozenset(client_touched),
                        n_touched=len(client_touched),
                        passes_over_data=1,
                        meta={},
                    )
                )
        for c in np.flatnonzero(weights):
            aggregate = sums[c] / weights[c]
            if self.momentum > 0 and np.isfinite(self._prototypes[c]).all():
                self._prototypes[c] = self.momentum * self._prototypes[c] + (1-self.momentum) * aggregate
            else:
                self._prototypes[c] = aggregate
        self._round += 1
        return records

    def predict(self, X) -> np.ndarray:
        proto = np.nan_to_num(self._prototypes, nan=1e12)
        dists = np.linalg.norm(X[:, None, :] - proto[None, :, :], axis=2)
        return np.argmin(dists, axis=1)
