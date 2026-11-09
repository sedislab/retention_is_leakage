"""R5 baseline `m0_dpfedavg_u2` (rebuttal/R5_PROTOCOL.md, amendment 1): client-level DP-FedAvg with
McMahan et al. (ICLR 2018)'s fixed-denominator weighted mean, on M0.

Each client runs M0's local training unchanged and clips its update Δ_c to Frobenius norm `clip_S`.
It gets weight w_c = min(n_c / w_cap, 1), where the cap `w_cap` is fixed in advance from public data.
The server adds N(0, σ² I) to Σ_c w_c·clip(Δ_c) and divides by the fixed denominator
W = Σ_c w_c of the task's base-stream clients, which is treated as a public constant (as in McMahan
et al.). Each weighted clipped update has norm ≤ S (w_c ≤ 1), so the sum has add/remove-one-client
sensitivity S, and σ = `analytic_gaussian_sigma(eps0, delta, clip_S)` (the routine M9 uses).

The release is the noisy average: one `client=-1` record per round, like M9, since the noise protects
only the aggregate. `touched` is every id of the task. Under U2 each (client, task) shard is read in
one record, so the lifelong guarantee equals the per-round one.

Other `aggregation` modes:
- `"weighted"`: M0's size-weighted mean, for the no-noise, no-clip test only (equal to M0 operation
  for operation).
- `"uniform"`: divide by the number of clients. This was the plan's first version. It failed the R5
  correctness gate (3/9 cells > 0.01) and is kept only to reproduce that record.
"""
from __future__ import annotations

import numpy as np

from .. import rng as rng_mod
from ..artifacts import ArtifactRecord, Family
from ..dp.mechanisms import analytic_gaussian_sigma
from .base import MethodSpec
from .m0_fedavg import FedAvgSequential

AGGREGATIONS = ("fixed_weighted", "weighted", "uniform")


def mcmahan_weights(sizes, w_cap: float) -> np.ndarray:
    """w_c = min(n_c / w_cap, 1)."""
    return np.minimum(np.asarray(sizes, dtype=float) / float(w_cap), 1.0)


class DPFedAvgU2(FedAvgSequential):
    spec = MethodSpec(
        name="m0_dpfedavg_u2",
        families=(Family.MODEL_DELTA,),
        cacheable=True,
        retention_type="individual",
        retention_knob_name="local_epochs",
        display_name="DP-FedAvg (client level, U2)",
    )

    def __init__(self, config: dict):
        super().__init__(config)
        self.clip_S = float(config.get("clip_S", np.inf))
        self.eps0 = float(config.get("eps", np.inf))
        self.delta = float(config.get("delta", 1e-5))
        self.aggregation = config.get("aggregation", "fixed_weighted")
        if self.aggregation not in AGGREGATIONS:
            raise ValueError(f"m0_dpfedavg_u2: aggregation must be one of {AGGREGATIONS}")
        self.w_cap = float(config.get("w_cap", np.nan))
        if self.aggregation == "fixed_weighted" and not (self.w_cap > 0):
            raise ValueError("m0_dpfedavg_u2: fixed_weighted needs a positive public w_cap")
        self.sigma = 0.0 if np.isinf(self.eps0) else analytic_gaussian_sigma(self.eps0, self.delta, self.clip_S)
        if self.aggregation == "weighted" and (self.sigma > 0 or np.isfinite(self.clip_S)):
            raise ValueError("m0_dpfedavg_u2: 'weighted' is only for the no-noise, no-clip equivalence check")
        self._noise_rng = rng_mod.seeded("m0_dpfedavg_u2::noise", int(config.get("noise_seed", 0)))
        self.frac_clipped_rounds: list = []
        self.frac_capped_rounds: list = []

    def fit_task(self, task_idx, X, y, ids, client_shards, rng) -> list:
        deltas, sizes, all_ids, n_clip = [], [], [], 0
        for shard in client_shards:
            idx = np.array(shard.ids, dtype=int)
            self.update_classes_seen(y[idx])
            delta = self._local_train(self.W, X[idx], y[idx]) - self.W
            if np.isfinite(self.clip_S):
                norm = float(np.linalg.norm(delta))
                n_clip += int(norm > self.clip_S)
                delta = delta * min(1.0, self.clip_S / max(norm, 1e-12))
            deltas.append(delta)
            sizes.append(len(idx))
            all_ids.extend(int(i) for i in idx)
        self.frac_clipped_rounds.append(n_clip / len(deltas) if deltas else 0.0)
        if not deltas:
            self._round += 1
            return []
        meta = {}
        if self.aggregation == "weighted":
            weights = np.array(sizes, dtype=float)
            avg = sum(w * d for w, d in zip(weights, deltas)) / float(weights.sum())
        elif self.aggregation == "uniform":
            total = sum(deltas)
            if self.sigma > 0:
                total = total + self._noise_rng.standard_normal(total.shape) * self.sigma
            avg = total / len(deltas)
        else:
            w = mcmahan_weights(sizes, self.w_cap)
            W_fixed = float(w.sum())  # base-stream denominator, a public constant
            total = sum(wc * d for wc, d in zip(w, deltas))
            if self.sigma > 0:
                total = total + self._noise_rng.standard_normal(total.shape) * self.sigma
            avg = total / W_fixed
            self.frac_capped_rounds.append(float(np.mean(np.asarray(sizes) > self.w_cap)))
            meta = {"w_cap": self.w_cap, "W_fixed": W_fixed}
        self.W = self.W + avg
        record = ArtifactRecord(
            round=self._round, task=task_idx, client=-1, family=Family.MODEL_DELTA, payload=avg,
            touched=frozenset(all_ids), n_touched=len(all_ids), passes_over_data=self.local_epochs,
            meta={"local_epochs": self.local_epochs, "lr": self.lr, "clip_S": self.clip_S, "sigma": self.sigma,
                  "eps": self.eps0, "unit": "U2", "aggregation": self.aggregation,
                  "aggregated_over_clients": [s.client for s in client_shards], "agg_weight": 1.0, **meta},
        )
        self._round += 1
        return [record]
