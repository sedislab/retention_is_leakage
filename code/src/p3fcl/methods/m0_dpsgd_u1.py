"""R5 baseline `m0_dpsgd_u1` (rebuttal/R5_PROTOCOL.md): M0 (FedAvg, one round per task, full-batch
local GD) with example-level DP-SGD inside every local step.

Each local step clips every per-example gradient of the linear head to Frobenius norm `clip_Cg`, sums
them, adds N(0, (z * clip_Cg)^2 I), and divides by the client's number of examples (treated as
public, as in M0's FedAvg weighting). `z` comes from the repo's RDP accountant: `local_epochs` steps
at sampling rate 1 give (eps0, delta) per round (`noise_multiplier_for`). Each example is read only in
its own task's round. So under U1 the ledger count per example is 1, and the lifelong guarantee equals
the per-round one. The client's delta is already DP, so per-client records are released exactly as
M0 releases them, with the same `touched`.

With `clip_Cg = inf` and `z = 0` the local update is M0's, operation for operation.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

from .. import rng as rng_mod
from ..dp.accountant import eps_gaussian_composed
from .base import MethodSpec
from .m0_fedavg import FedAvgSequential, _softmax


def noise_multiplier_for(eps0: float, steps: int, delta: float) -> float:
    """Smallest z with `eps_gaussian_composed(z, steps, delta) <= eps0` (bisection on a monotone
    function). 0 for eps0 = inf."""
    if np.isinf(eps0):
        return 0.0
    f = lambda z: eps_gaussian_composed(z, steps, delta) - eps0  # noqa: E731
    lo, hi = 1e-3, 1.0
    while f(hi) > 0:
        hi *= 2.0
    return float(brentq(f, lo, hi, xtol=1e-12))


def per_example_grad_norms(X: np.ndarray, G: np.ndarray) -> np.ndarray:
    """||x_i (p_i - y_i)^T||_F = ||x_i|| * ||p_i - y_i|| for the linear softmax head."""
    return np.linalg.norm(X, axis=1) * np.linalg.norm(G, axis=1)


class DPSGDFedAvgU1(FedAvgSequential):
    spec = MethodSpec(
        name="m0_dpsgd_u1",
        families=FedAvgSequential.spec.families,
        cacheable=True,
        retention_type="individual",
        retention_knob_name="local_epochs",
        display_name="FedAvg + DP-SGD (example level, U1)",
    )

    def __init__(self, config: dict):
        super().__init__(config)
        self.clip_Cg = float(config.get("clip_Cg", np.inf))
        self.z = float(config.get("noise_multiplier", 0.0))
        if self.z > 0 and not np.isfinite(self.clip_Cg):
            raise ValueError("m0_dpsgd_u1: noise needs a finite clip_Cg")
        self._noise_rng = rng_mod.seeded("m0_dpsgd_u1::noise", int(config.get("noise_seed", 0)))
        self.frac_clipped_steps: list = []

    def _local_train(self, W0: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        W = W0.copy()
        n = len(y)
        Y = np.zeros((n, self.n_classes))
        Y[np.arange(n), y] = 1.0
        for _ in range(self.local_epochs):
            P = _softmax(X @ W)
            Gm = P - Y
            if np.isfinite(self.clip_Cg):
                norms = per_example_grad_norms(X, Gm)
                Gm = Gm * np.minimum(1.0, self.clip_Cg / np.maximum(norms, 1e-12))[:, None]
                self.frac_clipped_steps.append(float(np.mean(norms > self.clip_Cg)))
            g = X.T @ Gm
            if self.z > 0:
                g = g + self._noise_rng.standard_normal(g.shape) * (self.z * self.clip_Cg)
            grad = g / n
            W = W - self.lr * grad
        return W

    def fit_task(self, task_idx, X, y, ids, client_shards, rng) -> list:
        records = super().fit_task(task_idx, X, y, ids, client_shards, rng)
        for rec in records:
            rec.meta.update({"clip_Cg": self.clip_Cg, "noise_multiplier": self.z, "unit": "U1"})
        return records
