"""M9 — the contractive DP analytic learner (family F5, `08_FIX_PLAN.md` §8, Algorithm 3 in the
paper). Same closed-form-ridge family as M8, with three changes that make it the paper's actual
proposal rather than a restatement of M8:

1. **Dimensionality reduction onto a public basis.** Features are projected with a PCA basis `P`
   fit on the `ref` split (disjoint from train/test, so this costs no privacy) before anything
   privacy-sensitive is computed -- `P` is passed in via `config["pca_basis"]`, never fit by this
   class, so `sim.run` stays feature-generic and the fitting stays auditable as its own step
   (`run_m9_hparam_selection.py`).
2. **DP noise on the released aggregate.** Exactly one record per task, `client=-1` (a genuine
   post-secure-aggregation release: the server only ever sees the noised SUM over clients, never a
   per-client term, so `touched` is honestly every id from every client that task -- Lemma 8, no
   under-reporting). Unit `U1` (per-example) has a closed-form sensitivity from the `B`-norm cap
   alone; unit `U2` (per-client) additionally joint-clips each client's own `(G_c, H_c)` pair to
   Frobenius norm `clip_C` before summing (`08_FIX_PLAN.md`: "U2 only. Clip the pair...").
3. **A contractive (not accumulating) running state.** `R <- gamma*R + G_tilde`, `Q <- gamma*Q +
   H_tilde`. `gamma = 1.0` recovers M8's unbounded running sum exactly (mod the ridge-lambda
   convention difference noted below); `gamma < 1` bounds any one release's influence on the
   long-run state, which is the whole C3/C4 point of this method existing.

`Pi_+` (clipping R's negative eigenvalues to 0 before ridge) guards against `gamma < 1` or DP noise
making the running Gram matrix non-PSD, which a plain `+lambda*I` cannot fix on its own once
eigenvalues go meaningfully negative.
"""
from __future__ import annotations

import numpy as np

from .. import rng as rng_mod
from ..artifacts import ArtifactRecord, Family
from ..dp.mechanisms import analytic_gaussian_sigma, sym_gaussian_noise
from .base import FCLMethod, MethodSpec


def project_and_cap(X: np.ndarray, P: np.ndarray, B: float = 1.0) -> np.ndarray:
    """PCA-project `X` onto `P` (`d x p`), then scale down (never up) any row whose L2 norm exceeds
    `B` -- "L2-normalise so that ||x_bar|| <= B", a cap, not a hard unit-normalisation."""
    Xp = X @ P
    norms = np.linalg.norm(Xp, axis=1, keepdims=True)
    scale = np.minimum(1.0, B / np.maximum(norms, 1e-12))
    return Xp * scale


def _one_hot(y, n_classes: int) -> np.ndarray:
    Y = np.zeros((len(y), n_classes))
    Y[np.arange(len(y)), y] = 1.0
    return Y


def client_stats(Xc_proj: np.ndarray, yc, n_classes: int) -> tuple:
    """`(G_c, H_c)` for one client's already-projected-and-capped shard: `G_c = X^T X`,
    `H_c = X^T Y` (Y one-hot). No ridge term here -- M9 adds ridge exactly once, at `predict()` time,
    unlike M8's per-client `+lambda*I` (see the class docstring's equivalence-test note)."""
    return Xc_proj.T @ Xc_proj, Xc_proj.T @ _one_hot(yc, n_classes)


def clip_pair(G: np.ndarray, H: np.ndarray, clip_C: float) -> tuple:
    """Joint Frobenius-norm clip of the stacked `(vech G, vec H)` pair to `clip_C` (U2's per-client
    clip, algorithm step 3) -- scales both matrices by the same factor so the clip is on the pair, not
    on each half independently."""
    norm = float(np.sqrt(np.sum(G**2) + np.sum(H**2)))
    scale = min(1.0, clip_C / max(norm, 1e-12))
    return G * scale, H * scale


def psd_clip(M: np.ndarray) -> np.ndarray:
    """Pi_+: clip `M`'s negative eigenvalues to 0 (symmetric `M` only). A no-op on an already-PSD
    matrix up to floating-point noise -- needed once `gamma < 1` or DP noise can push the running
    state off the PSD cone."""
    w, V = np.linalg.eigh(M)
    w = np.clip(w, 0.0, None)
    return (V * w) @ V.T


class ContractiveDPAnalytic(FCLMethod):
    spec = MethodSpec(
        name="m9_contractive",
        families=(Family.GRAM,),
        cacheable=True,
        retention_type="individual",
        retention_knob_name="gamma",
        display_name="Contractive DP analytic (ours)",
    )

    def __init__(self, config: dict):
        super().__init__(config)
        self.n_classes = int(config["n_classes"])
        self.P = np.asarray(config["pca_basis"], dtype=float)  # (feature_dim, p), fit on `ref`
        self.p = self.P.shape[1]
        self.gamma = float(config.get("gamma", 1.0))
        self.ridge_lambda = float(config.get("ridge_lambda", 1.0))
        self.unit = config.get("unit", "U1")
        if self.unit not in ("U1", "U2"):
            raise ValueError(f"m9_contractive: unit must be 'U1' or 'U2', got {self.unit!r}")
        self.eps = config.get("eps", float("inf"))
        self.delta = float(config.get("delta", 1e-5))
        self.clip_C = float(config.get("clip_C", 1.0))
        self.B = float(config.get("B", 1.0))
        # Noise seed is intentionally decoupled from `sim.run`'s data seed (which only ever reaches
        # this class via `fit_task`'s `rng` argument, which this class never touches) -- the same
        # (data, noise) seed pair must reproduce byte-identical output, and changing one without the
        # other must change only the corresponding half of the computation (FX5 test 5).
        noise_seed = int(config.get("noise_seed", 0))
        self._noise_rng = rng_mod.seeded("m9_contractive::noise", noise_seed)

        self.R = np.zeros((self.p, self.p))
        self.Q = np.zeros((self.p, self.n_classes))
        self._round = 0

    def rounds_per_task(self) -> int:
        return 1

    def _sigma(self) -> float:
        if self.eps is None or np.isinf(self.eps):
            return 0.0
        sensitivity = self.clip_C if self.unit == "U2" else float(np.sqrt(self.B**4 + self.B**2))
        return analytic_gaussian_sigma(self.eps, self.delta, sensitivity)

    def fit_task(self, task_idx, X, y, ids, client_shards, rng) -> list:
        G_sum = np.zeros((self.p, self.p))
        H_sum = np.zeros((self.p, self.n_classes))
        all_ids: list = []
        for shard in client_shards:
            idx = np.array(shard.ids, dtype=int)
            Xc, yc = X[idx], y[idx]
            self.update_classes_seen(yc)
            Xc_proj = project_and_cap(Xc, self.P, self.B)
            Gc, Hc = client_stats(Xc_proj, yc, self.n_classes)
            if self.unit == "U2":
                Gc, Hc = clip_pair(Gc, Hc, self.clip_C)
            G_sum += Gc
            H_sum += Hc
            all_ids.extend(int(i) for i in idx)

        sigma = self._sigma()
        if sigma > 0:
            G_tilde = G_sum + sym_gaussian_noise(self.p, sigma, self._noise_rng)
            H_tilde = H_sum + self._noise_rng.standard_normal(H_sum.shape) * sigma
        else:
            G_tilde, H_tilde = G_sum, H_sum

        self.R = self.gamma * self.R + G_tilde
        self.Q = self.gamma * self.Q + H_tilde

        record = ArtifactRecord(
            round=self._round,
            task=task_idx,
            client=-1,  # a genuine post-secure-aggregation release: no per-client identity survives
            family=Family.GRAM,
            payload={"R": G_tilde, "Q": H_tilde},
            touched=frozenset(all_ids),
            n_touched=len(all_ids),
            passes_over_data=1,
            meta={"gamma": self.gamma, "sigma": sigma, "unit": self.unit, "eps": self.eps, "clip_C": self.clip_C},
        )
        self._round += 1
        return [record]

    def predict(self, X) -> np.ndarray:
        Xp = project_and_cap(X, self.P, self.B)
        R_pos = psd_clip(self.R)
        W = np.linalg.solve(R_pos + self.ridge_lambda * np.eye(self.p), self.Q)
        return np.argmax(self.mask_unseen_logits(Xp @ W), axis=1)
