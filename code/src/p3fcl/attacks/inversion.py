"""A2 — analytic inversion of F5 (Gram statistics). Closed-form, no shadow models, no training.
Settles H5 (RESEARCH_PLAN.md §3.4), re-specified per `notes/2026-08-25_preliminary.md`'s toy-scale
finding as an n-threshold CURVE (cosine similarity vs n, per reference quality and feature
anisotropy) rather than a single "n<=64 is safe" threshold.

**Why there is no "oracle" curve reported here.** `R - lambda*I = X^T X` determines `X` up to a
single global orthogonal transform, always, for every n (RESEARCH_PLAN.md §3.4 point 1: "No
information about the span is lost. Ever."). Solving the orthogonal Procrustes problem directly
against the *true* targets therefore recovers cosine ~1.0 for every n by construction — that would
not be a finding, it would just restate the algebra. `oracle_reconstruction_selfcheck` exists only as
a correctness self-test (used in `tests/`, not reported as a result). The scientifically live question
is whether a *realistic* adversary — no ground truth, only a public reference set of same-class,
target-disjoint samples — can find a good-enough rotation. `reference_based_recovery` is that
adversary; its cosine similarity to the true targets is the curve FIG13 reports.
"""
from __future__ import annotations

import numpy as np

from .. import rng as rng_mod
from .base import Attack, ThreatModel


def gram_to_subspace(R: np.ndarray, ridge_lambda: float, n_samples: int) -> tuple:
    """Recover the top-`n_samples` eigenvector subspace and singular values from a released Gram
    statistic `R = X^T X + lambda*I`. Returns `(top_vecs (d,n), singular_vals (n,))`."""
    G = R - ridge_lambda * np.eye(R.shape[0])
    eigvals, eigvecs = np.linalg.eigh(G)
    order = np.argsort(eigvals)[::-1][:n_samples]
    top_vals = np.clip(eigvals[order], 0, None)
    top_vecs = eigvecs[:, order]
    return top_vecs, np.sqrt(top_vals)


def canonical_recovery(top_vecs: np.ndarray, singular_vals: np.ndarray) -> np.ndarray:
    """One valid point (choosing the arbitrary rotation `U = I`) in the rotation-ambiguity orbit.
    Shape `(n, d)` — row i is a candidate recovered feature vector, in no particular correspondence
    to any specific true sample until a rotation is resolved."""
    return (top_vecs * singular_vals[None, :]).T


def orthogonal_procrustes(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Solve `min_U ||U @ A - B||_F` s.t. `U^T U = I`, for `A, B` both `(n, d)`. Returns `U` (n x n).
    Standard solution: SVD of `M = B @ A^T` (n x n) gives `M = P @ S @ Q^T`; `U* = P @ Q^T`."""
    M = B @ A.T
    P, _, Qt = np.linalg.svd(M)
    return P @ Qt


def cosine_similarity_rows(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    num = np.sum(A * B, axis=1)
    denom = np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1) + 1e-12
    return num / denom


def oracle_reconstruction_selfcheck(recovered: np.ndarray, true_targets: np.ndarray) -> np.ndarray:
    """Correctness self-test only (see module docstring) -- not a reported result. Should be ~1.0
    for every n whenever `recovered` truly spans the same subspace as `true_targets`."""
    U = orthogonal_procrustes(recovered, true_targets)
    return cosine_similarity_rows(U @ recovered, true_targets)


def reference_based_recovery(recovered: np.ndarray, reference_pool: np.ndarray, n_iters: int = 15, seed: int = 0) -> np.ndarray:
    """The realistic adversary: resolve the rotation using only a public reference pool (same class,
    disjoint from the target). Alternates nearest-neighbour assignment of (rotated) recovered rows to
    reference rows and the orthogonal Procrustes solution for the current assignment -- the
    "alternating Procrustes" approach `notes/2026-08-25_preliminary.md` flagged as plausibly
    suboptimal but a real, working baseline. Returns the rotated `recovered`, shape `(n, d)`.
    """
    n = recovered.shape[0]
    r = rng_mod.seeded("attacks.inversion.reference_based_recovery", seed)
    idx = r.choice(len(reference_pool), size=n, replace=len(reference_pool) < n)
    U = np.eye(recovered.shape[0])
    for _ in range(n_iters):
        assigned = reference_pool[idx]
        U = orthogonal_procrustes(recovered, assigned)
        aligned = U @ recovered
        ref_norms = np.linalg.norm(reference_pool, axis=1)
        aligned_norms = np.linalg.norm(aligned, axis=1, keepdims=True)
        sims = (aligned @ reference_pool.T) / (aligned_norms * ref_norms[None, :] + 1e-12)
        idx = np.argmax(sims, axis=1)
    return U @ recovered


def feature_anisotropy(X: np.ndarray) -> float:
    """Eigenvalue-spectrum decay ratio: top eigenvalue / mean eigenvalue of the (centred) feature
    covariance. 1.0 = perfectly isotropic; larger = more anisotropic (energy concentrated in few
    directions), which the toy-scale note predicts makes rotation resolution easier."""
    Xc = X - X.mean(axis=0, keepdims=True)
    cov = (Xc.T @ Xc) / max(len(X) - 1, 1)
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.clip(eigvals, 0, None)
    mean_eig = eigvals.mean() + 1e-12
    return float(eigvals.max() / mean_eig)


class GramInversionAttack(Attack):
    threat_model = ThreatModel(
        who="hbc_server", active=False, view="task", auxiliary="public_data",
        target="reconstruction", survives_secure_agg=False,
    )

    def run(self, R: np.ndarray, ridge_lambda: float, n_samples: int, reference_pool: np.ndarray,
            true_targets: np.ndarray = None) -> dict:
        top_vecs, singular_vals = gram_to_subspace(R, ridge_lambda, n_samples)
        recovered = canonical_recovery(top_vecs, singular_vals)
        rotated = reference_based_recovery(recovered, reference_pool)
        out = {"recovered": recovered, "reference_recovered": rotated}
        if true_targets is not None:
            out["reference_cosine"] = cosine_similarity_rows(rotated, true_targets)
        return out
