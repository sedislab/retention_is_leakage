"""Noise mechanisms and their sensitivities.

`analytic_gaussian_sigma` implements the **analytic Gaussian mechanism** (Balle & Wang, ICML 2018):
it calibrates sigma against the *exact* privacy curve of the Gaussian mechanism, valid for every
eps >= 0 — not the classical `sqrt(2 ln(1.25/delta))/eps` bound, which is only valid for eps <= 1 and
silently understates the required noise above it.

On the continual-observation factor: this module ships **binary-tree aggregation**
(`tree_aggregation_factor`, Dwork-Naor-Pitassi-Rothblum STOC'10 / Chan-Shi-Song), not the SOTA banded
matrix-factorisation mechanism (Choquette-Choo et al., ICML 2023). That upgrade is P7's job
(build/00_BUILD_PLAN.md P7). Binary-tree aggregation is a real, fully-specified mechanism with an
honest polylog(T) cost — not a placeholder a reported number would silently depend on.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm


def gram_sensitivity(clip_norm: float) -> float:
    """L2 (Frobenius) sensitivity of `R = sum_i x_i x_i^T` under one-example add/remove (unit U1),
    given `||x_i|| <= clip_norm`. Exact: a rank-1 term `x x^T` has Frobenius norm `||x||^2`, so the
    worst-case one-example change has Frobenius norm exactly `clip_norm**2`."""
    if clip_norm <= 0:
        raise ValueError("clip_norm must be > 0")
    return float(clip_norm) ** 2


def _delta_of_z(eps: float, z: float) -> float:
    """Exact delta(eps) of the Gaussian mechanism at noise-to-sensitivity ratio z = sigma/sensitivity
    (Balle & Wang 2018; equivalently the Gaussian-DP exact tradeoff, Dong-Roth-Su 2019)."""
    a = 1.0 / (2.0 * z) - eps * z
    b = -1.0 / (2.0 * z) - eps * z
    return float(norm.cdf(a) - np.exp(eps) * norm.cdf(b))


def analytic_gaussian_sigma(eps: float, delta: float, sensitivity: float) -> float:
    """Smallest sigma such that `N(0, sigma^2 I)` added to a query of L2 sensitivity `sensitivity`
    is `(eps, delta)`-DP, using the exact (all-eps-valid) Gaussian mechanism calibration."""
    if eps <= 0:
        raise ValueError("eps must be > 0")
    if not (0.0 < delta < 1.0):
        raise ValueError("delta must be in (0, 1)")
    if sensitivity <= 0:
        raise ValueError("sensitivity must be > 0")

    lo, hi = 1e-8, 1.0
    while _delta_of_z(eps, hi) > delta:
        hi *= 2.0
        if hi > 1e12:
            raise RuntimeError("failed to bracket a root for the analytic Gaussian calibration")
    z_star = brentq(lambda z: _delta_of_z(eps, z) - delta, lo, hi, xtol=1e-14, rtol=1e-14)
    return float(z_star) * float(sensitivity)


def tree_aggregation_factor(T: int) -> float:
    """Binary-tree aggregation (continual counting): releasing a running sum over T rounds via a
    binary tree of partial sums touches each item in at most `ceil(log2(T)) + 1` tree nodes, so
    per-release noise scales the base per-round sigma by this factor rather than by `sqrt(T)` (naive
    independent noise) or `T` (fully sequential composition). See module docstring re: P7 upgrade."""
    if T < 1:
        raise ValueError("T must be >= 1")
    return float(np.ceil(np.log2(T)) + 1) if T > 1 else 1.0


def sym_gaussian_noise(d: int, sigma: float, rng) -> np.ndarray:
    """Symmetric Gaussian noise for a symmetric d x d statistic (e.g. a Gram matrix): draw noise for
    the upper triangle (incl. diagonal) and mirror it, so the perturbation is exactly symmetric and
    sensitivity is not double-counted between the two triangles."""
    if sigma < 0:
        raise ValueError("sigma must be >= 0")
    z = rng.standard_normal((d, d)) * sigma
    upper = np.triu(z)
    return upper + upper.T - np.diag(np.diag(upper))
