"""A7 — one-run canary audit (Steinke, Nasr & Jagielski, NeurIPS'23-style; the pure-eps,
non-abstaining case with independent 1/2-coin-flip canary inclusion).

Full wiring to M8/M9 with DP on (and a non-private control) is Phase P3 task 7; this module provides
the core statistical primitive so it exists and is unit-tested from P0 onward — it is the project's
insurance policy if every learned attack (A1, A3, A5, A6) comes back weak (RESEARCH_PLAN.md §7, R2).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class AuditResult:
    eps_lower_bound: float
    n_canaries: int
    n_guesses: int
    n_correct: int
    confidence: float


def _p_value(eps: float, r: int, v: int) -> float:
    """P[Binomial(v, p_eps) >= r], p_eps = e^eps / (1 + e^eps). Non-decreasing in eps."""
    p = np.exp(eps) / (1.0 + np.exp(eps))
    return float(stats.binom.sf(r - 1, v, p))


def audit_from_guesses(in_out, guesses, confidence: float = 0.95) -> AuditResult:
    """`in_out`: true IN(1)/OUT(0) bits for each canary (independent fair coin flips).
    `guesses`: the adversary's IN(1)/OUT(0) guess per canary, or -1 to abstain (excluded).

    Returns the largest eps for which a one-sided exact binomial test rejects, at level
    `1 - confidence`, the null "these guesses are no better than chance under eps-DP" — i.e. an
    `(eps, 1-confidence)` empirical lower bound on the mechanism's true eps.
    """
    in_out = np.asarray(in_out)
    guesses = np.asarray(guesses)
    if in_out.shape != guesses.shape:
        raise ValueError("in_out and guesses must have the same shape")
    n = len(in_out)
    mask = guesses >= 0
    v = int(mask.sum())
    if v == 0:
        return AuditResult(eps_lower_bound=0.0, n_canaries=n, n_guesses=0, n_correct=0, confidence=confidence)

    r = int(np.sum(guesses[mask] == in_out[mask]))
    alpha = 1.0 - confidence

    if _p_value(0.0, r, v) > alpha:
        return AuditResult(eps_lower_bound=0.0, n_canaries=n, n_guesses=v, n_correct=r, confidence=confidence)

    lo, hi = 0.0, 1.0
    while _p_value(hi, r, v) <= alpha:
        hi *= 2.0
        if hi > 1e6:
            break
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if _p_value(mid, r, v) <= alpha:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-9:
            break
    return AuditResult(eps_lower_bound=lo, n_canaries=n, n_guesses=v, n_correct=r, confidence=confidence)
