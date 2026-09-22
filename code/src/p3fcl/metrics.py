"""Accuracy, BWT, AUC, TPR@FPR, Clopper-Pearson. `membership_report` must never return AUC alone
(CLAUDE.md non-negotiable #5) — TPR@1%FPR and TPR@0.1%FPR are primary at this venue.
"""
from __future__ import annotations

import numpy as np
from scipy import stats

from . import rng as rng_mod


def accuracy(y_true, y_pred) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if len(y_true) == 0:
        return float("nan")
    return float(np.mean(y_true == y_pred))


def backward_transfer(acc_matrix) -> float:
    """`acc_matrix[t, k]` = accuracy on task k after finishing task t (0-indexed, t >= k).
    BWT = mean over k < T-1 of A[T-1, k] - A[k, k]. Negative = forgetting."""
    acc_matrix = np.asarray(acc_matrix, dtype=float)
    T = acc_matrix.shape[0]
    if T < 2:
        return float("nan")
    last = T - 1
    diffs = [acc_matrix[last, k] - acc_matrix[k, k] for k in range(last)]
    return float(np.nanmean(diffs))


def average_incremental_accuracy(acc_matrix) -> float:
    """Mean over t of the average accuracy on tasks 0..t after finishing task t."""
    acc_matrix = np.asarray(acc_matrix, dtype=float)
    T = acc_matrix.shape[0]
    per_t = [np.nanmean(acc_matrix[t, : t + 1]) for t in range(T)]
    return float(np.nanmean(per_t))


def roc_auc(scores, labels) -> float:
    """Rank-based AUC (Mann-Whitney U), tie-corrected. `labels` in {0,1}."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    n1 = int(np.sum(labels == 1))
    n0 = int(np.sum(labels == 0))
    if n1 == 0 or n0 == 0:
        return float("nan")
    ranks = stats.rankdata(scores)
    sum_ranks_pos = ranks[labels == 1].sum()
    return float((sum_ranks_pos - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def tpr_at_fpr(scores, labels, target_fpr: float) -> float:
    """TPR at the most permissive threshold whose FPR does not exceed `target_fpr` (conservative:
    never reports a TPR that required exceeding the stated FPR budget)."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    thresholds = np.unique(scores)[::-1]
    best_tpr = 0.0
    for thr in thresholds:
        fpr = float(np.mean(neg >= thr))
        if fpr <= target_fpr:
            best_tpr = max(best_tpr, float(np.mean(pos >= thr)))
    return best_tpr


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple:
    """Exact binomial confidence interval, two-sided, level `1 - alpha`."""
    if n == 0:
        return (float("nan"), float("nan"))
    lo = 0.0 if k == 0 else float(stats.beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(stats.beta.ppf(1 - alpha / 2, k + 1, n - k))
    return (lo, hi)


def membership_report(scores, labels, alpha: float = 0.05) -> dict:
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    n_pos = int(np.sum(labels == 1))
    n_neg = int(np.sum(labels == 0))
    tpr1 = tpr_at_fpr(scores, labels, 0.01)
    tpr01 = tpr_at_fpr(scores, labels, 0.001)
    k1 = int(round(tpr1 * n_pos)) if n_pos and not np.isnan(tpr1) else 0
    k01 = int(round(tpr01 * n_pos)) if n_pos and not np.isnan(tpr01) else 0
    lo1, hi1 = clopper_pearson(k1, n_pos, alpha)
    lo01, hi01 = clopper_pearson(k01, n_pos, alpha)
    return {
        "auc": roc_auc(scores, labels),
        "tpr_at_1pct_fpr": tpr1,
        "tpr_at_1pct_fpr_ci_lo": lo1,
        "tpr_at_1pct_fpr_ci_hi": hi1,
        "tpr_at_0.1pct_fpr": tpr01,
        "tpr_at_0.1pct_fpr_ci_lo": lo01,
        "tpr_at_0.1pct_fpr_ci_hi": hi01,
        "n_pos": n_pos,
        "n_neg": n_neg,
    }


def seed_ci(values, alpha: float = 0.05) -> tuple:
    """`(mean, ci_lo, ci_hi)` across a small number of per-seed point estimates (FIG01's "bands = 95%
    CI over seeds" -- a t-distribution interval on the seed-to-seed mean, appropriate for the small
    seed counts (3-5) this project uses; distinct from `clopper_pearson`, which is for a single seed's
    rate estimate on a finite test set, not for summarizing across seeds)."""
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    n = len(values)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(np.mean(values))
    if n == 1:
        return mean, mean, mean
    sem = float(np.std(values, ddof=1) / np.sqrt(n))
    tval = float(stats.t.ppf(1 - alpha / 2, df=n - 1))
    return mean, mean - tval * sem, mean + tval * sem


def fit_exponential_halflife(x, y) -> dict:
    """Fit `y = a * exp(-x / tau)` via linear regression of `log(y)` on `x` (robust and simple; `y`
    must be positive -- values `<= 0` are dropped before fitting, consistent with TPR/accuracy always
    living in `(0, 1]` in practice). Returns `{halflife, r2, fit_ok, slope}`.

    `fit_ok=False` (and `halflife=inf`) whenever the fitted trend is flat or *increasing*
    (`slope >= 0`) or there are fewer than 3 usable points -- `00_BUILD_PLAN.md`'s P4 protocol is
    explicit that "fitting a half-life to a flat curve is a reporting error": for an append-only
    family the honest answer is "no decay detected over the observed horizon, lower bound on
    half-life > the observed horizon," not a forced (and meaningless) exponential fit.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = y > 0
    x, y = x[mask], y[mask]
    if len(x) < 3:
        return {"halflife": float("inf"), "r2": float("nan"), "fit_ok": False, "slope": float("nan")}

    log_y = np.log(y)
    slope, intercept = np.polyfit(x, log_y, 1)
    if slope >= 0:
        return {"halflife": float("inf"), "r2": float("nan"), "fit_ok": False, "slope": float(slope)}

    pred = slope * x + intercept
    ss_res = float(np.sum((log_y - pred) ** 2))
    ss_tot = float(np.sum((log_y - np.mean(log_y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    tau = -1.0 / slope
    halflife = tau * np.log(2)
    return {"halflife": float(halflife), "r2": r2, "fit_ok": True, "slope": float(slope)}


def bootstrap_ci(fn, data, n: int = 2000, alpha: float = 0.05, seed=0) -> tuple:
    """Percentile bootstrap over matched-length arrays in `data` (each resampled with the same
    indices, so paired structure is preserved). Returns `(point_estimate, ci_lo, ci_hi)`."""
    r = seed if hasattr(seed, "integers") else rng_mod.seeded("metrics.bootstrap_ci", seed)
    arrays = [np.asarray(d) for d in data]
    m = len(arrays[0])
    point = fn(*arrays)
    boots = np.empty(n)
    for i in range(n):
        idx = r.integers(0, m, size=m)
        boots[i] = fn(*[a[idx] for a in arrays])
    lo = float(np.percentile(boots, 100 * alpha / 2))
    hi = float(np.percentile(boots, 100 * (1 - alpha / 2)))
    return float(point), lo, hi
