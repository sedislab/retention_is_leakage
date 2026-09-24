"""Empirical epsilon lower bound from exact one-sided pointwise binomial limits."""

import numpy as np
from scipy.stats import beta


def epsilon_lower_bound(scores, labels, delta=1e-5, alpha=0.05):
    scores, labels = np.asarray(scores), np.asarray(labels, dtype=int)
    keep = np.isfinite(scores)
    scores, labels = scores[keep], labels[keep]
    n1, n0 = int(labels.sum()), int((1 - labels).sum())
    if not n1 or not n0:
        raise ValueError("both membership classes are required")
    order = np.argsort(-scores, kind="stable")
    scores, labels = scores[order], labels[order]
    ends = np.r_[np.flatnonzero(scores[:-1] != scores[1:]), len(scores) - 1]
    tp = np.cumsum(labels)[ends]
    fp = (ends + 1) - tp
    # Tie groups are indivisible; these are real thresholds, never interpolated ROC points.
    tpr_lo = np.where(tp == 0, 0.0, beta.ppf(alpha, tp, n1 - tp + 1))
    fpr_hi = np.where(fp == n0, 1.0, beta.ppf(1 - alpha, fp + 1, n0 - fp))
    with np.errstate(divide="ignore", invalid="ignore"):
        forward = np.log((tpr_lo - delta) / fpr_hi)
        mirror = np.log((1 - fpr_hi - delta) / (1 - tpr_lo))
    values = np.stack([forward, mirror])
    best = np.unravel_index(np.nanargmax(values), values.shape)
    return dict(
        eps_lb=max(0.0, float(values[best])),
        threshold=float(scores[ends[best[1]]]),
        direction=["TPR/FPR", "(1-FPR)/(1-TPR)"][best[0]],
        thresholds=len(ends),
        tpr_lo=float(tpr_lo[best[1]]),
        fpr_hi=float(fpr_hi[best[1]]),
        interval_scope="one-sided 95% Clopper-Pearson pointwise; threshold maximum is not a simultaneous certificate",
    )
