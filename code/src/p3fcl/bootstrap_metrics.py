"""Exact order-statistic implementation of the existing conservative ROC statistic.

The public metrics implementation remains the reference. Bootstrap replicates only
need the TPR at a fixed FPR, so constructing every ROC threshold is unnecessary.
"""

import math

import numpy as np


def tpr_at_fpr(scores, labels, target_fpr):
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    pos, neg = scores[labels == 1], scores[labels == 0]
    if not len(pos) or not len(neg):
        return float("nan")
    if target_fpr < 0:
        return 0.0
    if target_fpr >= 1:
        return 1.0
    # Match the reference's floating comparison k / n_neg <= target_fpr even
    # when the multiplication rounds across an integer boundary.
    allowed = int(math.floor(len(neg) * target_fpr))
    while (allowed + 1) / len(neg) <= target_fpr:
        allowed += 1
    while allowed / len(neg) > target_fpr:
        allowed -= 1
    index = len(neg) - allowed - 1
    boundary = np.partition(neg, index)[index]
    # Including this boundary would exceed the FPR budget. Strict comparison
    # retains exactly the positives admitted by the next distinct threshold,
    # including correct handling of tied negative scores.
    return float(np.count_nonzero(pos > boundary) / len(pos))
