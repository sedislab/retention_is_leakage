"""Sample-count-invariant cross-entropy objective shared by the replay methods."""
import numpy as np


def mean_ce_gradient(W, X, y):
    if X is None or len(y) == 0:
        return np.zeros_like(W)
    logits = X @ W
    logits -= logits.max(axis=1, keepdims=True)
    probabilities = np.exp(logits)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    labels = np.zeros((len(y), W.shape[1]))
    labels[np.arange(len(y)), y] = 1.0
    return X.T @ (probabilities-labels) / len(y)


def balanced_gradient(W, X_current, y_current, X_replay, y_replay, replay_weight=1.0):
    return (mean_ce_gradient(W, X_current, y_current)
            + replay_weight * mean_ce_gradient(W, X_replay, y_replay))
