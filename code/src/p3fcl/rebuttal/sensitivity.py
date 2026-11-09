"""R7 (`rebuttal/R7_PROTOCOL.md`): summarise a test-split sweep of an existing selection grid into
selected / best / worst. This is a sensitivity analysis; nothing here feeds back into selection."""
from __future__ import annotations

import numpy as np
from scipy import stats

ORACLE_LABEL = "test-oracle upper bound, not used for selection"


def mean_ci(vals) -> tuple:
    v = np.asarray(vals, dtype=float)
    m = float(v.mean())
    if len(v) < 2:
        return m, float("nan"), float("nan")
    h = float(stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v)))
    return m, m - h, m + h


def summarize_cell(points: list, selected_key) -> dict:
    """`points`: list of `{"key": hashable, "accs": [per-seed acc]}` in grid order. Returns the
    best/worst points (first in grid order on ties), tie counts, and the selected point's rank
    (1 = best, by seed mean, with ties sharing the better rank)."""
    means = np.array([np.mean(p["accs"]) for p in points])
    i_best, i_worst = int(np.argmax(means)), int(np.argmin(means))  # argmax/argmin return the first index
    keys = [p["key"] for p in points]
    i_sel = keys.index(selected_key)
    return {
        "best": points[i_best], "worst": points[i_worst], "selected": points[i_sel],
        "n_tied_best": int(np.sum(means == means[i_best])), "n_tied_worst": int(np.sum(means == means[i_worst])),
        "selected_rank": int(np.sum(means > means[i_sel]) + 1), "n_points": len(points),
        "oracle_gap": float(means[i_best] - means[i_sel]),
    }
