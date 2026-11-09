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


# ---------------------------------------------------------------- R7b edge extension (R7b_PROTOCOL.md)

EXT_STEPS = {"lr_low": [0.005, 0.002], "lr_high": [2.0, 5.0], "mult_low": [0.02, 0.01], "mult_high": [8.0, 16.0]}


def best_point(points: list) -> int:
    """Index of the highest seed-mean accuracy; NaN ranks last; ties go to the first in `points` order."""
    means = np.array([np.mean(p["accs"]) for p in points], dtype=float)
    means = np.where(np.isnan(means), -np.inf, means)
    return int(np.argmax(means))


def edges_hit(lr: float, mult: float, lr_grid, mult_grid) -> list:
    out = []
    if lr == min(lr_grid):
        out.append("lr_low")
    if lr == max(lr_grid):
        out.append("lr_high")
    if mult == min(mult_grid):
        out.append("mult_low")
    if mult == max(mult_grid):
        out.append("mult_high")
    return out


def extend_grid(lr_grid: list, mult_grid: list, edges: list, used: dict) -> tuple:
    """One extension round: add the next unused step in each hit direction. `used` counts the steps
    taken per direction and is updated in place. Returns the new sorted grids and the directions
    actually extended (a direction whose steps are exhausted is not extended)."""
    lr_grid, mult_grid, done = list(lr_grid), list(mult_grid), []
    for e in edges:
        if used.get(e, 0) >= len(EXT_STEPS[e]):
            continue
        v = EXT_STEPS[e][used.get(e, 0)]
        used[e] = used.get(e, 0) + 1
        (lr_grid if e.startswith("lr") else mult_grid).append(v)
        done.append(e)
    return sorted(set(lr_grid)), sorted(set(mult_grid)), done
