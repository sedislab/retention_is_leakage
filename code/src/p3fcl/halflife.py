"""FX2 (`08_FIX_PLAN.md` §7b/7c): chance floors, normalization, half-life exactly per Definition 10 in
the paper, and the hierarchical bootstrap. Pure functions over already-computed per-elapsed-time
summary values or caller-supplied resampling callbacks -- this module never touches a shadow store or
runs an attack itself, so it can be reasoned about and tested independently of
`run_lira_pertask.py`'s scoring pipeline (built separately; this module is the self-contained
numerical core that pipeline's aggregation step wires real per-target/per-seed data into).
"""
from __future__ import annotations

import numpy as np

from . import rng as rng_mod

FLOOR_TPR1 = 0.01
FLOOR_TPR01 = 0.001
FLOOR_AUC = 0.5

_LEAK_FLOORS = {"tpr1": FLOOR_TPR1, "tpr01": FLOOR_TPR01, "auc": FLOOR_AUC}

NO_SIGNAL_LEAK_THRESHOLD = 0.005
NO_SIGNAL_ACC_THRESHOLD = 0.02


def leak_floor(metric: str) -> float:
    """beta: 0.01 for TPR@1%FPR, 0.001 for TPR@0.1%FPR, 0.5 for AUC."""
    if metric not in _LEAK_FLOORS:
        raise ValueError(f"leak_floor: unknown metric {metric!r}, expected one of {sorted(_LEAK_FLOORS)}")
    return _LEAK_FLOORS[metric]


def acc_floor_class_incremental(n_classes_seen: int) -> float:
    """a0(t) = 1 / |classes seen after task t|."""
    if n_classes_seen <= 0:
        raise ValueError("acc_floor_class_incremental: n_classes_seen must be > 0")
    return 1.0 / n_classes_seen


def acc_floor_majority_class(y_eval) -> float:
    """a0 for Camelyon17 (domain-incremental): the majority-class rate of the evaluation set."""
    y_eval = np.asarray(y_eval)
    if len(y_eval) == 0:
        raise ValueError("acc_floor_majority_class: y_eval is empty")
    _, counts = np.unique(y_eval, return_counts=True)
    return float(counts.max() / counts.sum())


def has_signal(base_value: float, base_ci_lo: float, floor: float, threshold: float) -> bool:
    """The 07b guard: `base_value - floor`'s lower 95% CI bound must exceed `threshold`, or the
    quantity is `no_signal` (report no half-life for it). Shared by leakage (threshold 0.005) and
    accuracy (threshold 0.02) -- only the threshold and the input CI differ."""
    return (base_ci_lo - floor) > threshold


def normalize(values_by_e: dict, floor: float) -> dict:
    """L(e) = (value(e) - floor) / (value(0) - floor), pooled numerator/denominator over K per
    `08_FIX_PLAN.md` §7b (callers pool across the fixed-k set before calling this, this function
    itself is agnostic to how `values_by_e` was pooled). Raises if the base value would make this a
    division by a non-positive number -- callers must apply `has_signal` first and skip this function
    entirely (report `no_signal`) rather than let it produce a nonsense ratio."""
    if 0 not in values_by_e:
        raise ValueError("normalize: values_by_e must include e=0 (the base value)")
    base = values_by_e[0] - floor
    if base <= 0:
        raise ValueError(
            f"normalize: base value(0)-floor={base} is not positive -- check has_signal() before calling this"
        )
    return {e: (v - floor) / base for e, v in values_by_e.items()}


def half_life(normalized_by_e: dict, E: int = 6) -> dict:
    """Definition 10: h is the first e >= 1 (integer elapsed time, `halflife_int`) at which the
    normalised value is <= 1/2, linearly interpolated between e-1 and e for the fractional `halflife`.
    Censored at `E` if no crossing occurs by then: `halflife` is reported as `E` itself with
    `status="censored"`, meaning "> E" -- never extrapolated past `E`, never reported as infinite.
    Requires `normalized_by_e[0] > 0.5` (e=0 is defined as 1.0 by construction of `normalize`, so this
    only fails if the caller passes an already-broken curve) and every integer e in `1..E` present."""
    if 0 not in normalized_by_e:
        raise ValueError("half_life: normalized_by_e must include e=0")
    for e in range(1, E + 1):
        if e not in normalized_by_e:
            raise ValueError(f"half_life: normalized_by_e missing e={e} (need every integer 0..{E})")

    for e in range(1, E + 1):
        v_prev = normalized_by_e[e - 1]
        v_curr = normalized_by_e[e]
        if v_curr <= 0.5:
            if v_prev == v_curr:
                h = float(e)
            else:
                frac = (v_prev - 0.5) / (v_prev - v_curr)
                h = (e - 1) + frac
            return {"halflife": h, "halflife_int": e, "status": "ok"}
    return {"halflife": float(E), "halflife_int": None, "status": "censored"}


def decoupling_ratio(h_leak: dict, h_acc: dict, by_construction: bool = False) -> dict:
    """rho = h_leak / h_acc. `ratio_type`: `by_construction` (caller-flagged: the per-client `full`
    view of F2/F5, constant by construction, per `08_FIX_PLAN.md` §4h) overrides everything else;
    else `undefined` if either side is `no_signal` or `h_acc` is censored (a censored denominator
    makes the ratio meaningless, not just a lower bound); else `lower_bound` if `h_leak` is censored
    (numerator is itself only known to be `> E`, so the ratio is a lower bound); else `point`."""
    if by_construction:
        return {"ratio": None, "ratio_type": "by_construction"}
    if h_leak["status"] == "no_signal" or h_acc["status"] == "no_signal":
        return {"ratio": None, "ratio_type": "undefined"}
    if h_acc["status"] == "censored":
        return {"ratio": None, "ratio_type": "undefined"}
    ratio = h_leak["halflife"] / h_acc["halflife"]
    if h_leak["status"] == "censored":
        return {"ratio": ratio, "ratio_type": "lower_bound"}
    return {"ratio": ratio, "ratio_type": "point"}


def no_signal_result() -> dict:
    """The `half_life` return shape's `no_signal` sibling -- callers that fail the 07b guard report
    this instead of calling `normalize`/`half_life` at all."""
    return {"halflife": None, "halflife_int": None, "status": "no_signal"}


def hierarchical_bootstrap(seed_groups: list, inner_resample_fn, compute_fn, n_replicates: int = 2000, seed: int = 0) -> dict:
    """FX2 §7c: "Resample seeds (outer), then targets within each seed (inner). A target keeps all of
    its eval-shadow pairs. ... Report the percentile 95% CI. If more than half the replicates are
    censored, report the CI as censored."

    Generic over what "seed" and "target" mean so the SAME function serves both the leakage side
    (`seed_groups[i]` = one seed's per-target eval-shadow score/label arrays, the inner unit is a
    target) and the accuracy side (`seed_groups[i]` = one seed's per-k accuracy/floor values, the
    inner unit is one of the fixed-k origin tasks) -- per `08_FIX_PLAN.md` §7c's own "Confidence
    intervals" note applying to both quantities without restating the resampling machinery twice.

    - `seed_groups`: one entry per available seed (>= 3, per CLAUDE.md non-negotiable #4).
    - `inner_resample_fn(seed_data, rng) -> resampled_seed_data`: resamples that seed's inner units
      WITH replacement (e.g. targets, or the k in K), preserving whatever grouping "a target keeps
      all of its eval-shadow pairs" requires -- this function decides what that means for its data.
    - `compute_fn(list_of_resampled_seed_data) -> float | None | object`: recomputes the statistic (a
      raw TPR/AUC, a half-life, whatever the caller is bootstrapping) from one full resampled draw
      (all outer-resampled seeds' inner-resampled data combined). Returns `None` for a replicate that
      is censored/undefined (e.g. a half-life that didn't cross by `E` in that replicate) -- these are
      excluded from the percentile CI but counted toward the "more than half censored" rule. May
      instead return a non-scalar object (e.g. a dict of several quantities computed from the SAME
      resampled draw, to avoid re-resampling once per quantity) -- in that case this function cannot
      derive `ci_lo`/`ci_hi`/`censored`/`n_valid` itself (those come back `None`) and the caller reads
      `replicates` directly to do its own aggregation, still getting the resampling loop shared.

    Returns `{"ci_lo", "ci_hi", "censored", "n_valid", "replicates"}`. `censored=True` (and
    `ci_lo`/`ci_hi`/`None`) if more than half of `n_replicates` were censored, or if every replicate
    was censored/undefined.
    """
    if len(seed_groups) < 1:
        raise ValueError("hierarchical_bootstrap: seed_groups must be non-empty")
    rng = rng_mod.seeded("halflife.hierarchical_bootstrap", seed)
    n_seeds = len(seed_groups)
    replicates: list = []
    for _ in range(n_replicates):
        resampled_seed_idx = rng.integers(0, n_seeds, size=n_seeds)
        resampled_data = [inner_resample_fn(seed_groups[i], rng) for i in resampled_seed_idx]
        replicates.append(compute_fn(resampled_data))

    try:
        valid = np.array([v for v in replicates if v is not None], dtype=float)
    except (TypeError, ValueError):
        # non-scalar replicates (e.g. dicts) -- the caller aggregates `replicates` itself.
        return {"ci_lo": None, "ci_hi": None, "censored": None, "n_valid": None, "replicates": replicates}
    n_censored = n_replicates - len(valid)
    if n_censored > n_replicates / 2 or len(valid) == 0:
        return {"ci_lo": None, "ci_hi": None, "censored": True, "n_valid": len(valid), "replicates": replicates}
    ci_lo, ci_hi = np.percentile(valid, [2.5, 97.5])
    return {"ci_lo": float(ci_lo), "ci_hi": float(ci_hi), "censored": False, "n_valid": len(valid), "replicates": replicates}
