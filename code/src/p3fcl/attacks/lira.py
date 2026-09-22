"""A1 -- cross-task LiRA over the transcript (`RESEARCH_PLAN.md` §3.3; H2/H3/H11's machinery;
`04_METHODS_AND_ATTACKS.md` §2's A1 row). Per-family, per-round *statistics* are computed by
`shadow_runner.py` (they need the ledger and the raw feature, same as any other attack); this module
is the LiRA *combination* step: given a target's real observed score trajectory and many shadow
federations' score trajectories with known IN/OUT membership for that same target, compute the
(log) likelihood ratio.

Simplified/"online" LiRA (Carlini et al. 2022): fit one Gaussian to the IN-shadow scores and one to
the OUT-shadow scores *per round*, and take the log-likelihood-ratio of the target's real score under
each. `trajectory=True` sums the per-round log-LR over every round the artifact has been released by
(the "transcript" arm, H3); `trajectory=False` uses only the most-recently-released round (the
"checkpoint" arm, FIG07's baseline) -- that comparison is hypothesis H3.

Calibration (`in_traj`/`out_traj`) must come from a shadow split disjoint from the evaluation targets
(CLAUDE.md non-negotiable #6) -- `scripts/run_lira.py` is responsible for keeping calibration and
evaluation shadows disjoint; this module trusts whatever it's given.
"""
from __future__ import annotations

import numpy as np
from scipy import stats

_EPS = 1e-6


def _fit_gaussian(x: np.ndarray) -> tuple:
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return float("nan"), float("nan")
    return float(np.mean(x)), float(np.std(x, ddof=1) + _EPS)


def per_round_log_lr(target_score: float, in_scores: np.ndarray, out_scores: np.ndarray) -> float:
    """log N(target | mu_in, sd_in) - log N(target | mu_out, sd_out) for one round.

    NaN if the target's own score is undefined (not yet released) or either shadow side lacks at
    least two observations -- callers must treat NaN as "no information for this round", not zero.
    """
    if np.isnan(target_score):
        return float("nan")
    mu_in, sd_in = _fit_gaussian(in_scores)
    mu_out, sd_out = _fit_gaussian(out_scores)
    if np.isnan(mu_in) or np.isnan(mu_out):
        return float("nan")
    return float(
        stats.norm.logpdf(target_score, mu_in, sd_in) - stats.norm.logpdf(target_score, mu_out, sd_out)
    )


def lira_score(
    target_traj: np.ndarray, in_traj: np.ndarray, out_traj: np.ndarray, trajectory: bool
) -> float:
    """`target_traj`: `(n_rounds,)`, the target's real observed score per round (NaN before release).
    `in_traj`/`out_traj`: `(n_shadows_in_or_out, n_rounds)`, shadow score trajectories for the SAME
    target (already restricted to whichever shadows had it IN / OUT respectively).

    `trajectory=True`: sum the per-round log-LR over every round released so far (H3's "transcript").
    `trajectory=False`: use only the last released round (H3's "checkpoint", FIG07's baseline).

    Returns NaN if the artifact has never been released under `target_traj`'s horizon (should not
    happen for a real target whose task has already occurred).
    """
    n_rounds = len(target_traj)
    released = [t for t in range(n_rounds) if not np.isnan(target_traj[t])]
    if not released:
        return float("nan")
    rounds = released if trajectory else [released[-1]]
    lrs = [per_round_log_lr(target_traj[t], in_traj[:, t], out_traj[:, t]) for t in rounds]
    lrs = [v for v in lrs if not np.isnan(v)]
    if not lrs:
        return float("nan")
    return float(np.sum(lrs))


def offline_log_lr(target_score: float, out_scores: np.ndarray) -> float:
    """Offline LiRA (OUT shadows only, Carlini et al. 2022's offline variant): `-log Pr[X_out >=
    target_score]`, i.e. the negative log-survival of the target's score under the OUT-only null.
    High when the target's score sits far in the "more IN-like" tail of the OUT distribution --
    **this assumes the score convention is already oriented so higher = more consistent with
    membership**. That is true of `shadow_runner.py`'s F2/prototype score (less distance = more IN)
    but is the OPPOSITE of its F5/Gram self-leverage score: by Sherman-Morrison, a point's quadratic
    form `x^T R^-1 x` against a Gram matrix that *excludes* it is always `h/(1-h) >= h`, i.e. strictly
    larger than against one that *includes* it (`h`) -- exclusion inflates self-leverage, it does not
    shrink it. Verified against a direct two-matrix comparison on real CIFAR-100 shadows (all 500 F5
    targets showed OUT-mean > IN-mean, matching the closed form exactly), so this is a property of the
    statistic, not a shadow_runner bug -- a target caught looking "too separated in the wrong
    direction" is worth checking against a known identity before assuming the code is broken, same as
    every other standing-heuristic instance this project has hit. The online path (`per_round_log_lr`,
    used for M4/M8 this pass) fits both distributions and is direction-agnostic, so this only matters
    if/when a Gram-family method is later wired into the offline path -- pass `-score` (or an
    equivalent sign flip) for F5 if that happens. Reserved for the non-cacheable methods (M3/M4-LoRA/M6/M7,
    `01_KODIAK.md §4.3`); not exercised by the M4/M8 pair scored this pass, which is cacheable and
    gets the full online treatment.
    """
    if np.isnan(target_score):
        return float("nan")
    mu_out, sd_out = _fit_gaussian(out_scores)
    if np.isnan(mu_out):
        return float("nan")
    return float(-stats.norm.logsf(target_score, mu_out, sd_out))
