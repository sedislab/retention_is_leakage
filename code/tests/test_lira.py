from __future__ import annotations

import numpy as np
from p3fcl.attacks.lira import lira_score, offline_log_lr, per_round_log_lr


def test_per_round_log_lr_favors_the_matching_distribution():
    rng = np.random.default_rng(0)
    in_scores = rng.normal(5.0, 1.0, size=200)
    out_scores = rng.normal(0.0, 1.0, size=200)
    # a target that looks like the IN population should get a positive log-LR ...
    assert per_round_log_lr(5.0, in_scores, out_scores) > 0
    # ... and a target that looks like the OUT population should get a negative one.
    assert per_round_log_lr(0.0, in_scores, out_scores) < 0


def test_per_round_log_lr_nan_before_release_or_without_calibration():
    in_scores = np.array([1.0, 1.1, 0.9])
    out_scores = np.array([0.0, 0.1, -0.1])
    assert np.isnan(per_round_log_lr(float("nan"), in_scores, out_scores))
    assert np.isnan(per_round_log_lr(1.0, np.array([1.0]), out_scores))  # only 1 IN observation


def test_lira_score_trajectory_sums_released_rounds_only():
    n_rounds = 4
    target = np.array([np.nan, 2.0, 2.0, 2.0])  # released starting round 1
    rng = np.random.default_rng(1)
    in_traj = np.stack([rng.normal(2.0, 0.5, n_rounds) for _ in range(50)])
    out_traj = np.stack([rng.normal(0.0, 0.5, n_rounds) for _ in range(50)])
    # trajectory sums 3 released rounds; last_round uses only round 3 -- trajectory should be a
    # strictly larger positive score since each released round independently favors IN.
    traj = lira_score(target, in_traj, out_traj, trajectory=True)
    last = lira_score(target, in_traj, out_traj, trajectory=False)
    assert traj > last > 0


def test_lira_score_nan_when_never_released():
    target = np.full(4, np.nan)
    in_traj = np.zeros((5, 4))
    out_traj = np.zeros((5, 4))
    assert np.isnan(lira_score(target, in_traj, out_traj, trajectory=True))


def test_offline_log_lr_high_for_outlier_from_the_out_distribution():
    out_scores = np.random.default_rng(2).normal(0.0, 1.0, size=300)
    assert offline_log_lr(5.0, out_scores) > offline_log_lr(0.0, out_scores)
    assert np.isnan(offline_log_lr(float("nan"), out_scores))
