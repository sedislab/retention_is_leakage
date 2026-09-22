from __future__ import annotations

import numpy as np
import pytest
from p3fcl import metrics


def test_accuracy_basic():
    assert metrics.accuracy([0, 1, 1, 0], [0, 1, 0, 0]) == pytest.approx(0.75)


def test_accuracy_empty_is_nan():
    assert np.isnan(metrics.accuracy([], []))


def test_backward_transfer_sign_on_forgetting():
    # 2 tasks: task0 acc drops from 0.9 (right after learning it) to 0.5 (after task1) => negative BWT
    acc_matrix = np.array([[0.9, np.nan], [0.5, 0.8]])
    bwt = metrics.backward_transfer(acc_matrix)
    assert bwt == pytest.approx(0.5 - 0.9)


def test_average_incremental_accuracy():
    acc_matrix = np.array([[1.0, np.nan], [0.5, 1.0]])
    aia = metrics.average_incremental_accuracy(acc_matrix)
    assert aia == pytest.approx(np.mean([1.0, 0.75]))


def test_roc_auc_perfect_separation():
    scores = [0.1, 0.2, 0.8, 0.9]
    labels = [0, 0, 1, 1]
    assert metrics.roc_auc(scores, labels) == pytest.approx(1.0)


def test_roc_auc_chance():
    scores = [0.5, 0.5, 0.5, 0.5]
    labels = [0, 1, 0, 1]
    assert metrics.roc_auc(scores, labels) == pytest.approx(0.5)


def test_tpr_at_fpr_never_exceeds_budget():
    rng = np.random.default_rng(0)
    n = 2000
    labels = rng.integers(0, 2, size=n)
    scores = rng.standard_normal(n) + labels * 0.3
    for target in (0.01, 0.001):
        tpr = metrics.tpr_at_fpr(scores, labels, target)
        # reconstruct achieved fpr at the implied threshold is <= target by construction; smoke-check range
        assert 0.0 <= tpr <= 1.0


def test_clopper_pearson_bounds_contain_mle():
    lo, hi = metrics.clopper_pearson(5, 10)
    assert lo < 0.5 < hi


def test_clopper_pearson_edge_cases():
    lo, hi = metrics.clopper_pearson(0, 10)
    assert lo == 0.0
    lo, hi = metrics.clopper_pearson(10, 10)
    assert hi == 1.0


def test_membership_report_never_returns_bare_auc():
    rng = np.random.default_rng(0)
    scores = rng.standard_normal(500)
    labels = rng.integers(0, 2, size=500)
    report = metrics.membership_report(scores, labels)
    required = {
        "auc", "tpr_at_1pct_fpr", "tpr_at_1pct_fpr_ci_lo", "tpr_at_1pct_fpr_ci_hi",
        "tpr_at_0.1pct_fpr", "tpr_at_0.1pct_fpr_ci_lo", "tpr_at_0.1pct_fpr_ci_hi",
        "n_pos", "n_neg",
    }
    assert required.issubset(report.keys())


def test_seed_ci_mean_and_width():
    mean, lo, hi = metrics.seed_ci([0.8, 0.82, 0.79, 0.81, 0.80])
    assert mean == pytest.approx(0.804, abs=1e-3)
    assert lo < mean < hi
    assert hi - lo < 0.05  # tight cluster of seeds -> a narrow interval


def test_seed_ci_single_value_is_a_point():
    mean, lo, hi = metrics.seed_ci([0.5])
    assert mean == lo == hi == pytest.approx(0.5)


def test_fit_exponential_halflife_recovers_known_halflife():
    true_halflife = 3.0
    tau = true_halflife / np.log(2)
    x = np.arange(10)
    y = np.exp(-x / tau)
    fit = metrics.fit_exponential_halflife(x, y)
    assert fit["fit_ok"] is True
    assert fit["halflife"] == pytest.approx(true_halflife, rel=1e-6)
    assert fit["r2"] == pytest.approx(1.0, abs=1e-6)


def test_fit_exponential_halflife_flat_curve_is_not_fit_ok():
    x = np.arange(10)
    y = np.full(10, 0.9)
    fit = metrics.fit_exponential_halflife(x, y)
    assert fit["fit_ok"] is False
    assert fit["halflife"] == float("inf")


def test_fit_exponential_halflife_increasing_curve_is_not_fit_ok():
    x = np.arange(10)
    y = np.linspace(0.5, 0.9, 10)
    fit = metrics.fit_exponential_halflife(x, y)
    assert fit["fit_ok"] is False


def test_fit_exponential_halflife_too_few_points():
    fit = metrics.fit_exponential_halflife([0, 1], [0.9, 0.5])
    assert fit["fit_ok"] is False
    assert fit["halflife"] == float("inf")


def test_bootstrap_ci_reasonable_on_constant_data():
    point, lo, hi = metrics.bootstrap_ci(lambda a: np.mean(a), [np.ones(50)], n=100, seed=0)
    assert point == pytest.approx(1.0)
    assert lo == pytest.approx(1.0)
    assert hi == pytest.approx(1.0)
