import numpy as np
import pytest
from p3fcl.audit.empirical import epsilon_lower_bound


def test_perfect_membership_has_positive_bound_and_ties_are_not_split():
    labels = np.repeat([0, 1], 1000)
    result = epsilon_lower_bound(labels.astype(float), labels)
    assert result["thresholds"] == 2
    assert result["eps_lb"] > 5


def test_equal_scores_have_zero_audited_epsilon():
    result = epsilon_lower_bound(np.zeros(2000), np.repeat([0, 1], 1000))
    assert result["eps_lb"] == 0


def test_mirror_is_included():
    # At this threshold, FPR=.50, TPR=.999; complement direction is much stronger.
    scores = np.r_[np.zeros(500), np.ones(500), np.zeros(1), np.ones(999)]
    result = epsilon_lower_bound(scores, np.repeat([0, 1], 1000))
    assert result["direction"] == "(1-FPR)/(1-TPR)"
    assert result["eps_lb"] > 3


def test_single_membership_class_is_rejected():
    with pytest.raises(ValueError, match="both membership"):
        epsilon_lower_bound([1, 2], [1, 1])
