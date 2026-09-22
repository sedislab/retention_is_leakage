from __future__ import annotations

import numpy as np
import pytest
from p3fcl.dp.mechanisms import (
    analytic_gaussian_sigma,
    gram_sensitivity,
    sym_gaussian_noise,
    tree_aggregation_factor,
)


def test_gram_sensitivity_is_clip_norm_squared():
    assert gram_sensitivity(3.0) == pytest.approx(9.0)


def test_gram_sensitivity_rejects_nonpositive():
    with pytest.raises(ValueError):
        gram_sensitivity(0.0)


def test_analytic_gaussian_sigma_valid_above_eps_1():
    # the classical sqrt(2 ln(1.25/delta))/eps bound is invalid for eps > 1; the analytic mechanism
    # must still return a finite, positive sigma here.
    sigma = analytic_gaussian_sigma(eps=3.0, delta=1e-5, sensitivity=1.0)
    assert sigma > 0 and np.isfinite(sigma)


def test_analytic_gaussian_sigma_decreases_as_eps_grows():
    s_small_eps = analytic_gaussian_sigma(eps=0.1, delta=1e-5, sensitivity=1.0)
    s_large_eps = analytic_gaussian_sigma(eps=5.0, delta=1e-5, sensitivity=1.0)
    assert s_large_eps < s_small_eps


def test_analytic_gaussian_sigma_scales_with_sensitivity():
    s1 = analytic_gaussian_sigma(eps=1.0, delta=1e-5, sensitivity=1.0)
    s2 = analytic_gaussian_sigma(eps=1.0, delta=1e-5, sensitivity=2.0)
    assert s2 == pytest.approx(2 * s1, rel=1e-6)


def test_analytic_gaussian_sigma_rejects_bad_inputs():
    with pytest.raises(ValueError):
        analytic_gaussian_sigma(eps=0.0, delta=1e-5, sensitivity=1.0)
    with pytest.raises(ValueError):
        analytic_gaussian_sigma(eps=1.0, delta=0.0, sensitivity=1.0)
    with pytest.raises(ValueError):
        analytic_gaussian_sigma(eps=1.0, delta=1e-5, sensitivity=0.0)


def test_tree_aggregation_factor_grows_like_log2():
    assert tree_aggregation_factor(1) == 1.0
    assert tree_aggregation_factor(2) == pytest.approx(2.0)
    assert tree_aggregation_factor(1024) == pytest.approx(np.ceil(np.log2(1024)) + 1)
    # honest polylog, not linear: factor at T=1000 is tiny compared to T
    assert tree_aggregation_factor(1000) < 20


def test_sym_gaussian_noise_is_exactly_symmetric():
    rng = np.random.default_rng(0)
    noise = sym_gaussian_noise(5, sigma=1.0, rng=rng)
    np.testing.assert_allclose(noise, noise.T)


def test_sym_gaussian_noise_zero_sigma_is_zero():
    rng = np.random.default_rng(0)
    noise = sym_gaussian_noise(4, sigma=0.0, rng=rng)
    np.testing.assert_allclose(noise, np.zeros((4, 4)))
