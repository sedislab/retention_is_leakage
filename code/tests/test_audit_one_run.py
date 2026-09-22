from __future__ import annotations

import numpy as np
from p3fcl.audit.one_run import audit_from_guesses


def test_perfect_guesser_gives_large_eps_lower_bound():
    rng = np.random.default_rng(0)
    n = 500
    in_out = rng.integers(0, 2, size=n)
    guesses = in_out.copy()  # perfect adversary
    result = audit_from_guesses(in_out, guesses, confidence=0.95)
    assert result.eps_lower_bound > 3.0  # perfect on 500 canaries should audit a large eps
    assert result.n_correct == n


def test_coin_flip_guesser_gives_near_zero_eps():
    rng = np.random.default_rng(1)
    n = 500
    in_out = rng.integers(0, 2, size=n)
    guesses = rng.integers(0, 2, size=n)  # independent of in_out
    result = audit_from_guesses(in_out, guesses, confidence=0.95)
    assert result.eps_lower_bound < 1.0


def test_all_abstain_gives_zero():
    n = 10
    result = audit_from_guesses(np.zeros(n), -np.ones(n, dtype=int))
    assert result.eps_lower_bound == 0.0
    assert result.n_guesses == 0


def test_shape_mismatch_raises():
    import pytest

    with pytest.raises(ValueError):
        audit_from_guesses(np.zeros(3), np.zeros(4))
