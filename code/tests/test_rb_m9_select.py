"""R2 helpers: the random projection is orthonormal, seeded and data-independent; the selection rule is
the paper's (first strict maximum over p ascending, then lambda ascending)."""
import numpy as np

from p3fcl.rebuttal.m9_select import random_orthonormal, select_first_max


def test_random_orthonormal_is_orthonormal_and_deterministic():
    Q = random_orthonormal(50, 8, 0)
    assert Q.shape == (50, 8)
    np.testing.assert_allclose(Q.T @ Q, np.eye(8), atol=1e-12)
    np.testing.assert_array_equal(Q, random_orthonormal(50, 8, 0))
    assert not np.allclose(Q, random_orthonormal(50, 8, 1))


def _paper_two_stage(grid):
    """`run_m9_hparam_selection.main`'s logic: best lambda within each p (strict >), then best across p (strict >)."""
    per_p = {}
    for g in grid:
        if g["p"] not in per_p or g["acc"] > per_p[g["p"]]["acc"]:
            per_p[g["p"]] = g
    best = None
    for p in sorted(per_p):
        if best is None or per_p[p]["acc"] > best["acc"]:
            best = per_p[p]
    return best


def test_select_first_max_matches_paper_two_stage_rule_including_ties():
    r = np.random.default_rng(0)
    for _ in range(200):
        grid = [{"p": p, "lambda": lam, "acc": float(r.integers(0, 4))} for p in (32, 64, 128) for lam in (0.01, 1.0, 100.0)]
        assert select_first_max(grid) is _paper_two_stage(grid)
