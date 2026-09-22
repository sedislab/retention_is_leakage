from __future__ import annotations

import numpy as np
import pytest
from p3fcl.attacks.inversion import (
    canonical_recovery,
    cosine_similarity_rows,
    feature_anisotropy,
    gram_to_subspace,
    oracle_reconstruction_selfcheck,
    orthogonal_procrustes,
    reference_based_recovery,
)


def test_orthogonal_procrustes_identity_case():
    rng = np.random.default_rng(0)
    A = rng.standard_normal((5, 8))
    U = orthogonal_procrustes(A, A)
    assert U.shape == (5, 5)
    np.testing.assert_allclose(U, np.eye(5), atol=1e-6)


def test_orthogonal_procrustes_recovers_known_rotation():
    rng = np.random.default_rng(1)
    A = rng.standard_normal((6, 10))
    Q, _ = np.linalg.qr(rng.standard_normal((6, 6)))
    B = Q @ A
    U = orthogonal_procrustes(A, B)
    np.testing.assert_allclose(U @ A, B, atol=1e-6)


def test_gram_to_subspace_exact_at_n1():
    rng = np.random.default_rng(42)
    d = 16
    x = rng.standard_normal(d)
    R = np.outer(x, x) + 1e-8 * np.eye(d)
    top_vecs, sv = gram_to_subspace(R, ridge_lambda=1e-8, n_samples=1)
    recovered = canonical_recovery(top_vecs, sv)
    cos = abs(cosine_similarity_rows(recovered, x[None, :])[0])
    assert cos == pytest.approx(1.0, abs=1e-6)


def test_oracle_selfcheck_is_perfect_for_any_n():
    """CLAUDE.md-relevant algebraic fact (RESEARCH_PLAN.md §3.4): the Gram statistic loses no
    information about the span, ever -- oracle reconstruction against the true targets must be ~1.0
    cosine similarity for every sample, at every n, whenever n <= d."""
    rng = np.random.default_rng(0)
    for n in (1, 2, 5, 10):
        d = 32
        X = rng.standard_normal((n, d))
        R = X.T @ X + 1e-6 * np.eye(d)
        top_vecs, sv = gram_to_subspace(R, ridge_lambda=1e-6, n_samples=n)
        recovered = canonical_recovery(top_vecs, sv)
        cos = oracle_reconstruction_selfcheck(recovered, X)
        np.testing.assert_allclose(cos, np.ones(n), atol=1e-5)


def test_reference_based_recovery_shape_and_range():
    rng = np.random.default_rng(0)
    n, d = 4, 16
    X = rng.standard_normal((n, d))
    R = X.T @ X + 1e-6 * np.eye(d)
    top_vecs, sv = gram_to_subspace(R, ridge_lambda=1e-6, n_samples=n)
    recovered = canonical_recovery(top_vecs, sv)
    ref_pool = np.concatenate([X, rng.standard_normal((20, d))], axis=0)
    rotated = reference_based_recovery(recovered, ref_pool)
    assert rotated.shape == (n, d)
    cos = cosine_similarity_rows(rotated, X)
    assert np.all(cos >= -1.0 - 1e-9) and np.all(cos <= 1.0 + 1e-9)


def test_reference_based_recovery_finds_exact_targets_when_pool_is_a_clean_anisotropic_subset():
    # a favourable, structured (non-isotropic) case: reference pool = a small class-clustered set
    # including the true targets, well separated from distractors -- recovery should be excellent
    # here even though the isotropic-Gaussian stress case (this module's docstring) is not.
    rng = np.random.default_rng(3)
    d = 8
    center = rng.standard_normal(d) * 5
    X = center + rng.standard_normal((3, d)) * 0.05  # tight cluster = low effective dimension
    R = X.T @ X + 1e-8 * np.eye(d)
    top_vecs, sv = gram_to_subspace(R, ridge_lambda=1e-8, n_samples=3)
    recovered = canonical_recovery(top_vecs, sv)
    ref_pool = np.concatenate([X, center + rng.standard_normal((10, d)) * 0.05], axis=0)
    rotated = reference_based_recovery(recovered, ref_pool, n_iters=30)
    cos = cosine_similarity_rows(rotated, X)
    assert np.mean(cos) > 0.9, cos


def test_feature_anisotropy_higher_for_clustered_than_isotropic():
    rng = np.random.default_rng(0)
    isotropic = rng.standard_normal((200, 16))
    clustered = np.concatenate([
        rng.standard_normal((100, 16)) * 0.1 + 5,
        rng.standard_normal((100, 16)) * 0.1 - 5,
    ])
    assert feature_anisotropy(clustered) > feature_anisotropy(isotropic)
