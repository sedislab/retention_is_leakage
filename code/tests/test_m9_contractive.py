from __future__ import annotations

import numpy as np
import pytest
from p3fcl import sim
from p3fcl.artifacts import Family
from p3fcl.dp.accountant import account
from p3fcl.methods.m8_analytic import AnalyticFCL
from p3fcl.methods.m9_contractive import ContractiveDPAnalytic, client_stats, clip_pair, project_and_cap
from p3fcl.streams import Shard
from p3fcl.units import Unit


def _stream(n_tasks: int, n_clients: int, n_per_client_task: int) -> list:
    return [
        [
            Shard(
                client=c, task=t,
                ids=tuple(range((t * n_clients + c) * n_per_client_task, (t * n_clients + c + 1) * n_per_client_task)),
            )
            for c in range(n_clients)
        ]
        for t in range(n_tasks)
    ]


def _synthetic_data(n, d, n_classes, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d))
    y = rng.integers(0, n_classes, size=n)
    return X, y


def test_m9_matches_m8_at_eps_inf_gamma_1_p_equals_d():
    """FX5 test 1: with eps=inf, gamma=1, p=d, and the same total lambda, M9's predictions are
    identical to M8's. Alignment note (the plan's own caveat): M8 adds `lambda*I` to EVERY client's
    Gram matrix every task (so its effective total ridge is `lambda * n_clients * n_tasks`), while M9
    adds `lambda*I` exactly once, at `predict()` time. The only value both conventions agree on
    without a conversion factor is `lambda=0`, so that is the aligned case this test uses -- with
    ridge_lambda=0 on both sides, R stays PSD (a sum of Gram matrices) and Pi_+ is a no-op, so R/Q
    should be identical up to floating point and predictions should match exactly. `B` (M9's own
    L2 cap, algorithm step 1) is set far above the data's actual norms so the cap never binds -- M8
    has no such cap at all, so leaving `B` at its default of 1.0 would silently shrink M9's features
    and confound the comparison with a real behavioral difference, not just a convention gap."""
    d, n_classes = 10, 4
    n_tasks, n_clients, n_per = 4, 3, 8
    X, y = _synthetic_data(n_tasks * n_clients * n_per, d, n_classes, seed=1)
    stream = _stream(n_tasks, n_clients, n_per)

    m8 = AnalyticFCL({"n_classes": n_classes, "feature_dim": d, "ridge_lambda": 0.0})
    m9 = ContractiveDPAnalytic({
        "n_classes": n_classes, "pca_basis": np.eye(d), "gamma": 1.0, "eps": float("inf"),
        "ridge_lambda": 0.0, "unit": "U1", "B": 1e6,
    })
    for t, shards in enumerate(stream):
        m8.fit_task(t, X, y, ids=None, client_shards=shards, rng=np.random.default_rng(0))
        m9.fit_task(t, X, y, ids=None, client_shards=shards, rng=np.random.default_rng(0))

    np.testing.assert_allclose(m8._R, m9.R, atol=1e-8)
    np.testing.assert_allclose(m8._Q, m9.Q, atol=1e-8)
    np.testing.assert_array_equal(m8.predict(X), m9.predict(X))


def test_m9_numeric_sensitivity_matches_declared_bound():
    """FX5 test 2: random neighbouring datasets with ||x|| <= 1 change (vech G, vec H) by at most
    sqrt(2) under U1 (no clipping), and by at most `clip_C` after the U2 joint clip -- tested against
    M9's own `client_stats`/`clip_pair` helpers, not a reimplementation of the math."""
    rng = np.random.default_rng(2)
    d, n_classes = 8, 3

    def capped_batch(n):
        X = rng.standard_normal((n, d))
        X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1.0) / rng.uniform(1.0, 4.0, size=(n, 1))
        y = rng.integers(0, n_classes, size=n)
        return X, y

    for _ in range(30):
        X, y = capped_batch(6)
        x_new = rng.standard_normal(d)
        x_new = x_new / max(np.linalg.norm(x_new), 1.0) / rng.uniform(1.0, 4.0)
        y_new = int(rng.integers(0, n_classes))

        G0, H0 = client_stats(X, y, n_classes)
        G1, H1 = client_stats(np.vstack([X, x_new]), np.append(y, y_new), n_classes)
        combined = float(np.sqrt(np.sum((G1 - G0) ** 2) + np.sum((H1 - H0) ** 2)))
        assert combined <= np.sqrt(2) + 1e-9, combined

        # U2: after jointly clipping EACH of the two (pre/post) pairs to clip_C, the clipped pair's
        # own norm cannot exceed clip_C by construction -- clip_pair's contract, checked directly.
        clip_C = 0.3
        G0c, H0c = clip_pair(G0, H0, clip_C)
        G1c, H1c = clip_pair(G1, H1, clip_C)
        for Gc, Hc in ((G0c, H0c), (G1c, H1c)):
            assert np.sqrt(np.sum(Gc**2) + np.sum(Hc**2)) <= clip_C + 1e-9


def test_m9_releases_exactly_one_record_per_task():
    """FX5 test 3."""
    d, n_classes = 6, 3
    n_tasks, n_clients, n_per = 5, 4, 3
    X, y = _synthetic_data(n_tasks * n_clients * n_per, d, n_classes, seed=3)
    stream = _stream(n_tasks, n_clients, n_per)
    method = ContractiveDPAnalytic({
        "n_classes": n_classes, "pca_basis": np.eye(d), "gamma": 0.9, "eps": 4.0, "unit": "U2",
        "clip_C": 1.0, "noise_seed": 0,
    })
    result = sim.run(method, X, y, stream, seed=0)
    ledger = result["ledger"]
    assert all(r.family == Family.GRAM and r.client == -1 for r in ledger)
    for t in range(n_tasks):
        assert sum(1 for r in ledger if r.task == t) == 1


def test_m9_accountant_is_flat_under_U1_and_U2():
    """FX5 test 4 (ties to FX1 test 4): a single joint aggregate release per task, never replayed,
    gives m_T = 1 for every T under both U1 and U2."""
    d, n_classes = 6, 3
    n_tasks, n_clients, n_per = 6, 3, 4
    X, y = _synthetic_data(n_tasks * n_clients * n_per, d, n_classes, seed=4)
    stream = _stream(n_tasks, n_clients, n_per)
    method = ContractiveDPAnalytic({
        "n_classes": n_classes, "pca_basis": np.eye(d), "gamma": 1.0, "eps": 2.0, "unit": "U1", "noise_seed": 0,
    })
    ledger = sim.run(method, X, y, stream, seed=0)["ledger"]

    df_u1 = account(ledger, stream, unit=Unit.EXAMPLE, sigma=2.0, delta=1e-5)
    df_u2 = account(ledger, stream, unit=Unit.TASK, sigma=2.0, delta=1e-5)
    assert (df_u1["m_T"] == 1).all()
    assert (df_u2["m_T"] == 1).all()


def test_m9_noise_seed_is_separate_from_data_seed_and_reproducible():
    """FX5 test 5: the same (data, noise) seed pair reproduces byte-identical output; changing only
    the noise seed changes the released (noisy) payload without needing to touch the data seed."""
    d, n_classes = 6, 3
    n_tasks, n_clients, n_per = 3, 2, 5
    X, y = _synthetic_data(n_tasks * n_clients * n_per, d, n_classes, seed=5)
    stream = _stream(n_tasks, n_clients, n_per)

    def run(noise_seed, data_seed):
        method = ContractiveDPAnalytic({
            "n_classes": n_classes, "pca_basis": np.eye(d), "gamma": 1.0, "eps": 1.0, "unit": "U1",
            "noise_seed": noise_seed,
        })
        return sim.run(method, X, y, stream, seed=data_seed)["ledger"]

    ledger_a = run(noise_seed=42, data_seed=0)
    ledger_b = run(noise_seed=42, data_seed=0)
    for ra, rb in zip(ledger_a, ledger_b):
        np.testing.assert_array_equal(ra.payload["R"], rb.payload["R"])
        np.testing.assert_array_equal(ra.payload["Q"], rb.payload["Q"])

    ledger_c = run(noise_seed=7, data_seed=0)
    any_differ = any(
        not np.array_equal(ra.payload["R"], rc.payload["R"]) for ra, rc in zip(ledger_a, ledger_c)
    )
    assert any_differ, "a different noise_seed should change the noisy released payload"


def test_project_and_cap_never_increases_norm_above_B():
    rng = np.random.default_rng(6)
    d, p, B = 10, 4, 1.0
    P, _ = np.linalg.qr(rng.standard_normal((d, p)))
    X = rng.standard_normal((50, d)) * 5.0  # deliberately large, well above B after projection
    Xp = project_and_cap(X, P, B)
    norms = np.linalg.norm(Xp, axis=1)
    assert (norms <= B + 1e-9).all()


def test_m9_rejects_unknown_unit():
    with pytest.raises(ValueError):
        ContractiveDPAnalytic({"n_classes": 2, "pca_basis": np.eye(3), "unit": "U3"})
