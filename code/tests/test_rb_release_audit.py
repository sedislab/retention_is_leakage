"""R1 release audit (rebuttal/R1_PROTOCOL.md): the cheap-path statistics must equal the methods' own
releases. These run on synthetic data; `scripts/rb_r1_equivalence.py` repeats the check on real
features and masked paper streams inside PBS."""
import numpy as np
import pytest
from scipy.stats import norm

from p3fcl import streams
from p3fcl.artifacts import Family
from p3fcl.methods.m2_target import TARGET
from p3fcl.methods.m3_fot import FOT
from p3fcl.rebuttal import release_audit as ra
from p3fcl.shadow_runner import _apply_population_mask, _population_mask, build_targets

N_TASKS, N_CLIENTS, D, N_CLASSES = 4, 3, 12, 8


def _data(seed=0, n=240):
    r = np.random.default_rng(seed)
    y = np.repeat(np.arange(N_CLASSES), n // N_CLASSES)
    X = r.standard_normal((n, D)) + 0.5 * y[:, None] * r.standard_normal(D)[None, :]
    return X, y


def _masked_stream(X, y, shadow_id=3, method="m2_target"):
    base = streams.build_stream(y, np.arange(len(y)), n_tasks=N_TASKS, n_clients=N_CLIENTS, beta=0.5, seed=1)
    mask = _population_mask("synthetic", method, shadow_id, len(y), 0.5)
    return base, _apply_population_mask(base, mask)


def test_m2_release_equals_target_fit_task():
    X, y = _data()
    _, stream = _masked_stream(X, y)
    m = TARGET({"n_classes": N_CLASSES, "feature_dim": D, "local_epochs": 2, "lr": 0.5})
    r = np.random.default_rng(0)
    worst = 0.0
    for t in range(N_TASKS):
        recs = m.fit_task(t, X, y, ids=None, client_shards=stream[t], rng=r)
        client_stats, generators = ra.m2_task_release(X, y, stream[t])
        gen_recs = [rec for rec in recs if rec.family == Family.GENERATIVE]
        n_payload = 0
        for rec in gen_recs:
            for key, val in rec.payload.items():
                kind, c = key.split("_")
                mean, var = client_stats[(rec.client, int(c))]
                worst = max(worst, np.max(np.abs((mean if kind == "mean" else var) - val)))
                n_payload += 1
        assert n_payload == 2 * len(client_stats)
        for c, (gm, gv) in generators.items():
            mm, mv = m._generators[c]
            worst = max(worst, np.max(np.abs(gm - mm)), np.max(np.abs(gv - mv)))
    assert worst < 1e-10


def test_m3_release_equals_fot_fit_task():
    X, y = _data(1)
    _, stream = _masked_stream(X, y, method="m3_fot")
    m = FOT({"n_classes": N_CLASSES, "feature_dim": D, "local_epochs": 2, "lr": 0.5, "subspace_rank": 3})
    r = np.random.default_rng(0)
    for t in range(N_TASKS):
        recs = m.fit_task(t, X, y, ids=None, client_shards=stream[t], rng=r)
        R, V = ra.m3_task_release(X, stream[t], rank=3)
        gram = [rec for rec in recs if rec.family == Family.GRAM]
        assert len(gram) == 1
        assert np.max(np.abs(gram[0].payload["R"] - R)) < 1e-10
        # The new directions lie in the method's updated projection subspace.
        assert np.max(np.abs(V - m.U @ (m.U.T @ V))) < 1e-10


def test_diag_gauss_loglik_matches_scipy():
    r = np.random.default_rng(2)
    f, mean, var = r.standard_normal(5), r.standard_normal(5), r.uniform(0.1, 2.0, 5)
    assert ra.diag_gauss_loglik(f, mean, var) == pytest.approx(norm.logpdf(f, mean, np.sqrt(var)).sum(), abs=1e-12)


def test_m3_scores_match_explicit_formulas():
    r = np.random.default_rng(3)
    A = r.standard_normal((30, 6))
    R = A.T @ A
    V = np.linalg.qr(r.standard_normal((6, 2)))[0]
    G = r.standard_normal((4, 6))
    frac = ra.subspace_energy_fraction(G, V)
    assert np.all((frac >= 0) & (frac <= 1 + 1e-12))
    for i in range(4):
        g = G[i]
        assert frac[i] == pytest.approx(np.linalg.norm(V.T @ g) ** 2 / np.linalg.norm(g) ** 2)
        assert ra.ridge_leverage(G, R, 1.0)[i] == pytest.approx(g @ np.linalg.inv(R + np.eye(6)) @ g)


@pytest.mark.parametrize("method", ["m2_target", "m3_fot"])
def test_release_scores_nan_before_task_and_constant_after(method):
    X, y = _data(4)
    base, stream = _masked_stream(X, y, method=method)
    targets = build_targets(base, 2, 1)
    task_classes = streams.class_incremental_tasks(y, N_TASKS, 1)
    centres = ra.ref_task_centres(X, y, task_classes)
    out = ra.release_scores(method, X, y, stream, targets, ref_centres=centres)
    assert set(out) == set(ra.RELEASE_VIEWS[method])
    for view, s in out.items():
        assert s.shape == (len(targets), N_TASKS)
        for j, t in enumerate(targets):
            k = t["task"]
            assert np.all(np.isnan(s[j, :k]))
            row = s[j, k:]
            assert np.all(np.isnan(row)) or np.all(row == row[0])
        assert np.isfinite(s).any()


def test_m2_client_view_nan_when_class_absent_from_client():
    X, y = _data(5)
    shard = streams.Shard(client=0, task=0, ids=(0, 1, 2))  # all class 0
    targets = [{"target_id": 40, "client": 0, "task": 0}]   # class 1, not in client 0's shard
    out = ra.release_scores("m2_target", X, y, [[shard]], targets)
    assert np.isnan(out["client"][0, 0]) and np.isnan(out["broadcast"][0, 0])


def test_combine_log_lr_counts_release_once_or_per_round():
    Lm = np.array([[[np.nan, 1.0, 2.0, 3.0]]])
    Lb = np.array([[[np.nan, 0.5, 0.5, 0.5]]])
    lit = ra.combine_log_lr(Lm, Lb, once=False)
    once = ra.combine_log_lr(Lm, Lb, once=True)
    np.testing.assert_allclose(lit["log_lr"][0, 0, 1:], [1.5, 2.5, 3.5])
    np.testing.assert_allclose(lit["trajectory"][0, 0, 1:], [1.5, 4.0, 7.5])
    np.testing.assert_allclose(once["log_lr"][0, 0, 1:], [1.5, 2.5, 3.5])
    np.testing.assert_allclose(once["trajectory"][0, 0, 1:], [1.5, 3.5, 6.5])
    assert np.isnan(lit["trajectory"][0, 0, 0]) and np.isnan(once["trajectory"][0, 0, 0])
    # A missing release LLR contributes nothing.
    np.testing.assert_allclose(ra.combine_log_lr(Lm, np.full_like(Lb, np.nan), once=True)["trajectory"][0, 0, 1:], [1.0, 3.0, 6.0])


def test_m3_subspace_exact_on_float32_features():
    """Real features are float32 (R1_PROTOCOL.md amendment 1): V must still equal FOT's own
    eigendecomposition of its released R exactly; only the span-in-U check is float32-limited."""
    X, y = _data(6)
    X = X.astype(np.float32)
    _, stream = _masked_stream(X, y, method="m3_fot")
    m = FOT({"n_classes": N_CLASSES, "feature_dim": D, "local_epochs": 1, "lr": 0.5, "subspace_rank": 3})
    r = np.random.default_rng(0)
    for t in range(N_TASKS):
        recs = m.fit_task(t, X, y, ids=None, client_shards=stream[t], rng=r)
        R, V = ra.m3_task_release(X, stream[t], rank=3)
        payload = [rec for rec in recs if rec.family == Family.GRAM][0].payload["R"]
        ev, evec = np.linalg.eigh(payload)
        np.testing.assert_array_equal(V, evec[:, np.argsort(ev)[::-1][:3]])
        assert np.max(np.abs(V - m.U @ (m.U.T @ V))) < 1e-5
