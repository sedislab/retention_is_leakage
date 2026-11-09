"""R3 helpers (rebuttal/R3_PROTOCOL.md)."""
import numpy as np

from p3fcl import streams
from p3fcl.methods.m9_contractive import ContractiveDPAnalytic
from p3fcl.rebuttal import client_scaling as cs
from p3fcl.rng import seeded


def _data(n=400, d=10, k=8, seed=0):
    r = np.random.default_rng(seed)
    y = np.repeat(np.arange(k), n // k)
    return r.standard_normal((n, d)), y


def test_mean_client_size():
    assert cs.mean_client_size(40000, 10, 10) == 400
    assert cs.mean_client_size(40000, 10, 2000) == 2
    assert cs.mean_client_size(4994, 10, 100) == 5
    assert cs.mean_client_size(100, 10, 1000) == 1


def test_pseudo_clients_have_size_m_task_classes_and_are_seeded():
    X, y = _data()
    tasks = [[0, 1, 2, 3], [4, 5, 6, 7]]
    ids = cs.pseudo_client_ids(y, tasks, m=7, dataset="syn", n=10, n_per_task=5)
    assert len(ids) == 10
    for j, a in enumerate(ids):
        assert len(a) == 7
        assert set(y[a]) <= set(tasks[j // 5])
    again = cs.pseudo_client_ids(y, tasks, m=7, dataset="syn", n=10, n_per_task=5)
    assert all(np.array_equal(a, b) for a, b in zip(ids, again))


def test_pair_norm_grows_with_client_size():
    X, y = _data()
    P = np.linalg.qr(np.random.default_rng(1).standard_normal((10, 4)))[0]
    tasks = [list(range(8))]
    small = cs.size_matched_clip(X, y, cs.pseudo_client_ids(y, tasks, 5, "syn", 1, 50), P, 8)
    big = cs.size_matched_clip(X, y, cs.pseudo_client_ids(y, tasks, 50, "syn", 2, 50), P, 8)
    assert big > 3 * small


def test_u2_diagnostics_match_m9_noiseless_aggregate():
    X, y = _data()
    stream = streams.build_stream(y, np.arange(len(y)), n_tasks=2, n_clients=5, beta=0.5, seed=0)
    P = np.linalg.qr(np.random.default_rng(2).standard_normal((10, 4)))[0]
    C = 3.0
    m = ContractiveDPAnalytic({"n_classes": 8, "pca_basis": P, "unit": "U2", "eps": float("inf"), "clip_C": C, "gamma": 1.0})
    m.fit_task(0, X, y, ids=None, client_shards=stream[0], rng=seeded("x", 0))
    rec = m.R  # eps = inf: R after task 0 is the noiseless clipped aggregate G_sum
    diag = cs.u2_diagnostics(X, y, stream, P, 8, C, clip=True)
    H = m.Q
    assert abs(diag["agg0_frob"] - float(np.sqrt(np.sum(rec**2) + np.sum(H**2)))) < 1e-10
    assert 0.0 <= diag["frac_clipped"] <= 1.0
    assert cs.u2_diagnostics(X, y, stream, P, 8, C, clip=False)["frac_clipped"] is None
