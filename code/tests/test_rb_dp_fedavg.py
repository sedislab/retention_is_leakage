"""R5 baselines (rebuttal/R5_PROTOCOL.md): m0_dpsgd_u1 and m0_dpfedavg_u2."""
import numpy as np
import pytest

from p3fcl import sim, streams
from p3fcl.dp.accountant import account, eps_gaussian_composed
from p3fcl.dp.mechanisms import analytic_gaussian_sigma
from p3fcl.methods.m0_dpfedavg_u2 import DPFedAvgU2, mcmahan_weights
from p3fcl.methods.m0_dpsgd_u1 import DPSGDFedAvgU1, noise_multiplier_for, per_example_grad_norms
from p3fcl.methods.m0_fedavg import FedAvgSequential
from p3fcl.units import Unit

D, K = 12, 8


def _setup(seed=0, n=320):
    r = np.random.default_rng(seed)
    y = np.repeat(np.arange(K), n // K)
    X = r.standard_normal((n, D)) + 0.7 * r.standard_normal((K, D))[y]
    stream = streams.build_stream(y, np.arange(n), n_tasks=4, n_clients=5, beta=0.5, seed=seed)
    return X, y, stream


BASE = {"n_classes": K, "feature_dim": D, "local_epochs": 5, "lr": 0.5}


def test_dpsgd_without_clip_or_noise_is_m0_exactly():
    X, y, stream = _setup()
    a = sim.run(FedAvgSequential(BASE), X, y, stream, seed=0)
    b = sim.run(DPSGDFedAvgU1(BASE), X, y, stream, seed=0)
    for wa, wb in zip(a["w_history"], b["w_history"]):
        np.testing.assert_array_equal(wa, wb)


def test_dpfedavg_weighted_without_clip_or_noise_is_m0_exactly():
    X, y, stream = _setup(1)
    a = sim.run(FedAvgSequential(BASE), X, y, stream, seed=0)
    b = sim.run(DPFedAvgU2({**BASE, "aggregation": "weighted"}), X, y, stream, seed=0)
    for wa, wb in zip(a["w_history"], b["w_history"]):
        np.testing.assert_array_equal(wa, wb)


def test_dpfedavg_weighted_mode_refuses_noise_or_clip():
    with pytest.raises(ValueError):
        DPFedAvgU2({**BASE, "aggregation": "weighted", "eps": 1.0, "clip_S": 1.0})


def test_per_example_norm_formula_and_clip_bound():
    r = np.random.default_rng(2)
    X, G = r.standard_normal((6, D)), r.standard_normal((6, K))
    for i in range(6):
        assert per_example_grad_norms(X, G)[i] == pytest.approx(np.linalg.norm(np.outer(X[i], G[i])))
    Cg = 0.3
    Gc = G * np.minimum(1.0, Cg / per_example_grad_norms(X, G))[:, None]
    assert np.all(per_example_grad_norms(X, Gc) <= Cg + 1e-12)


@pytest.mark.parametrize("eps0", [1.0, 4.0, 8.0])
def test_noise_multiplier_hits_target_eps(eps0):
    z = noise_multiplier_for(eps0, 30, 1e-5)
    assert eps_gaussian_composed(z, 30, 1e-5) <= eps0 + 1e-9
    assert eps_gaussian_composed(z * 0.999, 30, 1e-5) > eps0
    assert noise_multiplier_for(float("inf"), 30, 1e-5) == 0.0


def test_dpfedavg_sigma_is_analytic_gaussian():
    m = DPFedAvgU2({**BASE, "eps": 4.0, "clip_S": 0.7, "w_cap": 50.0})
    assert m.sigma == analytic_gaussian_sigma(4.0, 1e-5, 0.7)


def test_dp_runs_are_seeded_and_noise_changes_the_model():
    X, y, stream = _setup(3)
    cfg = {**BASE, "clip_Cg": 0.5, "noise_multiplier": 2.0, "noise_seed": 7}
    a = sim.run(DPSGDFedAvgU1(cfg), X, y, stream, seed=0)["w_history"][-1]
    b = sim.run(DPSGDFedAvgU1(cfg), X, y, stream, seed=0)["w_history"][-1]
    c = sim.run(DPSGDFedAvgU1({**cfg, "noise_seed": 8}), X, y, stream, seed=0)["w_history"][-1]
    np.testing.assert_array_equal(a, b)
    assert not np.allclose(a, c)


def test_ledger_count_is_one_per_unit_instance():
    X, y, stream = _setup(4)
    led1 = sim.run(DPSGDFedAvgU1({**BASE, "clip_Cg": 0.5, "noise_multiplier": 1.0}), X, y, stream, seed=0)["ledger"]
    assert account(led1, stream, Unit.EXAMPLE, sigma=1.0, delta=1e-5)["m_T"].max() == 1
    led2 = sim.run(DPFedAvgU2({**BASE, "eps": 4.0, "clip_S": 0.5, "w_cap": 15.0}), X, y, stream, seed=0)["ledger"]
    assert account(led2, stream, Unit.TASK, sigma=1.0, delta=1e-5)["m_T"].max() == 1
    assert all(r.client == -1 for r in led2)
    assert frozenset().union(*(r.touched for r in led2)) == frozenset(range(len(y)))


def test_fixed_weighted_without_cap_clip_or_noise_matches_m0():
    """Amendment 1 gate condition: if w_cap >= every client size, the fixed-denominator mean is M0's
    size-weighted mean (equal up to float rounding)."""
    X, y, stream = _setup(5)
    cap = max(len(s.ids) for t in stream for s in t)
    a = sim.run(FedAvgSequential(BASE), X, y, stream, seed=0)
    b = sim.run(DPFedAvgU2({**BASE, "w_cap": float(cap)}), X, y, stream, seed=0)
    for wa, wb in zip(a["w_history"], b["w_history"]):
        np.testing.assert_allclose(wa, wb, rtol=1e-10, atol=1e-12)


def test_mcmahan_weights_capped_and_sum_sensitivity_is_S():
    np.testing.assert_array_equal(mcmahan_weights([5, 20, 80], 20.0), [0.25, 1.0, 1.0])
    X, y, stream = _setup(6)
    S = 0.05
    shards = stream[0]

    def weighted_sum(sh):
        m = DPFedAvgU2({**BASE, "w_cap": 10.0, "clip_S": S})  # eps = inf: noiseless
        rec = m.fit_task(0, X, y, ids=None, client_shards=sh, rng=None)[0]
        return rec.payload * rec.meta["W_fixed"]

    full = weighted_sum(shards)
    for drop in range(len(shards)):
        # Removing one client moves the (pre-noise) weighted sum by at most S.
        assert np.linalg.norm(full - weighted_sum(shards[:drop] + shards[drop + 1:])) <= S * (1 + 1e-9)


def test_fixed_weighted_requires_cap():
    with pytest.raises(ValueError):
        DPFedAvgU2({**BASE})


def test_public_weight_cap_is_deterministic_and_in_range():
    from p3fcl.rebuttal.client_scaling import public_client_sizes, public_weight_cap
    y_ref = np.repeat(np.arange(20), 10)
    sizes = public_client_sizes(y_ref, 4000, 4, 5, 0.5, seeds=[1000, 1001])
    assert sizes.sum() == 2 * 4000
    cap = public_weight_cap(y_ref, 4000, 4, 5, 0.5)
    assert cap == public_weight_cap(y_ref, 4000, 4, 5, 0.5)
    assert 4000 / (4 * 5) < cap <= sizes.max() * 10
