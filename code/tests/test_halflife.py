from __future__ import annotations

import numpy as np
import pytest
from p3fcl.halflife import (
    acc_floor_class_incremental,
    acc_floor_majority_class,
    decoupling_ratio,
    half_life,
    has_signal,
    hierarchical_bootstrap,
    leak_floor,
    no_signal_result,
    normalize,
)


def test_leak_floors_match_the_spec():
    assert leak_floor("tpr1") == 0.01
    assert leak_floor("tpr01") == 0.001
    assert leak_floor("auc") == 0.5
    with pytest.raises(ValueError):
        leak_floor("bogus")


def test_acc_floor_class_incremental():
    assert acc_floor_class_incremental(10) == pytest.approx(0.1)
    with pytest.raises(ValueError):
        acc_floor_class_incremental(0)


def test_acc_floor_majority_class():
    y = [0, 0, 0, 1, 1]
    assert acc_floor_majority_class(y) == pytest.approx(0.6)


def test_has_signal_threshold():
    # leak: threshold 0.005
    assert has_signal(base_value=0.5, base_ci_lo=0.02, floor=0.01, threshold=0.005)  # 0.02-0.01=0.01 > 0.005
    assert not has_signal(base_value=0.5, base_ci_lo=0.012, floor=0.01, threshold=0.005)  # 0.002 <= 0.005
    # acc: threshold 0.02
    assert has_signal(base_value=0.9, base_ci_lo=0.5, floor=0.1, threshold=0.02)  # 0.4 > 0.02
    assert not has_signal(base_value=0.9, base_ci_lo=0.11, floor=0.1, threshold=0.02)  # 0.01 <= 0.02


def test_normalize_base_is_one_and_scales_linearly():
    values = {0: 0.9, 1: 0.7, 2: 0.5, 3: 0.3}
    norm = normalize(values, floor=0.1)
    assert norm[0] == pytest.approx(1.0)
    assert norm[1] == pytest.approx((0.7 - 0.1) / (0.9 - 0.1))
    assert norm[3] == pytest.approx((0.3 - 0.1) / (0.9 - 0.1))


def test_normalize_rejects_nonpositive_base():
    with pytest.raises(ValueError):
        normalize({0: 0.1, 1: 0.05}, floor=0.1)  # base = 0.1-0.1 = 0, not > 0


def test_normalize_requires_e0():
    with pytest.raises(ValueError):
        normalize({1: 0.5}, floor=0.1)


def test_half_life_exact_crossing_at_integer_e():
    # normalized value hits exactly 0.5 at e=2 -> halflife == 2.0, halflife_int == 2
    norm = {0: 1.0, 1: 0.75, 2: 0.5, 3: 0.4, 4: 0.3, 5: 0.2, 6: 0.1}
    result = half_life(norm, E=6)
    assert result["status"] == "ok"
    assert result["halflife"] == pytest.approx(2.0)
    assert result["halflife_int"] == 2


def test_half_life_interpolates_between_integers():
    # crosses between e=1 (0.6) and e=2 (0.4): frac = (0.6-0.5)/(0.6-0.4) = 0.5 -> halflife = 1.5
    norm = {0: 1.0, 1: 0.6, 2: 0.4, 3: 0.3, 4: 0.2, 5: 0.1, 6: 0.05}
    result = half_life(norm, E=6)
    assert result["status"] == "ok"
    assert result["halflife"] == pytest.approx(1.5)
    assert result["halflife_int"] == 2


def test_half_life_censored_at_E_when_no_crossing():
    norm = {0: 1.0, 1: 0.95, 2: 0.9, 3: 0.85, 4: 0.8, 5: 0.75, 6: 0.7}
    result = half_life(norm, E=6)
    assert result["status"] == "censored"
    assert result["halflife"] == pytest.approx(6.0)
    assert result["halflife_int"] is None


def test_half_life_never_reports_infinity():
    norm = {e: 1.0 for e in range(7)}  # perfectly flat, never decays at all
    result = half_life(norm, E=6)
    assert result["status"] == "censored"
    assert result["halflife"] == 6.0
    import math
    assert math.isfinite(result["halflife"])


def test_half_life_requires_every_integer_e_present():
    with pytest.raises(ValueError):
        half_life({0: 1.0, 1: 0.5}, E=6)  # missing e=2..6


def test_decoupling_ratio_point_case():
    h_leak = {"halflife": 1.5, "halflife_int": 2, "status": "ok"}
    h_acc = {"halflife": 4.5, "halflife_int": 5, "status": "ok"}
    result = decoupling_ratio(h_leak, h_acc)
    assert result["ratio_type"] == "point"
    assert result["ratio"] == pytest.approx(1.5 / 4.5)


def test_decoupling_ratio_lower_bound_when_leak_censored():
    h_leak = {"halflife": 6.0, "halflife_int": None, "status": "censored"}
    h_acc = {"halflife": 2.0, "halflife_int": 2, "status": "ok"}
    result = decoupling_ratio(h_leak, h_acc)
    assert result["ratio_type"] == "lower_bound"
    assert result["ratio"] == pytest.approx(3.0)


def test_decoupling_ratio_undefined_when_acc_censored():
    h_leak = {"halflife": 1.5, "halflife_int": 2, "status": "ok"}
    h_acc = {"halflife": 6.0, "halflife_int": None, "status": "censored"}
    result = decoupling_ratio(h_leak, h_acc)
    assert result["ratio_type"] == "undefined"
    assert result["ratio"] is None


def test_decoupling_ratio_undefined_when_either_side_no_signal():
    h_leak = no_signal_result()
    h_acc = {"halflife": 2.0, "halflife_int": 2, "status": "ok"}
    result = decoupling_ratio(h_leak, h_acc)
    assert result["ratio_type"] == "undefined"


def test_decoupling_ratio_by_construction_overrides_everything():
    h_leak = {"halflife": 1.5, "halflife_int": 2, "status": "ok"}
    h_acc = {"halflife": 6.0, "halflife_int": None, "status": "censored"}
    result = decoupling_ratio(h_leak, h_acc, by_construction=True)
    assert result["ratio_type"] == "by_construction"
    assert result["ratio"] is None


def test_no_signal_result_shape_matches_half_life_keys():
    ns = no_signal_result()
    assert set(ns.keys()) == {"halflife", "halflife_int", "status"}
    assert ns["status"] == "no_signal"
    assert ns["halflife"] is None


# ---------------------------------------------------------------------------------------------
# hierarchical_bootstrap
# ---------------------------------------------------------------------------------------------


def _identity_resample(seed_data, rng):
    # inner resample: sample len(seed_data) targets with replacement from seed_data itself
    idx = rng.integers(0, len(seed_data), size=len(seed_data))
    return [seed_data[i] for i in idx]


def test_hierarchical_bootstrap_recovers_a_known_mean_with_a_sane_ci():
    rng = np.random.default_rng(0)
    true_mean = 5.0
    # 5 "seeds", each with 50 "targets" drawn from the same N(5, 1) population.
    seed_groups = [rng.normal(true_mean, 1.0, size=50).tolist() for _ in range(5)]

    def compute_fn(resampled_seeds):
        flat = [v for seed_data in resampled_seeds for v in seed_data]
        return float(np.mean(flat))

    result = hierarchical_bootstrap(seed_groups, _identity_resample, compute_fn, n_replicates=500, seed=1)
    assert not result["censored"]
    assert result["ci_lo"] < true_mean < result["ci_hi"]
    assert result["n_valid"] == 500


def test_hierarchical_bootstrap_is_reproducible_given_the_same_seed():
    rng = np.random.default_rng(0)
    seed_groups = [rng.normal(0, 1, size=20).tolist() for _ in range(4)]

    def compute_fn(resampled_seeds):
        flat = [v for seed_data in resampled_seeds for v in seed_data]
        return float(np.mean(flat))

    r1 = hierarchical_bootstrap(seed_groups, _identity_resample, compute_fn, n_replicates=200, seed=42)
    r2 = hierarchical_bootstrap(seed_groups, _identity_resample, compute_fn, n_replicates=200, seed=42)
    assert r1["ci_lo"] == r2["ci_lo"]
    assert r1["ci_hi"] == r2["ci_hi"]
    assert r1["replicates"] == r2["replicates"]


def test_hierarchical_bootstrap_reports_censored_when_majority_of_replicates_are_none():
    seed_groups = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

    call_count = {"n": 0}

    def mostly_none_compute_fn(resampled_seeds):
        call_count["n"] += 1
        # 60% of replicates return None (censored) -- more than half -> overall result is censored.
        return None if call_count["n"] % 5 < 3 else 1.0

    result = hierarchical_bootstrap(seed_groups, _identity_resample, mostly_none_compute_fn, n_replicates=100, seed=0)
    assert result["censored"]
    assert result["ci_lo"] is None and result["ci_hi"] is None


def test_hierarchical_bootstrap_not_censored_when_minority_of_replicates_are_none():
    seed_groups = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    rng_state = {"n": 0}

    def mostly_valid_compute_fn(resampled_seeds):
        rng_state["n"] += 1
        return None if rng_state["n"] % 10 == 0 else float(rng_state["n"])  # 10% None

    result = hierarchical_bootstrap(seed_groups, _identity_resample, mostly_valid_compute_fn, n_replicates=100, seed=0)
    assert not result["censored"]
    assert result["n_valid"] == 90


def test_hierarchical_bootstrap_requires_nonempty_seed_groups():
    with pytest.raises(ValueError):
        hierarchical_bootstrap([], _identity_resample, lambda x: 1.0, n_replicates=10)


def test_hierarchical_bootstrap_supports_non_scalar_compute_fn_for_shared_resampling():
    """FX2's `build_fx2_summary.py` needs ONE resampling pass per replicate to feed several
    downstream quantities (per-e TPR/AUC/TPR01 across 7 elapsed times, reused for both the per-e
    summary CI and the half-life CI) rather than a separate `hierarchical_bootstrap` call per
    quantity -- `compute_fn` returning a dict (non-scalar) must not raise, and the caller reads
    `replicates` directly to do its own aggregation."""
    seed_groups = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

    def dict_compute_fn(resampled_seeds):
        flat = [v for seed_data in resampled_seeds for v in seed_data]
        return {"mean": float(np.mean(flat)), "max": float(np.max(flat))}

    result = hierarchical_bootstrap(seed_groups, _identity_resample, dict_compute_fn, n_replicates=50, seed=0)
    assert result["ci_lo"] is None and result["ci_hi"] is None and result["censored"] is None
    assert len(result["replicates"]) == 50
    assert all(isinstance(r, dict) and "mean" in r and "max" in r for r in result["replicates"])


def test_hierarchical_bootstrap_inner_resample_actually_varies_the_result():
    """Regression guard: if `inner_resample_fn` were accidentally a no-op (or `compute_fn` ignored its
    argument), every replicate would be identical and the CI would collapse to a point -- a real
    resampling with real per-target variance should not do that."""
    rng = np.random.default_rng(3)
    seed_groups = [rng.normal(0, 5.0, size=30).tolist() for _ in range(4)]  # high target-level variance

    def compute_fn(resampled_seeds):
        flat = [v for seed_data in resampled_seeds for v in seed_data]
        return float(np.mean(flat))

    result = hierarchical_bootstrap(seed_groups, _identity_resample, compute_fn, n_replicates=300, seed=7)
    assert result["ci_hi"] - result["ci_lo"] > 1e-6
