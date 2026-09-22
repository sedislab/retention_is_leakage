from __future__ import annotations

import numpy as np
import pytest
from p3fcl.artifacts import ArtifactRecord, Family, Ledger
from p3fcl.dp.accountant import (
    account,
    check_disjointness,
    eps_gaussian_composed,
    filter_exhaustion,
    rdp_gaussian,
    rdp_to_dp,
)
from p3fcl.units import Unit


def test_rdp_gaussian_basic_formula():
    assert rdp_gaussian(alpha=2.0, sigma=1.0) == pytest.approx(1.0)


def test_rdp_gaussian_rejects_bad_alpha_or_sigma():
    with pytest.raises(ValueError):
        rdp_gaussian(alpha=1.0, sigma=1.0)
    with pytest.raises(ValueError):
        rdp_gaussian(alpha=2.0, sigma=0.0)


def test_rdp_to_dp_monotone_in_delta():
    e1 = rdp_to_dp(rdp=1.0, alpha=2.0, delta=1e-3)
    e2 = rdp_to_dp(rdp=1.0, alpha=2.0, delta=1e-6)
    assert e2 > e1  # smaller delta demands a larger eps for the same RDP


def test_eps_gaussian_composed_monotone_in_k():
    e1 = eps_gaussian_composed(sigma=1.0, k=1, delta=1e-5)
    e10 = eps_gaussian_composed(sigma=1.0, k=10, delta=1e-5)
    e100 = eps_gaussian_composed(sigma=1.0, k=100, delta=1e-5)
    assert e1 < e10 < e100


def test_eps_gaussian_composed_zero_k_is_zero():
    assert eps_gaussian_composed(sigma=1.0, k=0, delta=1e-5) == 0.0


def test_check_disjointness_empty_ledger_refuses():
    report = check_disjointness(Ledger())
    assert not report.task_disjoint
    assert "V0" in report.violations


def test_check_disjointness_no_touched_ids_refuses_to_certify():
    ledger = Ledger()
    ledger.add(
        ArtifactRecord(round=0, task=0, client=0, family=Family.GRAM, payload=np.zeros(2), touched=frozenset())
    )
    report = check_disjointness(ledger)
    assert not report.task_disjoint
    assert report.violations == ["V0"]


def test_check_disjointness_v1_round_mixes_tasks():
    ledger = Ledger()
    ledger.add(
        ArtifactRecord(round=0, task=0, client=0, family=Family.GRAM, payload=np.zeros(2), touched=frozenset({1}))
    )
    ledger.add(
        ArtifactRecord(round=0, task=1, client=1, family=Family.GRAM, payload=np.zeros(2), touched=frozenset({2}))
    )
    report = check_disjointness(ledger)
    assert "V1" in report.violations


def test_check_disjointness_v5_multi_round_same_task():
    ledger = Ledger()
    ledger.add(
        ArtifactRecord(round=0, task=0, client=0, family=Family.GRAM, payload=np.zeros(2), touched=frozenset({1}))
    )
    ledger.add(
        ArtifactRecord(round=1, task=0, client=0, family=Family.GRAM, payload=np.zeros(2), touched=frozenset({1}))
    )
    report = check_disjointness(ledger)
    assert "V5" in report.violations


def test_check_disjointness_v5b_multi_pass_flagged():
    ledger = Ledger()
    ledger.add(
        ArtifactRecord(
            round=0, task=0, client=0, family=Family.GRAM, payload=np.zeros(2),
            touched=frozenset({1}), passes_over_data=3,
        )
    )
    report = check_disjointness(ledger)
    assert "V5b" in report.violations


def test_filter_exhaustion_increases_with_budget():
    k_small = filter_exhaustion(eps_budget=1.0, sigma=2.0, delta=1e-5)
    k_large = filter_exhaustion(eps_budget=10.0, sigma=2.0, delta=1e-5)
    assert k_large > k_small


def test_account_multi_epoch_gives_higher_eps_than_single_epoch():
    """Regression test for a real under-reporting bug (2026-09-16, found via FIG05 on real ledgers):
    `_max_task_touch_multiplicity` counted ledger records, not `passes_over_data`, so a method doing
    30 local SGD epochs per release got the identical eps(T) curve as one doing 1 epoch -- silently
    ignoring real sequential composition within a single release."""
    def make_ledger(passes: int) -> Ledger:
        ledger = Ledger()
        for t in range(3):
            ledger.add(
                ArtifactRecord(
                    round=t, task=t, client=0, family=Family.MODEL_DELTA, payload=np.zeros(2),
                    touched=frozenset(range(t * 5, (t + 1) * 5)), passes_over_data=passes,
                )
            )
        return ledger

    result_1 = account(make_ledger(1), sigma=2.0, unit=Unit.CLIENT_BOUNDED, delta=1e-5)
    result_30 = account(make_ledger(30), sigma=2.0, unit=Unit.CLIENT_BOUNDED, delta=1e-5)
    assert result_30.eps_of_T(10) > result_1.eps_of_T(10)
