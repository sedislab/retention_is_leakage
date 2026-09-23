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
from p3fcl.streams import Shard
from p3fcl.units import Unit


def _stream(n_tasks: int, n_clients: int = 1, n_per_client_task: int = 5) -> list:
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


def test_account_m_T_passes_reflects_passes_but_eps_does_not():
    """FX1 (`08_FIX_PLAN.md` §6) deliberately narrows the pre-fix accountant: the primary `eps` column
    is now driven by `m_T` (a plain count of releases that touched the unit), and
    `passes_over_data` only shows up in the secondary `m_T_passes` appendix column -- it no longer
    feeds `eps` at all. This inverts what used to be a regression test here (2026-09-16, pre-fix
    `_max_task_touch_multiplicity` folded `passes_over_data` into the composed count `k`, so 30 local
    epochs and 1 local epoch gave different eps(T)); under the new, plan-mandated definition they must
    now give the SAME eps(T) but a DIFFERENT m_T_passes. Flagged as an intentional, plan-directed
    narrowing in `notes/2026-09-22_fx1_accountant.md`, not a silent regression."""
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

    stream = _stream(n_tasks=3, n_clients=1)
    df_1 = account(make_ledger(1), stream, unit=Unit.CLIENT_BOUNDED, sigma=2.0, delta=1e-5)
    df_30 = account(make_ledger(30), stream, unit=Unit.CLIENT_BOUNDED, sigma=2.0, delta=1e-5)
    np.testing.assert_array_equal(df_1["m_T"].to_numpy(), df_30["m_T"].to_numpy())
    np.testing.assert_allclose(df_1["eps"].to_numpy(), df_30["eps"].to_numpy())
    assert (df_30["m_T_passes"].to_numpy() > df_1["m_T_passes"].to_numpy()).any()


# ---------------------------------------------------------------------------------------------
# FX1 (08_FIX_PLAN.md §6) tests 1-4.
# ---------------------------------------------------------------------------------------------


def test_eps_ordering_U1_le_U2_le_U3_le_U4_on_synthetic_and_real_ledgers():
    """Test 1: eps_U1 <= eps_U2 <= eps_U3 <= eps_U4 for every T, on every real and synthetic ledger --
    a direct consequence of D(U1) subseteq D(U2) subseteq D(U3-window) subseteq D(U4) for the
    matching instance, so m_T can only grow (never shrink) as the unit widens."""
    def assert_ordering(ledger, stream, window=3):
        dfs = {
            u: account(ledger, stream, unit=u, sigma=2.0, delta=1e-5, window=window)
            for u in (Unit.EXAMPLE, Unit.TASK, Unit.CLIENT_BOUNDED, Unit.CLIENT_LIFELONG)
        }
        for T in dfs[Unit.EXAMPLE]["T"]:
            e1 = dfs[Unit.EXAMPLE].loc[dfs[Unit.EXAMPLE]["T"] == T, "eps"].item()
            e2 = dfs[Unit.TASK].loc[dfs[Unit.TASK]["T"] == T, "eps"].item()
            e3 = dfs[Unit.CLIENT_BOUNDED].loc[dfs[Unit.CLIENT_BOUNDED]["T"] == T, "eps"].item()
            e4 = dfs[Unit.CLIENT_LIFELONG].loc[dfs[Unit.CLIENT_LIFELONG]["T"] == T, "eps"].item()
            assert e1 <= e2 + 1e-9 <= e3 + 1e-9 <= e4 + 1e-9, (T, e1, e2, e3, e4)

    # Synthetic, with real replay so the units actually separate (all equal-and-flat would pass any
    # ordering vacuously).
    ledger = Ledger()
    stream = _stream(n_tasks=6, n_clients=2)
    for t, shards in enumerate(stream):
        for shard in shards:
            ledger.add(
                ArtifactRecord(
                    round=t, task=t, client=shard.client, family=Family.MODEL_DELTA, payload=np.zeros(2),
                    touched=frozenset(shard.ids), passes_over_data=1,
                )
            )
            if t > 0:  # a replay term reads a slice of this client's own previous-task shard too
                prev_ids = frozenset(stream[t - 1][shard.client].ids[:2])
                ledger.add(
                    ArtifactRecord(
                        round=t, task=t, client=shard.client, family=Family.EXEMPLAR, payload=np.zeros(2),
                        touched=prev_ids, passes_over_data=1,
                    )
                )
    assert_ordering(ledger, stream)

    # Real ledger: an actual M0 FedAvgSequential run on synthetic features.
    from p3fcl import sim
    from p3fcl.methods.m0_fedavg import FedAvgSequential

    rng = np.random.default_rng(0)
    d, n_classes, n_tasks_real, n_clients_real = 6, 4, 4, 2
    n = 40
    X = rng.standard_normal((n, d))
    y = rng.integers(0, n_classes, size=n)
    real_stream = _stream(n_tasks=n_tasks_real, n_clients=n_clients_real, n_per_client_task=n // (n_tasks_real * n_clients_real))
    method = FedAvgSequential({"n_classes": n_classes, "feature_dim": d, "local_epochs": 2, "lr": 0.1})
    real_ledger = sim.run(method, X, y, real_stream, seed=0)["ledger"]
    assert_ordering(real_ledger, real_stream)


def _no_replay_stream_and_ledger(n_tasks: int, n_per_task: int = 5):
    """Single client, one release per task, touching only that task's own (never-again-touched) ids
    -- the M9-style pattern FX1's test 4 pins exactly."""
    stream = _stream(n_tasks=n_tasks, n_clients=1, n_per_client_task=n_per_task)
    ledger = Ledger()
    for t, shards in enumerate(stream):
        ledger.add(
            ArtifactRecord(
                round=t, task=t, client=0, family=Family.GRAM, payload=np.zeros((2, 2)),
                touched=frozenset(shards[0].ids), passes_over_data=1,
            )
        )
    return stream, ledger


def test_no_replay_ledger_U1_U2_flat_U3_window_fold_U4_linear():
    """Test 2: on a synthetic no-replay ledger (each id touched only in its own task), U1 and U2 are
    flat, U3 plateaus at the window width W, and U4 is exactly linear (m_T = T)."""
    window = 3
    n_tasks = 6
    stream, ledger = _no_replay_stream_and_ledger(n_tasks)

    df_u1 = account(ledger, stream, unit=Unit.EXAMPLE, sigma=2.0, delta=1e-5)
    df_u2 = account(ledger, stream, unit=Unit.TASK, sigma=2.0, delta=1e-5)
    df_u3 = account(ledger, stream, unit=Unit.CLIENT_BOUNDED, sigma=2.0, delta=1e-5, window=window)
    df_u4 = account(ledger, stream, unit=Unit.CLIENT_LIFELONG, sigma=2.0, delta=1e-5)

    assert (df_u1["m_T"] == 1).all()
    assert (df_u2["m_T"] == 1).all()
    # U3: min(T, window) -- ramps up to the window width, then plateaus.
    expected_u3 = [min(T, window) for T in df_u3["T"]]
    np.testing.assert_array_equal(df_u3["m_T"].to_numpy(), expected_u3)
    assert (df_u3.loc[df_u3["T"] >= window, "m_T"] == window).all()
    # U4: exactly T (every past task's own release is still the only thing touching its own data,
    # but ALL of them now count since D(u) is the client's entire history so far).
    np.testing.assert_array_equal(df_u4["m_T"].to_numpy(), df_u4["T"].to_numpy())


def test_replay_ledger_U1_and_U2_grow_with_T():
    """Test 3: on a synthetic ledger with real replay (a later task's release re-touches an earlier
    task's ids), U1 and U2's m_T grow with T instead of staying flat at 1."""
    n_tasks = 6
    stream, ledger = _no_replay_stream_and_ledger(n_tasks)
    # Every task from 1 onward also replays task 0's first id.
    replayed_id = stream[0][0].ids[0]
    for t in range(1, n_tasks):
        ledger.add(
            ArtifactRecord(
                round=t, task=t, client=0, family=Family.EXEMPLAR, payload=np.zeros(2),
                touched=frozenset({replayed_id}), passes_over_data=1,
            )
        )

    df_u1 = account(ledger, stream, unit=Unit.EXAMPLE, sigma=2.0, delta=1e-5)
    df_u2 = account(ledger, stream, unit=Unit.TASK, sigma=2.0, delta=1e-5)
    assert df_u1["m_T"].iloc[-1] > df_u1["m_T"].iloc[0]
    assert df_u2["m_T"].iloc[-1] > df_u2["m_T"].iloc[0]
    # replayed_id's own m_T is exactly 1 (its own release) + however many later tasks replayed it.
    np.testing.assert_array_equal(df_u1["m_T"].to_numpy(), [min(T, n_tasks) for T in df_u1["T"]])


def test_m9_style_ledger_matches_the_exact_closed_form():
    """Test 4 (M9 doesn't exist yet -- FX5 -- so this pins the exact closed-form pattern FX5's real M9
    ledger must reproduce once it lands, per `08_FIX_PLAN.md` §6: m_T = 1 for U1 and U2 at every T;
    m_T = window for U3; m_T = T for U4. M9's whole design point (a bounded-influence, non-accumulating
    retention operator) is to behave exactly like the no-replay ledger in the test above -- one clean
    release per task, no cross-task carry -- so this is the same construction, asserted as exact
    equalities rather than qualitative flat/linear checks."""
    window = 3
    n_tasks = 7
    stream, ledger = _no_replay_stream_and_ledger(n_tasks)

    df_u1 = account(ledger, stream, unit=Unit.EXAMPLE, sigma=2.0, delta=1e-5)
    df_u2 = account(ledger, stream, unit=Unit.TASK, sigma=2.0, delta=1e-5)
    df_u3 = account(ledger, stream, unit=Unit.CLIENT_BOUNDED, sigma=2.0, delta=1e-5, window=window)
    df_u4 = account(ledger, stream, unit=Unit.CLIENT_LIFELONG, sigma=2.0, delta=1e-5)

    assert (df_u1["m_T"] == 1).all()
    assert (df_u2["m_T"] == 1).all()
    assert (df_u3.loc[df_u3["T"] >= window, "m_T"] == window).all()
    np.testing.assert_array_equal(df_u4["m_T"].to_numpy(), df_u4["T"].to_numpy())
