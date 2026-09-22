from __future__ import annotations

import numpy as np
from p3fcl.artifacts import ArtifactRecord, Family, Ledger, load_npz, save_npz


def _rec(round_, task, client, family=Family.PROTOTYPE, payload=None, touched=(0, 1)):
    return ArtifactRecord(
        round=round_,
        task=task,
        client=client,
        family=family,
        payload=np.arange(4, dtype=float) if payload is None else payload,
        touched=frozenset(touched),
    )


def test_touched_over_report_defaults_n_touched():
    rec = _rec(0, 0, 0, touched=(1, 2, 3))
    assert rec.n_touched == 3


def test_family_coerces_from_string():
    rec = ArtifactRecord(round=0, task=0, client=0, family="F5", payload=np.zeros(2), touched=frozenset())
    assert rec.family == Family.GRAM


def test_ledger_views():
    ledger = Ledger()
    ledger.add(_rec(0, 0, 0))
    ledger.add(_rec(1, 0, 0))
    ledger.add(_rec(2, 1, 0))
    assert len(ledger.view("full")) == 3
    assert {r.round for r in ledger.view("final")} == {2}
    task_view = ledger.view("task")
    assert {(r.task, r.round) for r in task_view} == {(0, 1), (1, 2)}


def test_ledger_by_family_and_client_and_tasks():
    ledger = Ledger()
    ledger.add(_rec(0, 0, 0, family=Family.GRAM))
    ledger.add(_rec(0, 0, 1, family=Family.PROTOTYPE))
    assert len(ledger.by_family(Family.GRAM)) == 1
    assert len(ledger.by_client(1)) == 1
    assert ledger.tasks() == [0]


def test_aggregate_view_sums_over_clients():
    ledger = Ledger()
    ledger.add(_rec(0, 0, 0, payload=np.array([1.0, 2.0]), touched=(0,)))
    ledger.add(_rec(0, 0, 1, payload=np.array([3.0, 4.0]), touched=(1,)))
    agg = ledger.aggregate_view()
    assert len(agg) == 1
    np.testing.assert_allclose(agg[0].payload, [4.0, 6.0])
    assert agg[0].touched == frozenset({0, 1})
    assert agg[0].client == -1


def test_aggregate_view_dict_payload():
    ledger = Ledger()
    ledger.add(_rec(0, 0, 0, payload={"R": np.eye(2)}, touched=(0,)))
    ledger.add(_rec(0, 0, 1, payload={"R": np.eye(2)}, touched=(1,)))
    agg = ledger.aggregate_view()
    np.testing.assert_allclose(agg[0].payload["R"], 2 * np.eye(2))


def test_aggregate_view_dict_payload_with_mismatched_keys():
    """Regression test: a per-class payload (F2's prototype dict) can have a different key set per
    client whenever a client's Dirichlet share holds zero samples of some class -- real at low beta /
    high client count (found via A4's real run on CIFAR-100, n_clients=50, beta=0.02: `KeyError` on
    aggregate_view() before the fix). A record missing a key must not contribute to that key's sum,
    not crash."""
    ledger = Ledger()
    ledger.add(_rec(0, 0, 0, payload={"3": np.array([1.0, 1.0]), "7": np.array([2.0, 2.0])}, touched=(0,)))
    ledger.add(_rec(0, 0, 1, payload={"3": np.array([5.0, 5.0])}, touched=(1,)))  # no class "7" this round
    agg = ledger.aggregate_view()
    assert len(agg) == 1
    np.testing.assert_allclose(agg[0].payload["3"], [6.0, 6.0])
    np.testing.assert_allclose(agg[0].payload["7"], [2.0, 2.0])


def test_npz_roundtrip_array_payload(tmp_path):
    ledger = Ledger()
    ledger.add(_rec(0, 0, 0, payload=np.arange(6.0).reshape(2, 3), touched=(5, 6, 7)))
    path = tmp_path / "ledger.npz"
    save_npz(ledger, path)
    loaded = load_npz(path)
    assert len(loaded) == 1
    rec = loaded.records[0]
    np.testing.assert_allclose(rec.payload, np.arange(6.0).reshape(2, 3))
    assert rec.touched == frozenset({5, 6, 7})
    assert rec.family == Family.PROTOTYPE


def test_npz_roundtrip_dict_payload_and_meta(tmp_path):
    ledger = Ledger()
    ledger.add(
        ArtifactRecord(
            round=1,
            task=2,
            client=3,
            family=Family.GRAM,
            payload={"R": np.eye(3), "Q": np.ones((3, 2))},
            touched=frozenset({10, 11}),
            passes_over_data=1,
            meta={"ridge_lambda": 0.5},
        )
    )
    path = tmp_path / "ledger.npz"
    save_npz(ledger, path)
    loaded = load_npz(path)
    rec = loaded.records[0]
    np.testing.assert_allclose(rec.payload["R"], np.eye(3))
    np.testing.assert_allclose(rec.payload["Q"], np.ones((3, 2)))
    assert rec.meta == {"ridge_lambda": 0.5}
    assert rec.round == 1 and rec.task == 2 and rec.client == 3


def test_npz_roundtrip_is_atomic_no_leftover_tmp(tmp_path):
    ledger = Ledger()
    ledger.add(_rec(0, 0, 0))
    path = tmp_path / "ledger.npz"
    save_npz(ledger, path)
    assert path.exists()
    assert not (tmp_path / "ledger.npz.tmp").exists()
