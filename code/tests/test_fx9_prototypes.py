from types import SimpleNamespace

import numpy as np
from p3fcl.artifacts import Family
from p3fcl.methods.m4_proto import PrototypeFCL
from p3fcl.shadow_runner import _reconstruct_prototype_aggregate, _reconstruct_prototype_global, _score_target


def test_overlapping_clients_bank_and_domain_momentum_match_hand_calculation():
    X = np.array([[0.0, 0.0], [2.0, 4.0], [4.0, 8.0], [10.0, 20.0], [14.0, 28.0]])
    y = np.zeros(5, dtype=int)
    method = PrototypeFCL(dict(n_classes=1, feature_dim=2, prototype_momentum=0.5))
    ledger = []
    cases = [
        ([SimpleNamespace(client=0, ids=[0]), SimpleNamespace(client=1, ids=[1, 2])], np.array([2.0, 4.0])),
        ([SimpleNamespace(client=0, ids=[3]), SimpleNamespace(client=1, ids=[4])], np.array([7.0, 14.0])),
    ]
    for r, (shards, expected) in enumerate(cases):
        records = method.fit_task(r, X, y, np.arange(5), shards, None)
        ledger.extend(records)
        np.testing.assert_allclose(method._prototypes[0], expected, atol=1e-10)
        reconstructed = _reconstruct_prototype_global(ledger, r + 1, 1, 2)
        np.testing.assert_allclose(reconstructed[-1], method._prototypes, atol=1e-10)
        for rec in records:
            if rec.family == Family.PROTOTYPE:
                ids = next(s.ids for s in shards if s.client == rec.client)
                np.testing.assert_allclose(rec.payload["0"], X[ids].mean(axis=0))
                assert rec.touched == frozenset(ids)


def test_class_incremental_global_and_aggregate_scores_agree():
    X = np.array([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0], [7.0, 8.0], [9.0, 10.0], [11.0, 12.0]])
    y = np.array([0, 0, 0, 1, 1, 1])
    method = PrototypeFCL(dict(n_classes=2, feature_dim=2, prototype_momentum=0.5))
    ledger = []
    for r in range(2):
        shards = [
            SimpleNamespace(client=0, ids=[3 * r]),
            SimpleNamespace(client=1, ids=[3 * r + 1, 3 * r + 2]),
        ]
        ledger.extend(method.fit_task(r, X, y, np.arange(6), shards, None))
    banks = _reconstruct_prototype_global(ledger, 2, 2, 2)
    for k in range(2):
        agg = _reconstruct_prototype_aggregate(ledger, k, 2, 2)
        for bank in banks[k:]:
            np.testing.assert_allclose(bank[k], agg[k], atol=1e-10)

        target = dict(task=k, target_id=3 * k, client=0)
        global_score = _score_target(ledger, Family.PROTOTYPE, target, X, y, 2, context=banks, view="global")
        aggregate_score = _score_target(
            ledger, Family.PROTOTYPE, target, X, y, 2, context=agg, view="aggregate"
        )
        np.testing.assert_allclose(global_score, aggregate_score, atol=1e-10, equal_nan=True)
