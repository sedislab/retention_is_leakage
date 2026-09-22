"""The four CLAIM tests (06_PACKAGE_SPEC.md §9). These are not unit tests: each encodes a
load-bearing argument, and a failure means an argument in the paper has broken, not that a helper
regressed. If one ever fails, say so rather than patching the test.
"""
from __future__ import annotations

import numpy as np
import pytest
from p3fcl.artifacts import ArtifactRecord, Family, Ledger
from p3fcl.dp.accountant import account, check_disjointness
from p3fcl.units import Unit


def _task_disjoint_ledger(n_tasks: int, n_per_task: int = 5) -> Ledger:
    """A synthetic ledger where task t's single release touches only ids [t*n_per_task, (t+1)*n_per_task)."""
    ledger = Ledger()
    for t in range(n_tasks):
        ids = frozenset(range(t * n_per_task, (t + 1) * n_per_task))
        ledger.add(
            ArtifactRecord(
                round=t,
                task=t,
                client=0,
                family=Family.GRAM,
                payload=np.zeros((2, 2)),
                touched=ids,
                passes_over_data=1,
            )
        )
    return ledger


class TestT1Fwd:
    """CLAIM: a synthetic task-disjoint ledger accounts to an eps that is *constant* in T under U2,
    for T in {1, 10, 100, 1000}."""

    def test_eps_constant_in_T_under_U2(self):
        ledger = _task_disjoint_ledger(n_tasks=5)
        report = check_disjointness(ledger)
        assert report.task_disjoint, f"expected task-disjoint, got violations={report.violations}"

        result = account(ledger, sigma=2.0, unit=Unit.TASK, delta=1e-5)
        assert result.lifelong
        values = [result.eps_of_T(T) for T in (1, 10, 100, 1000)]
        assert all(v == pytest.approx(values[0], rel=1e-9) for v in values), values


class TestH1:
    """CLAIM: the same ledger under U4 gives an eps that is strictly increasing and unbounded in T."""

    def test_eps_unbounded_in_T_under_U4(self):
        ledger = _task_disjoint_ledger(n_tasks=5)
        result = account(ledger, sigma=2.0, unit=Unit.CLIENT_LIFELONG, delta=1e-5)
        assert not result.lifelong
        v1 = result.eps_of_T(1)
        v50 = result.eps_of_T(50)
        v1000 = result.eps_of_T(1000)
        assert v1 < v50 < v1000, (v1, v50, v1000)
        # unbounded: eps must keep growing well past any fixed budget as T grows further
        assert result.eps_of_T(5000) > v1000


class TestAnalyticTaskDisjointness:
    """CLAIM: M8's single-pass ledger is certified task_disjoint=True; a variant with a replay
    buffer (a later task re-touching an earlier task's ids) is certified False with a V2/V3 violation."""

    def test_m8_style_ledger_is_disjoint(self):
        from p3fcl.methods.m8_analytic import AnalyticFCL
        from p3fcl.streams import Shard

        rng = np.random.default_rng(0)
        d, n_classes = 8, 3
        X = rng.standard_normal((30, d))
        y = rng.integers(0, n_classes, size=30)
        stream = [
            [Shard(client=0, task=t, ids=tuple(range(t * 10, (t + 1) * 10)))] for t in range(3)
        ]
        method = AnalyticFCL({"n_classes": n_classes, "feature_dim": d, "ridge_lambda": 1.0})
        ledger = Ledger()
        for t, shards in enumerate(stream):
            ledger.extend(method.fit_task(t, X, y, ids=None, client_shards=shards, rng=rng))

        report = check_disjointness(ledger)
        assert report.task_disjoint, report.violations

    def test_replay_variant_violates_v2_v3(self):
        ledger = _task_disjoint_ledger(n_tasks=3, n_per_task=5)
        # inject a "replay" record at task 2 that re-touches task 0's ids
        ledger.add(
            ArtifactRecord(
                round=2,
                task=2,
                client=0,
                family=Family.EXEMPLAR,
                payload=np.zeros(3),
                touched=frozenset(range(0, 5)),
                passes_over_data=1,
            )
        )
        report = check_disjointness(ledger)
        assert not report.task_disjoint
        assert "V2/V3" in report.violations


class TestH5ExactAtNEquals1:
    """CLAIM: Gram inversion recovers a single feature vector to |cos| = 1 up to sign, exactly, as
    the linear algebra requires (RESEARCH_PLAN.md §3.4: rank(X^T X) = n, so at n=1 the sample
    subspace is one-dimensional and R - lambda*I is an exact outer product)."""

    def test_gram_inversion_exact_at_n1(self):
        rng = np.random.default_rng(42)
        d = 16
        x = rng.standard_normal(d)
        lam = 1e-6  # negligible regulariser, isolates the pure linear-algebra claim
        R = np.outer(x, x) + lam * np.eye(d)

        eigvals, eigvecs = np.linalg.eigh(R - lam * np.eye(d))
        top = eigvecs[:, np.argmax(eigvals)]
        recovered = top * np.sqrt(max(eigvals.max(), 0.0))

        cos_sim = abs(np.dot(recovered, x) / (np.linalg.norm(recovered) * np.linalg.norm(x)))
        assert cos_sim == pytest.approx(1.0, abs=1e-9)
