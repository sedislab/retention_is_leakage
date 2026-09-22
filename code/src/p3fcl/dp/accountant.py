"""RDP composition and the task-disjointness checker (V1-V5, RESEARCH_PLAN.md §4.3).

`check_disjointness` reads only the `Ledger` — never a method's internals — which is what makes it
usable as an external certifier rather than something a method can quietly satisfy by construction.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..artifacts import Ledger
from ..units import Unit


def rdp_gaussian(alpha: float, sigma: float) -> float:
    """RDP of order `alpha` for the Gaussian mechanism at noise multiplier `sigma`
    (`sigma = std / L2-sensitivity`): `eps_RDP(alpha) = alpha / (2 * sigma^2)`."""
    if alpha <= 1:
        raise ValueError("alpha must be > 1")
    if sigma <= 0:
        raise ValueError("sigma must be > 0")
    return alpha / (2.0 * sigma**2)


def rdp_to_dp(rdp: float, alpha: float, delta: float) -> float:
    """Standard RDP -> (eps, delta)-DP conversion (Mironov 2017): `eps = rdp + ln(1/delta)/(alpha-1)`."""
    if alpha <= 1:
        raise ValueError("alpha must be > 1")
    if not (0.0 < delta < 1.0):
        raise ValueError("delta must be in (0, 1)")
    return rdp + np.log(1.0 / delta) / (alpha - 1.0)


_DEFAULT_ALPHA_GRID = np.concatenate([np.linspace(1.01, 10.0, 200), np.linspace(10.0, 1000.0, 200)])


def eps_gaussian_composed(sigma: float, k: int, delta: float, alpha_grid=None) -> float:
    """eps for k-fold sequential composition of the Gaussian mechanism, minimised over `alpha_grid`."""
    if k <= 0:
        return 0.0
    grid = _DEFAULT_ALPHA_GRID if alpha_grid is None else alpha_grid
    best = np.inf
    for alpha in grid:
        eps = rdp_to_dp(k * rdp_gaussian(alpha, sigma), alpha, delta)
        if eps < best:
            best = eps
    return float(best)


@dataclass
class DisjointnessReport:
    task_disjoint: bool
    violations: list
    details: dict = field(default_factory=dict)


def check_disjointness(ledger: Ledger) -> DisjointnessReport:
    """V0 no `touched` ids anywhere -> unverifiable, refuse to certify.
    V1 a round mixes more than one task index (clients out of phase).
    V2/V3 a datum influences more than one task-epoch (replay, regulariser, persistent state).
    V5  a datum is touched in more than one round within a single task.
    V5b `passes_over_data > 1` on some record.
    """
    records = list(ledger)
    violations: list = []
    details: dict = {}

    if len(records) == 0:
        return DisjointnessReport(task_disjoint=False, violations=["V0"], details={"V0": "empty ledger"})

    if all(len(r.touched) == 0 for r in records):
        return DisjointnessReport(
            task_disjoint=False,
            violations=["V0"],
            details={"V0": "no touched ids recorded anywhere in the ledger; disjointness is unverifiable"},
        )

    round_tasks: dict = {}
    for r in records:
        round_tasks.setdefault(r.round, set()).add(r.task)
    v1 = {rnd: sorted(tasks) for rnd, tasks in round_tasks.items() if len(tasks) > 1}
    if v1:
        violations.append("V1")
        details["V1"] = v1

    datum_tasks: dict = {}
    for r in records:
        for d in r.touched:
            datum_tasks.setdefault(d, set()).add(r.task)
    v23 = {d: sorted(tasks) for d, tasks in datum_tasks.items() if len(tasks) > 1}
    if v23:
        violations.append("V2/V3")
        details["V2/V3"] = {
            "n_data_touched_across_tasks": len(v23),
            "example": dict(list(v23.items())[:5]),
        }

    datum_task_rounds: dict = {}
    for r in records:
        for d in r.touched:
            datum_task_rounds.setdefault((d, r.task), set()).add(r.round)
    v5 = {k: sorted(v) for k, v in datum_task_rounds.items() if len(v) > 1}
    if v5:
        violations.append("V5")
        details["V5"] = {"n_datum_task_pairs_multi_round": len(v5)}

    multi_pass = [r for r in records if r.passes_over_data > 1]
    if multi_pass:
        violations.append("V5b")
        details["V5b"] = {"n_records_multi_pass": len(multi_pass)}

    return DisjointnessReport(task_disjoint=(len(violations) == 0), violations=violations, details=details)


def _max_task_touch_multiplicity(ledger: Ledger) -> int:
    """How many times, in the worst case, a single datum is effectively touched within one task --
    across records (repeated release) AND within a single record (`passes_over_data`, e.g. multiple
    local SGD epochs over the same shard in one release). Both are real sequential-composition
    events; counting only one of them would under-report exactly the way CLAUDE.md non-negotiable #2
    warns against (a method with `local_epochs=30` must not get the same eps(T) curve as one with
    `local_epochs=1` just because both happen to emit one ledger record per task)."""
    counts: dict = {}
    for r in ledger:
        for d in r.touched:
            key = (d, r.task)
            counts[key] = counts.get(key, 0) + max(1, r.passes_over_data)
    return max(counts.values()) if counts else 1


@dataclass
class LifelongResult:
    eps_of_T: object  # Callable[[int], float]
    regime: str
    lifelong: bool
    disjointness: DisjointnessReport


def account(ledger: Ledger, sigma: float, unit, delta: float) -> LifelongResult:
    """Route each release to the right accounting regime:
    task-disjoint + U2 -> parallel composition, eps constant in T (**T1-fwd**).
    U4 (client-level, unbounded) -> sequential composition over all T tasks, eps unbounded (**H1**).
    otherwise -> sequential RDP composition at `max_task_touches x rounds_per_task` uses per T.
    """
    unit = unit if isinstance(unit, Unit) else Unit(unit)
    report = check_disjointness(ledger)

    if report.task_disjoint and unit == Unit.TASK:
        eps_single = eps_gaussian_composed(sigma, k=1, delta=delta)
        return LifelongResult(
            eps_of_T=lambda T, _e=eps_single: _e,
            regime="parallel-composition (task-disjoint, U2)",
            lifelong=True,
            disjointness=report,
        )

    if unit == Unit.CLIENT_LIFELONG:
        def eps_of_T(T, _sigma=sigma, _delta=delta):
            return eps_gaussian_composed(_sigma, k=T, delta=_delta)

        return LifelongResult(
            eps_of_T=eps_of_T,
            regime="sequential-composition (U4, unbounded client-level)",
            lifelong=False,
            disjointness=report,
        )

    max_touches = _max_task_touch_multiplicity(ledger)

    def eps_of_T(T, _sigma=sigma, _delta=delta, _m=max_touches):
        return eps_gaussian_composed(_sigma, k=max(1, T * _m), delta=_delta)

    return LifelongResult(
        eps_of_T=eps_of_T,
        regime=f"sequential-composition ({unit.value}, {max_touches} touches/task)",
        lifelong=False,
        disjointness=report,
    )


def filter_exhaustion(eps_budget: float, sigma: float, delta: float, alpha_grid=None) -> int:
    """Number of task-epochs `k` a per-client privacy filter survives before
    `eps_gaussian_composed(sigma, k, delta)` would first exceed `eps_budget` (H9 / FIG21)."""
    k = 0
    while eps_gaussian_composed(sigma, k + 1, delta, alpha_grid) <= eps_budget:
        k += 1
        if k > 10**7:
            return k
    return k
