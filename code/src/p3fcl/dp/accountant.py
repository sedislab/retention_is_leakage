"""RDP composition, the task-disjointness checker (V1-V5, RESEARCH_PLAN.md §4.3), and the lifelong
accountant (`account`/`extrapolate_lifelong`, FX1 rewrite, `08_FIX_PLAN.md` §6).

`check_disjointness` reads only the `Ledger` — never a method's internals — which is what makes it
usable as an external certifier rather than something a method can quietly satisfy by construction.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

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


def _client_task_ids(stream) -> dict:
    """`(client, task) -> frozenset(ids)` -- one client's own shard at one task-epoch, read directly
    off the `stream` (list of lists of `streams.Shard`) that produced the ledger. This is the
    per-instance "D(u)" for U2, and the building block U3/U4 union over."""
    out: dict = {}
    for t, task_shards in enumerate(stream):
        for shard in task_shards:
            out[(shard.client, t)] = frozenset(int(i) for i in shard.ids)
    return out


def _instance_m_T(records_upto_T: list, ids: frozenset) -> tuple:
    """`(m_T, m_T_passes)` for one unit instance with private data `ids`: the count of records in
    `records_upto_T` whose `touched` intersects `ids` at all (m_T, per Lemma 8 -- a record either
    read some of this unit's data or it didn't, no double-counting within one record), and the same
    count weighted by `max(1, passes_over_data)` (m_T_passes, secondary/appendix only per
    `08_FIX_PLAN.md` §6)."""
    if not ids:
        return 0, 0
    m, mp = 0, 0
    for r in records_upto_T:
        if r.touched & ids:
            m += 1
            mp += max(1, r.passes_over_data)
    return m, mp


def account(ledger: Ledger, stream, unit, sigma: float, delta: float, window: int = 3) -> pd.DataFrame:
    """FX1 (`08_FIX_PLAN.md` §6): the unified lifelong accountant. For every observed horizon
    `T = 1..len(stream)`, and every "unit instance" u of the given `unit` (an id for U1/U5, a
    `(client, task)` shard for U2, a `window`-wide rolling window of one client's shards for U3, or
    one client's entire history-so-far for U4), computes

        m_T(u) = #{records r : r.task < T and touched(r) & D(u) != empty}

    and reports `eps(T) = max_u eps_gaussian_composed(sigma, m_T(u), delta)` -- the worst-case unit.
    No unit gets a hard-coded routing branch (the old accountant's task-disjoint-implies-parallel-
    composition special case and its unconditional-sequential U4 case are both now just what falls
    out of this one rule): a task-disjoint ledger under U2 has every shard's own release be the only
    record that ever touches it, so m_T(u) is 1 forever and eps(T) is flat by construction, not by an
    `if report.task_disjoint` branch.

    `m_T_passes` (the same count weighted by `max(1, passes_over_data)`) is reported alongside `m_T`
    for the *argmax* instance only, as a secondary/appendix column -- per the plan, the primary `eps`
    column is driven by the unweighted record count `m_T`, not by passes. This is a real, deliberate
    narrowing versus the pre-fix accountant's `_max_task_touch_multiplicity` (which folded
    `passes_over_data` directly into the composed count `k`); flagged in `notes/` and STATE.md rather
    than silently carried over, since CLAUDE.md non-negotiable #2 makes under-reporting privacy loss
    a serious failure mode to watch for even when the plan itself specifies the narrower definition.

    Returns one row per `T` with columns `T, unit, window, m_T, m_T_passes, argmax_instance, eps`.
    U5 is not given special handling -- `08_FIX_PLAN.md` states U5 = U1 under this project's
    one-person-one-example renewal model, so callers pass `Unit.EXAMPLE` for both and do not plot U5
    separately.
    """
    unit = unit if isinstance(unit, Unit) else Unit(unit)
    if unit == Unit.INDIVIDUAL:
        unit = Unit.EXAMPLE
    n_tasks = len(stream)
    records = list(ledger)
    ct_ids = _client_task_ids(stream)
    clients = sorted({c for c, _ in ct_ids})

    rows = []
    for T in range(1, n_tasks + 1):
        records_upto_T = [r for r in records if r.task < T]

        if unit == Unit.EXAMPLE:
            # U1 (== U5): one instance per private example id -- cheaper to invert the loop (walk
            # records once, accumulate per-id counts) than to re-scan all records per id.
            counts: dict = {}
            passes: dict = {}
            for r in records_upto_T:
                w = max(1, r.passes_over_data)
                for d in r.touched:
                    counts[d] = counts.get(d, 0) + 1
                    passes[d] = passes.get(d, 0) + w
            if counts:
                best_id = max(counts, key=counts.get)
                m_T, m_T_passes, argmax = counts[best_id], passes[best_id], f"id={best_id}"
            else:
                m_T, m_T_passes, argmax = 0, 0, None

        elif unit == Unit.TASK:
            # U2: each (client, task) shard is its own fixed instance.
            best_m, best_mp, argmax = -1, 0, None
            for (c, k), ids in ct_ids.items():
                m, mp = _instance_m_T(records_upto_T, ids)
                if m > best_m:
                    best_m, best_mp, argmax = m, mp, f"client={c},task={k}"
            m_T, m_T_passes = max(best_m, 0), best_mp

        elif unit == Unit.CLIENT_BOUNDED:
            # U3: a rolling window of `window` consecutive tasks for one client, worst case over
            # every start s (windows may be shorter than `window` at the tail of the stream).
            best_m, best_mp, argmax = -1, 0, None
            for c in clients:
                for s in range(n_tasks):
                    ks = range(s, min(s + window, n_tasks))
                    ids = frozenset().union(*(ct_ids.get((c, k), frozenset()) for k in ks))
                    m, mp = _instance_m_T(records_upto_T, ids)
                    if m > best_m:
                        best_m, best_mp, argmax = m, mp, f"client={c},window=[{s},{min(s + window, n_tasks)})"
            m_T, m_T_passes = max(best_m, 0), best_mp

        elif unit == Unit.CLIENT_LIFELONG:
            # U4: one client's ENTIRE footprint revealed so far (tasks 0..T-1) -- an unbounded window
            # that grows with T, unlike U3's fixed width.
            best_m, best_mp, argmax = -1, 0, None
            for c in clients:
                ids = frozenset().union(*(ct_ids.get((c, k), frozenset()) for k in range(T)))
                m, mp = _instance_m_T(records_upto_T, ids)
                if m > best_m:
                    best_m, best_mp, argmax = m, mp, f"client={c}"
            m_T, m_T_passes = max(best_m, 0), best_mp

        else:
            raise ValueError(f"account: unsupported unit {unit}")

        eps = eps_gaussian_composed(sigma, k=m_T, delta=delta)
        rows.append({
            "T": T, "unit": unit.value, "window": window, "m_T": m_T, "m_T_passes": m_T_passes,
            "argmax_instance": argmax, "eps": eps,
        })

    return pd.DataFrame(rows)


def extrapolate_lifelong(
    df: pd.DataFrame, sigma: float, delta: float, T_max: int = 1000, n_extrap_points: int = 25,
    T_values=None,
) -> pd.DataFrame:
    """FX1: extends `account()`'s observed rows (`T = 1..n_observed`) out to `T_max` by classifying
    the regime from the *second half* of the observed range and extrapolating analytically -- never
    by re-running `account()` at large T, which would need a stream/ledger that doesn't exist yet.

    **Contractive**: `m_T` is constant over the second half of the observed range -> held flat
    forever (`eps` flat too). **Accumulating**: otherwise -> a degree-1 fit of `m_T` vs `T` over the
    second half is extended linearly. Adds `observed` (1 for real rows, 0 for extrapolated ones) and
    `regime` (`"contractive"` or `"accumulating"`, same for every row of this `(method, unit)`) so a
    plot can draw the observed part solid and the extrapolated part dashed.

    Extrapolated `T` values default to a log-spaced grid from the last observed `T` to `T_max`
    (rounded to distinct integers) -- `T_max` can be 1000 while the observed range is only 10-50
    tasks, so a dense per-integer sweep there would be almost entirely redundant, flat or
    near-linear points. Pass `T_values` (e.g. from a test that needs `eps` at specific, exact
    horizons) to extrapolate at exactly those `T` instead of the default grid.
    """
    df = df.sort_values("T").reset_index(drop=True)
    n_observed = len(df)
    if n_observed == 0:
        raise ValueError("extrapolate_lifelong: empty observed DataFrame")

    half = max(1, -(-n_observed // 2))  # ceil(n_observed / 2): "the second half" of the observed range
    tail = df.iloc[-half:]
    m_vals = tail["m_T"].to_numpy(dtype=float)
    T_vals = tail["T"].to_numpy(dtype=float)

    last_T = int(df["T"].iloc[-1])
    last_m = float(df["m_T"].iloc[-1])
    is_constant = bool(np.allclose(m_vals, m_vals[0]))
    if is_constant or len(T_vals) < 2:
        regime = "contractive"
        slope = 0.0
    else:
        slope = float(np.polyfit(T_vals, m_vals, 1)[0])
        regime = "contractive" if np.isclose(slope, 0.0, atol=1e-9) else "accumulating"

    rows = df.to_dict("records")
    for r in rows:
        r["observed"] = 1
        r["regime"] = regime

    if T_values is not None:
        extra_T = sorted({int(t) for t in T_values if int(t) > last_T})
    elif last_T < T_max:
        extra_T = sorted(set(np.unique(np.geomspace(last_T + 1, T_max, num=n_extrap_points)).round().astype(int).tolist()))
        extra_T = [t for t in extra_T if t > last_T]
    else:
        extra_T = []
    if extra_T:
        for T in extra_T:
            m_T = last_m if regime == "contractive" else max(0.0, last_m + slope * (T - last_T))
            eps = eps_gaussian_composed(sigma, k=max(0, int(round(m_T))), delta=delta)
            rows.append({
                "T": T, "unit": df["unit"].iloc[0], "window": df["window"].iloc[0], "m_T": m_T,
                "m_T_passes": float("nan"), "argmax_instance": None, "eps": eps,
                "observed": 0, "regime": regime,
            })

    return pd.DataFrame(rows)


def filter_exhaustion(eps_budget: float, sigma: float, delta: float, alpha_grid=None) -> int:
    """Number of task-epochs `k` a per-client privacy filter survives before
    `eps_gaussian_composed(sigma, k, delta)` would first exceed `eps_budget` (H9 / FIG21)."""
    k = 0
    while eps_gaussian_composed(sigma, k + 1, delta, alpha_grid) <= eps_budget:
        k += 1
        if k > 10**7:
            return k
    return k
