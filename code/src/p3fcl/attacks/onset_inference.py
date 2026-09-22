"""A4 -- onset inference (`RESEARCH_PLAN.md` §3.1.3/§3.3; H4's deciding test; `[LOCKED: onset
inference stays in Paper A even if it costs space]`). From the transcript alone, with **no auxiliary
data**, infer the round at which a new client's data first enters the federation -- the "St. Mary's
began seeing cases of disease X in week 32" harm, concrete and reportable in a way membership
inference is not.

**Scope and threat model, stated precisely because more than one reading of "onset" is defensible**:
this module tests *client-arrival* onset -- the round a client that was never active before first
contributes -- not per-client task-boundary onset. Under this project's class-incremental streams with
`beta` large enough that every client gets a nonzero Dirichlet share of every task's classes (the
config every other P3 attack has used, `n_tasks=10, n_clients=10, beta=0.5`), *every* client is active
in *every* round -- there is nothing to infer, the "event" would be trivial and constant. Client-arrival
onset only exists as a real, non-trivial event when the partition is skewed enough that some clients
are genuinely absent from early rounds (verified empirically before running anything: `beta=0.5` gives
zero such events across 10 seeds; `beta=0.02, n_clients=50` gives ~12/seed) -- `run_onset_inference.py`
uses that skewed configuration deliberately, not the shared A1 one, and says so.

**The adversary's observation, by design, survives secure aggregation**: the signal is the aggregate
per-round F1 (`MODEL_DELTA`) update norm (`ledger.aggregate_view()`'s summed payload norm) and the
aggregate participant count (`meta["aggregated_over_clients"]`'s length) -- both are quantities real
secure-aggregation protocols with dropout resilience (e.g. Bonawitz et al.) typically still reveal, even
though they hide *which* client contributed what. No per-client record is read.

Real attack: two-sided CUSUM change-point detection on the aggregate norm series (the actual attack,
calibrated on a held-out calibration set of seeds -- CLAUDE.md non-negotiable #6). Two baselines, named
explicitly in `RESEARCH_PLAN.md §3.3`: a **random guess** of the same number of flagged rounds, and a
**norm-of-update-spike heuristic** (flag the top-k highest-norm rounds, no change-point structure at
all). H4's bar: precision > 0.8 at ±2-round tolerance.
"""
from __future__ import annotations

import numpy as np

from ..artifacts import Family, Ledger
from .base import Attack, ThreatModel


def aggregate_signal_series(ledger: Ledger, n_rounds: int, family: Family = Family.MODEL_DELTA) -> tuple:
    """`(norm_series, count_series)`, each `(n_rounds,)`. `norm_series[r]` is the L2 norm of the
    round-`r` aggregate (summed-over-clients) payload -- for a dict payload (F2/F5's per-class or
    per-key statistics), the norm of every value concatenated, since there is no single natural
    "the" array to pick; `count_series[r]` is the number of distinct clients that contributed that
    round, per `aggregate_view()`'s own bookkeeping. Both are quantities a secure-aggregation observer
    plausibly still sees (CLAUDE.md non-negotiable #1: this reads only `ledger.aggregate_view()`,
    never a per-client record)."""
    norm_series = np.zeros(n_rounds)
    count_series = np.zeros(n_rounds, dtype=int)
    for rec in ledger.aggregate_view():
        if rec.family != family or not (0 <= rec.round < n_rounds):
            continue
        if isinstance(rec.payload, dict):
            flat = np.concatenate([np.asarray(v).ravel() for v in rec.payload.values()])
        else:
            flat = np.asarray(rec.payload).ravel()
        norm_series[rec.round] = float(np.linalg.norm(flat))
        count_series[rec.round] = len(rec.meta.get("aggregated_over_clients", []))
    return norm_series, count_series


def true_arrival_rounds(stream: list, skip_round_zero: bool = True) -> list:
    """Ground truth (computed by the experimenter from the real stream, not something the attack may
    read): the sorted list of task indices at which at least one client contributes for the first time
    ever. Round 0's arrivals are trivial (any adversary already knows round 0 is the start) and
    excluded by default -- only *later* arrivals are a real inference target."""
    seen: set = set()
    arrivals = []
    for t, shards in enumerate(stream):
        clients_here = {s.client for s in shards}
        if (clients_here - seen) and not (skip_round_zero and t == 0):
            arrivals.append(t)
        seen |= clients_here
    return arrivals


def cusum_changepoints(series: np.ndarray, threshold: float, drift: float = 0.0) -> list:
    """Two-sided CUSUM (Page 1954): flags round `r` when the cumulative deviation of `series` from its
    own mean exceeds `threshold` in either direction, then resets -- a real change-point detector, not
    a fixed-window heuristic."""
    series = np.asarray(series, dtype=float)
    mean = float(np.mean(series))
    pos = neg = 0.0
    flags = []
    for i, x in enumerate(series):
        pos = max(0.0, pos + (x - mean) - drift)
        neg = min(0.0, neg + (x - mean) + drift)
        if pos > threshold or -neg > threshold:
            flags.append(i)
            pos = neg = 0.0
    return flags


def norm_spike_baseline(series: np.ndarray, n_flags: int) -> list:
    """The "norm-of-update spike" heuristic named explicitly as A4's baseline
    (`RESEARCH_PLAN.md §3.3`): flag the `n_flags` highest-norm rounds, no change-point structure."""
    series = np.asarray(series, dtype=float)
    n_flags = min(n_flags, len(series))
    order = np.argsort(series)[::-1]
    return sorted(int(i) for i in order[:n_flags])


def random_baseline(n_rounds: int, n_flags: int, rng: np.random.Generator) -> list:
    n_flags = min(n_flags, n_rounds)
    return sorted(int(i) for i in rng.choice(n_rounds, size=n_flags, replace=False))


def precision_at_tolerance(flagged: list, true_rounds: list, tolerance: int = 2) -> tuple:
    """`(hits, n_flagged)` -- a flagged round is a "hit" if it lies within `tolerance` of some true
    onset round. Returns raw counts (not just a ratio) so callers can pool across seeds and compute an
    exact Clopper-Pearson interval on the pooled total, per CLAUDE.md non-negotiable #4."""
    hits = sum(1 for f in flagged if any(abs(f - t) <= tolerance for t in true_rounds))
    return hits, len(flagged)


def recall_at_tolerance(flagged: list, true_rounds: list, tolerance: int = 2) -> tuple:
    """`(hits, n_true)` -- a true onset round is "covered" if some flagged round lies within
    `tolerance` of it."""
    hits = sum(1 for t in true_rounds if any(abs(f - t) <= tolerance for f in flagged))
    return hits, len(true_rounds)


class OnsetInferenceAttack(Attack):
    threat_model = ThreatModel(
        who="participating_client", active=False, view="full", auxiliary="none",
        target="onset", survives_secure_agg=True,
    )

    def run(self, ledger: Ledger, n_rounds: int, threshold: float, family: Family = Family.MODEL_DELTA) -> list:
        norm_series, _ = aggregate_signal_series(ledger, n_rounds, family)
        return cusum_changepoints(norm_series, threshold)
