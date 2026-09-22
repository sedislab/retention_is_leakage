"""A3 — prototype-difference attribution (04_METHODS_AND_ATTACKS.md §"A3"). Successive released
prototypes (F2) plus released counts (F7, the join key) let an adversary solve for the sum of newly
contributed features between releases: `n_t*mu_t - n_{t-1}*mu_{t-1}`. `methods/m4_proto.py` at
`prototype_momentum=0` uses the *per-round-mean* convention (each release is the mean of exactly that
round's new shard), which makes the attack exact with no differencing at all: `mean * count` is the
literal sum of that shard's features. At `momentum > 0` the release is a *running* (EMA-blended)
mean, and the attack must invert the blend first using two consecutive releases for the same
(client, class). Either way the quantity that governs leakage is the **increment size** (the count) —
at count=1 the recovered "sum" is an exact single feature vector; this is the "approaches individual
reconstruction" case named in the spec.

**Why the F7 (counts) ablation is measured on scaled reconstruction error, not cosine similarity.**
Cosine similarity is scale-invariant: `cos(mean * count, target) == cos(mean, target)` for any
`count > 0`, so removing the F7 release cannot change a cosine-similarity number at all — the
*direction* (the mean itself) is exact either way. What counts (F7) actually buys the adversary is
the ability to correctly *scale* the recovered direction back into a sum with the right magnitude,
and to know for certain how many samples contributed (which is what lets a small shard be recognised
and treated as a near-exact individual reconstruction rather than an unscaled direction of unknown
provenance). `run()` reports both, so the counts-on/off comparison has a metric that can actually move.
"""
from __future__ import annotations

import numpy as np

from ..artifacts import Family, Ledger
from .base import Attack, ThreatModel


def recover_direction_and_sum(mean_t: np.ndarray, count_t, mean_prev=None, count_prev=None, momentum: float = 0.0):
    """Recover the newly-contributed direction (always available, exact) and, if `count_t` (and, for
    momentum>0, `count_prev`) is known, the scaled sum of newly-contributed features.

    Returns `(direction, sum_or_none, increment_or_none)`. `direction` is exact regardless of whether
    counts are known -- it is simply the released mean (momentum=0) or the inverted new-mean
    (momentum>0). `sum_or_none`/`increment_or_none` are `None` when `count_t` is not supplied (the
    "counts off" ablation), which is the point: direction survives, scale does not.
    """
    if momentum == 0.0 or mean_prev is None:
        direction = mean_t
        increment = count_t
    else:
        direction = (mean_t - momentum * mean_prev) / (1 - momentum)
        increment = None if count_t is None or count_prev is None else count_t - count_prev

    if increment is None:
        return direction, None, None
    return direction, direction * increment, increment


class PrototypeDifferenceAttack(Attack):
    threat_model = ThreatModel(
        who="hbc_server", active=False, view="full", auxiliary="none",
        target="reconstruction", survives_secure_agg=True,
    )

    def run(self, ledger: Ledger, client: int, class_id: int, momentum: float = 0.0, use_counts: bool = True) -> list:
        """Scans one client's F2 (+ optionally F7) releases across rounds for `class_id`, recovering
        the newly-contributed direction (and, if `use_counts`, the scaled sum) at every round where
        that class appears. Returns a list of per-round recovery dicts, in round order."""
        proto_recs = sorted(
            [r for r in ledger if r.client == client and r.family == Family.PROTOTYPE and str(class_id) in r.payload],
            key=lambda r: r.round,
        )
        count_recs = {
            r.round: r.payload["counts"][class_id]
            for r in ledger
            if r.client == client and r.family == Family.COUNTS
        }

        out = []
        prev_mean, prev_count = None, None
        for rec in proto_recs:
            mean_t = rec.payload[str(class_id)]
            count_t = count_recs.get(rec.round) if use_counts else None
            direction, recovered_sum, increment = recover_direction_and_sum(
                mean_t, count_t, prev_mean, prev_count, momentum=momentum
            )
            out.append({
                "round": rec.round, "task": rec.task, "direction": direction,
                "recovered_sum": recovered_sum, "increment": increment,
            })
            prev_mean = mean_t
            prev_count = count_recs.get(rec.round)
        return out
