"""A6 — property inference over time (04_METHODS_AND_ATTACKS.md; H10's deciding test, carries claim
C1). From artifacts observed up to round T, decide whether client c held class y at task k, and
estimate c's task-k class histogram. Reported against the **marginal-frequency baseline** (always
predict using the population base rate for class y), never raw accuracy, which is trivially high
under class imbalance (CLAUDE.md's own point about TPR-vs-AUC applies here in spirit).

**Scope of this implementation**: F2+F7 (prototype + counts) methods, where the released `COUNTS`
artifact is a direct, honest join key for "did client c contribute to class y" (RESEARCH_PLAN.md
§2.2: F7 is "high as a join key"). This is the cleanest, most directly-specified case; F5-based
(Gram) and F1-based (model-delta) property inference would need a different, less direct signal
(e.g. testing whether Q_c's structure is consistent with class y's presence) and are not attempted
here — a scoped, stated cut, not a silent omission.

**A well-defined "round-T artifact" (worth stating explicitly, since the ledger structure makes more
than one reading defensible)**: a client's evidence for holding class y at task k is the existence of
a `COUNTS` (or `PROTOTYPE`) record from that *specific client*, for that *specific class*, with
`round <= T` (the adversary has observed the transcript up to round T and retains what it saw — the
honest-but-curious, persistent-observer model, `V_full`). Under the methods currently built
(single-pass, class-incremental, no retraction of a class's evidence once released), this predicts
**no decay** once `T >= k` for F2/F7 — that is a real, reportable finding in its own right if the
data confirms it, not something to force otherwise.
"""
from __future__ import annotations

import numpy as np

from ..artifacts import Family, Ledger
from .base import Attack, ThreatModel


def infer_client_class_histogram(ledger: Ledger, client: int, T: int) -> dict:
    """Estimated per-class counts for `client`, using only records with `round <= T`. Uses the last
    (highest-round) `COUNTS` record observed for this client, if any -- `COUNTS` records are already
    cumulative per client-task by construction (`methods.m4_proto`), so the most recent one under the
    horizon is the adversary's best estimate."""
    recs = [r for r in ledger if r.client == client and r.round <= T and r.family == Family.COUNTS]
    if not recs:
        return {}
    latest = max(recs, key=lambda r: r.round)
    counts = latest.payload["counts"]
    return {int(c): int(n) for c, n in enumerate(counts) if n > 0}


def infer_held_class(ledger: Ledger, client: int, class_id: int, T: int) -> bool:
    """Did `client` release evidence (a PROTOTYPE or COUNTS record naming `class_id`) at any round
    `<= T`? Returns a plain boolean prediction -- the property-inference decision."""
    for r in ledger:
        if r.client != client or r.round > T:
            continue
        if r.family == Family.COUNTS and r.payload["counts"][class_id] > 0:
            return True
        if r.family == Family.PROTOTYPE and str(class_id) in r.payload:
            return True
    return False


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=bool)
    y_pred = np.asarray(y_pred, dtype=bool)
    tp = np.sum(y_true & y_pred)
    tn = np.sum(~y_true & ~y_pred)
    n_pos = np.sum(y_true)
    n_neg = np.sum(~y_true)
    tpr = tp / n_pos if n_pos > 0 else float("nan")
    tnr = tn / n_neg if n_neg > 0 else float("nan")
    return float(np.nanmean([tpr, tnr]))


def marginal_frequency_baseline(y_true: np.ndarray, seed_rng) -> np.ndarray:
    """The baseline A6 must beat: predict "held" by an independent coin flip at the population base
    rate of `y_true`, never using any per-client evidence at all."""
    base_rate = float(np.mean(y_true))
    return seed_rng.random(len(y_true)) < base_rate


class PropertyInferenceAttack(Attack):
    threat_model = ThreatModel(
        who="hbc_server", active=False, view="full", auxiliary="none",
        target="property", survives_secure_agg=False,
    )

    def run(self, ledger: Ledger, client: int, class_id: int, T: int) -> bool:
        return infer_held_class(ledger, client, class_id, T)


def infer_holding_client(ledger: Ledger, query_feature: np.ndarray, class_id: int, task: int) -> int | None:
    """A5 — client attribution (the "smaller sibling", 04_METHODS_AND_ATTACKS.md: builds on A6's
    machinery). Given a known sample's feature vector and true class, identify which client's
    per-client `PROTOTYPE` record for that class+task is the nearest match. Needs *per-client*
    records to exist in the ledger at all -- it therefore does NOT survive secure aggregation, which
    replaces individual client records with a single aggregate broadcast (`Ledger.aggregate_view`);
    that is a structural fact about this attack, not a tuning choice, and should be demonstrated by
    running it against `ledger.aggregate_view()` and observing it can no longer identify a `client`
    (the aggregate has no client field) rather than asserted.
    """
    candidates = [r for r in ledger if r.task == task and r.family == Family.PROTOTYPE and str(class_id) in r.payload]
    if not candidates:
        return None
    dists = [np.linalg.norm(r.payload[str(class_id)] - query_feature) for r in candidates]
    return candidates[int(np.argmin(dists))].client


class ClientAttributionAttack(Attack):
    threat_model = ThreatModel(
        who="hbc_server", active=False, view="full", auxiliary="public_data",
        target="attribution", survives_secure_agg=False,
    )

    def run(self, ledger: Ledger, query_feature: np.ndarray, class_id: int, task: int):
        return infer_holding_client(ledger, query_feature, class_id, task)
