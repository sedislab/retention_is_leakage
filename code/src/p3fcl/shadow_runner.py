"""The shadow-federation runner (P3's core infrastructure, `04_METHODS_AND_ATTACKS.md` §3 /
`build/00_BUILD_PLAN.md` P3 task 1). For each shadow index, independently subsample the *entire*
candidate pool (every datum in the base stream, not just a tracked few) at `p_in`, rerun the
federation on cached features with that subsample, and write only the score trajectory of a fixed
pool of tracked *targets* the attack needs (`attacks/lira.py`) -- never the whole ledger.

**Population-level resampling is load-bearing, not an arbitrary choice** (found the hard way --
`notes/2026-09-16_p3_a1_population_resampling.md`): an earlier version of this module resampled only
the ~5 tracked targets per shard and left everyone else fixed. For a method whose release is
partitioned by class (F2's per-class prototype), a target sampled from a shard spanning many classes
is, with high probability, the *only* ever-resampled member of its own class -- so the "OUT" and "IN"
distributions each collapse to a single deterministic value (no other source of randomness ever
touches that class's statistic), and LiRA gets a perfect, textbook-clean AUC=1.0 for a reason that has
nothing to do with real leakage strength: the null distribution has zero variance by construction, not
because the attack is genuinely unbeatable. Standard LiRA (Carlini et al. 2022) resamples a random
half of the *whole* candidate pool per shadow precisely to avoid this -- every shadow's released
statistic reflects genuine population-level sampling noise, and a target's own membership must be
detected against that real noise floor, not against a point mass. `_population_mask` below does that:
one seeded draw over every id in the base stream per shadow, applied uniformly.

Design requirements, all load-bearing (§3):
1. Write statistics, not ledgers -- one small `.npz` per shadow.
2. Deterministic -- `shadow_id` seeds every draw via `rng.seeded`, so re-running index i reproduces
   its `.npz` byte-identically (`test_shadow_runner.py` pins this).
3. Resumable -- skip if the output exists; write `.tmp` then `os.replace`.
4. Process-parallel, single-threaded BLAS -- the caller sets `OMP_NUM_THREADS=1`; this module forks
   worker processes (`multiprocessing.get_context("fork")`) so the (potentially large) feature cache
   and base stream are shared via copy-on-write, never re-pickled per shadow.
5. Chunked for the scheduler -- `run_shadow_range(start, count, ...)` is one array task's worth.

**Scope**: M4 (F2, prototype), M8 (F5, Gram), M0 (F1, FedAvg-sequential model delta) — matching
`04_METHODS_AND_ATTACKS.md` §4's "A6 then A1, on M4-proto/M8" ordering, extended to M0 specifically to
give H3 a non-degenerate test. M4/M8 are deterministic given the stream (no SGD noise) and release
each target's evidence in exactly one round (class-incremental, momentum/lambda fixed), so a target's
score trajectory is a constant repeated from its release round onward -- the same structural fact
`attacks/property_inference.py` already found for H10, and it makes the trajectory-vs-last-round-only
ablation (H3, FIG07) degenerate for those two (summing K copies of the same per-round log-LR is a
monotone rescale, doesn't change ranking -- confirmed exactly on real data,
`notes/2026-09-16_p3_a1_lira_results.md`). **M0 is different**: its released `MODEL_DELTA` is one
piece of an ever-changing global model, reconstructible round-by-round from the ledger alone
(`_reconstruct_running_w`, replaying M0's own FedAvg aggregation using only ledger-visible `payload`
and `n_touched`) -- a target's logit-margin trajectory under that reconstructed model genuinely varies
across rounds regardless of its own task, since later tasks keep updating the shared weight. This is
what finally makes `trajectory` vs `last_round` a real comparison instead of an algebraic identity.
`METHOD_REGISTRY` extends the same F1 reconstruction to M1/M2/M3/M5 (all FedAvg-style, all
`Family.MODEL_DELTA`) -- toward H2's real dose-response sweep across retention mechanisms. M6/M7 are
out of scope for this CPU array design entirely (`cacheable=False`, GPU-only): they need the offline
LiRA variant with N~64-128 shadows, per `01_KODIAK.md §4.3`, not this module.
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import warnings
from pathlib import Path

import numpy as np

from . import features, streams
from . import rng as rng_mod
from .artifacts import Family
from .methods.m0_fedavg import FedAvgSequential
from .methods.m1_glfc import GLFC
from .methods.m2_target import TARGET
from .methods.m3_fot import FOT
from .methods.m4_proto import PrototypeFCL
from .methods.m5_hybrid_replay import HybridReplay
from .methods.m8_analytic import AnalyticFCL
from .methods.m9_contractive import ContractiveDPAnalytic, project_and_cap
from .sim import run as sim_run

BACKBONE = "vit_base_patch16_224.augreg_in21k"
REPO_ROOT = Path(__file__).resolve().parents[3]
FEATURES_DIR = REPO_ROOT / "features"

METHOD_REGISTRY = {
    "m4_proto": (PrototypeFCL, Family.PROTOTYPE),
    "m8_analytic": (AnalyticFCL, Family.GRAM),
    "m0_fedavg": (FedAvgSequential, Family.MODEL_DELTA),
    "m1_glfc": (GLFC, Family.MODEL_DELTA),
    "m2_target": (TARGET, Family.MODEL_DELTA),
    "m3_fot": (FOT, Family.MODEL_DELTA),
    "m5_hybrid_replay": (HybridReplay, Family.MODEL_DELTA),
    "m9_contractive": (ContractiveDPAnalytic, Family.GRAM),
}


def _method_config(method_name: str, n_classes: int, feature_dim: int) -> dict:
    if method_name == "m4_proto":
        return {"n_classes": n_classes, "feature_dim": feature_dim, "prototype_momentum": 0.0}
    if method_name == "m8_analytic":
        return {"n_classes": n_classes, "feature_dim": feature_dim, "ridge_lambda": 1.0}
    # M0/M1/M2/M3/M5 all share local_epochs=30, lr=0.5 -- the real TAB05 headline config
    # (run_utility_baseline.py), not an attack-specific weakening. Each method's own extra
    # retention-strength knob is also pinned at its TAB05 headline value.
    if method_name == "m0_fedavg":
        return {"n_classes": n_classes, "feature_dim": feature_dim, "local_epochs": 30, "lr": 0.5}
    if method_name == "m1_glfc":
        return {
            "n_classes": n_classes, "feature_dim": feature_dim, "local_epochs": 30, "lr": 0.5,
            "replay_weight": 1.0, "exemplar_budget": 10, "distillation_weight": 1.0, "temperature": 2.0,
        }
    if method_name == "m2_target":
        return {
            "n_classes": n_classes, "feature_dim": feature_dim, "local_epochs": 30, "lr": 0.5,
            "replay_weight": 1.0, "replay_ratio": 1.0, "n_synthetic_per_class": 20,
        }
    if method_name == "m3_fot":
        return {
            "n_classes": n_classes, "feature_dim": feature_dim, "local_epochs": 30, "lr": 0.5,
            "subspace_rank": 8, "projection_strength": 1.0,
        }
    if method_name == "m5_hybrid_replay":
        return {
            "n_classes": n_classes, "feature_dim": feature_dim, "local_epochs": 30, "lr": 0.5,
            "replay_weight": 1.0, "buffer_size_per_class": 10,
        }
    if method_name == "m9_contractive":
        # Deliberately excludes `pca_basis`: only a caller with dataset (`ref` split) access can fit
        # a real one, and it must be fit ONCE outside the per-shadow worker loop (`run_m9_audit.py`),
        # never here (`_method_config` runs inside `_run_one_shadow`, once per shadow). A caller that
        # forgets to supply `method_config_override["pca_basis"]` gets a loud `KeyError` from
        # `ContractiveDPAnalytic.__init__`, not a silently-wrong default.
        return {"n_classes": n_classes}
    raise ValueError(
        f"shadow_runner: unsupported method {method_name!r} "
        f"(scoped to {sorted(METHOD_REGISTRY)} this pass)"
    )


def build_targets(base_stream: list, targets_per_shard: int, base_seed: int) -> list:
    """One entry per selected target: `{target_id, client, task}`. A pure function of
    `(base_stream, targets_per_shard, base_seed)` -- every shadow-runner process recomputes the same
    pool independently rather than reading it from disk, so there is no cross-process race on the
    pool's *content* (only on the convenience copy written to `targets.json`).

    These are the datums whose score gets *recorded*; every datum in `base_stream` (targets included)
    is still subject to the same population-level resampling in `_population_mask`, so a target is not
    special-cased in how its own membership is drawn -- only in that we bother to write down its score.
    """
    targets = []
    for t, shards in enumerate(base_stream):
        for shard in shards:
            r = rng_mod.seeded(f"shadow_runner.build_targets::{shard.client}::{t}", base_seed)
            ids = np.array(shard.ids, dtype=int)
            n_pick = min(targets_per_shard, len(ids))
            picked = ids[r.permutation(len(ids))[:n_pick]]
            for i in picked:
                targets.append({"target_id": int(i), "client": int(shard.client), "task": int(t)})
    return targets


def _population_mask(dataset: str, method: str, shadow_id: int, n_total: int, p_in: float) -> np.ndarray:
    """One boolean draw per id in `0..n_total-1`, from a single seeded generator per shadow (not one
    `rng.seeded` call per id -- that would mean tens of thousands of SHA-256 hashes per shadow for a
    real dataset). Every id -- targets and non-targets alike -- is resampled the same way, which is
    exactly the property the module docstring's "population-level resampling" note depends on."""
    r = rng_mod.seeded(f"shadow_runner.population_mask::{dataset}::{method}", shadow_id)
    return r.random(n_total) < p_in


def _apply_population_mask(base_stream: list, keep_mask: np.ndarray) -> list:
    out = []
    for shards in base_stream:
        new_shards = []
        for shard in shards:
            kept = tuple(i for i in shard.ids if keep_mask[i])
            if kept:
                new_shards.append(streams.Shard(client=shard.client, task=shard.task, ids=kept))
        out.append(new_shards)
    return out


def _reconstruct_running_w(ledger, n_rounds: int, feature_dim: int, n_classes: int) -> list:
    """Replays FedAvg aggregation (every F1 method's `fit_task` does the same weighted average of
    per-client deltas) from `MODEL_DELTA` ledger records alone -- `payload` (the delta) is
    ledger-visible, so this is a legitimate honest-but-curious-server reconstruction, not a read of
    the method's internal state (CLAUDE.md non-negotiable #1: attacks only ever consume the ledger).
    Unlike M4/M8's one-shot release, F1's global model keeps changing every round regardless of a
    given target's own task -- this is what finally gives A1's `trajectory` vs `last_round` ablation
    (H3) something real to measure, instead of the constant-repeated-value degeneracy
    `attacks/property_inference.py` and M4/M8 both hit.

    **FX4b fix (08_FIX_PLAN.md R3): uses `meta["agg_weight"]` -- the exact weight the server gave that
    client in that round's FedAvg, now released honestly by every F1 method -- falling back to
    `n_touched` (with a one-time logged warning) only for pre-fix ledgers that don't carry it.** Before
    this fix, `n_touched` was used as a *proxy* for the true weight and was honestly inflated beyond
    the raw shard size for M1/M2/M3/M5 by buffer/subspace/replay bookkeeping that has since been
    corrected (touched no longer accumulates post-processing reads, per Lemma 8) -- `agg_weight` is
    the real value, not an approximation, for every method after this fix. Returns one
    `(feature_dim, n_classes)` weight matrix per round, `0..n_rounds-1`."""
    w = np.zeros((feature_dim, n_classes))
    w_by_round = []
    warned_fallback = False
    for r in range(n_rounds):
        recs = [rec for rec in ledger if rec.family == Family.MODEL_DELTA and rec.round == r]
        if recs:
            if all("agg_weight" in rec.meta for rec in recs):
                weights = np.array([float(rec.meta["agg_weight"]) for rec in recs], dtype=float)
            else:
                if not warned_fallback:
                    warnings.warn(
                        "shadow_runner._reconstruct_running_w: falling back to n_touched as the "
                        "FedAvg weight -- this ledger predates FX4b's meta['agg_weight'] release and "
                        "n_touched is only an approximation for M1/M2/M3/M5 (08_FIX_PLAN.md R3).",
                        stacklevel=2,
                    )
                    warned_fallback = True
                weights = np.array([rec.n_touched for rec in recs], dtype=float)
            if weights.sum() > 0:
                deltas = np.stack([np.asarray(rec.payload) for rec in recs])
                avg_delta = np.tensordot(weights, deltas, axes=(0, 0)) / weights.sum()
                w = w + avg_delta
        w_by_round.append(w.copy())
    return w_by_round


def _reconstruct_running_w_secure_agg(ledger, n_rounds: int, feature_dim: int, n_classes: int) -> list:
    """H11's secure-aggregation variant of `_reconstruct_running_w`: same contract, but computed from
    `Ledger.aggregate_view()` (per-round SUM over clients, `client=-1`) instead of per-client records
    -- what an honest-but-curious server sees if the transport genuinely only reveals the aggregate,
    not each client's individual update. `aggregate_view()`'s sum is *unweighted*; the true FedAvg
    average is weighted by client shard size, which this view does not expose (no per-client
    `n_touched` survives a sum) -- this is deliberately the *more conservative* (more protective of
    privacy) reading: an unweighted mean over however many distinct clients contributed that round
    (`meta["aggregated_over_clients"]`, the one piece of non-sensitive participation metadata a real
    secure-aggregation deployment does not hide). See `code/scripts/check_secure_agg_a1.py` and
    `notes/2026-09-21_secure_agg_a1.md` for why this and not a weighted reconstruction was chosen, and
    for the measured degradation this approximation introduces relative to the true model."""
    w = np.zeros((feature_dim, n_classes))
    w_by_round = []
    agg_records = ledger.aggregate_view()
    for r in range(n_rounds):
        recs = [rec for rec in agg_records if rec.family == Family.MODEL_DELTA and rec.round == r]
        if recs:
            rec = recs[0]
            n_clients = len(rec.meta.get("aggregated_over_clients", [])) or 1
            w = w + np.asarray(rec.payload) / n_clients
        w_by_round.append(w.copy())
    return w_by_round


def _reconstruct_prototype_global(ledger, n_rounds: int, n_classes: int, d: int) -> list:
    """Reconstruct the count-weighted server bank using only round-local F2/F7 records."""
    ledger = list(ledger)
    bank = np.full((n_classes, d), np.nan)
    out = []
    for r in range(n_rounds):
        records = [rec for rec in ledger if rec.round == r]
        counts = {rec.client: rec.payload["counts"] for rec in records if rec.family == Family.COUNTS}
        sums, weights = np.zeros_like(bank), np.zeros(n_classes)
        momenta = set()
        for rec in records:
            if rec.family != Family.PROTOTYPE:
                continue
            momenta.add(float(rec.meta.get("server_momentum", 0.0)))
            if rec.payload and rec.client not in counts:
                raise ValueError("prototype global reconstruction requires released F7 counts")
            for c_str, vec in rec.payload.items():
                c = int(c_str)
                w = float(counts[rec.client][c])
                if w <= 0:
                    raise ValueError("released prototype must have a positive count")
                sums[c] += w * vec
                weights[c] += w
        if len(momenta) > 1:
            raise ValueError("inconsistent server momentum within round")
        momentum = next(iter(momenta), 0.0)
        for c in np.flatnonzero(weights):
            agg = sums[c] / weights[c]
            bank[c] = momentum*bank[c] + (1-momentum)*agg if momentum > 0 and np.isfinite(bank[c]).all() else agg
        out.append(bank.copy())
    return out


def _reconstruct_prototype_aggregate(ledger, task_k: int, n_classes: int, d: int) -> np.ndarray:
    """FX4h: "the count-weighted mean over clients of class y at task k -- what secure aggregation
    reveals." Combines every client's PROTOTYPE release at `task==task_k` with its paired COUNTS
    release (same client, same round) as the weight. A class released by only one client that round
    is unaffected (the "mean" is just that one client's value)."""
    proto_recs = [r for r in ledger if r.family == Family.PROTOTYPE and r.task == task_k]
    counts_recs = {r.client: r for r in ledger if r.family == Family.COUNTS and r.task == task_k}
    agg = np.full((n_classes, d), np.nan)
    sums = np.zeros((n_classes, d))
    weights = np.zeros(n_classes)
    for rec in proto_recs:
        counts_rec = counts_recs.get(rec.client)
        for c_str, vec in rec.payload.items():
            c = int(c_str)
            w = float(counts_rec.payload["counts"][c]) if counts_rec is not None else 1.0
            if w <= 0:
                w = 1.0
            sums[c] += w * vec
            weights[c] += w
    has_weight = weights > 0
    agg[has_weight] = sums[has_weight] / weights[has_weight, None]
    return agg


def _reconstruct_gram_global(ledger, n_rounds: int, d: int, gamma: float = 1.0) -> list:
    """FX4h: "the running state after round r, exactly the matrix `predict` inverts." Ledger-only:
    at `gamma=1.0` (M8's only mode, and the default here), the cumulative sum of every GRAM record's
    `R` payload up to and including round r -- exactly mirroring M8's own `self._R += Rc`
    accumulation. Generalized (`08_FIX_PLAN.md` §8's M9 audit) to `R <- gamma*R + round_sum`, applied
    once per round transition regardless of how many client records land in that round -- this
    exactly mirrors `ContractiveDPAnalytic.fit_task`'s own `self.R = self.gamma*self.R + G_tilde`
    update (M9 always releases exactly one `client=-1` record per round, so "sum this round's
    records, then decay-and-add" and "decay-and-add per record" coincide for M9; they do NOT coincide
    for M8's multiple-per-round client records, which is why the sum happens first)."""
    R = np.zeros((d, d))
    out = []
    for r in range(n_rounds):
        recs = [rec for rec in ledger if rec.family == Family.GRAM and rec.round == r]
        round_sum = np.zeros((d, d))
        for rec in recs:
            round_sum = round_sum + rec.payload["R"]
        R = gamma * R + round_sum
        out.append(R.copy())
    return out


def _reconstruct_gram_aggregate(ledger, task_k: int, d: int) -> np.ndarray:
    """FX4h: "Sigma_c R_c for task k" -- the sum over every client's GRAM release at that one task,
    which is exactly what a secure-aggregation broadcast of that round's contributions would reveal."""
    recs = [r for r in ledger if r.family == Family.GRAM and r.task == task_k]
    R = np.zeros((d, d))
    for rec in recs:
        R = R + rec.payload["R"]
    return R


def _score_target(
    ledger, family: Family, target: dict, X: np.ndarray, y: np.ndarray, n_rounds: int, context=None,
    view: str = "full", pca_basis=None, B: float = 1.0,
) -> np.ndarray:
    """Per-round score, rounds `0..n_rounds-1`. NaN before the target's task and (for PROTOTYPE/GRAM
    `full`/`aggregate`) if the target's entire shard vanished this shadow (e.g. a singleton shard
    whose only member was dropped OUT). `context` is family-specific shared state computed once per
    shadow (F1's `w_by_round`, or for PROTOTYPE/GRAM `global`: the per-round reconstructed bank/R).

    `pca_basis`/`B` (`08_FIX_PLAN.md` §8's M9 audit): M9's running state lives in a `p`-dimensional
    PCA-projected space (`p < d`), not the raw feature space M8's GRAM state uses -- `None` (the
    default) leaves every existing GRAM caller (M8) untouched; a real basis projects-and-caps the raw
    feature the same way `m9_contractive.py::project_and_cap` does before computing leverage, so the
    matrix dimensions actually match. Detected by presence of a real `pca_basis`, not by method name,
    so this function stays family-generic rather than gaining a method-name special case.

    `view` (FX4h, 08_FIX_PLAN.md §4h) only matters for PROTOTYPE/GRAM -- F1 (MODEL_DELTA) has one view
    (`full`) by construction, since the logit-margin attack only ever reads the global model, which is
    identical under secure aggregation once weights are exact (Prop. 3). For PROTOTYPE/GRAM:
    `full` = the target's own client's task-k release (constant by construction, the pre-fix-only
    behavior); `aggregate` = the count-weighted/summed combination across every client at task k (what
    secure aggregation reveals, still constant by construction); `global` = the running server state
    after round r (the one view that is NOT constant by construction -- the true retention
    measurement)."""
    scores = np.full(n_rounds, np.nan)
    k = target["task"]
    tid = target["target_id"]
    feat = X[tid]
    label = int(y[tid])

    if family == Family.MODEL_DELTA:
        # Logit margin (true-class logit minus the best competing logit) under the round-r global
        # model, reconstructed from the full transcript -- genuinely varies round to round, since W
        # keeps absorbing every later task's updates regardless of this target's own task.
        for r in range(k, n_rounds):
            logits = feat @ context[r]
            true_logit = logits[label]
            other_max = np.max(np.delete(logits, label))
            scores[r] = float(true_logit - other_max)
        return scores

    if family == Family.PROTOTYPE and view == "global":
        # `context` is `_reconstruct_prototype_global`'s per-round bank list.
        for r in range(k, n_rounds):
            proto = context[r][label]
            if np.any(np.isnan(proto)):
                continue
            scores[r] = -float(np.linalg.norm(feat - proto))
        return scores

    if family == Family.GRAM and view == "global":
        # `context` is `_reconstruct_gram_global`'s per-round R list.
        f = feat if pca_basis is None else project_and_cap(feat.reshape(1, -1), pca_basis, B)[0]
        for r in range(k, n_rounds):
            Rc = context[r]
            if not np.any(Rc):
                continue
            scores[r] = float(f @ np.linalg.solve(Rc, f))
        return scores

    if view == "aggregate":
        if family == Family.PROTOTYPE:
            agg = context  # `_reconstruct_prototype_aggregate`'s (n_classes, d) array for this task
            proto = agg[label]
            if np.any(np.isnan(proto)):
                return scores
            scores[k:] = -float(np.linalg.norm(feat - proto))
        elif family == Family.GRAM:
            Rc = context  # `_reconstruct_gram_aggregate`'s (d, d) array for this task
            f = feat if pca_basis is None else project_and_cap(feat.reshape(1, -1), pca_basis, B)[0]
            scores[k:] = float(f @ np.linalg.solve(Rc, f))
        else:
            raise ValueError(f"shadow_runner: no aggregate view for family {family}")
        return scores

    # view == "full" (the only view PROTOTYPE/GRAM had before FX4h): the target's own client's
    # record at its own task -- structurally the SAME identity a real per-client transmission would
    # have carried under secure aggregation, i.e. NOT what a real secure-aggregation-restricted
    # adversary could ever see (see FX3/FIG11 v2's "these are structurally unattackable" finding).
    recs = [r for r in ledger if r.family == family and r.client == target["client"] and r.task == k]
    if not recs:
        return scores
    rec = recs[0]
    if family == Family.PROTOTYPE:
        proto = rec.payload.get(str(label))
        if proto is None:
            return scores
        val = -float(np.linalg.norm(feat - proto))
    elif family == Family.GRAM:
        # Self-leverage x^T R^-1 x. By Sherman-Morrison this is *lower* when x was included in R than
        # when it was excluded (excluding x inflates its own leverage: h/(1-h) >= h) -- the opposite
        # of the naive "more like IN => bigger" intuition. Confirmed on real CIFAR-100 shadows and
        # documented in `attacks/lira.py::offline_log_lr`; harmless for the online LiRA path used
        # here (direction-agnostic), load-bearing if this family is ever wired into the offline path.
        Rc = rec.payload["R"]
        val = float(feat @ np.linalg.solve(Rc, feat))
    else:
        raise ValueError(f"shadow_runner: no score function for family {family}")
    scores[k:] = val
    return scores


_WORKER = {}


def _init_worker(
    dataset, method_name, X, y, base_stream, targets, n_classes, feature_dim, p_in, out_dir,
    method_config_override=None, adversary_view="full",
):
    _WORKER.update(
        dataset=dataset, method_name=method_name, X=X, y=y, base_stream=base_stream,
        targets=targets, n_classes=n_classes, feature_dim=feature_dim, p_in=p_in, out_dir=Path(out_dir),
        method_config_override=dict(method_config_override or {}), adversary_view=adversary_view,
    )


def _run_one_shadow(shadow_id: int) -> dict:
    s = _WORKER
    out_path = s["out_dir"] / f"shadow_{shadow_id:06d}.npz"
    if out_path.exists():
        return {"shadow_id": shadow_id, "status": "skipped"}

    targets = s["targets"]
    keep_mask = _population_mask(s["dataset"], s["method_name"], shadow_id, len(s["X"]), s["p_in"])
    modified = _apply_population_mask(s["base_stream"], keep_mask)

    method_cls, family = METHOD_REGISTRY[s["method_name"]]
    method_cfg = {
        **_method_config(s["method_name"], s["n_classes"], s["feature_dim"]),
        **s.get("method_config_override", {}),
    }
    # M9's DP noise seed is deliberately decoupled from `sim.run`'s data seed (its own docstring:
    # never touches `fit_task`'s `rng` argument) -- ContractiveDPAnalytic reads `config["noise_seed"]`
    # only, defaulting to a fixed 0. Since `method_config_override` is ONE dict shared by every shadow
    # in this `run_shadow_range` call, leaving that default in place would give every single shadow
    # the exact SAME noise draw -- a fixed additive term carries no differential privacy at all, which
    # would make the M9 LiRA audit (08_FIX_PLAN.md §8) meaningless at best and silently wrong at worst
    # (the empirical mechanism under test would not be the DP-guaranteed one). Default it to the
    # shadow's own id -- each shadow is a separate hypothetical world and must draw its own
    # independent noise, exactly like the population-resampling seed already does -- while still
    # letting an explicit override win, consistent with every other config key here.
    method_cfg.setdefault("noise_seed", shadow_id)
    method = method_cls(method_cfg)
    result = sim_run(method, s["X"], s["y"], modified, seed=shadow_id)
    ledger = result["ledger"]
    n_rounds = len(modified)

    target_ids = np.array([t["target_id"] for t in targets], dtype=int)
    in_out = keep_mask[target_ids]

    # FX4h (08_FIX_PLAN.md §4h): F1 has one view (full) by construction -- the logit-margin attack
    # only ever reads the global model, identical under secure aggregation once weights are exact.
    # F2 (M4)/F5 (M8) get all three views; `global` is the one NOT constant by construction and is
    # the real retention measurement.
    score_arrays: dict = {}
    if family == Family.MODEL_DELTA:
        if s.get("adversary_view", "full") == "secure_agg":
            context = _reconstruct_running_w_secure_agg(ledger, n_rounds, s["feature_dim"], s["n_classes"])
        else:
            context = _reconstruct_running_w(ledger, n_rounds, s["feature_dim"], s["n_classes"])
        score_arrays["full"] = np.stack(
            [_score_target(ledger, family, t, s["X"], s["y"], n_rounds, context, view="full") for t in targets]
        )
    elif family == Family.PROTOTYPE:
        score_arrays["full"] = np.stack(
            [_score_target(ledger, family, t, s["X"], s["y"], n_rounds, view="full") for t in targets]
        )
        global_ctx = _reconstruct_prototype_global(ledger, n_rounds, s["n_classes"], s["feature_dim"])
        score_arrays["global"] = np.stack(
            [_score_target(ledger, family, t, s["X"], s["y"], n_rounds, global_ctx, view="global") for t in targets]
        )
        agg_by_task = {t["task"]: _reconstruct_prototype_aggregate(ledger, t["task"], s["n_classes"], s["feature_dim"])
                       for t in targets}
        score_arrays["aggregate"] = np.stack(
            [_score_target(ledger, family, t, s["X"], s["y"], n_rounds, agg_by_task[t["task"]], view="aggregate")
             for t in targets]
        )
    elif family == Family.GRAM:
        # M9's running state lives in the PCA-projected p-dim space (p < d), not the raw feature
        # space M8's GRAM state uses -- detect it by the presence of a real `pca_basis` in the
        # method config (set once by the caller, e.g. `run_m9_audit.py`, never fit here), not by
        # method name, so this stays a family-generic branch.
        pca_basis = method_cfg.get("pca_basis")
        gram_B = float(method_cfg.get("B", 1.0))
        gram_gamma = float(method_cfg.get("gamma", 1.0))
        gram_d = pca_basis.shape[1] if pca_basis is not None else s["feature_dim"]
        score_arrays["full"] = np.stack(
            [_score_target(ledger, family, t, s["X"], s["y"], n_rounds, view="full",
                            pca_basis=pca_basis, B=gram_B) for t in targets]
        )
        global_ctx = _reconstruct_gram_global(ledger, n_rounds, gram_d, gamma=gram_gamma)
        score_arrays["global"] = np.stack(
            [_score_target(ledger, family, t, s["X"], s["y"], n_rounds, global_ctx, view="global",
                            pca_basis=pca_basis, B=gram_B) for t in targets]
        )
        agg_by_task = {t["task"]: _reconstruct_gram_aggregate(ledger, t["task"], gram_d) for t in targets}
        score_arrays["aggregate"] = np.stack(
            [_score_target(ledger, family, t, s["X"], s["y"], n_rounds, agg_by_task[t["task"]], view="aggregate",
                            pca_basis=pca_basis, B=gram_B) for t in targets]
        )
    else:
        raise ValueError(f"shadow_runner: no view set for family {family}")

    tmp = out_path.with_name(out_path.name + ".tmp")
    with open(tmp, "wb") as f:
        np.savez_compressed(
            f, shadow_id=shadow_id, target_ids=target_ids,
            target_task=np.array([t["task"] for t in targets], dtype=int),
            target_client=np.array([t["client"] for t in targets], dtype=int),
            in_out=in_out, views=np.array(sorted(score_arrays), dtype="U16"),
            **{f"scores_{v}": arr for v, arr in score_arrays.items()},
        )
    os.replace(tmp, out_path)
    return {"shadow_id": shadow_id, "status": "written"}


def run_shadow_range(dataset: str, method: str, start: int, count: int, workers: int, out_dir, config: dict) -> dict:
    """One array task's worth of shadows: `[start, start+count)`. Returns a small summary dict; the
    caller (`cli.py`/`scripts/run_shadows.py`) is responsible for logging it via `provenance`.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cache = features.load_cache(FEATURES_DIR, dataset, BACKBONE, "train")
    X, y = cache["features"], cache["labels"]
    n_classes = int(y.max()) + 1
    feature_dim = X.shape[1]

    stream_cfg = config.get("stream", {})
    n_tasks = int(stream_cfg.get("n_tasks", 10))
    n_clients = int(stream_cfg.get("n_clients", 10))
    beta = float(stream_cfg.get("beta", 0.5))
    base_seed = int(stream_cfg.get("base_seed", 0))
    # FIG18 (Camelyon17 natural-federation check, `00_BUILD_PLAN.md`'s "never cut" list):
    # `stream.use_domain_field=true` switches from the default class-incremental split to
    # domain-incremental (one task per real hospital/domain, per `datasets/<dataset>/index.json`'s
    # `"domain"` field per sample, keyed by sample id) -- `n_clients=1` then gives the natural
    # federation (client = hospital, no Dirichlet), `n_clients>1` gives the Dirichlet-subpartitioned
    # comparison arm, both using the exact same already-generic `streams.build_stream`.
    # Wave V3 (08_FIX_PLAN.md's H13 fix, "matched-N-client redesign"): `stream.client_field=<name>`
    # switches the WITHIN-task client split from `dirichlet_partition`'s random per-class draw to
    # `natural_chunked_partition`'s field-grouped one (e.g. `"slide"` for Camelyon17), reading that
    # named field the exact same way `use_domain_field` already reads `"domain"` -- both the "natural"
    # and "dirichlet" comparison arms can now share the same `n_clients`, isolating partition
    # STRATEGY from client COUNT.
    domain_field = None
    if stream_cfg.get("use_domain_field", False):
        index_path = REPO_ROOT / "datasets" / dataset / "index.json"
        index = json.loads(index_path.read_text())
        id_to_domain = {s["id"]: s["domain"] for s in index["samples"]}
        domain_field = np.array([id_to_domain[int(i)] for i in cache["ids"]])
    client_field = None
    client_field_name = stream_cfg.get("client_field")
    if client_field_name:
        index_path = REPO_ROOT / "datasets" / dataset / "index.json"
        index = json.loads(index_path.read_text())
        id_to_client_field = {s["id"]: s[client_field_name] for s in index["samples"]}
        client_field = np.array([id_to_client_field[int(i)] for i in cache["ids"]])
    base_stream = streams.build_stream(
        y, np.arange(len(y)), n_tasks=n_tasks, n_clients=n_clients, beta=beta, seed=base_seed,
        domain_field=domain_field, client_field=client_field,
    )

    targets_cfg = config.get("targets", {})
    targets_per_shard = int(targets_cfg.get("per_shard", 5))
    p_in = float(targets_cfg.get("p_in", 0.5))
    targets = build_targets(base_stream, targets_per_shard, base_seed)

    # P5 dose-response pilot support: `--set method_config_override.<key>=<value>` lets a caller
    # override one of the method's own retention-strength knobs (e.g. buffer_size_per_class) without
    # touching _method_config's headline TAB05 defaults -- the override is applied on top of them.
    method_config_override = dict(config.get("method_config_override", {}))
    # H11 secure-aggregation check support: `--set adversary_view=secure_agg` restricts the
    # MODEL_DELTA reconstruction to `Ledger.aggregate_view()` instead of per-client records --
    # PROTOTYPE/GRAM are not wired here since that answer is already deterministic (0 records ever
    # match under the aggregate view), see `code/scripts/check_secure_agg_a1.py`.
    adversary_view = str(config.get("adversary_view", "full"))

    targets_path = out_dir / "targets.json"
    if not targets_path.exists():
        # Unique per (process, array task) tmp name: concurrent PBS array tasks race to write this
        # file, and a shared ".tmp" name lets one task's os.replace delete the file out from under
        # another's (found for real on Kodiak -- job 157138, FileNotFoundError on the rename).
        # `targets` is a pure function of `config`, so whichever writer's replace lands last is fine.
        tmp = out_dir / f"targets.json.tmp.{os.getpid()}"
        tmp.write_text(json.dumps(targets))
        os.replace(tmp, targets_path)

    shadow_ids = list(range(start, start + count))
    ctx = mp.get_context("fork")
    results = []
    with ctx.Pool(
        processes=max(1, workers),
        initializer=_init_worker,
        initargs=(
            dataset, method, X, y, base_stream, targets, n_classes, feature_dim, p_in, str(out_dir),
            method_config_override, adversary_view,
        ),
    ) as pool:
        for r in pool.imap_unordered(_run_one_shadow, shadow_ids):
            results.append(r)

    written = [out_dir / f"shadow_{r['shadow_id']:06d}.npz" for r in results if r["status"] == "written"]
    return {
        "n_written": len(written),
        "n_skipped": sum(1 for r in results if r["status"] == "skipped"),
        "n_targets": len(targets),
        "n_rounds": len(base_stream),
        "outputs": written,
    }
