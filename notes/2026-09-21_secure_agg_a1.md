# 2026-09-21 — H11: does A1 (the flagship attack) survive secure aggregation?

## Why this is the most important remaining gap

`agents/OPEN_QUESTIONS.md`'s H11 entry already flagged this precisely: A3/A5/A6 have real
secure-aggregation findings, A4 was built secure-agg-native from the start (H4), but **A1 — the
attack behind claim C1, the whole paper's flagship result — has never been tested against
`Ledger.aggregate_view()`**, and the register's own note says "Blast radius: High for reviewer
reception... Prioritize it." Any reviewer's first defensive instinct will be "just use secure
aggregation" — this is the test that pre-empts that dismissal, or tells us honestly that it doesn't.

## Two different questions, two different kinds of answer

**Question 1 (definedness, PROTOTYPE/GRAM families — M4, M8): can the attack even compute a score
from the aggregate view at all?**

`shadow_runner._score_target`'s PROTOTYPE/GRAM branch filters
`[r for r in ledger if r.client == target["client"] and r.task == k]` — it needs one **specific
client's own record**. `Ledger.aggregate_view()` (`artifacts.py`) sets `client=-1` on every record it
returns, by construction: it is defined as a SUM over clients, so no individual client's identity
survives the aggregation. A real target's `client` field is always a real client index, never -1.

**This is a deterministic, structural fact, not a probabilistic one — verified directly** via
`code/scripts/check_secure_agg_a1.py` on real CIFAR-100 federations:

```
{'method': 'm4_proto',    'score_definable_from_full_ledger': True, 'score_definable_from_aggregate': False}
{'method': 'm8_analytic', 'score_definable_from_full_ledger': True, 'score_definable_from_aggregate': False}
```

**M4 and M8 — exactly the two methods with the most extreme leakage findings anywhere in this project
(AUC≈1.0000, infinite decoupling ratios on every dataset) — cannot be attacked by A1 at all once
restricted to the secure-aggregation view.** Not "weakly" — the score is provably NaN for every
target, every round, 100% of the time. Also captured as a permanent regression test:
`test_prototype_and_gram_scores_undefined_under_secure_agg` (`code/tests/test_shadow_runner.py`).

**Why this is real and not an artifact of how this project happens to store records**: any actual
secure-aggregation deployment of a prototype- or Gram-statistic-based method would face the identical
structural choice — the whole point of secure aggregation is that the server never learns any single
client's contribution, only a sum/aggregate. A per-(client, class) prototype or a per-client Gram
matrix is *inherently* a per-client-keyed quantity; there is no way to route it through a genuine
secure-sum protocol without either (a) losing the per-client identity (making A1 as currently
constructed impossible, exactly as found here), or (b) not actually using secure aggregation for that
release channel at all (in which case the method isn't really "secure-aggregated," and the finding
still holds under its actual, honest transport). This is a finding about the family (F2/F5), not an
implementation quirk.

**Question 2 (degradation, MODEL_DELTA family — M0): if the attack CAN still compute a score, how
much does it degrade?**

`shadow_runner._reconstruct_running_w` computes a *weighted* average of per-client deltas
(`Σ(n_touched_i · Δ_i) / Σ n_touched_i`) — mathematically the same quantity a real secure-aggregation
deployment gives if clients pre-weight their update by their (typically public, non-sensitive)
sample count before the secure sum, the standard Bonawitz-et-al.-style SecAgg+FedAvg design.
`Ledger.aggregate_view()` as it already exists (built for A3/A4/A6) does an **unweighted** sum instead
— the more conservative, more-protective-of-privacy reading (no per-client weights survive at all,
only a plain sum and a participant count). Added `_reconstruct_running_w_secure_agg` in
`shadow_runner.py`: same contract as the existing reconstruction, but from `aggregate_view()`, dividing
the summed delta by the number of distinct contributing clients that round instead of by
per-client-weighted totals.

**Measured on a real CIFAR-100 M0 federation** (`code/scripts/check_secure_agg_a1.py`):

```
{'method': 'm0_fedavg', 'mean_relative_error_w': 0.140, 'final_relative_error_w': 0.147,
 'sample_acc_full_recon': 1.0, 'sample_acc_secure_agg_recon': 1.0}
```

The unweighted-aggregate reconstruction differs from the true weighted model by ~14% (relative
Frobenius error) on average across rounds, but a coarse sanity check (classification accuracy on a
handful of task-9 examples) is *unaffected* — the reconstructed model is still functionally close to
the true one. This is a real but *modest* degradation, not a collapse — worth a calibrated,
shadow-scale TPR@1%FPR number rather than stopping at this single-federation check.

## The pilot

Submitted 2026-09-21: 3 PBS array jobs (183365-183367), `DATASET=cifar100, METHOD=m0_fedavg,
adversary_view=secure_agg` (new `shadow_runner.py` support, threaded exactly like
`method_config_override`), one job per seed (`SEED in {0,1,2}`), reduced **1,024-shadow budget** (not
the 4,096 headline budget — this is a new, previously-untested code path, treated with the same pilot
caution as the C2 dose-response pilots), output isolated under
`shadows/cifar100/m0_fedavg/secure_agg/seed<N>/` so it never collides with the existing full-view M0
shadows (already computed at the full 4,096-shadow budget). Sanity-checked end to end with a 2-shadow
local run before submitting (confirmed `--set adversary_view=secure_agg` reaches the worker, produces
the same population mask but different scores than the existing full-view shadow for the same
`shadow_id` — mean absolute score difference ≈20.7, consistent with the ~14% weight-reconstruction
error measured above).

**New unit test for the plumbing itself**: `test_run_shadow_range_respects_adversary_view_secure_agg`.

## Real result — Question 2 (M0, complete)

All 3 (seed) shadow stores completed at the full 1,024/1,024 budget on the first attempt.
`results/secure_agg_pilot_m0.csv`, compared directly against the existing full-view
`results/a1_lira_cifar100_m0_fedavg[_seed<N>].csv` (5-seed headline data; only seeds 0-2 used here for
an apples-to-apples n=3 comparison against the pilot):

| elapsed | full ledger, mean [95% CI] | secure agg, mean [95% CI] |
|---|---|---|
| 0 | 0.0275 [0.0223, 0.0326] | 0.0287 [0.0240, 0.0333] |
| 5 | 0.0274 [0.0237, 0.0311] | 0.0292 [0.0232, 0.0351] |
| 9 | 0.0206 [0.0075, 0.0337] | 0.0240 [0.0116, 0.0364] |

**The CIs overlap almost completely at every elapsed value, and the point estimates are
indistinguishable from noise (secure_agg is if anything marginally *higher*, not lower).** Despite the
~14% relative-error degradation found in the single-federation weight-reconstruction check above,
**the actual downstream membership-inference signal is statistically unaffected**: secure aggregation
provides no detectable protection against A1 for M0. Figure: `figs/fig11_secure_agg.pdf`
(`analysis/fig11_secure_agg.py`), visually inspected — nearly-identical grouped bars with heavily
overlapping error bars at all 3 elapsed values, plus a text annotation carrying the M4/M8 structural
finding (no bar to draw for an undefined score).

## Bottom line for the paper — a complete, nuanced answer to H11/FIG11 for the flagship attack

**Secure aggregation is not a uniform defense against this project's findings — it depends entirely on
which artifact family is being attacked, and the split is exactly along the lines the rest of this
project's evidence already predicts:**
- **F1 (model-delta) methods — M0 confirmed empirically, and the same reconstruction logic applies
  identically to M1/M2/M3/M5 (not separately re-run at pilot scale here, but no reason to expect a
  different mechanism) — secure aggregation provides no protection.** The attack only ever needed the
  round-by-round *global* model, which any real FL deployment broadcasts to every client every round
  regardless of aggregation method — that is the whole point of federated learning, and secure
  aggregation was never designed to hide it.
- **F2/F5 (prototype/Gram) methods — M4 and M8, exactly the two methods with the most extreme leakage
  findings anywhere in this project — become completely unattackable via A1's current mechanism: the
  score is provably undefined for every target, every round, not just weaker.** This is real
  protection, but it comes from a structural property of what these methods release (a per-client
  keyed statistic that a genuine sum-only secure-aggregation protocol cannot preserve the identity
  of), not from any weakness in the attack.

**This is a stronger and more publication-ready answer than either a blanket "secure aggregation
doesn't help" or "secure aggregation fixes everything" would be** — it says precisely which mechanisms
are protected and why, in the same mechanistically-grounded style as this project's other findings
(H4's gradient-vs-closed-form split, M3's three-way dataset-dependent split). See
`paper/PAPER_BRIEFING.md`'s discussion section and `agents/OPEN_QUESTIONS.md`'s H11 entry for the
final write-up.

## Honest scope limits

- M0 is the only F1 method empirically re-tested at shadow scale; M1/M2/M3/M5 were not (the
  mechanistic argument for why they should behave the same is sound, but stated as an argument, not
  re-verified empirically for each one — a well-scoped follow-up, not claimed as tested).
- 3-seed pilot budget (1,024 shadows), not the 5-seed/4,096-shadow headline budget.
- The "weights are public" assumption behind treating `_reconstruct_running_w`'s existing computation
  as secure-agg-legitimate is a standard, realistic modeling choice (participation counts are
  typically not protected by SecAgg, only gradient content is) but is a modeling *choice*, stated as
  such, not an empirically-verified property of any specific real deployment.
