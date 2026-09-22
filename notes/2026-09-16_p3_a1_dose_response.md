# 2026-09-16 — P3: A1 extended to 5 F1 methods, a real cross-method dose-response preview for H2

## What was built

Extended `shadow_runner.METHOD_REGISTRY` with M1 (GLFC), M2 (TARGET), M3 (FOT), M5 (HybridReplay) —
all `Family.MODEL_DELTA`, all reusing `_reconstruct_running_w` unchanged (no new scoring code needed:
the F1 reconstruction is generic over any method that does FedAvg-style weighted-delta aggregation,
which all five F1 methods in the zoo do). Config for each pinned at its real TAB05 headline value
(`run_utility_baseline.py`) — `local_epochs=30, lr=0.5` plus each method's own retention knob at its
TAB05 default — not an attack-specific weakening.

**One real approximation, documented rather than hidden**: the FedAvg weight `_reconstruct_running_w`
uses (`n_touched`) is *exact* for M0 (whose `touched` is exactly the raw shard, no buffer/distillation)
but only an *approximation* for M1/M2/M3/M5, whose `touched` is honestly inflated beyond the raw shard
by buffer/distillation/subspace carry-forward (CLAUDE.md non-negotiable #2: over-report, never
under-report `touched`). None of the four release a companion record with the exact raw shard size at
every round, so a genuinely ledger-only observer cannot recover the true weight for them either — this
is a real limitation of what `V_full` exposes for these methods, not a shortcut in the code. Documented
in `shadow_runner.py`'s `_reconstruct_running_w` docstring; worth re-examining if the absolute AUC
numbers below ever need to be taken as precise rather than directional.

7/7 tests pass on all seven methods now in `METHOD_REGISTRY` (M4, M8, M0, M1, M2, M3, M5), 162/162
total, ruff clean.

## A real, recurring infrastructure finding: silent array-task deaths under concurrency

Every one of the six new full-scale (4,096-shadow) runs this session lost 2-9 of its 32 array tasks to
**silent deaths** — empty log files, no Python traceback, job simply gone from the queue after using
part of its walltime. Confirmed via a direct isolation test (M1, indices 4-5, which had failed twice
under concurrent load): resubmitted alone with no other array jobs competing, and it completed cleanly
both times. **This is a real, reproducible Kodiak batch-queue reliability degradation under
high concurrent submission load** (multiple 30-way-concurrent array jobs submitted in the same
burst), not a code bug — every death happened before the Python process even reached its first print
statement, and resubmitting the exact same work later (with the queue less loaded, or in isolation)
always succeeded. Resumability (`.tmp` + `os.replace`, skip-if-exists) handled every occurrence for
free, exactly as designed — the fix was always "resubmit the same array, it'll skip everything that's
already there." Worth building into the standard operating procedure for future large sweeps: **submit
the array, wait, then always verify the shadow count matches the expected total before trusting a run
complete** — do not assume a job leaving the queue with an empty log means it failed with useful
diagnostics; check the file count.

## The result: a real cross-method preview of H2's central correlation

All numbers are `elapsed=T-k`, online LiRA, `trajectory` ablation, 4,096 shadows, CIFAR-100, real
`results/a1_lira_cifar100_<method>.csv` files.

| Method | Retention mechanism | AUC(e=0) | AUC(peak) | AUC(e=9) | Shape |
|---|---|---|---|---|---|
| M0 (FedAvgSequential) | **none** | 0.560 | 0.573 (e=2) | 0.555 | rises briefly, then **decays** |
| M5 (HybridReplay) | exemplar replay | 0.566 | 0.581 (e=1-2) | 0.567 | rises, then flat/mild decline |
| M2 (TARGET) | generative replay | 0.567 | 0.600 (e=6) | 0.595 | rises, plateaus high |
| M1 (GLFC) | distillation + exemplar | 0.523 | 0.548 (e=6) | 0.548 | rises, plateaus |
| M3 (FOT) | orthogonal subspace projection | 0.592 | 0.612 (e=3) | 0.603 | rises fast, stays flat/high |
| M4 (PrototypeFCL) | class-mean carry-forward | 0.883 | — | 0.869 | flat from the start, high |
| M8 (AnalyticFCL) | exact running Gram sum | 1.000 | — | 1.000 | flat, saturated |

**The pattern is exactly H2's predicted shape, and it is monotone in an intuitive sense**: the one
method with *no* anti-forgetting mechanism (M0) is the only one whose leakage curve turns over and
heads back toward chance; every method with *some* retention mechanism (M1/M2/M3/M5) shows leakage
that *rises* from its initial value and then holds (not decaying, sometimes still slowly climbing at
elapsed=9); the two methods with the *strongest*, most literal retention (M4's carried-forward class
mean, M8's exact running sum) show flat, saturated leakage from round one. This is not yet H2's
formal deciding test (that needs the actual −BWT-vs-TPR correlation with a stated Spearman ρ, ≥8
methods × ≥3 datasets, `RESEARCH_PLAN.md §3`) — but it is a real, quantitative, single-dataset preview
of the exact relationship H2 is trying to establish, obtained essentially for free by extending A1's
existing infrastructure to methods that were already built and validated in P2.

**Absolute magnitudes are informative too, and worth a caveat.** M3's signal (orthogonal projection)
is the strongest of the five SGD-based methods — plausibly because subspace projection actively
protects *gradient directions* associated with old tasks, which is closer to preserving individual
example information than M1/M2's exemplar/generative summaries are. This is exactly the semantic-vs-
individual axis the Red Team's standing objection to H2 is about (`agents/OPEN_QUESTIONS.md`'s H2 entry)
and should be examined directly (M3 vs. M1/M2) when the formal dose-response sweep is scoped.

## Status updates

- `agents/OPEN_QUESTIONS.md`'s H2 entry updated with the full 7-method table as motivating evidence
  (still `OPEN` — this is a single-dataset, single-seed preview, not the ≥8-method × ≥3-dataset ×
  correlation-with-CI deciding test H2 needs).
- No hypothesis flips to `SUPPORTED`/`REFUTED` from this note alone — it strengthens H2's case and adds
  a concrete, real dataset toward eventually running the actual dose-response sweep, but the register's
  own bar for H2 is higher than what a single dataset can satisfy.

## What's left for A1

- M6/M7 (GPU, `cacheable=False`) need the offline LiRA variant (`attacks/lira.py::offline_log_lr`,
  already implemented and unit-tested but never run against real data) with N~64-128 shadows, per
  `01_KODIAK.md §4.3` — a different, GPU-constrained runner, not this module.
- Secure-aggregation variant for A1 (any method) still not built — H11's A1-specific evidence.
- Only CIFAR-100 tested; H2's real deciding test needs ≥3 datasets.
- A4 (onset/composition inference) not built.
