# 2026-09-16 — P3: A1 cross-task LiRA, headline numbers on real CIFAR-100

## What was built and run

`src/p3fcl/shadow_runner.py` (shadow-federation runner) + `src/p3fcl/attacks/lira.py` (LiRA
combination) + `code/scripts/run_shadows.py` / `python -m p3fcl.cli shadows` (same underlying
function) + `code/scripts/run_lira.py` (report driver). Scoped to M4 (F2/prototype) and M8 (F5/Gram)
per `04_METHODS_AND_ATTACKS.md §4`'s "A6 then A1, on M4-proto/M8" ordering — both are deterministic
given the stream (no SGD noise), which isolates the IN/OUT variable A1 needs cleanly and made this the
right pair to build first.

Real run: 4,096 shadows each (target budget 4,000; `CHUNK=128 × 32 array tasks` overshoots slightly,
harmless), CIFAR-100, 10 tasks × 10 clients, 500 tracked targets, population resampled at `p_in=0.5`
per shadow. Calibration/evaluation split 80/20 by shadow (3,277 calibration / 819 evaluation),
CLAUDE.md non-negotiable #6. Results: `results/a1_lira_cifar100_{m4_proto,m8_analytic}.csv`.

**Two real bugs caught and fixed before these numbers were trustworthy** — both written up in
`notes/2026-09-16_p3_a1_population_resampling.md`:
1. A `targets.json` write race across concurrent PBS array tasks (real crash, job 157138[2]).
2. A shadow-design flaw that gave an artifactual `AUC=1.0000` everywhere: resampling only the tracked
   targets (not the whole population) made each target the sole source of randomness for its own
   class's statistic under M4's class-partitioned release, collapsing the null to a point mass.
   Fixed by resampling the entire candidate pool per shadow (standard LiRA practice), which is what the
   numbers below are computed from.

## Headline numbers

**M4 (F2/prototype), online LiRA, `view=full`:**

| elapsed (T−k) | AUC | TPR@1%FPR | TPR@0.1%FPR | n (pos≈neg) |
|---|---|---|---|---|
| 0 | 0.883 | 0.345 | 0.192 | ~205k |
| 1 | 0.887 | 0.355 | 0.198 | ~184k |
| 2 | 0.887 | 0.357 | 0.199 | ~164k |
| 5 | 0.886 | 0.350 | 0.189 | ~102k |
| 9 | 0.869 | 0.299 | 0.153 | ~20k |

Real, non-degenerate, and stable across the full observed horizon (no sharp decay out to 9 elapsed
tasks) — a positive result for **H3** in the sense that matters here: `trajectory` and `last_round`
ablations are **exactly identical** at every elapsed value (to the printed precision), confirmed by
direct inspection of the CSV, not just visually. This was the pre-registered prediction
(`shadow_runner.py`'s module docstring, written before this run): M4 releases each target's evidence
in exactly one round, so summing K identical per-round log-LR terms is a per-target positive rescale
that does not change the pooled ranking. **H3's real test needs a method with genuine multi-round
re-exposure (F1's iteratively-updated model, e.g. M0) — not built this pass, stated follow-up.**

**M8 (F5/Gram), online LiRA, `view=full`:**

`AUC = 1.0000`, `TPR@1%FPR = TPR@0.1%FPR = 1.0000` at **every** elapsed value, over ~400k pooled
evaluation instances at elapsed=0 alone (819 eval shadows × 500 targets) — checked before trusting it,
exactly as the M4 fix above demanded: this is not a residual version of the same population-resampling
bug (verified the OUT-side has real, substantial between-shadow variance now, std ~125-430, not a point
mass) but a real, mathematically-explained result. Self-leverage `x^T R^-1 x` for an *included* point
is bounded below 1 and saturates very close to it whenever the per-class sample count is small relative
to the feature dimension (median per-class shard size ~84 *before* the 50% population halving, against
`d=768`) — the same `n << d` regime `RESEARCH_PLAN.md §3.4` already identified as giving near-exact
analytic recovery, and the same regime `notes/2026-09-16_p3_fig13_h5.md`'s real-data H5 run
independently found gives near-exact reconstruction via direct Gram inversion. **A1/M8 is a second,
methodologically independent confirmation of the same underlying fact (F5 leaks almost everything in
the realistic n<<d regime) — this time via a calibrated, shadow-based membership attack rather than
direct analytic inversion.** Worth citing both together in the paper as a robustness cross-check: two
unrelated attack methodologies agree.

## What this means for the register

- **H2 (retention-leakage flagship correlation)**: A1's TPR@1%FPR numbers (M4 ~0.30-0.36, M8 ~1.00) are
  now available as real inputs to the eventual −BWT-vs-leakage correlation, alongside A6's balanced
  accuracy — still needs ≥8 methods × ≥3 datasets per H2's own stated target, so H2 itself stays `OPEN`;
  this is infrastructure toward it, not the deciding test.
- **H3 (accumulation leakage, transcript vs checkpoint)**: real evidence gathered, but it's a **null
  result by design** for this method pair — `trajectory` and `last_round` are identical because M4/M8
  release once per target. Not evidence against H3 in general; the real test needs M0 (F1). Status
  stays `OPEN`, noted precisely rather than forced.
- **H11 (secure aggregation doesn't help)**: A1's secure-aggregation variant (scoring against
  `ledger.aggregate_view()` instead of per-client records) is **not built yet** for A1 specifically —
  A3/A5/A6 already have secure-agg findings, A1/A4 don't. Stated follow-up, not done this pass.
- **H5/H6 (analytic FCL's Gram statistics in the n<<d regime)**: A1/M8's AUC=1.0 result is additional,
  independent supporting evidence, worth citing alongside the existing FIG13 finding.

## Stated scope, for whoever picks this up next

- Only M4/M8 scored (both cacheable, deterministic, single-release). M0/M1/M2/M3/M5/M6 are not yet
  wired into `shadow_runner.py`'s `METHOD_REGISTRY` — extending to M0 (F1) is the natural next step and
  the one that actually tests H3.
- Offline LiRA (`attacks/lira.py::offline_log_lr`) is implemented and unit-tested but not exercised by
  this run — reserved for the non-cacheable methods (M3/M4-LoRA/M6/M7) per `01_KODIAK.md §4.3`. Its
  docstring already documents that F5's self-leverage statistic needs a sign flip if it's ever used
  there (Sherman-Morrison inverts the naive "higher = more IN" intuition for that specific family).
- Secure-aggregation variant for A1 not built.
