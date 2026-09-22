# 2026-09-18 — P4 extended to all 7 A1-validated methods — decoupling holds for 6/7

## What was done

Extended P4's first pass (M0 vs M8 only, `notes/2026-09-17_p4_fig01_decoupling.md`) to all 7 methods
A1 already covers (M0/M1/M2/M3/M4/M5/M8), same dataset (CIFAR-100), same 5 seeds
(`{0,1,2,3,4}`). Reused identical infrastructure with no new code beyond wiring the extra methods
into `build_fig01.py`/`build_fig02.py`'s `METHOD_FAMILY` dict and `run_accuracy_matrix.py`'s
`METHODS` dict — this was cheap precisely because A1's shadow-runner and LiRA report machinery is
already generic over any registered method.

This meant generating 20 new (method, seed) shadow stores (5 methods x 4 new seeds, 4,096 shadows
each) and their LiRA reports. **The recurring silent-array-task-death pattern
(`notes/2026-09-16_p3_a1_population_resampling.md` et seq.) showed up at higher intensity than any
single-batch run so far** — submitting 20 concurrent 32-task arrays at once (640 total subtasks
competing for the cluster's ~64-task concurrent-execution ceiling under the 512-core cap) produced a
much higher rate of silent deaths per round than the earlier 2-array (M0+M8) batches did. Needed three
full resubmission rounds plus one small targeted backfill before every store reached 4,096/4,096 --
still zero data loss (resumability worked exactly as designed every time), just more rounds than
usual. **Lesson for future large sweeps: stagger big multi-method submissions rather than firing them
all in one burst, or budget more backfill rounds up front.** One session hiccup, caught immediately
and harmless: a stale `/tmp` copy of the "bigmem" PBS template (predating the SEED-parameter code)
briefly ran against the wrong output directory during an isolated-retry attempt -- a costless no-op
since every shadow it touched already existed there, but worth a reminder to regenerate `/tmp`
scratch templates from the current source file rather than reusing old copies across a session.

## The result: decoupling holds for 6 of 7 methods, with real quantitative variation

`results/decoupling_ratio.csv` (paired bootstrap over seeds, per `notes/2026-09-17_p4_fig01_decoupling.md`'s
methodology):

| Method | Retention mechanism | Decoupling ratio (h_leak / h_acc) | 95% CI | Excludes 1.0? |
|---|---|---|---|---|
| M0 FedAvgSequential | none | 3.75 | [2.60, 6.87] | **yes** |
| M1 GLFC | distillation + exemplar | ∞ (no leak decay) | — | **yes** (trivially) |
| M2 TARGET | generative replay | ∞ (no leak decay) | — | **yes** (trivially) |
| M3 FOT | orthogonal projection | 2.80 | [0.72, 26.12] | **no** |
| M4 PrototypeFCL | class-mean carry-forward | 3.71 | [1.81, 176.57] | **yes** (wide) |
| M5 HybridReplay | exemplar replay | ∞ (no leak decay) | — | **yes** (trivially) |
| M8 AnalyticFCL | exact running Gram sum | ∞ (no leak decay) | — | **yes** (trivially) |

**6/7 methods show real evidence of decoupling** (leakage decays slower than accuracy, in some cases
not at all). **Only M3's confidence interval includes 1.0** — its point estimate (2.80) still points
the same direction as everyone else's, but with only 5 seeds the bootstrap CI is wide enough
(`[0.72, 26.12]`) that "leakage decays exactly as fast as accuracy" cannot be ruled out at 95%
confidence for this specific method. Reported as inconclusive, not forced into either a confirm or a
null — the honest reading is "directionally consistent, not yet significant."

## A real complication worth flagging precisely: M1's accuracy curve isn't monotonic

M1 (GLFC)'s accuracy-half-life fit has R²=0.043 — essentially no fit at all. Looking at the raw
normalised-accuracy curve directly (`results/fig01_decoupling.csv`): `1.00, 1.79, 2.11, 2.29, 2.29,
2.38, 2.46, 1.77, 1.33, 0.83`. **Accuracy on task k more than doubles by elapsed=6 before declining**
-- a real, substantial *positive backward transfer* effect (consistent with the accuracy-matrix run's
own `bwt` numbers for M1: +0.27 to +0.37, the only method with positive BWT among the seven), plausibly
from GLFC's knowledge-distillation term actively consolidating/improving old-task representations as
training continues, not just protecting them from forgetting.

**An exponential-decay model is the wrong shape for a curve that rises before it falls**, so M1's
reported `acc_halflife=26.20` should not be trusted as a meaningful number (the code correctly still
marks `fit_ok=True` -- it only checks that the fitted trend is net negative over the full range, not
that the model is a *good* description -- so a low R² is the reader's own signal to distrust the
number, and it should always be checked alongside the point estimate, not just fit_ok). **This does
NOT undermine M1's headline finding**: `ratio=inf` is driven entirely by the leakage side being
censored (no decay detected at all, independent of whatever the accuracy curve is doing), so "M1's
leakage never decays" stands on its own regardless of the accuracy-side complication. Only the
*specific* "3.75x-style" ratio *magnitude* would be unreliable for a method like M1, and no such
magnitude is being claimed for it (it's reported as infinite/censored, not a finite ratio).

## Visualization notes

`figs/fig01_decoupling.pdf`: 5 of 7 methods share family F1 (`Family.MODEL_DELTA`) and therefore share
one color under `plotting.py`'s "one line per family" convention -- distinguished from each other by
linestyle instead. At the scale needed to show M1's dramatic accuracy overshoot, the closely-clustered
low-leakage F1 methods (M0/M1/M2/M3/M5, all in the 0.02-0.05 TPR@1%FPR range) are visually
indistinguishable from each other in the bottom panel -- an honest reflection of the real data (they
really are all similarly low), not a rendering defect, but a real presentation limitation worth
revisiting (e.g. a log-scale leakage axis, or splitting F1 methods into their own sub-panel) before
this goes in the paper.

`figs/fig02_halflife.pdf`: switched to a log-x axis (`03_RESULTS_SPEC.md`'s own "log scale where the
data is multiplicative" rule) after M4/leak's real bootstrap CI upper bound came back at ~4,300 elapsed
tasks against a 9-task horizon -- a genuine consequence of the `tau = -1/slope` reciprocal transform
blowing up when some resampled seed subset's fitted decay is very shallow, not a bug. On the log axis,
every method's leak-half-life point (or censored arrow) visibly sits to the right of its own
acc-half-life point -- the decoupling pattern is now visible at a glance across all 7 methods.

## Status

No hypothesis status change from this note alone (H2/C1 already updated in `agents/OPEN_QUESTIONS.md`
after the M0/M8 first pass) -- this note substantially *strengthens* that evidence (6/7 methods now
support it, with real quantitative variation and one honest null) and documents the M1 accuracy-model
caveat precisely so it doesn't get silently overclaimed later.

## What's left for P4

- M3's inconclusive result: more seeds would narrow its CI; whether it would end up excluding or
  including 1.0 with more data is a real, stated open question, not assumed either way.
- Still 1 dataset only -- H2's own bar needs ≥3.
- FIG06 (long-horizon T=50 variant) not attempted.
- No A6/property panel in FIG01 (would need M4's A6 numbers, which exist from the earlier P3 pass,
  wired into this same pipeline -- a cheap, stated next increment).
- The visualization limitation noted above (F1-family methods hard to tell apart at this scale) is
  worth addressing before any of this goes in the paper, even though the underlying data is correct.
