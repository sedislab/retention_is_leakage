# 2026-09-22 — FX4g automatic gate: results and a real, unresolved finding

## What happened

Job 183846 (FX4g gate, `code/scripts/run_fx4g_gate.py`) completed: 3 datasets x {M0 baseline + M1,
M2, M3, M5} x 3 seeds, post-fix test accuracy, against the plan's pass condition (BWT <= +0.02,
last-task accuracy >= M0's last-task accuracy - 0.15, final average accuracy >= M0's). Full results
in `results/fx4_gate.csv` (`config_id` column: `default`, `tuned`, or `n/a (baseline)` for M0 rows).

## Pass/fail summary

| dataset | M1 | M2 | M3 | M5 |
|---|---|---|---|---|
| cifar100 | **PASS** (tuned: lr=0.1, local_epochs=30, distillation_weight=0.5) | FAIL | PASS (default) | PASS (default) |
| cub200 | FAIL (tuning grid found no config with val_bwt<=0.02) | FAIL | PASS (default) | FAIL |
| imagenet_r | FAIL (same) | FAIL | PASS (default) | FAIL |

M3 (orthogonal projection, no buffer, no distillation) passes on every dataset. M2 fails everywhere.
M1 and M5 pass only on CIFAR-100.

## The real finding: CUB-200/ImageNet-R failures are a different failure mode than CIFAR-100's

CIFAR-100's pre-tuning M1 failure looked like the "expected" kind: moderate negative BWT, somewhat
low last-task accuracy, fixed by lowering `distillation_weight`. CUB-200 and ImageNet-R do NOT fail
that way. For M1/M2/M5 on both datasets, **BWT is strongly POSITIVE** (M1: +0.25 to +0.27 on cub200,
+0.08 to +0.10 on imagenet_r; M2: +0.69 to +0.73 on cub200, +0.25 on imagenet_r; M5: +0.45 to +0.49
on cub200, +0.13 on imagenet_r) **while last-task accuracy collapses toward 0** (M1 on cub200: 0.03-
0.08, vs. M0's 0.71-0.93; M2 on cub200: 0.0-0.002). The tuning grid (`results/fx4_gate_grid.csv`)
confirms this isn't a config artifact: every one of M1's 8 grid points on cub200/imagenet_r has
val_bwt well above the 0.02 threshold (cub200: 0.24-0.56; imagenet_r: 0.10-0.42) -- there is no
config in the searched grid that fixes it, so `run_fx4g_gate.py` correctly fell through to "keep the
best configuration, mark GATE FAILED" per the plan's own fallback, not a bug in the gate script.

## Why this is worth flagging as its own thing, not just "gate failed"

A large POSITIVE BWT combined with near-zero last-task accuracy is not the same failure as ordinary
forgetting (which would show negative BWT). It looks like these retention mechanisms (KD toward old
classes for M1, generative replay for M2, exemplar replay for M5) bias the model so heavily toward
previously-seen classes on these harder, more fine-grained/class-numerous datasets (200 classes vs.
CIFAR-100's 100) that the model barely predicts the CURRENT task's brand-new classes at all right
after learning them -- and only "recovers" on old tasks later, which reads as positive backward
transfer relative to an already-poor initial fit. CIFAR-100 doesn't show this because M1's default
`distillation_weight=1.0` combined with only `exemplar_budget=10` per class evidently isn't enough
retention pressure to trigger it there, but IS on the two harder datasets. Untested here: whether
this is primarily a class-count effect, a per-class-sample-count effect, or specific to how KD/replay
interact with ViT features on fine-grained categories. NOT EVIDENCE toward or against any of C1-C4 by
itself -- it's a utility-side finding about these retention mechanisms' practical behavior on
harder-fine-grained streams, not a privacy result.

## Status / next steps

Logged here per CLAUDE.md non-negotiable #7 (negative results are results). Not fixing further this
session -- the plan's own fallback ("keep the best configuration; mark GATE FAILED... continue. Do
not stop") applies, and a third tuning variant is explicitly against the plan's "do not try a third
variant" rule for repeated same-reason failures. `agents/OPEN_QUESTIONS.md` should get a line noting
this once its hypothesis mapping is next touched (FX7 errata pass). `results/accuracy_matrix_*.csv`
regeneration (jobs 183855/183856/183857) uses `fx4_gate.csv`'s tuned config for CIFAR-100 M1 and each
method's plain default everywhere else, including the two GATE FAILED datasets -- their numbers will
visibly show this collapse, which is the honest thing for FIG01/FIG02/TAB05 to show rather than
hiding it behind a config search that didn't find a fix.
