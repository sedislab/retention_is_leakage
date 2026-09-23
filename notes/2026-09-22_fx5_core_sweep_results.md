# 2026-09-22 — FX5 M9 core sweep: clean gamma dose-response; client-scaling barely helps U2

## What happened

Jobs 183914/183915/183916 (`run_m9_sweep.py`, one per dataset) completed: 360 total rows in
`results/m9_sweep.csv` (main grid: 2 units x 6 eps x 3 gammas x 3 seeds x 3 datasets = 324; M0/M8
references: 3 datasets x 3 seeds x 2 methods = 18; CIFAR-100 U2 n_clients-scaling arm: 2 n_clients x
3 eps x 3 seeds = 18).

## Finding 1: gamma (retention strength) dose-response is clean and monotonic on ALL 3 datasets

At eps=inf, U1 (isolating the retention-strength effect from DP noise):

| dataset | gamma=1.0 acc / bwt | gamma=0.9 acc / bwt | gamma=0.7 acc / bwt |
|---|---|---|---|
| cifar100 | 0.884 / -0.047 | 0.867 / -0.082 | 0.735 / -0.249 |
| cub200 | 0.868 / -0.056 | 0.802 / -0.161 | 0.530 / -0.482 |
| imagenet_r | 0.634 / -0.094 | 0.604 / -0.179 | 0.399 / -0.460 |

Lower gamma (more contraction, i.e. more aggressive decay of the running statistic) monotonically
costs BOTH accuracy and BWT, on every dataset, with 3/3 seeds agreeing on direction every time. This
is exactly the dose-response claim C2 is about, now demonstrated concretely for M9's own retention
knob (gamma) rather than only for M1/M5's exemplar-count knobs from the earlier pilot. NOT the
leakage side of the dose-response yet (that needs the M9 LiRA audit, not run this pass) -- this is
the utility/forgetting side only.

## Finding 2: U2 (client-level DP) utility barely improves with more clients, and only at loose eps

CIFAR-100, U2, gamma=1.0, mean final_avg_acc over 3 seeds:

| eps | n_clients=10 | n_clients=50 | n_clients=100 |
|---|---|---|---|
| 1 | 0.015 | 0.016 | 0.016 |
| 4 | 0.016 | 0.019 | 0.018 |
| 8 | 0.017 | 0.042 | 0.048 |

Scaling from 10 to 100 clients roughly triples utility at eps=8 (0.017 -> 0.048) but does essentially
nothing at eps=1 or eps=4 -- and even the best case (0.048) is nowhere near the non-private ceiling
(~0.88 for cifar100 at eps=inf). This directly answers `08_FIX_PLAN.md` §8's "report what it shows"
instruction: **federation size helps U2 utility only modestly, and only once the privacy budget is
already fairly loose; it does not rescue client-level DP in this regime.** Matches, and sharpens,
`notes/2026-09-22_fx5_hparam_selection_results.md`'s earlier observation that U2 was pinned near
chance during hyperparameter selection -- that was not solely a small-federation artifact.

## Status and next steps

Real evidence for C2 (gamma dose-response) and a real, reportable negative result for the
client-scaling question -- not yet turned into FIG08 v2/TAB08 (analysis/plotting scripts not written
this pass) or cross-checked against the M9 LiRA audit (not run). `results/m9_sweep.csv` is the single
merged source (`merge_m9_sweep.py`, pure concatenation of the 3 per-dataset files, no computation) --
any figure/table built from this should read that file, not recompute.
