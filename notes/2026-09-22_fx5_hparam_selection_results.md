# 2026-09-22 — FX5 M9 hyperparameter selection (`ref` split): U2 is stuck near chance, as expected

## What happened

Job 183861 (`run_m9_hparam_selection.py`) completed: 900 rows in `results/m9_hparam_selection.csv`
(3 datasets x 2 units x 6 eps x 5 p x 5 lambda, gamma=1.0 fixed, selected on `ref-val`).

## The pattern

- **eps=inf (non-private)**: scales sensibly with `p` (more PCA dimensions -> better accuracy) and
  reaches 0.72-0.88 on cifar100/cub200 at full dimension, 0.51 on imagenet_r (harder, 200 classes,
  fewer `ref` samples per class).
- **U1 (example-level), finite eps**: on CIFAR-100 (most `ref` samples per shard, ~50), a real,
  smooth privacy-utility tradeoff: 0.043 at eps=0.5 climbing to 0.66 at eps=8, approaching the
  non-private 0.88. On cub200/imagenet_r (fewer `ref` samples per shard, 200 classes), U1 stays much
  closer to chance even at eps=8 (0.02 and 0.03 respectively) -- consistent with "less data per shard,
  more classes -> DP hurts more," not obviously a bug.
- **U2 (client-level), every eps 0.5-8, every dataset**: pinned near chance (0.007-0.025) regardless
  of eps -- essentially NO learning signal survives client-level DP at 10 clients, even at the least
  private tested budget (eps=8), while the same (dataset, p) combination's non-private ceiling is
  0.72-0.88.

## Why this is not (yet) treated as a bug

`08_FIX_PLAN.md` §8 anticipates exactly this: "U2 with 10 clients is expected to be hard, because
client-level DP with few clients usually is," and prescribes the fix as part of the core sweep, not
the hparam selection step: "Also run U2 on CIFAR-100 with n_clients in {50, 100} and eps in {1, 4, 8}.
The figure then shows how U2 utility scales with federation size." The observed U1-vs-U2 gap here
matches a well-known, expected property of client-level DP (fixed per-release noise divided across
more, independently-clipped client contributions improves SNR as client count grows) rather than
looking like an implementation defect -- reassuring given how stark the U2 collapse is (0.02 vs. 0.88
non-private) that a bug would also produce.

## What would change this assessment

If the CORE SWEEP's n_clients-scaling arm (U2 at n_clients in {50, 100}) does NOT show U2 utility
recovering as client count grows, that would be the point to seriously suspect an implementation bug
in M9's clipping/noise calibration (`m9_contractive.py::clip_pair`/`_sigma`) rather than accepting
"client-level DP is just hard" as the whole story. Flagging this now so whoever runs/reads that part
of the sweep next has the expected-vs-suspicious threshold written down in advance, before seeing the
result and being tempted to rationalize it either way.

## Status

NOT EVIDENCE toward C4 by itself (this is a small-`ref`-split hyperparameter selection pass, not the
real train-split core sweep) -- but a useful sanity check that `m9_contractive.py`'s DP mechanics
produce a plausible, plan-anticipated pattern rather than an obviously broken one, before spending
real train-split compute on the full core sweep.
