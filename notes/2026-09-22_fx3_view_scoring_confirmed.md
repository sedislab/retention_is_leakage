# 2026-09-22 — FX3 needs no new attack code: `view=` already does the aggregate/global scoring

## What happened

Before assuming FX3 (aggregate/global-state attacks, FIG11 v2/FIG19) needs a new pipeline, checked
whether `run_lira_pertask.py`'s existing `view` argument (added for FX4h's multi-view shadow scoring)
already produces correct, meaningfully different numbers for M4's three views on real wave V2 data
(cifar100, m4_proto, seed0). It does, with no code changes beyond the `_shadow_dir` fix in the
sibling note (`2026-09-22_fx2_stale_shadow_dir_bug.md`).

## Real numbers (job 183918)

| view | auc | tpr1 |
|---|---|---|
| full (per-client, constant by construction) | 0.880 | 0.325 |
| aggregate (count-weighted combination across clients) | 0.682 | 0.054 |
| global (running server prototype bank) | 0.558 | 0.067 |

Exactly the expected ordering and shape: `full` is the "structurally unattackable in practice" view
(no real adversary restricted to secure aggregation could ever see a single client's own release) and
overstates leakage; `aggregate` is what secure aggregation actually reveals and is substantially
weaker; `global` (the one view FX4h's docstring calls "the real retention measurement") is weaker
still. All three views are flat across every elapsed `e` in 0..6 for this (method, stream) config --
consistent with M4's shadow-generation config using `prototype_momentum=0.0` (`shadow_runner.py`'s
`_method_config`) on a class-incremental stream: each target's own class column is written exactly
once and never touched again under any of the three views, so nothing changes with elapsed time. Not
a bug -- matches the already-known "M4 is task-disjoint by construction under this project's
class-incremental streams" finding.

## Implication for FX3

The plan's "aggregate/global-state attacks" deliverable is mostly a **reporting exercise on top of
already-built infrastructure**, not a new attack pipeline: `run_lira_pertask.py view=aggregate` /
`view=global` and `build_fx2_summary.py` (unchanged) already produce correct, real per-view leak
numbers for M4/M8 (the only families with more than one view). What's still needed is FIG11 v2 itself
(grouped bars: full/aggregate/global x M0/M5/M4/M8) and FIG19 (a new figure, "retention vs. release")
-- plotting/analysis work, not attack-scoring work. Flagging this now so FX3 isn't over-scoped as a
from-scratch build when it starts.

## Status

Real, verified finding (one real job, not synthetic) -- not yet run at the full scale FX3 needs
(all 3 datasets x seeds x M4/M8 x 3 views), but the mechanism is confirmed correct on real data.
