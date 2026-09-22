# 2026-09-16 — P3: FIG05, and a real under-reporting bug in the accountant caught by real ledgers

## What happened

Built `run_fig05.py`: runs each of the 7 cacheable methods once on real CIFAR-100 features (10
tasks, seed 0) and accounts the resulting real ledger under units U1-U5 at matched sigma=2.0 over
T in {1,5,10,50,100,500,1000}. First run (job 151548) completed cleanly and the qualitative pattern
matched the claim tests exactly: M8 and M4 (both certified `task_disjoint=True`) show a perfectly
flat eps(T) under U2 (parallel composition, T1-fwd); every method shows strictly increasing,
unbounded eps(T) under U4 (H1) regardless of disjointness, matching T1's converse (task-disjointness
does not save a single client's cumulative loss under U4).

**But inspecting the actual numbers surfaced a real bug**: M0 (30 local SGD epochs per release,
`passes_over_data=30`, correctly flagged `V5b` by the disjointness checker) produced the *identical*
eps(T) curve as M8 (a single-pass, `passes_over_data=1` method) under every unit except U2. That's
wrong — 30 local epochs over the same shard within one release is real, additional sequential
composition, and it should cost more privacy budget than one epoch, not the same.

Root cause: `dp/accountant.py`'s `_max_task_touch_multiplicity` (which drives eps computation for
every unit except the task-disjoint+U2 fast path) counted how many *ledger records* touch each datum
per task, but never looked at `passes_over_data` on those records. `check_disjointness` reads
`passes_over_data` correctly (that's how V5b gets flagged at all) — but the number that actually
computes eps silently ignored it. A method could be flagged non-disjoint for exactly the right reason
and still get an eps(T) curve that pretends the extra passes never happened.

This is the same failure family as M5's under-reported `touched` and M6's dead `pool_keys` gradient:
a real, load-bearing piece of information was being *tracked* somewhere in the system but not
*used* where it mattered, and nothing crashed — it just quietly under-stated a privacy cost. Fixed by
having `_max_task_touch_multiplicity` count `passes_over_data` (clamped to >=1) instead of a flat 1
per touching record. Added a regression test
(`test_account_multi_epoch_gives_higher_eps_than_single_epoch`) that fails without the fix. Re-ran
FIG05 (job 151915) with the fix in place.

## Standing pattern worth naming explicitly

Three times now in one session (M5's `touched`, M6's `pool_keys` gradient, and now this), the bug
had the same shape: a field or code path existed and was *read* by one part of the system (the
disjointness checker, in two of three cases) but silently ignored by another part that should also
have used it (the release, the optimizer, the eps computation). None of these crashed; all three
would have produced plausible-looking, wrong numbers if not caught. The checklist this suggests for
anything new in this codebase: **whenever a field like `touched` or `passes_over_data` exists
specifically because it's privacy-critical, grep for every place that reads it and confirm each one
that *should* actually changes behavior when the field's value changes** — a field that's computed
correctly but not load-bearing everywhere it should be is worse than no field at all, because it
looks like the system accounted for something it didn't.

## What the next person needs to know

- `results/fig05_eps_of_T.csv` from job 151548 (pre-fix) should be treated as superseded once job
  151915 (post-fix) lands — do not cite numbers from the first run.
- The fix only affects the "otherwise" (non-task-disjoint-U2) accounting path. The task-disjoint+U2
  parallel-composition fast path was already correct (it uses `k=1` regardless of touch count, which
  is right per T1-fwd — a single task-epoch's release is one release regardless of how many local
  epochs happened locally, since parallel composition doesn't care about within-task composition by
  definition of the unit U2 adjacency).
