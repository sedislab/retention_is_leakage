# 2026-09-22 — Caught before use: `_shadow_dir` would have scored stale pre-fix M4/M8/etc. shadows

## What happened

While checking whether FX3 (aggregate/global-state attacks) could reuse `run_lira_pertask.py`'s
existing `view` parameter against wave V2's multi-view M4/M8 shadows, I re-examined `_shadow_dir`'s
resolution logic and found a real bug that had not yet been exercised: it picked a shadow directory
by **existence on disk**, preferring `shadows/<dataset>/<method>` (no seed suffix) over
`shadows_v2/<dataset>/<method>/seed<S>` whenever the former existed at all.

`shadows/cifar100/m4_proto`, `shadows/cifar100/m8_analytic`, `shadows/cifar100/m1_glfc` (and
presumably the same for m2_target/m3_fot/m5_hybrid_replay, and for cub200/imagenet_r) are all real,
pre-fix directories left over from before FX4b-e's method fixes and FX4h's multi-view scoring. Wave
V2 (`shadows_v2/.../seed0`) now ALSO exists for every one of these methods. Since `_shadow_dir`
checked existence first, it would have returned the **stale pre-fix** directory for every non-M0
method at seed 0 -- silently scoring old, buggy-method-era shadow data (wrong `touched` sets, F8
records that shouldn't exist for M1, non-disjoint M2/M3, single-view-only M4/M8) as if it were wave
V2's fixed data, with no error or warning.

## Why this didn't corrupt anything yet

Job 183899 (`build_fx2_summary.py` real-data run) only used `m0_fedavg`, which is the ONE method
`_shadow_dir` is *supposed* to route to the legacy store for (FX4i, `08_FIX_PLAN.md` §4i: "M0: no
new shadows if it passed test (i)."). That path was and remains correct. The bug would only have
fired the moment FX2's leak pipeline ran against any of M1/M2/M3/M4/M5/M8 at seed 0 -- which was the
very next planned step (STATE.md's "run FX2's leak pipeline on the other ~50 combinations"). Caught
during a pre-check, not after producing a wrong number.

## Fix

`_shadow_dir` (`code/scripts/run_lira_pertask.py`) now only considers the legacy `shadows/` path at
all when `method == "m0_fedavg"`; every other method goes straight to `shadows_v2/.../seed<S>`,
unconditionally, regardless of what else exists on disk. Regression test added:
`test_shadow_dir_never_uses_the_stale_legacy_store_for_non_m0_methods` (constructs both a stale
legacy dir and a real v2 dir for `m4_proto`, asserts the v2 one is chosen).

## Lesson

"Prefer whichever path exists" is a dangerous default whenever two DIFFERENT eras of the same kind of
data can coexist on disk with different meanings under the same naming scheme minus one path
component (here, the presence/absence of a `seed0` subdirectory). The safe version needed an explicit
allowlist (`method == "m0_fedavg"`) for the one case where reusing old data is actually intended,
rather than a general "check if it's there" fallback.
