# 2026-09-23 — M9 LiRA audit (FX5 §8, FIG10 v2): design notes before implementing

## Status

NOT implemented yet. Started scoping the `shadow_runner.py` integration while job 184034 (TAB05
regen) was running, found three real subtleties worth resolving carefully rather than rushing, and
stopped to write them down instead of shipping a half-correct integration. This is exactly the kind
of change (a leakage-measurement pipeline) where CLAUDE.md non-negotiable #2's "if unsure, over-report
and note it" cuts hardest against getting it subtly wrong.

## What the audit needs (`08_FIX_PLAN.md` §8, M9 audit)

Online LiRA against M9 on CIFAR-100, U1, eps in {1, 4, inf}, gamma=1, 1024 shadows x 1 seed, scored
by leverage on the `global` state. Plot empirical ROC (log-log) against the DP bound TPR <= e^eps *
FPR + delta. If the empirical TPR exceeds the bound beyond its CI, there's a bug -- stop and debug.

## Three real subtleties found while scoping the `shadow_runner.py` integration

1. **PCA-space mismatch.** `_score_target`'s existing GRAM-family `global` branch
   (`shadow_runner.py`) computes leverage as `feat @ solve(Rc, feat)` using the RAW feature vector.
   That's correct for M8 (`Rc` is `d x d`, matching raw features). For M9, the running state `R` is
   `p x p` (`p < d`, the PCA-projected dimension) -- `_score_target` would need to PROJECT (and
   B-cap) the feature the same way `m9_contractive.py::project_and_cap` does before computing the
   leverage, or the matrix dimensions won't even match. This is not just a parameter to thread
   through -- it means `_score_target`'s GRAM branch needs a method-aware projection step, which it
   currently has no hook for.

2. **Gamma-decay reconstruction.** `_reconstruct_gram_global` currently does a PLAIN cumulative sum
   (`R = R + rec.payload["R"]` every round) -- correct for M8 (no decay) and, by coincidence, also
   correct for M9 specifically AT `gamma=1` (the audit's own fixed config). It would be WRONG for any
   M9 audit run at `gamma != 1` (state update is `R <- gamma*R + G_tilde`, not a plain sum). Since the
   audit spec fixes `gamma=1`, this doesn't block THIS audit, but the function should still be
   generalized to accept a `gamma` parameter (default 1.0, preserving M8's exact behavior) rather than
   leaving a silent trap for the next person who reuses it at `gamma < 1`.

3. **Fitting the PCA basis once, not once per shadow.** `_method_config` is called INSIDE
   `_run_one_shadow` (once per shadow, 1024 times for this audit) -- it must NOT fit a fresh PCA basis
   there. The basis needs to be fit ONCE (same `_fit_pca` helper `run_m9_hparam_selection.py`/
   `run_m9_sweep.py` already use) in the PARENT process (`run_shadow_range`, before forking workers),
   then handed to workers via `_init_worker`'s existing `method_config_override` mechanism -- forked
   worker processes share it via copy-on-write, so passing a real numpy array through is fine (no
   serialization through a YAML `--set` string needed, which couldn't carry an array anyway).

## Proposed plan (not yet built)

- `shadow_runner.py`: register `"m9_contractive": (ContractiveDPAnalytic, Family.GRAM)` in
  `METHOD_REGISTRY`; add an `_method_config` branch returning sensible defaults EXCLUDING
  `pca_basis` (which only a caller with `dataset` access can supply -- calling `run_shadow_range` for
  M9 without a `method_config_override["pca_basis"]` should fail loudly with `KeyError` inside
  `ContractiveDPAnalytic.__init__`, not silently do something wrong).
- Generalize `_reconstruct_gram_global(ledger, n_rounds, d, gamma=1.0)` to apply the decay; M8's call
  site stays unchanged (default `gamma=1.0`); M9's call site in `_run_one_shadow` passes
  `method_cfg["gamma"]`.
- Extend `_score_target`'s GRAM `global`/`aggregate` branches to accept an optional projection
  (`pca_basis`, `B`) and apply `project_and_cap` before the leverage computation when scoring M9
  specifically -- needs a clean way to detect "this is an M9-family shadow" (e.g. pass `pca_basis` as
  `None` for M8, a real array for M9, and branch on that rather than on method name, keeping
  `_score_target` family-generic rather than adding a method-name special case).
- A new `run_m9_audit.py` (audit-specific caller): fits the PCA basis once per (dataset, unit, eps)
  via the hparam table, injects it into `config["method_config_override"]`, calls
  `run_shadow_range` for 1024 shadows x each of eps in {1, 4, inf}, then reuses
  `run_lira.py`/`attacks/lira.py`'s existing online-LiRA machinery scored against the `global` view,
  and compares empirical TPR/FPR against the analytic DP bound `TPR <= e^eps * FPR + delta`.
- Tests needed before trusting any of this: a synthetic check that `_score_target`'s M9 leverage
  computation matches a hand-computed value on a tiny case; a check that `_reconstruct_gram_global`'s
  `gamma` parameter reduces to the old plain-sum behavior at `gamma=1.0` (regression-safe for M8); an
  end-to-end smoke test generating a handful of M9 shadows on synthetic features and confirming the
  npz output loads and has sane `global` scores.

## Why this is logged instead of built right now

Real risk of a subtle, hard-to-catch correctness bug (wrong-dimension leverage, or a silently-stale
gamma=1 assumption baked in without the generalization) in a component whose whole job is measuring
privacy leakage -- worth doing carefully in a dedicated pass with the tests above, not shipped
partway while multitasking with an unrelated TAB05 regeneration job. Next session picking this up
should start from this note, not from scratch.
