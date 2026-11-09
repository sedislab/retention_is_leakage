# R1 protocol: release-only audit of M2/M3 (fixed 2026-11-06, before any R1 job ran)

This file adds detail to `REBUTTAL_EXPERIMENTS.md` §5 where that section leaves a choice open. Nothing
here changes after a result is seen.

## Inputs (paired with the paper)
- Paper shadow stores (via `p3fcl.paths.shadow_dir`): M2 = `shadows_v3/<ds>/m2_target/seed{0,1,2}`,
  M3 = `shadows_v2/<ds>/m3_fot/seed{0,1,2}`, 1,024 shadows each, ids 0–1023.
- Same base stream (`build_stream(y, arange, 10, 10, 0.5, seed)`), same target pool
  (`build_targets(stream, 5, seed)`), same mask (`_population_mask(ds, method, sid, n_train, 0.5)`).
- **Pairing check.** Each `shadows_rb` file must have `target_ids` and `in_out` equal to the paper
  shadow file with the same id. Otherwise the array task fails.

## Released statistics (target's task k only; classes are disjoint across tasks)
- **M2 `client`**: the target's own client, class c = the target's label. Mean = `feats.mean(0)`,
  var = `feats.var(0) + 1e-6` over that client's members of class c. Same expressions as
  `TARGET.fit_task` (the F6 GENERATIVE record payload).
- **M2 `broadcast`**: the count-weighted aggregate over every client with members of class c:
  `agg_mean`, `agg_var = max(E[var+mean^2] - agg_mean^2, 1e-6)`. This is `TARGET.fit_task`'s
  `_generators[c]`.
  Note: the ledger has no record for this broadcast generator, and per-class counts are not
  in the GENERATIVE meta. So it is computed from member features with the method's formula, and the
  equivalence test checks it against `_generators`.
- **M2 score**: diagonal Gaussian log-likelihood
  `sum_j [-0.5 log(2π var_j) - 0.5 (f_j - mean_j)^2 / var_j]`.
- **M3 `server`**: `R = Xc^T Xc`, where `Xc` is the task's member features (sorted ids) minus their
  own mean. This is the GRAM payload `"R"`.
- **M3 `broadcast`**: the top `subspace_rank = 8` eigenvectors of R (`eigh`, sorted descending) =
  `new_dirs` in `FOT.fit_task`. The paper's V2 M3 shadows used `_method_config` with no override, so
  `subspace_rank = 8`.
- **Centre for M3 scores**: `mu_ref` = mean of the public `ref` features whose label is in task k's
  class set. Centred target feature `g = f - mu_ref`. The same centre is used for both M3 views.
- **M3 `broadcast` score**: `||V^T g||^2 / ||g||^2`.
- **M3 `server` score**: `g^T (R + λ I)^{-1} g` with **λ = 1.0**. The existing GRAM scorer
  (`shadow_runner._score_target`) solves on the released payload as is, and every GRAM payload it
  scored in the paper (M8) carries `ridge_lambda = 1.0`. M3's payload carries no ridge and is
  singular whenever a task has fewer members than d = 768 (CUB-200: about 250 per shadow). So λ = 1.0
  is the paper's GRAM-family convention, chosen without looking at any score. It is not tuned.
- **Missing statistic → NaN.** This follows `_score_target`'s PROTOTYPE `full` convention. NaN
  occurs when, for example, the target's client has no member of class c in this shadow. NaN scores
  drop out of LiRA exactly as in the paper pipeline. The NaN count per (ds, method, view) is logged.
- Scores are constant from round k onward and NaN before k (each release happens once). They are
  written for all 500 targets and all 10 tasks.

## LiRA and bootstrap (existing code, imported, not copied)
- `run_lira.compute_log_lr_surfaces` with the calibration split from
  `rng.seeded(run_lira_pertask._calibration_key(ds, method, "full"), 0).permutation(1024)[:819]`.
  This is the paper's `full` split, so the comparison is paired shadow by shadow.
- K = {0,1,2,3}, E = 6, pooled over K. Both ablations (`trajectory`, `last_round`).
- Hierarchical bootstrap: `build_fx2_summary` internals (`_seed_arrays`, `_resample_targets`,
  `_replicate_all_metrics`, `hierarchical_bootstrap`), 2,000 replicates, seed 0, seeds 0–2.

## Prediction (from §5, fixed)
For each (dataset, method, release view): release-only TPR@1%FPR at e = 0 (point estimate,
`trajectory` ablation) ≥ the paper's margin TPR@1%FPR at e = 6, from
`results/a1_lira_fixedk_summary_<ds>_<method>_full.csv` (`ablation = trajectory`, `elapsed = 6`).
Trajectory is the paper's primary ablation (`build_fx2_summary` builds the half-life and curves from
it). The `last_round` paper value is also copied for reference. `prediction_holds` is computed for
each release view separately. The e = 0 and e = 6 release values should be equal (a constant release
under a monotone pooled transform).

## Combined arm (original shadows exist, so it runs)
Paper margin scores `scores_full` (paper shadow files) and the **broadcast** release score are
scored separately with the same calibration split, giving per-round log-LRs `L_m(r)` and `L_b(r)`.
- `view = combined` (the spec, literal): per-round sum `L_m(r) + L_b(r)`, then the paper's
  trajectory cumsum and last-round readout. The release LLR is counted once per round from k on.
- `view = combined_once` (secondary): `trajectory(e) = cumsum L_m up to k+e, plus L_b once`;
  `last_round(e) = L_m(k+e) + L_b`. Counting the single release once is the more principled
  version, so both are reported.
- In both, a missing (NaN) release LLR contributes 0 (no information). The combined LLR is defined
  wherever the margin LLR is defined.

## Outputs
- `shadows_rb/<ds>/<method>/seed<S>/shadow_XXXXXX.npz`: existing schema (`shadow_id`,
  `target_ids`, `target_task`, `target_client`, `in_out`, `views`, `scores_<view>`), plus a
  `targets.json` copy.
- `results/rb_release_audit.csv` (+ sidecar, RUN_LOG). Columns per §5.5, plus `ablation`, CI columns
  for tpr01/auc, `n_seeds`, `paper_margin_tpr1_e6_last_round`, and `nan_frac`. `prediction_holds`
  is `n/a` except for release views at e = 0.

## Gates inside R1
1. Synthetic unit tests in `make test` (statistics vs. the method's own `fit_task`).
2. Real-data equivalence job: 3 shadow ids × 6 (ds, method), running the real `fit_task` (paper
   config) on the masked stream for tasks 0..3. Asserts max |diff| < 1e-10 on
   M2 client mean/var, M2 `_generators`, M3 R, and M3 subspace projector. The array depends on it
   (`afterok`).

## Amendment 1 (2026-11-06, after equivalence job 263193 failed; the statistics themselves are unchanged)
Job 263193 checked 144 values. All 108 statistic checks (M2 client mean/var, M2 broadcast generator,
M3 R) matched with max |diff| = 0. All 36 failures were the M3 check "V lies in FOT's updated
subspace U" (`max|V - U U^T V|`), with residuals between 8.9e-8 and 4.2e-7. The cause is the
features, which are float32, so FOT's `np.linalg.qr` runs in float32. That is rounding in the
comparison, not a difference in the released statistic. The check is replaced by:
- `broadcast_subspace_exact`: V against FOT's own eigendecomposition lines applied to FOT's released R
  (tolerance 1e-10). This is the statistic check.
- `broadcast_subspace_in_U`: the old span check with tolerance 1e-5 (about 100 × float32 eps), kept as
  a secondary consistency check.
The failed attempt's output is kept as `rebuttal/r1_equivalence_attempt1_263193.csv`. No score,
statistic or prediction changed.
