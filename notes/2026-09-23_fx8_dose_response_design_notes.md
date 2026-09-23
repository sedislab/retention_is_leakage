# 2026-09-23 — FX8 (C2 dose-response rerun): design notes before implementing

## Status

NOT implemented yet. Scoped carefully instead of rushed, for the same reason the M9 audit got a design
note first: this is a leakage-measurement pipeline (claim C2's headline evidence), and a subtly wrong
population/normalization choice here would produce a misleading dose-response curve, not just a noisy
one. Picking this up while the M9 audit's real shadow generation ran in the background.

## What the plan asks for (`08_FIX_PLAN.md` §10)

- Dataset: CIFAR-100. 512 shadows x 3 seeds per level (this phase's seed-count override, not
  `03_RESULTS_SPEC.md`'s stricter >=5 -- `08_FIX_PLAN.md` §0.10 explicitly overrides it for
  shadow-based figures).
- **M5** (individual retention): `exemplar_budget` in {1, 2, 5, 10, 20, 50}.
- **M2** (semantic retention): `n_synthetic_per_class` in the same six levels.
- Drop the M4 arm (its knob, `prototype_momentum`, was dead code under class-incremental streams --
  H6's own finding, already documented: momentum never fires because a class is assigned to exactly
  one task). Say so in the errata, not silently.
- FIG03 v2 (`results/fig03_dose_response.csv`): `dataset, method, knob_name, knob_value,
  retention_bwt, tpr1, tpr01, final_acc, seed, ci_lo, ci_hi` -- per `03_RESULTS_SPEC.md` §... this is
  a **per-seed row schema** (has both `seed` and `ci_lo`/`ci_hi` columns), meaning each row's CI is
  that row's OWN within-seed Clopper-Pearson interval on `tpr1` (matching `run_lira_pertask.py`'s
  `cp_lo`/`cp_hi` per-seed convention), not a pre-aggregated cross-seed CI. A separate step (the
  FIG03 plot script, or a small aggregator) does cross-seed aggregation for the actual figure, mirroring
  how `a1_lira_fixedk_summary.csv` aggregates the raw per-seed `a1_lira_pertask_*.csv` files.
- FIG04 v2 (`results/fig04_semantic_vs_individual.csv`): `dataset, method, retention_type
  {semantic|individual}, knob_value, tpr1, retention_bwt, seed, ci_lo, ci_hi` -- same per-seed
  convention. `retention_type` is already a `MethodSpec` field (`m2_target`="semantic",
  `m5_hybrid_replay`="individual", confirmed in the method source) so this is a direct lookup, not a
  new classification.
- Retention measured as **-BWT on TEST accuracy** -- the old pilot used `bwt_train` as an FX4a
  stopgap; this needs the real `eval_sets` fix like every other post-FX4a accuracy computation.
- FIG03's x-axis is "retention strength (normalised per method)" -- M5's `exemplar_budget` and M2's
  `n_synthetic_per_class` are literally the same six integer levels {1,2,5,10,20,50} by construction
  (the plan picked matching level sets specifically so this works), so "normalised" likely just means
  plotting against the shared level INDEX (0..5) or the raw value on a shared log axis, not a nontrivial
  per-method rescaling -- confirm this reading before building the plot script, but it does not block
  building the DATA pipeline, which only needs `knob_value` recorded as-is.

## What already exists and can be reused directly

- `shadow_runner.py`'s `METHOD_REGISTRY` already has both `m2_target` and `m5_hybrid_replay`
  (`Family.MODEL_DELTA`/GENERATIVE and EXEMPLAR/MODEL_DELTA respectively) -- no shadow_runner code
  changes needed, unlike M9's integration. `method_config_override` already threads a knob override
  through cleanly (`test_run_shadow_range_respects_method_config_override` already covers this pattern
  for M5's `buffer_size_per_class`; the same mechanism works for `exemplar_budget` and
  `n_synthetic_per_class` -- confirm these are the exact config keys each class's `__init__` reads,
  not e.g. a differently-named knob, before generating anything).
- `run_lira_pertask.py`'s K_SET/E_MAX fixed-population design, `_pooled_scores_labels`, `_report_row`,
  and the whole per-seed CSV + npz-sidecar output shape are exactly what FX8 needs for the leakage
  side -- FX2 already solved "how do you get a leakage number that's comparable across different
  configurations of the same method," which is precisely FX8's cross-LEVEL comparability problem too.
  Reuse by parametrizing `_shadow_dir` (currently hardcoded to
  `shadows_v2/<dataset>/<method>/seed<S>`) to accept a caller-supplied directory, OR write a thin
  FX8-specific wrapper that imports `K_SET`, `E_MAX`, `_pooled_scores_labels`, `_report_row` from
  `run_lira_pertask` (matching `build_fig16.py`'s existing import-from-run_lira_pertask pattern) and
  only reimplements the shadow-dir resolution and the outer level/seed loop.
- `run_accuracy_matrix.py`'s real `eval_sets` pattern (`streams.task_eval_sets` + `sim.run(...,
  eval_sets=...)`) is the exact fix needed for the accuracy/BWT side; CIFAR-100 already has real test
  features cached, no new extraction needed.
- `metrics.membership_report`/`clopper_pearson` for the per-row CI, matching the target schema exactly.

## What's new / needs building

1. **A manifest-driven PBS array** for shadow generation, matching wave V2's pattern
   (`build_wave_v2_manifest.py`/`shadow_wave.pbs`): rows = 6 levels x 2 methods x 3 seeds = 36 combos,
   each needing 512 shadows -- probably chunked (e.g. `CHUNK=64` -> 8 array tasks/combo, matching the
   FIG18 pilot's own chunk size) rather than one task per combo, since M2 is already the KNOWN
   slowest method in the whole zoo (wave V2's own timing note: "Slowest chunk (m2_target) took ~14 min"
   for a 64-shadow chunk at the TAB05 headline config -- dose-response levels may cost more or less
   than that headline config depending on how `n_synthetic_per_class` trades off against per-shadow
   cost, unmeasured here, so a real small-scale timing check on THIS knob range is needed before
   committing to the full 512x3x6x2 = 18,432-shadow run, exactly like the M9 audit's own 16-shadow
   timing test before its full run).
2. **New shadow output paths**: `shadows_v2/cifar100/m5_hybrid_replay/dose_exemplar_budget_<level>/
   seed<S>/` and `shadows_v2/cifar100/m2_target/dose_n_synthetic_per_class_<level>/seed<S>/` (mirroring
   the old pilot's `dose_<knob>_<level>` naming, moved under `shadows_v2/` per the standing rule).
3. **A real-accuracy script** (`run_dose_response_accuracy.py`, replacing the old pilot's
   `_run_accuracy` stopgap) that computes real test-split BWT per (method, level, seed) via
   `eval_sets` -- cheap, CPU-only, comparable cost to `run_accuracy_matrix.py`, can run BEFORE the
   expensive shadow generation finishes (independent).
4. **A leakage-scoring script** (`run_dose_response_lira.py` or a parametrized reuse of
   `run_lira_pertask.py`) producing the per-seed `tpr1`/`tpr01`/`ci_lo`/`ci_hi` at whichever
   elapsed value(s) FIG03/FIG04 actually need -- the old pilot used a single `ELAPSED_FOR_LEAK=5`;
   confirm whether the spec wants one elapsed value or the full curve before building (the schema has
   no `elapsed` column, unlike FIG01/TAB03/TAB04, suggesting ONE representative elapsed value per row
   is intended, most likely elapsed=0 for consistency with this project's other "headline" leakage
   points, or the max observed elapsed for a "full transcript" reading -- needs a decision, not a
   guess, before generating 18k shadows around the wrong scoring convention).
5. **FIG03 v2 / FIG04 v2 plot scripts** (`analysis/fig03_dose_response.py`, `analysis/
   fig04_semantic_vs_individual.py` -- both currently read the OLD pilot schema and need a rewrite for
   the new one, same shape of change FX6 already made for FIG01/FIG02/TAB03/TAB04).
6. **Errata note** for dropping the M4 arm (dead knob under class-incremental streams), to go in
   `paper/PAPER_BRIEFING.md`'s errata section once that's touched.

## Open design questions to resolve before generating any shadows (not guessed here)

- ~~Exact config key names~~ **RESOLVED (checked directly, 2026-09-23)**: the plan's prose
  "exemplar_budget" is descriptive, not the literal key -- `HybridReplay.__init__` reads
  `config.get("buffer_size_per_class", 5)` (also `MethodSpec.retention_knob_name==
  "buffer_size_per_class"`, confirming this IS the method's own official retention knob, just named
  differently in the plan's prose than in code). `TARGET.__init__` reads
  `config.get("n_synthetic_per_class", 20)` -- this one DOES match the plan's literal wording. Both
  match `shadow_runner._method_config`'s existing TAB05 defaults exactly
  (`buffer_size_per_class=10`, `n_synthetic_per_class=20`), so the override mechanism
  (`method_config_override={"buffer_size_per_class": level}` / `{"n_synthetic_per_class": level}`)
  is exactly what to use -- no shadow_runner.py changes needed, `_method_config`'s defaults are
  overridden on top exactly as `test_run_shadow_range_respects_method_config_override` already
  proves works for this exact M5 key.
- Which elapsed value(s) FIG03/FIG04 want (see point 4 above) -- affects both the scoring script and
  how many targets/rounds worth of shadow data are actually needed (if only elapsed=0 is wanted,
  `E_MAX` could be smaller than FX2's 6, changing the target-selection/timing math).
- Real per-shadow timing for M2/M5 across this specific knob range, not assumed from the TAB05-config
  timing note -- a small (e.g. 16-shadow) real-scale PBS timing test per method, before committing to
  the full run, exactly like the M9 audit did.

## Why this is logged instead of built right now

Same reasoning as the M9 audit's design note: this is real, leakage-measurement infrastructure for a
headline claim (C2), currently mid-way through the M9 audit's own shadow generation running in the
background. Building 18k shadows' worth of new PBS infrastructure around an unconfirmed elapsed-value
convention or an unverified config-key assumption risks generating a large, expensive batch of shadows
that then need regenerating -- worth the extra half hour to confirm the two or three unknowns above
first. Next session picking this up should start here, not from scratch, and should do the M2/M5
config-key check and the small timing test as its literal first two actions.
