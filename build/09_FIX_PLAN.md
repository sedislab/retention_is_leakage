# 09_FIX_PLAN.md: second fix round (FX9)

**Status:** ACTIVE from 2026-09-23 evening (CDT). This file overrides `08_FIX_PLAN.md` wherever they conflict. Everything in `CLAUDE.md` still applies. The operating rules of `08_FIX_PLAN.md` §0 still apply unchanged: autonomy, PBS only, no git, and honesty.

**Deadlines**
- **Soft target:** every P0 output is final by **2026-09-25 06:00 CDT**. Opus needs time to write.
- **Hard freeze:** **2026-09-25 12:00 CDT**, using the same freeze procedure as `08_FIX_PLAN.md` §12.

**Why this round exists.** The FX0–FX8 work is mostly good, but a review of the actual CSVs and rendered figures found three bugs that change headline numbers, one pipeline inconsistency, and several items reported as done that are not done. They are listed in §1.

**Before you start, read this.** In the last round, several items were marked done while their output was wrong or blank. In this round, an item is done only when its acceptance check (listed under each item) prints the expected numbers and you have pasted them into STATE.md.

---

## 0. Extra operating rules for this round

1. **No login-node compute that writes into `results/`, `figs/` or `tables/`.** This happened five times last round. Every script that writes an output file goes through PBS, including "quick" CSV joins. Pytest under 2 minutes is still fine on the login node.
2. **Generous walltimes.** Use 24 h for any bootstrap or LiRA scoring job, and 4 h for shadow chunks. An over-long walltime costs nothing; an exceeded one wastes everything that ran.
3. **One shadow-path registry.**
   - Add `SHADOW_ROOT = {method: path}` in one module, `p3fcl/paths.py`. It maps M0 → `shadows/`, M3 and M8 → `shadows_v2/`, and M1, M2, M4, M5 → `shadows_v3/`.
   - Every consumer imports it: `run_lira_pertask.py`, `build_fig16.py`, the FX8 scripts, the M9 audit, and any future script.
   - Add a test that greps `code/` for any other hard-coded `shadows` path and fails if it finds one.
4. **Rebuild shared summaries from scratch.** `a1_lira_fixedk_summary.csv`, `fig02_halflife.csv` and every other merged CSV must be rebuilt only from current per-combo files. Never fold in an existing shared file. Archive the old shared file first.
5. **Keep STATE.md honest.** Add `## FX9 (09_FIX_PLAN)` at the top with a checklist of FX9-0 to FX9-10. For each item give status, job ids, outputs, and the printed acceptance numbers.

---

## 1. What the review found

| ID | Finding | Evidence | Fix |
|---|---|---|---|
| B1 | **The FIG01 accuracy curve is wrong.** `build_fig01_decoupling.py::_accuracy_curve` looks up `acc_by_key.get((s, k, T))` with `T = k + e`, but the dict is keyed `(seed, task_k, elapsed)`. Every k > 0 therefore reads elapsed k + e instead of e. <br>• M0 on CIFAR shows 0.499 at e = 0; the true pooled diagonal is 0.93. <br>• FIG19's accuracy line reads the same CSV, so it is also wrong. <br>• FIG02 and TAB04 half-lives come from `build_fx2_accuracy_summary.py`, which is correct: M0 on CIFAR is h = 0.87, verified by hand. | `results/fig01_decoupling.csv` vs `accuracy_matrix_cifar100.csv` | FX9-1 |
| B2 | **M4's server prototype bank is overwritten per client.** `self._prototypes[c] = mean_c` runs inside the client loop, so the bank holds the *last* client's local mean for each class. `predict` and `_reconstruct_prototype_global` both use this bank. <br>• M4 accuracy is understated (ImageNet-R 0.29, CIFAR 0.66). <br>• The `global` leakage view is wrong: CUB `global` 0.157 against `aggregate` 0.832, although in class-incremental streams they should be equal. | `methods/m4_proto.py`, `shadow_runner._reconstruct_prototype_global` | FX9-2 |
| B3 | **M1, M2 and M5 still learn each task one task late on CUB and ImageNet-R.** <br>• CUB last-task accuracy is 0.051 / 0.001 / 0.103 against M0's 0.836. <br>• BWT is +0.257 / +0.710 / +0.474. <br>• The gate failed on these combos. <br>• **Root cause:** local loss is one mean over current ∪ replay, with one FedAvg round per task. The new-class gradient is scaled down by the old:new sample ratio. For M2 (20 synthetic per old class per client), the ratio on CUB is 4:1 at task 1 and 38:1 at task 9. <br>• The lag also shows up on CIFAR in FX8 at buffer 20 and 50 (BWT +0.012, +0.047). <br>• It confounds the rising-leakage curves of replay methods on CUB and ImageNet-R. | `fx4_gate.csv`, `fx4_lag_diagnostic.csv`, `m1_glfc.py`, `m2_target.py`, `m5_hybrid_replay.py` `_local_train` | FX9-3 |
| B4 | **TAB05 describes a different system from the one that was attacked.** `run_utility_baseline.py` z-scores the features (`_standardize`). The shadows, accuracy matrices, gate and M9 sweep all use raw features. <br>• M0 CIFAR final accuracy: TAB05 0.732, attacked configuration 0.333. <br>• TAB07 inherits the problem. | `code/scripts/run_utility_baseline.py` | FX9-4 |
| B5 | **M5 releases its raw exemplar buffer as an F8 record.** The original Hybrid Replay (Nori et al., ICLR 2025) keeps exemplars on the client. M1 already had its buffer record removed for the same reason. | `m5_hybrid_replay.py` | FX9-3 |
| I1 | **M9 outputs are incomplete.** <br>• FIG05's M9 panel says "pending FX5". <br>• TAB08's `eps_certified_*` columns are blank. <br>• FIG08's `eps_audited_*` columns are blank. | | FX9-7 |
| I2 | **`paper/PAPER_BRIEFING.md` was never updated.** It still describes pre-fix results and has no errata, final-results or CUT sections. | | FX9-10 |
| I3 | **Decoupling-ratio CIs are blank.** No joint bootstrap was done. | `decoupling_ratio.csv` | FX9-9 |
| I4 | **Several figures break the style rules.** <br>• FIG02 is still a 44-row portrait of censored arrows. <br>• Every figure has a title inside it. <br>• FIG03 and FIG04 label −BWT as "retention"; −BWT measures forgetting. <br>• FIG13 was not redone. | | FX9-8 |

**Do not touch** anything that is correct: M0, M3 and M8 shadows and results; the FX1 accountant; M9 and its sweep and audit; FIG18 v2; the M0 budget check.

---

## 2. Schedule

| # | Item | Prio | Target (CDT) |
|---|---|---|---|
| 1 | FX9-0 archive | P0 | Sep 23 +0.5 h |
| 2 | FX9-1 FIG01 bug fix + unified retention curves (code) | P0 | +1 h |
| 3 | FX9-2 M4 bank fix + tests | P0 | +2 h |
| 4 | FX9-3 balanced replay for M1/M2/M5, M5 F8 removal, tests | P0 | +3.5 h |
| 5 | FX9-3 gate (sim-only PBS array) | P0 | +4.5 h |
| 6 | FX9-4 TAB05 on the attacked pipeline; regenerate all accuracy matrices | P0 | +5.5 h |
| 7 | FX9-5 wave V3 shadows (M1, M2, M4, M5): pilot, then full array; FX9-6 FX8 rerun in parallel | P0 | launch by Sep 24 02:00 |
| 8 | While waves run: FX9-7 (M9 completions), FX9-9 (joint bootstrap code), FX9-8 figure code, FIG13 job | P0 / P1 | Sep 24 morning |
| 9 | FX2 scoring on the 36 new combos; rebuild all summaries; all figures and tables | P0 | Sep 24 evening |
| 10 | FX9-10 briefing, `paper_numbers.csv`, OPEN_QUESTIONS, method descriptions | P0 | Sep 25 06:00 |
| 11 | Freeze | P0 | Sep 25 12:00 |

**Core budget.** Wave V3 uses `%40` × 8 cores = 320. FX8 uses `%16` × 8 = 128. Everything else shares the remaining 64.

---

## 3. FX9-0: Archive (P0)

1. Copy with `cp -a` into `archive/2026-09-23b_pre_fx9/`:
   - `results/`, `figs/`, `tables/`
   - `paper/PAPER_BRIEFING.md`, `agents/OPEN_QUESTIONS.md`, `build/STATE.md`
2. Write `SHA256SUMS` over the copy and a one-paragraph `README.md` that points to §1 of this file.
3. Delete nothing, and leave `shadows/` and `shadows_v2/` untouched.
4. The outputs that this round will replace are overwritten in place, and only after their replacements pass their acceptance checks.

---

## 4. FX9-1: Fix FIG01 and unify all retention curves (P0)

### The problem

The accuracy curve used by FIG01 and FIG19 was computed separately from the one behind the half-lives, and it had a lookup bug. Replace both with **one** function and **one** CSV.

### The shared function

- **Name and location:** `p3fcl/halflife.py::accuracy_curve(acc_rows, K=(0,1,2,3), E=6, floor="classes_seen")`.
- **Output:** for each elapsed e, the raw pooled accuracy and the normalised accuracy A(e), using exactly the `08_FIX_PLAN.md` §7b definition:

  A(e) = mean_k[acc_k(k+e) − a0(k+e)] / mean_k[acc_k(k) − a0(k)].

- **Lookup rule:** index rows by `(seed, task_k, elapsed)` and look up `(s, k, e)`. Never use `task_T` in the elapsed slot.
- **Who uses it:** `build_fx2_accuracy_summary.py` (half-lives), the new curve builder below, and nothing else re-derives accuracy curves.

### The single source of truth

Write `results/retention_curves.csv` with columns:

`dataset, method, family, view, quantity{acc|leak_tpr1|leak_auc}, elapsed, raw, raw_ci_lo, raw_ci_hi, norm, norm_ci_lo, norm_ci_hi, n_seeds`

- Accuracy rows come from the accuracy bootstrap: resample seeds, then tasks within K.
- Leakage rows come from the existing FX2 hierarchical bootstrap: resample seeds, then targets. They are normalised with the §7b floors.
- FIG01, the new FIG02 and FIG19 read only this file.

### Tests

- For M0 on CIFAR-100, `accuracy_curve(...).raw[0]` equals the mean of the diagonal accuracies over seeds 0–4 and k ∈ K. That value is 0.93 ± 0.01 in the current matrix.
- `norm` at e = 0 is exactly 1.
- The half-life computed from this function equals the value in `fig02_halflife.csv` for every row, to 1e-9.

### Accept when

STATE.md shows M0's raw accuracy at e = 0 and e = 6 for all three datasets, and the e = 0 values match the diagonal means.

---

## 5. FX9-2: M4 prototype bank (P0)

### Method fix

- **Per-client releases stay as they are:** each client releases its local class mean (F2) plus counts (F7).
- **New server bank rule:** after all clients in a round, for each class c released that round:
  - compute the count-weighted mean over clients of the client prototypes for c, using that round's F7 counts;
  - if `prototype_momentum > 0` and c was already in the bank: `bank[c] = m·bank[c] + (1−m)·agg_c`;
  - otherwise: `bank[c] = agg_c`.
- `predict` uses this bank.

### Reconstruction fix

Rewrite `_reconstruct_prototype_global` to apply exactly the same per-round, count-weighted rule from ledger records only. No more "last write wins".

### Tests

- The reconstructed global bank equals `method._prototypes` after every round (to 1e-10) on a small stream with overlapping client classes.
- In a class-incremental stream, `global` scores equal `aggregate` scores.
- With `momentum = 0.5` on a domain-incremental toy stream, the bank equals the hand-computed blend.

### Accept when

STATE.md shows M4's new test final average accuracy for all three datasets, next to the old values (0.66 / 0.56 / 0.29 from TAB05).

---

## 6. FX9-3: Balanced replay loss for M1, M2, M5; remove M5's F8 release (P0)

### Balanced local objective

Standard experience replay (Chaudhry et al., 2019) gives current data and replay data equal weight, whatever their sizes. For client c at task t:

  L = CE_mean(current shard) + ρ · CE_mean(replay set) [ + M1 only: λ_KD · KD_mean(current ∪ buffer, old classes only) ]

The gradient is `X_cur.T @ (P_cur − Y_cur) / n_cur + ρ · X_rep.T @ (P_rep − Y_rep) / n_rep`, plus the unchanged M1 KD term.

- Default ρ = 1.0. Add `replay_weight` to each method's config.
- The replay set is:
  - M1 and M5: the client's own buffer;
  - M2: the synthetic samples.
- With an empty replay set, L is exactly M0's local objective.

### M5: stop releasing the buffer

- Remove M5's F8 exemplar record from the released ledger. Its buffer is private local state, as in the original method.
- The F1 delta's `touched` must still include every buffer id read that round, so the accountant still sees the re-reads.
- Update the module docstring, `MethodSpec.families` (F1 only) and `method_descriptions.csv`.
- FIG03 and FIG04 legends become "M5 exemplar replay (individual)" and "M2 Gaussian replay (semantic)".

### Tests

- With an empty buffer, one round of M1, M2 or M5 equals one M0 round, to 1e-10.
- **Balance invariance:** duplicating every replay sample (so n_rep doubles) leaves the local gradient unchanged, to 1e-10. The same holds for duplicating the current shard. This is the property the old single-mean loss violated.
- M5 emits no F8 record, and its F1 `touched` still includes buffer ids.

### Gate (automatic, simulation only, PBS)

**Run:** datasets {cifar100, cub200, imagenet_r} × seeds {0,1,2} × methods {M0, M1, M2, M3, M5}, test accuracy.

- Drop the old `(cifar100, m1_glfc)` tuned override. Start every method from defaults.

**Pass condition** (same as before; means over seeds):
- BWT ≤ +0.02;
- last-task accuracy ≥ M0's − 0.15;
- final average accuracy ≥ M0's.

**If the gate fails:** tune on the train-side validation split only, over ρ ∈ {0.5, 1, 2} × lr ∈ {0.1, 0.5}.

**If it still fails:** keep the best configuration, record GATE FAILED with the numbers, and continue.

**Outputs:**
- `results/fx9_gate.csv`, with the same schema as `fx4_gate.csv` plus `replay_weight`.
- `results/fx9_lag_diagnostic.csv`: the CUB seed-0 diagonal for M1, M2 and M5 before FX9 (from the archive) and after.

**Accept when** STATE.md shows, for all three datasets and M1/M2/M5, the last-task accuracy and BWT before FX9 and after.

---

## 7. FX9-4: TAB05 on the attacked pipeline (P0)

- **Match the attacked configuration.**
  - `run_utility_baseline.py` must build each method from `shadow_runner._method_config(...)` plus the FX9 gate override. Import it; do not keep a second config table.
  - It must use raw features: remove `_standardize`.
  - Add a test that asserts the configurations are identical for every method.
- **Regenerate accuracy matrices.** Rerun `run_accuracy_matrix.py` for all methods (M0 to M8, seeds 0 to 4) on the three datasets, using the FX9 method code. Include both `acc` and `acc_train`.
- **Regenerate TAB05 and TAB07.** Use 10 tasks. Do not run 20-task configs. Drop the Camelyon17 rows from TAB05 and put the FIG18 pipeline's accuracy numbers in the appendix instead.

**Accept when** TAB05's `final_avg_acc` for each (dataset, method) equals the value derived from `accuracy_matrix_<dataset>.csv` (last row mean) to 1e-9. Add this as a check in `check_fx9_consistency.py` (§13).

---

## 8. FX9-5: Wave V3 shadows (P0)

**Scope.**
- Methods M1, M2 and M5 (balanced replay; M5 without F8) and M4 (fixed bank).
- Datasets cifar100, cub200, imagenet_r. Seeds 0, 1, 2. 1024 shadows each.
- Unchanged settings: `calibration_frac` 0.8, 5 targets per shard, `p_in` 0.5, 10 tasks × 10 clients, β = 0.5.

**Views.** M1, M2, M5: `full`. M4: `full`, `aggregate`, `global`.

**Array.**
- Output goes to `shadows_v3/<dataset>/<method>/seed<S>/`.
- Manifest: `build/waves/v3_main.tsv`, with 4 × 3 × 3 × 16 = **576 rows**. It reads `fx9_gate.csv` overrides.
- Submit as `-J 0-575%40`.
- Run a pilot first: one subjob per method on imagenet_r.

**Accept when** all 36 combos have 1024 loadable npz files. Check with `verify_wave_v2.py`, generalised to take a root path.

**Then score them.**
1. Run FX2 per-task LiRA plus `build_fx2_summary.py` on the 36 new (dataset, method, view) combos: M1/M2/M5 × 3 datasets × 1 view, plus M4 × 3 datasets × 3 views = 18 rows. Use a `-J` array with 24 h walltime.
2. Rebuild `a1_lira_fixedk_summary.csv`, `fig02_halflife.csv` and `retention_curves.csv` **from per-combo files only**:
   - M0, M3 and M8 per-combo files stay as they are;
   - M1, M2, M4 and M5 per-combo files must come from this wave.
   The rebuild script must assert this using each per-combo file's `.meta.json` `utc_start`.
3. Then rebuild FIG01, FIG02, FIG19, TAB03, TAB04, `decoupling_ratio.csv`, `fx3_views_summary.csv`, FIG11 (plus its appendix variants), FIG16 and FIG17.

---

## 9. FX9-6: FX8 dose–response rerun (P0)

- Same design as FX8:
  - CIFAR-100;
  - M5 with `buffer_size_per_class` ∈ {1, 2, 5, 10, 20, 50} and M2 with `n_synthetic_per_class` ∈ {1, 2, 5, 10, 20, 50};
  - 512 shadows × 3 seeds; e = 6; pooled K.
- Use the FX9 method code (balanced replay, M5 without F8).
- Submit in parallel with wave V3 at `%16`.
- Rebuild FIG03 and FIG04. The y-axis label becomes **"forgetting (−BWT)"**; −BWT is forgetting, not retention.
- Report the result as it comes out. Under balanced replay the knob changes *what* is retained (how many individuals, or how many synthetic samples), not how much weight replay gets. If M2's curve flattens, that is a finding.

---

## 10. FX9-7: Complete the M9 outputs (P0)

- **FIG05 M9 panel.**
  - Build M9 ledgers with `sim.run` (CIFAR-100, 10 tasks, U1, γ = 1) at ε₀ ∈ {1, 4}.
  - Compute certified ε_U(T) with the FX1 accountant, using M9's **actual** noise multiplier z = σ/Δ from its analytic-Gaussian calibration, not σ = 2.
  - Also build 50-task ledgers so the U4 curve is observed further.
  - At m = 1, also record the exact analytic (ε₀, δ₀) as `eps0_analytic`. The RDP conversion is looser at m = 1; state both values.
  - Replace the "pending FX5" panel with this plot.
- **TAB08.** Fill `eps_certified_T10_*` and `eps_certified_T50_*` for U1 to U4, from the same computation.
- **FIG08 audited ε.**
  - Compute from the M9 audit scores the empirical lower bound ε_lb = max_t ln((TPR_lo(t) − δ) / FPR_hi(t)), with one-sided 95% Clopper–Pearson bounds.
  - Take the max over both directions: the formula above and its (1 − FPR)/(1 − TPR) mirror.
  - Write it to `results/m9_audit_summary.csv` and into FIG08's `eps_audited_*` columns.

**Accept when** STATE.md shows the certified ε for U1 to U4 at T = 10 and T = 50 for ε₀ = 1 and 4, and ε_lb for ε ∈ {1, 4, ∞}.

---

## 11. FX9-8: Figures (P0 unless marked)

**Global rules.**
- Remove every suptitle and in-figure title such as "FIG01 v2 — …". Panel labels (dataset names) stay.
- Fonts must be ≥ 7 pt at final size, including legends. Width ≤ 5.5 in.
- Display names come only from `method_descriptions.csv`.

**Per figure.**
- **FIG01 (retention curves).** Read `retention_curves.csv`.
  - Row 1: normalised accuracy A(e).
  - Row 2: normalised leakage L(e) for TPR@1%FPR.
  - Columns: datasets. CI bands.
  - M4 and M8 `full` view: dashed, labelled "per-client release (constant by construction)".
  - Size ≤ 5.5 × 3.4 in.
- **FIG02 (replace the forest plot): "retention at horizon".**
  - One panel per dataset. x = A(6), y = L(6) (TPR@1%FPR), one marker per (method, view), with 95% CI bars on both axes.
  - Reference lines: y = 1 ("leakage unchanged") and y = x.
  - Put the half-life numbers in TAB04, not in a figure.
  - Size ≤ 5.5 × 2.4 in.
  - Data in `results/fig02_retention_at_horizon.csv`.
- **FIG19.** Read `retention_curves.csv` (M4 and M8; views `full` and `global`; plus A(e)).
- **FIG03 and FIG04.** Axis label and legend changes from §9.
- **FIG11.** With the M4 fix, `global` equals `aggregate` for M4. Keep all three bars and let the numbers show it.
- **FIG13 (P1).** 25 trials per cell, median with bootstrap CI. Remove the anisotropy panel. Rebuild through PBS.
- **FIG18.** No rerun. Relabel it honestly: "clients = slide groups within each hospital task (natural) vs Dirichlet (β = 0.5), 5 clients, M0, 5k-patch subsample". Keep it as an appendix figure.

---

## 12. FX9-9: Joint bootstrap for the M0 decoupling ratio (P1)

The ratio is defined only for M0 (on all three datasets). Compute its CI with one bootstrap that shares the same resampled draw:
- outer: resample seeds;
- inner: resample tasks from K;
- within each drawn task: resample its targets.

From each replicate, compute h_acc, h_leak and ρ. Fill `ratio_ci_lo` and `ratio_ci_hi`, and label lower-bound ratios as such. Use the same machinery to write CIs for A(6) and L(6) into `fig02_retention_at_horizon.csv` for every row.

---

## 13. FX9-10: Paper-facing documents (P0)

### `code/scripts/check_fx9_consistency.py`

Called by `make verify`, run through PBS. It must fail if any of the following does not hold:
1. FIG01 and `retention_curves.csv` raw accuracy at e = 0 equals the accuracy-matrix diagonal mean for every (dataset, method).
2. TAB05 equals the accuracy-matrix final row (see §7).
3. Every M1, M2, M4 and M5 row in any summary CSV traces to an FX9-phase per-combo file.
4. Every `paper_numbers.csv` row's `value` equals the value in its `source_csv` under its `row_filter`.
5. No figure PDF is older than the CSV it reads.

### `paper/PAPER_BRIEFING.md`: rewrite it (≤ 400 lines)

Move the old file to the archive. The new file has these sections:
1. **ERRATA:** R1 to R12 (08 plan) and B1 to B5 (this plan). For each: old claim or number → new number, or "withdrawn".
2. **PROTOCOL:** the facts Opus needs for the Method section, each one line:
   - frozen ViT-B/16 features, raw (not standardised);
   - one FedAvg round per task, 30 local epochs, lr 0.5;
   - 10 tasks × 10 clients, Dirichlet β = 0.5;
   - class-incremental streams;
   - test-split accuracy, predictions restricted to seen classes;
   - LiRA: 1024 shadows × 3 seeds, `calibration_frac` 0.8, K = {0,1,2,3}, E = 6;
   - accountant σ = 2, δ = 1e-5, W = 3;
   - M9 hyperparameters selected on the `ref` split.
3. **FINAL RESULTS:** one block per figure and table: path, what it shows (2–4 sentences), key numbers with CIs, caveats.
4. **CUT:** everything not delivered, with the reason. At least: DP-FedAvg baselines, FIG09, 20-task TAB05, FIG18 hospital-as-client design.

### `results/paper_numbers.csv`

Regenerate from final CSVs only, adding keys for:
- A(6) and L(6) with CIs for every (dataset, method, view);
- raw TPR@1%FPR at e = 0 and e = 6;
- M9 certified ε (U1–U4, T = 10 and 50) and audited ε_lb;
- the FX9 gate outcomes;
- M4's new accuracy.

### Other documents

- **`agents/OPEN_QUESTIONS.md`:** update H2, H7, H11 and H13 with the final numbers. H8 → CUT (no DP baselines).
- **`results/method_descriptions.csv`:** balanced replay for M1, M2 and M5; M5 without F8; M4's count-weighted bank. Add one protocol row stating "one FedAvg round per task" as a shared limitation.

---

## 14. Definition of done

- [ ] FX9-0 archive with checksums.
- [ ] B1 fixed; `retention_curves.csv` is the single source; tests pass; M0 e = 0 accuracy matches the diagonal.
- [ ] B2 fixed; M4 accuracy and `global` view regenerated; `global` = `aggregate` in class-incremental streams.
- [ ] B3 fixed; gate recorded; lag diagnostic shows the before/after numbers.
- [ ] B4 fixed; TAB05 and TAB07 match the attacked pipeline.
- [ ] B5 done: M5 releases no F8 record.
- [ ] Wave V3 complete (36 combos × 1024) and scored; all summaries rebuilt from per-combo files.
- [ ] FX8 rerun complete; FIG03 and FIG04 relabelled.
- [ ] FIG05 M9 panel, TAB08 certified ε, FIG08 audited ε filled.
- [ ] New FIG02; FIG01 and FIG19 from `retention_curves.csv`; titles removed; FIG13 (P1).
- [ ] Joint-bootstrap CIs for the M0 ratio and for A(6)/L(6) (P1).
- [ ] Briefing rewritten; `paper_numbers.csv` regenerated; OPEN_QUESTIONS and method descriptions updated.
- [ ] `make figures` and `make verify` (with `check_fx9_consistency.py`) pass through PBS.