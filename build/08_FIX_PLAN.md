# 07_FIX_PLAN.md: post-review fix phase

**Status:** ACTIVE from 2026-09-23. This file overrides 00–06 wherever they conflict on the items below. Everything else in `CLAUDE.md` still applies.

**Hard results freeze: 2026-09-25 12:00 CDT.** After the freeze you submit no new experiment jobs. You only run `make figures`, run `make verify`, and update the briefing. The paper is due the same day (AoE). The human and Opus need the afternoon and night to write.

**Who does what.** Opus (a separate session) wrote this plan and writes the paper. You write all the code, run every job, and produce every CSV, figure and table. You also keep `paper/PAPER_BRIEFING.md` and `results/paper_numbers.csv` accurate, because Opus writes the paper from those two files.

---

## 0. Operating rules for this phase

1. **Autonomy.** Run FX0 to FX8 in the order given in §2 without asking.
   - Every gate in this plan is automatic. If a gate passes, continue. If it fails, apply its documented fallback, log it, and continue.
   - Stop and ask the human in only two cases:
     - (a) a blocker you cannot route around, such as the cluster being down, disk or quota full, or a missing module;
     - (b) any action that would delete data.
   - For every other decision, decide, log the decision in `build/STATE.md`, and keep going.
2. **Compute policy (Kodiak).**
   - **Allowed on the login node:** editing, `qsub`, `qstat`, `qdel`, `ls`, `du`, `df`, and `pytest` runs that finish in under 2 minutes.
   - **Everything else runs through PBS.** That means every result, CSV, figure or table, including analysis and plotting scripts.
   - **Why this matters:** the earlier `run_lira.py` runs happened on the login node, so their meta.json has `pbs_jobid: null`. That broke the rule, and those outputs are regenerated here.
   - **Generic job runner.** Write each command to `build/jobs/<name>.sh`. Submit `code/scripts/pbs/run_job.pbs` with `-v JOBFILE=/data/islamm/retention_leakage/build/jobs/<name>.sh`. This avoids the comma-splitting problem with `-v`.
   - **Queues.** Use `batch` only. Never target condo queues. This phase needs no GPU.
3. **Core budget.** The per-user cap is 512 concurrent cores.
   - Shadow waves use `select=1:ncpus=8:mem=32gb` with throttle `%56`, which is 448 cores.
   - All other jobs share the remaining 64 cores (for example, at most 16 jobs at `ncpus=4`).
   - Log live core use in STATE.md each time you launch.
4. **Paths.**
   - Working directory: `/data/islamm/retention_leakage`.
   - Environment: `module load python/3.10.4` then `source envs/p3fcl/bin/activate`.
   - New shadows go to `shadows_v2/`. Never write into `shadows/`, which is now read-only.
   - Run `df -h /data` before each wave. It was 88% full.
5. **No git.** Provenance stays `source_hash()` plus meta.json. The human syncs the repo to GitHub himself. Do not run git commands.
6. **No paper prose.** Your paper deliverables are the briefing and `paper_numbers.csv` (FX7). Do not edit `paper/main.tex`.
7. **Honesty.**
   - Never tune toward an expected result.
   - Select hyperparameters on a train-side validation split or on the public `ref` split, never on `test`.
   - If a result contradicts the story, report it plainly. It is still a result.
8. **STATE.md.** Add a section `## FIX PHASE (07_FIX_PLAN)`.
   - It holds a checklist FX0 to FX8. Each item records its status, PBS job ids, output paths and a one-line result.
   - Update it after every meaningful step, not at the end.
   - Always keep "Next action" filled in. If your context resets, this section is how you resume.
9. **Tests.** Keep the existing suite green. Every fix below lists new tests. Write them before you run any experiment that depends on that fix.
10. **Seeds under the deadline.** Shadow-based results use 3 seeds (0, 1, 2) × 1024 shadows. Accuracy-only results keep 5 seeds (0 to 4). This overrides the "≥ 5 seeds" line in `03_RESULTS_SPEC.md` for shadow-based figures.

---

## 1. What the review found (why this phase exists)

| ID | Finding | Evidence | Fix |
|---|---|---|---|
| R1 | **Accuracy is measured on training examples.** `sim.run` builds `task_eval_ids` from the stream, so every accuracy matrix is train-set accuracy. That covers TAB05, BWT and every accuracy half-life. `get_data.py` already defines `test` and `ref` splits. First check that their feature caches exist for every dataset. If one is missing, extract it with `extract_features.pbs` before 4a. | `sim.py` (`task_eval_ids`), `get_data.py` fracs | FX4a |
| R2 | **M1, M2 and M5 learn each task one task late.** On CUB, last-task accuracy is 0.000 / 0.002 / 0.087 against 0.884 for M0, and BWT is positive (+0.281 / +0.363 / +0.178). There are two causes: every client shares one global replay buffer, and KD runs over all classes. The teacher knows nothing about the new classes, so KD suppresses them. | `accuracy_matrix_*.csv`; `m1_glfc.py`: `self._buffer[c]`, KD `softmax(X_all@W_old/T)` | FX4c–e |
| R3 | **`touched` and `n_touched` are inflated,** by M1's `_all_touched_ever` and M3's `_subspace_touched`. `_reconstruct_running_w` weights FedAvg by `n_touched`, so the adversary's reconstructed model is not the real global model for M1, M2, M3 and M5. | `shadow_runner.py:165` | FX4b |
| R4 | **M4 and M8 per-client leakage is constant by construction** (`scores[k:] = val` from the task-k record). Their ∞ ratios measure transcript persistence, not retention. M4's finite half-lives are pooling artefacts. | `shadow_runner._score_target` | FX3, FX2 |
| R5 | **The FIG05 accountant is wrong.** M0 under U2 goes 16.9 → 4166 but should be flat. M8 under U1 goes 2.5 → 201 but should be flat. For M1, U4 < U2, which is impossible. | `dp/accountant.account()` | FX1 |
| R6 | **The half-life is an extrapolated exponential fit.** Examples: M8 accuracy 53 / 223; M4-CUB leakage 2738; M0-ImageNet-R 606 with R² = 0.007. There is no chance floor. `run_lira.py` pools targets across tasks by elapsed time, so the target population changes with e. None of this matches Definition 10 in the paper. | `metrics.fit_exponential_halflife`, `run_lira.py` | FX2 |
| R7 | **LiRA confidence intervals are far too narrow.** They are Clopper–Pearson intervals over (target, shadow) pairs, and those pairs are not independent. | `a1_lira_*.csv` | FX2 |
| R8 | **FIG18 is confounded.** The natural arm has 1 client and the Dirichlet arm has 10. The Dirichlet arm is near chance, and the natural arm's CI at e = 4 is [−0.08, 0.19]. | `fig18_natural_federation.csv` | FX6 |
| R9 | **Secure aggregation is a split by artifact family, not a blanket result.** The F1 attack uses only the global model, so secure aggregation cannot affect it. Per-client F2 and F5 attacks are removed. Aggregate-level attacks on F2 and F5 were never run. | FIG11 | FX3 |
| R10 | **The C2 pilot is weak.** M5 levels are 0.024 / 0.043 / 0.049 with overlapping CIs. The M4 arm is a null experiment because its knob is dead code. M5 itself is invalid because of R2. | FIG03 | FX8 |
| R11 | **Minor issues:** <br>• FIG01 draws all F1 methods in one colour. <br>• FIG02 is 42 rows tall. <br>• FIG13's anisotropy contradicts its causal story (CUB 34 < CIFAR 86, yet CUB inverts better). <br>• A4 is 0.716 [0.593, 0.820] against a random baseline of 0.625 [0.458, 0.773]. <br>• A6 = 1.0 comes straight from the counts. <br>• The Camelyon17 MANIFEST `url_used` is wrong (the HF mirror was used). | | FX6, FX7 |
| R12 | **There is no M9.** C4 is theory only, but the paper states Algorithm 3 and Theorem 11 and needs an empirical privacy–utility result. | `methods/` | FX5 |

**Solid results.** Do not break these, and keep their tests:

- H5: exact inversion at n = 1.
- M4 on CIFAR: AUC 0.88.
- The counts join key.
- The TAB07 backbone confound.
- All the DP theorems.
- M8 per-client AUC 1.000 (it stays as the `full` view).

---

## 2. Priorities and schedule

- **P0:** must land before the freeze.
- **P1:** should land.
- **P2:** only if cores and time are free.

| # | Item | Prio | Depends on | Target (CDT) |
|---|---|---|---|---|
| 1 | FX0 archive and quarantine | P0 | none | Sep 23, start + 0.5 h |
| 2 | FX4a–f code, FX3 scoring code (4h), all new tests | P0 | 1 | start + 5 h |
| 3 | FX4g gate (simulation only, PBS array) | P0 | 2 | start + 6 h |
| 4 | Wave V2 shadows: pilot, then the full array | P0 | 3 | Sep 23 night |
| 5 | FX1 accountant + FIG05 | P0 | 2 | while the wave runs |
| 6 | FX5 M9 code, tests, core sweep; FIG08, TAB08 | P0 | 2 | while the wave runs |
| 7 | FX2 code, first run on M0 (existing shadows) | P0 | 2 | while the wave runs |
| 8 | FX2 on V2 → FIG01, FIG02, TAB03, TAB04, `decoupling_ratio` | P0 | 4 | Sep 24 |
| 9 | FX3 → FIG11 v2, FIG19 | P1 | 4 | Sep 24 |
| 10 | Wave V3: FIG18 rerun, M9 audit (FIG10); then FX8 | P1 / P2 | cores free | Sep 24 |
| 11 | FX5 stretch: DP baselines, FIG09 50-task | P2 | 6 | Sep 24–25 |
| 12 | FX6 remaining figure and table fixes | P0 / P1 | 8 | Sep 25 morning |
| 13 | FX7 errata, final results, `paper_numbers.csv` | P0 | all | continuous; final at freeze |
| 14 | Freeze: `make figures` and `make verify` via PBS | P0 | none | **Sep 25 12:00** |

Anything unfinished at the freeze is cut. Mark it `CUT` in STATE.md and in the briefing, with the reason. Never leave a half-updated figure. Either the v2 figure exists with its CSV, or the item is listed as withdrawn.

---

## 3. FX0: Archive and quarantine (P0, about 30 min)

1. Create `archive/2026-09-23_pre_fix/` and copy into it with `cp -a`:
   - `results/`, `figs/`, `tables/`
   - `paper/PAPER_BRIEFING.md`
   - `agents/OPEN_QUESTIONS.md`
   - `build/STATE.md`
2. Write `archive/2026-09-23_pre_fix/SHA256SUMS` over the copied files.
3. Write `archive/2026-09-23_pre_fix/README.md`. It records the date, the `source_hash()` at archive time, and one paragraph pointing to §1 of this file.
4. Only after the copy verifies, move the stale outputs from the live tree to `archive/2026-09-23_pre_fix/stale/` with `mv`:
   - all `results/a1_lira_*` (FX2 regenerates them per task)
   - `results/accuracy_matrix_*`
   - results and figures for `fig01*`, `fig02*`, `fig03*`, `fig04*`, `fig05*`, `fig16*`, `fig17*`, `fig18*`
   - `tab03*`, `tab04*`, `tab05*`, `tab07*`
   - `results/decoupling_ratio.csv`
5. Keep in place:
   - raw A2, A3, A4 and A6 results
   - `tab02` and `tab10`
   - `disjointness_report`, `MANIFEST`, `JOBS.jsonl`, `RUN_LOG.jsonl`
6. Leave `shadows/` untouched. Delete nothing anywhere.

**Accept when:** the checksums verify, and no stale CSV remains where a figure script could read it.

---

## 4. FX4: Simulator and method fixes (P0, critical path)

### 4a. Test-split accuracy (R1)

**Simulator change.**
- `sim.run(method, X, y, stream, seed, eval_sets=None)` takes a new optional argument, `eval_sets`, with one `(X_eval_k, y_eval_k)` per task.
- When `eval_sets` is given, `acc_matrix[t, k]` is computed on `eval_sets[k]`.
- Keep the old train-id evaluation and return it as `acc_matrix_train`. Return both matrices.

**Building the evaluation sets.**
- Add a helper `streams.task_eval_sets(...)` and build the sets from the `test` feature cache.
- Class-incremental: task k's set is the test samples whose label is in task k's classes. Use exactly the same `task_classes` partition as `build_stream`.
- Domain-incremental (Camelyon17): task k's set is the test samples of domain k.

**Prediction over seen classes only.**
- Every method's `predict` must choose among the classes seen so far. Check each method.
- For example, zero-initialised columns for unseen classes give logit 0. That can beat negative logits for seen classes. Fix any method where this happens and record it in the errata.

**Reporting.** From now on, test accuracy is primary (`acc`) and train accuracy is secondary (`acc_train`). Every CSV that has an accuracy column gets both.

**Tests:**
- Task k's evaluation set contains only task-k classes.
- `predict` never outputs a class that has not been seen yet.
- On a toy stream, `acc_matrix` differs from `acc_matrix_train`.

### 4b. Honest ledger and exact reconstruction (R3)

**Rule (Lemma 8 in the paper).** A record's `touched` is exactly the set of private example ids whose values were read to compute that record's payload in that round. Reading any of the following adds nothing, because it is post-processing:
- a broadcast model;
- a released statistic;
- anything computed from earlier releases.

**Changes.**
- Remove `_all_touched_ever` (M1) and `_subspace_touched` (M3) from `touched`.
- Every F1 record gets `meta["agg_weight"]`: the exact weight the server gave that client in that round's FedAvg.
- `_reconstruct_running_w` uses `meta["agg_weight"]`. Fall back to `n_touched`, with a logged warning, only for pre-fix ledgers.
- **M3 (FOT).** The server subspace is computed from task data. Record it as its own release at task k: family GRAM, `client = -1`, `touched` = the task-k ids, payload = the covariance (or basis increment). Later rounds use U as post-processing.
- **Local buffers are not releases.** A local replay buffer is private local state. If M1 emits an F8 record that only represents local storage, remove it from the released ledger; you may log it under `result["local_state"]`. Keep only records that correspond to real transmissions.
- `sim.run` also returns `w_history`, the global W after each round, for F1 methods.

**Tests:**
- (i) For every F1 method on a small stream: `max |_reconstruct_running_w(ledger)[r] − w_history[r]| < 1e-8` for every round r.
- (ii) For M3, `touched` in later rounds is a subset of that client's current shard.
- (iii) For M1 and M5, `touched` equals the current shard plus the ids actually read from that client's own buffer in that round.

**Using test (i) to decide which shadows to regenerate.**
- Run test (i) on the pre-fix code for M0, M1, M2, M3 and M5, and log the result.
- If M0 passes, its existing shadows are valid and are reused (seeds 0 to 2, `shadow_id < 1024`).
- Any method that fails gets new shadows in wave V2. M1, M2, M3 and M5 are expected to fail.

### 4c. M1 (GLFC-style) (R2)

- **Per-client buffer.** Use `self._buffer[(client, cls)]` with a budget of `exemplar_budget` (keep 10) per class per client.
- **Filling the buffer.** Fill it at the end of each of the client's tasks, from its own current shard only. Selection is deterministic, seeded by `(seed, client, task)`. Use random selection or herding, and state which.
- **Local objective at task t:**
  - CE over the current shard plus the client's own buffer, with logits restricted to classes seen so far;
  - plus KD over **old classes only**: `KL( softmax(X W_old[:, old] / τ) || softmax(X W[:, old] / τ) )`, with τ = 2.0 and weight `distillation_weight`;
  - `W_old` is the global model broadcast at the start of task t;
  - KD is evaluated on the current shard plus the buffer.
- No client ever reads another client's buffer.

**Tests:**
- Buffer isolation: client c never holds ids from another client c′.
- The KD gradient is exactly zero on new-class columns.
- With `distillation_weight = 0` and an empty buffer, one M1 round equals one M0 round.

### 4d. M5 (hybrid replay)

- Use the same per-client buffer keyed by `(client, class)`.
- Keep whatever else M5 does, but it may read only its own client's buffer.
- If M5 has a KD term, restrict it to old classes.
- Class-balanced CE (inverse class frequency over the current shard plus the buffer) is allowed if the gate needs it.
- Tests: the same isolation and `touched` tests as M1.

### 4e. M2: relabel it; do not reimplement TARGET

- The current M2 is class-conditional Gaussian feature replay, not TARGET's data-free generator.
  - Set its display name to **"Gaussian feature replay (TARGET-style)"**.
  - Keep `spec.name = m2_target` so paths stay stable, and add a `display_name` field.
- Make its ledger honest:
  - Each client fits per-class (mean, diagonal variance) on its own current shard.
  - It releases them once, at task k, as a family-F6 record whose `touched` is that shard.
  - The server aggregates the statistics weighted by counts and broadcasts them.
  - Replay samples drawn from the broadcast Gaussians are post-processing and add nothing to `touched`.
- If M2 has a KD term, restrict it to old classes.

### 4f. Lag diagnostic

Write `results/fx4_lag_diagnostic.csv` for CUB, seed 0. It gives the per-task accuracy rows for M1, M2 and M5 before the fix (from the archived CSVs) and after the fix (test accuracy), so the briefing can show the lag fix in one table.

### 4g. Automatic gate (simulation only)

**Run.** A PBS array over:
- datasets {cifar100, cub200, imagenet_r};
- seeds {0, 1, 2};
- methods {M0, M1, M2, M3, M5};
- test accuracy.

**Pass condition,** per (dataset, method) for M1, M2 and M5, using means over seeds:
- BWT ≤ +0.02;
- last-task accuracy ≥ M0's last-task accuracy − 0.15;
- final average accuracy ≥ M0's final average accuracy.

**If the gate fails,** run one tuning pass on a validation split carved from train (10%, stratified, fixed seed). Never tune on test.
- Grid: lr ∈ {0.1, 0.5}, local_epochs ∈ {5, 30}, distillation_weight ∈ {0.5, 1.0}, class-balanced CE ∈ {off, on}.
- Select by validation final average accuracy, subject to BWT ≤ 0.02.
- Rerun the gate with the selected configuration.

**If it still fails,**
- keep the best configuration;
- mark **GATE FAILED** in STATE.md and in the briefing errata, with the numbers;
- continue. Do not stop.

**Outputs:**
- `results/fx4_gate.csv` with columns `dataset, method, seed, config_id, final_avg_acc, last_task_acc, bwt, aia, m0_final_avg_acc, m0_last_task_acc, pass`.
- `results/fx4_gate_grid.csv`, if the grid ran.

**Then regenerate utility results:**
- `results/accuracy_matrix_<dataset>.csv` for every method (M0 to M8), seeds 0 to 4, with both `acc` and `acc_train`. Include Camelyon17.
- `results/tab05_utility_baselines.csv` for 10 tasks. Add 20 tasks only if time allows, and mark which.

### 4h. Shadow runner v2 (includes the FX3 views)

**Output file.** `_run_one_shadow` writes `shadows_v2/<dataset>/<method>/seed<S>/shadow_<id:06d>.npz`. Keys:
- `shadow_id`, `target_ids`, `target_task`, `target_client`, `in_out`, `views`;
- one `scores_<view>` array per view, shaped targets × rounds.

**Views by family:**

| Family | Methods | Views |
|---|---|---|
| F1 | M0, M1, M2, M3, M5 | `full` only. The logit-margin attack reads only the global model. With exact weights (test i), that model is identical under secure aggregation. FIG11 labels F1 bars "identical by construction (Prop. 3)". |
| F2 | M4 | `full`: the client's task-k prototype (constant by construction). <br>`aggregate`: the count-weighted mean over clients of class y at task k; this is what secure aggregation reveals. <br>`global`: the server prototype bank after round r, exactly as `predict` uses it. |
| F5 | M8 | `full`: the client's R_c (constant by construction). <br>`aggregate`: Σ_c R_c for task k. <br>`global`: the running state after round r, exactly the matrix `predict` inverts. <br>Every view is scored by leverage xᵀR⁻¹x. |

`global` is the only view that is not constant by construction. It is the true retention measurement for M4 and M8.

**Tests:**
- With a single task, `global` after task k equals `aggregate`.
- `aggregate` equals the count-weighted combination of the per-client payloads.
- Old npz files with the key `scores` load as view `full`.

### 4i. Wave V2 (P0)

**Scope.**
- Methods: M1, M2, M3 and M5 (fixed code) plus M4 and M8 (multi-view).
- Datasets: cifar100, cub200, imagenet_r.
- Seeds: 0, 1, 2.
- 1024 shadows each.
- Unchanged settings: `calibration_frac` 0.8, 5 targets per shard, `p_in` 0.5, 10 tasks × 10 clients, β = 0.5.

**Manifest-driven array.**
- The manifest is `build/waves/v2_main.tsv` with columns `dataset, method, seed, start, count, out_dir`.
- Chunks of 64 give 16 subjobs per (dataset, method, seed), so 6 × 3 × 3 × 16 = **864 subjobs**.
- `code/scripts/pbs/shadow_wave.pbs` reads line `$PBS_ARRAY_INDEX` and calls the existing `python -m p3fcl.cli shadows ... --workers 8`.
- Resources: `-l select=1:ncpus=8:mem=32gb`, submitted as `-J 0-863%56`.

**Pilot first.**
1. Submit 6 subjobs, one per method, on the largest dataset.
2. Time one chunk. Set walltime to 3 × the measured time.
3. Project the disk use from the pilot output sizes.
4. If the projection exceeds free space minus 100 GB, stop and ask. This counts as a blocker.

**Then** submit the full array. Resubmit failed subjobs by index list.

**Accept when** every (dataset, method, seed) has 1024 npz files that all load.

**M0:** no new shadows if it passed test (i). Use `shadows/` seeds 0 to 2 with `shadow_id < 1024`.

---

## 5. FX3: Aggregate and global-state attacks (P1; the code ships with 4h)

After wave V2 finishes:

1. FX2's per-task LiRA runs on every view (§7).
2. Write `results/fx3_views_summary.csv` with columns `dataset, method, family, view, elapsed, auc, tpr1, tpr01` (each metric with `_ci_lo` / `_ci_hi`), `n_seeds, n_shadows`.
3. **FIG11 v2 (secure aggregation):**
   - Grouped bars of TPR@1%FPR at e = 0 and e = 6 (fixed-k set).
   - Bars are {per-client (`full`), `aggregate` (what secure aggregation reveals), `global` state}.
   - Groups are M0 and M5 (F1), M4 (F2) and M8 (F5).
   - Draw a chance line at 0.01 and hierarchical-bootstrap CIs.
   - Put CIFAR-100 in the main panel and the other datasets in an appendix variant.
4. **FIG19 (new): retention versus release.**
   - For M8 and M4, plot normalised leakage against e under `full` (dashed; flat by construction) and under `global` (measured).
   - Overlay normalised test accuracy.
   - This figure separates transcript persistence from state retention.
5. Report the outcome as it is. If M8's `global` leakage stays near its e = 0 value, say so. If it decays, say so.

---

## 6. FX1: Lifelong accountant rewrite (P0)

### Definitions

A unit instance u has data D(u):

- **U1:** {x}, a single example. In our streams, **U5 ≡ U1**, because each person is one example. State this; do not plot U5 separately.
- **U2:** D_{c,k}, client c's shard in task k.
- **U3** (window W, default 3): ∪_{k=s}^{s+W−1} D_{c,k} for every start s.
- **U4:** ∪_k D_{c,k}.

**Influence count.** m_T(u) is the number of released records r with round(r) ≤ the end of task T−1 and touched(r) ∩ D(u) ≠ ∅.

**Release-level model.** Each released record is one Gaussian mechanism with noise multiplier σ relative to the unit's sensitivity. Then

  ε_U(T) = max_u ε( Gaussian(σ) composed m_T(u) times, δ ),

using the existing RDP → DP conversion. Also write `m_T_passes`, the count weighted by `passes_over_data`, as a secondary appendix column only.

### Implementation

- New signature: `account(ledger, stream, unit, sigma, delta, window=3) -> DataFrame[T, m_T, argmax_instance, eps]`.
- Build `home[id] = (client, task)` from the stream.
- Parallel composition falls out of the max over instances. Delete the old special cases: `_max_task_touch_multiplicity` and the U4 branch that ignored passes.
- **Extrapolating to T = 1000:**
  - A curve is *contractive* if m_T is constant over the second half of the observed T range.
  - Otherwise it is *accumulating*, with slope from a linear fit of m_T over the second half.
  - Draw the observed part solid and the extrapolated part dashed. Add a column `observed ∈ {1, 0}`.
- **Ledgers:**
  - Post-fix `sim.run` ledgers, seed 0, CIFAR-100, 10 tasks, for M0, M1, M2, M3, M4, M5, M8 and M9 (M9 at U1 and U2).
  - Also the 50-task CIFAR ledgers for M0, M5, M8 and M9, if FX5 produced them.
- **Noise and δ:** use the σ recorded in the archived `fig05_eps_of_T.csv` and state it. δ = 1e-5.

### Tests (add to `code/tests/test_dp_accountant.py`)

1. ε_U1 ≤ ε_U2 ≤ ε_U3 ≤ ε_U4 for every T, on every real and synthetic ledger.
2. Synthetic no-replay ledger (each id touched only in its own task): U1 and U2 are flat, U3 is the W-fold value, and U4 is linear in T.
3. Synthetic replay ledger: U1 and U2 grow with T.
4. M9 ledger: m_T = 1 for U1 and U2 at every T; m_T = W for U3; m_T = T for U4.
5. The existing claim tests still pass. If one of them encoded the buggy behaviour, fix the test and log it in the errata.

### Outputs

- `results/fig05_eps_of_T.csv` with columns `method, dataset, n_tasks, ledger_hash, unit, window_W, sigma, delta, T, m_T, m_T_passes, eps, regime, observed`.
- **FIG05 v2:** 2 × 3 small multiples (M0, M1, M5, M4, M8, M9); log-y ε against log-x T; four lines (U1 to U4); the U5 ≡ U1 note in the CSV meta. Width 5.5 in.

---

## 7. FX2: Per-task LiRA, chance floors, non-parametric half-life (P0)

### 7a. Per-task LiRA

- New script `code/scripts/run_lira_pertask.py`. **PBS only.**
- **Input:** one (dataset, method, seed, view) shadow set with n_shadows = 1024. For M0, use the first 1024 old shadows.
- **Method:** the same online LiRA as now (`attacks/lira.py`) and the same calibration/eval shadow split (`calibration_frac` 0.8).
- **Old npz files:** recover `target_task` and `target_client` by rebuilding the targets with `build_targets`. Assert that the rebuilt `target_ids` match the stored ones.
- **Scoring:**
  - Score at the round that ends task k + e.
  - **Fixed-k set K = {0, 1, 2, 3}, elapsed e ∈ {0, …, 6}.** Every e then has the same target population.
  - Also emit per-k rows for every valid (k, e), for the appendix.
- **Output per shadow set:** `results/a1_lira_pertask_<dataset>_<method>_seed<S>_<view>.csv` with columns:
  - `dataset, method, family, view, seed, n_shadows, n_eval_shadows, task_k, elapsed, auc, tpr1, tpr01, n_targets, n_pos, n_neg`;
  - the per-pair Clopper–Pearson interval only as `cp_lo, cp_hi`, marked not for reporting in the meta.

### 7b. Chance floors and normalisation

**Leakage.**
- The floor β is 0.01 for TPR@1%FPR, 0.001 for TPR@0.1%FPR, and 0.5 for AUC.
- Normalised leakage: L(e) = (TPR(e) − β) / (TPR(0) − β), with numerator and denominator each pooled over K.

**Accuracy (test).**
- The floor a0(t) = 1 / |classes seen after task t| for class-incremental streams. For Camelyon17 it is the majority-class rate of the evaluation set.
- Normalised accuracy: A(e) = mean_k[ acc_k(k+e) − a0(k+e) ] / mean_k[ acc_k(k) − a0(k) ], with k ranging over K.

**Guard.** If the lower 95% bound of TPR(0) − β is ≤ 0.005, or the lower bound of acc_k(k) − a0 is ≤ 0.02, the quantity has no signal. Set status `no_signal` and report no half-life.

### 7c. Half-life (matches Definition 10 in the paper)

- **Definition.** h is the first e ≥ 1 at which the normalised value is ≤ 1/2, linearly interpolated between e − 1 and e. Also report the integer first crossing as `halflife_int`.
- **Censoring.** If there is no crossing by E = 6, the value is censored and reported as a lower bound `> 6`. Never report ∞, and never extrapolate.
- **The exponential fit** stays only as secondary columns (`halflife_expfit`, `r2`) for the appendix.
- **Decoupling ratio** ρ = h_leak / h_acc. `ratio_type` is one of:
  - `point`;
  - `lower_bound`, when h_leak is censored;
  - `undefined`, when h_acc is censored or either side is `no_signal`;
  - `by_construction`, for the per-client `full` view of F2 and F5.
- **Confidence intervals.** Hierarchical bootstrap with 2000 replicates.
  - Resample seeds (outer), then targets within each seed (inner). A target keeps all of its eval-shadow pairs.
  - In each replicate, recompute TPR and AUC, the curves, and h.
  - Report the percentile 95% CI. If more than half the replicates are censored, report the CI as censored.
  - Also report the minimum and maximum across seeds.

### 7d. Outputs

- `results/a1_lira_fixedk_summary.csv`: `dataset, method, family, view, elapsed, auc, tpr1, tpr01` (each with `_ci_lo` / `_ci_hi`), `n_seeds, n_shadows`.
- `results/fig02_halflife.csv`: `dataset, method, family, view, quantity{acc|leak_tpr1|leak_auc}, halflife, halflife_int, ci_lo, ci_hi, status{ok|censored|no_signal}, horizon_E, base_value, base_ci_lo, floor, halflife_expfit, r2`.
- `results/decoupling_ratio.csv`: `dataset, method, family, view, h_acc, h_leak, ratio, ratio_ci_lo, ratio_ci_hi, ratio_type`.
- `results/fig01_decoupling.csv`: the spec schema plus `view, acc_norm, leak_norm, acc_train`.
- `results/a1_m0_budget_check.csv`: M0 at 1024 shadows × 3 seeds against 4096 shadows × 5 seeds (existing shadows), as an appendix robustness row.

---

## 8. FX5: M9, the contractive DP analytic learner (P0 core, P2 stretch)

This implements Algorithm 3 in the paper.

**Registration.**
- File: `code/src/p3fcl/methods/m9_contractive.py`.
- Class: `ContractiveDPAnalytic(FCLMethod)`, registered as `m9_contractive`, family GRAM.
- `MethodSpec`: use the same `retention_type` as M8, with `retention_knob_name = "gamma"`.
- Display name: "Contractive DP analytic (ours)".

### Algorithm

1. **Preprocess.** Project the features with a PCA basis P ∈ ℝ^{d×p} fit on the public `ref` split, which is disjoint from train and test and so costs no privacy. Then L2-normalise so that ‖x̄‖ ≤ B = 1.
2. **Client statistics.** Client c at task k computes G_c = Σ x̄x̄ᵀ and H_c = Σ x̄yᵀ, with y one-hot.
3. **U2 only.** Clip the pair (vech G_c, vec H_c) jointly to Frobenius norm C.
4. **Aggregation and noise.**
   - The server receives Σ_c through secure aggregation.
   - It adds N(0, σ²) to vech(G) and to vec(H), then symmetrises G̃ by mirroring.
   - σ = `analytic_gaussian_sigma(eps0, delta0, Δ)`, with Δ_U1 = √(B⁴ + B²) = √2 and Δ_U2 = C.
   - **Trust model:** secure aggregation with distributed noise, simulated as central noise.
   - **Ledger:** only the noisy aggregate is released, as one record per task with `client = -1`, `touched` = all task-k ids and `passes_over_data = 1`.
5. **State update.** R ← γR + G̃ and Q ← γQ + H̃. The prediction weights are W = (Π₊(R) + λI)⁻¹Q, where Π₊ clips negative eigenvalues to 0.
6. **Non-private limit.** ε = ∞ means σ = 0.

### Hyperparameters

- The grid: p ∈ {32, 64, 128, 256, d}, λ ∈ {1e-2, 1e-1, 1, 10, 100}, and C.
- Select them by running the full M9 pipeline on the `ref` split only:
  - split `ref` 50/50 into ref-train and ref-val;
  - use the same task and client structure;
  - select per (dataset, unit, ε).
- Set C to the 95th percentile of pseudo-client norms on ref-train.
- Never select on train or test data.
- Log every choice in `results/m9_hparam_selection.csv`.

### Tests

1. With ε = ∞, γ = 1, p = d and the same total λ, M9's predictions are identical to M8's. Align the λ convention first: M8 adds λI per client.
2. **Numeric sensitivity.** Random neighbouring datasets with ‖x‖ ≤ 1 change (vech G, vec H) by at most √2 under U1, and by at most C after client clipping under U2.
3. There is exactly one released record per task.
4. The accountant on M9's ledger is flat under U1 and U2 (ties to FX1 test 4).
5. The noise seed is separate from the data seed. The same seeds give identical output.

### Core sweep (P0)

**Grid.**
- ε ∈ {0.5, 1, 2, 4, 8, ∞};
- unit ∈ {U1, U2};
- γ ∈ {1.0, 0.9, 0.7};
- datasets {cifar100, cub200, imagenet_r};
- seeds {0, 1, 2};
- 10 tasks × 10 clients.

**Metrics:** test final average accuracy, BWT, AIA, and the full accuracy matrix.

**References** on the same streams: M8 (non-private) and M0 (non-private).

**U2 with 10 clients** is expected to be hard, because client-level DP with few clients usually is. Also run U2 on CIFAR-100 with n_clients ∈ {50, 100} and ε ∈ {1, 4, 8}. The figure then shows how U2 utility scales with federation size. Report what it shows.

### Outputs (P0)

- `results/m9_sweep.csv`: raw, one row per run.
- **FIG08 v2, privacy–utility:**
  - Test final average accuracy against ε (log-x), one panel per dataset.
  - Lines for U1 and U2 at γ = 1.0.
  - Horizontal reference lines for M8 and M0.
  - Data in `results/fig08_pareto.csv` (spec schema). Fill the `eps_audited_*` columns from the M9 audit if it ran; otherwise leave them empty.
- **TAB08:** dataset × ε × unit → final average accuracy (mean and CI over seeds).
  - Add the certified ε at T = 10 and T = 50 for U1, U2, U3 (W = 3) and U4, taken from the FX1 accountant.
  - Files: `tables/tab08_dp_utility.csv` and `.tex`.

### M9 audit (P1): the empirical check of Theorem 11

- Online LiRA against M9 on CIFAR-100, U1, ε ∈ {1, 4, ∞}, γ = 1.
- 1024 shadows × 1 seed, scored by leverage on the `global` state.
- Plot the empirical ROC (log-log) against the DP bound TPR ≤ e^ε · FPR + δ.
- Outputs: `results/fig10_m9_audit.csv` and FIG10 v2.
- If the empirical TPR exceeds the bound beyond its CI anywhere, there is a bug. Stop the M9 line, debug it, and log what you found.

### Stretch (P2): DP baselines and horizon

- **FIG09 v2.** Task-0 accuracy against T for γ ∈ {1.0, 0.9, 0.7} at ε = 4 (U1), on CIFAR-100 with 50 tasks (2 classes per task). Add the final accuracy at 10 and 20 tasks. Data in `results/fig09_advantage_vs_T.csv`.
- **DP-FedAvg linear probe:** M0 with client-level clipping and Gaussian noise each round, under U2. Choose σ so the FX1 accountant gives the target total ε at horizon T.
- **DP-M5:** the fixed M5 with DP-FedAvg. Under U2, buffer re-reads raise m_T, so σ must grow with T at fixed ε.
- Plot accuracy at fixed ε ∈ {1, 4} against T ∈ {10, 20, 50} on CIFAR-100, next to M9. Add the lines to FIG09 and the rows to TAB08.

---

## 9. FX6: Figures and tables

### Global style (enforced in `plotting.py`, used by every figure)

- **Sizes:** figure width ≤ 5.5 in, which is the ICLR text width. Height ≤ 3.4 in unless stated otherwise.
- **Text:** fonts ≥ 7 pt at final size. No titles inside the figure files; captions belong to the paper.
- **Formats:** vector PDF plus PNG at 300 dpi.
- **Method styles:** one `METHOD_STYLE` dict gives each method a distinct colour (Okabe–Ito palette) and marker, with linestyle by family. Every figure uses the same style for the same method.
- **Display names** come from `results/method_descriptions.csv` (FX7).
- **Data:** every figure has a CSV of exactly the plotted numbers plus a meta.json. Figure scripts read only CSVs.

### Per figure and table

- **FIG01, decoupling (P0):**
  - Rows: {normalised test accuracy, normalised TPR@1%FPR}. Columns: datasets. x-axis: e from 0 to 6. One line per method.
  - M4 and M8 `full` are dashed, with a small label "per-client release: constant by construction". M4 and M8 `global` are solid.
  - Clip y to [−0.1, 1.2]. Draw CI bands.
- **FIG02, half-life (P0):**
  - A compact forest plot, one panel per dataset (3 across), one row per method.
  - Each row has two markers with CIs: h_acc (●) and h_leak (■).
  - A censored value is drawn as a right arrow at 6 labelled "> 6". A `no_signal` value is drawn as "n/s".
  - Size ≤ 5.5 × 3.0 in.
- **FIG05** as in §6. **FIG08, FIG09 and FIG10** as in §8. **FIG11 and FIG19** as in §5.
- **FIG13, Gram inversion (P1):**
  - 25 trials per cell, reported as median with a bootstrap CI.
  - Remove the causal claim about anisotropy. If the spectrum panel fits, keep it only as a descriptive appendix panel.
- **FIG16 ROC and FIG17 seed variance (P1):** regenerate from the v2 per-task data (fixed-k, e = 0).
- **FIG18, natural versus Dirichlet (P1), rerun:**
  - **Clients:** both arms have 5 clients. In the natural arm, client i is hospital i. In the Dirichlet arm, the same samples are pooled and split into 5 clients by Dirichlet(β = 0.5) on labels.
  - **Tasks:** within each client, order samples by slide id and cut them into 5 contiguous chunks. Task t is chunk t of every client. Use the WILDS metadata field `slide`; if it is missing, use a fixed seeded order and document it.
  - **Methods:** M0 and the fixed M5.
  - **Budget:** 1024 shadows × 3 seeds × 2 arms × 2 methods.
  - **Output:** the spec schema plus `n_clients`.
  - Report what it shows. The same pattern in both arms is itself the robustness answer.
- **TAB03, leakage:** method × dataset; AUC, TPR@1% and TPR@0.1% at e = 0 and e = 6 (fixed-k); a view column; CIs.
- **TAB04, half-life:** h_acc, h_leak, ρ and status.
- **TAB05, utility (test accuracy):** final average accuracy, BWT and AIA for 10 tasks, plus 20 tasks if they ran.
- **TAB07, reproduction gap:** recompute from the new TAB05.
- **Every table** as `.csv` plus `.tex`, using booktabs, no vertical rules, and a width ≤ 5.5 in.

---

## 10. FX8: C2 dose–response rerun (P2)

- Run only after wave V2 finishes and cores are free.
- Dataset: CIFAR-100. 512 shadows × 3 seeds per level.
- Methods and knobs:
  - fixed M5 (individual retention): `exemplar_budget` ∈ {1, 2, 5, 10, 20, 50};
  - M2 (semantic retention): synthetic samples per class ∈ the same six levels.
- Drop the M4 arm, because its knob was dead code. Say so in the errata.
- Produce FIG03 v2 and FIG04 v2 with the spec schemas. Measure retention as −BWT on test accuracy.

---

## 11. FX7: Documentation for the paper (P0, continuous)

- **`data/MANIFEST.json`:** set Camelyon17 `url_used` to the Hugging Face mirror actually used (take it from the logs), and add a note that codalab was unreachable.
- **`results/method_descriptions.csv`:** one row per method with these columns:
  - `method_id` and `display_name`;
  - `implemented` (one sentence on what the code does);
  - `differs_from_original` (one sentence);
  - `retention_mechanism`;
  - `released_families`.

  Opus cites this file in the paper.
- **`paper/PAPER_BRIEFING.md`,** in this order:
  1. `## ERRATA (2026-09-23 fix phase)` at the top. For each of R1 to R12: the old claim or number, then the new number or "withdrawn".
  2. `## FINAL RESULTS (post-fix)`. For every figure and table: its path; 2 to 4 plain sentences on what it shows; the key numbers with CIs; and its caveats. Keep the tone factual, with no adjectives like "dramatic".
  3. `## CUT`: each cut item and its reason.
- **`results/paper_numbers.csv`:**
  - Columns: `key, value, ci_lo, ci_hi, unit, source_csv, row_filter, note`.
  - One row for every number the paper may quote. That includes the headline AUC and TPR, half-lives, ratios, ε values, M9 accuracies and the gate outcome.
  - Make the keys descriptive, for example `m8_cifar100_global_tpr1_e6`.
  - This is the file Opus pulls numbers from.
- **`agents/OPEN_QUESTIONS.md`:**
  - H13 → OPEN (confounded; the natural arm had 1 client). Update it after FIG18 v2.
  - H11 → "SUPPORTED as a family split: secure aggregation removes the per-client F2/F5 attacks and does not affect the F1 attack, which uses only the global model." Add the FX3 aggregate-view numbers.
  - H2 → OPEN until FX8.
  - H4 → NOT SUPPORTED at current power (the A4 CI overlaps random).
  - Update every status after FX2, FX3 and FX5.
- **`analysis/verify_provenance.py`:** fail if any CSV, figure or table produced after the start of this phase has `pbs_jobid: null`.

---

## 12. Freeze procedure (2026-09-25 12:00 CDT)

1. **11:00 CDT:** submit nothing new except the figure and verify job. Let running jobs finish only if they will end before 12:00. Otherwise `qdel` them and mark the item `CUT`.
2. Submit `make figures` then `make verify` as one PBS job. Both must pass.
3. Do a final pass on `PAPER_BRIEFING.md` and `paper_numbers.csv`. Set the STATE.md phase to `FROZEN`.
4. Print a one-screen summary for the human covering:
   - what landed;
   - what was cut;
   - the gate outcome;
   - five to ten headline numbers with CIs.

---

## 13. Definition of done

- [ ] FX0: archive with verified checksums; live tree quarantined.
- [ ] Test suite green; every new test from 4a–4h, §6 and §8 present.
- [ ] Gate recorded as pass, or as a documented fail with the best configuration.
- [ ] Accuracy matrices and TAB05 regenerated with test accuracy.
- [ ] Wave V2 complete: 6 methods × 3 datasets × 3 seeds × 1024 shadows, all loadable.
- [ ] FIG01, FIG02, FIG05, TAB03, TAB04 and `decoupling_ratio.csv` regenerated from v2 data.
- [ ] M9 implemented and tested; FIG08 and TAB08 produced.
- [ ] FIG11 v2 and FIG19 (P1).
- [ ] FIG10, FIG18, FIG09 and FIG03/FIG04 each either produced or marked CUT.
- [ ] Errata, final results, CUT list and `paper_numbers.csv` complete.
- [ ] `make verify` passes, with no `pbs_jobid: null` after the start of this phase.