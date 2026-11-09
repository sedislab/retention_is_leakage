# Rebuttal experiments: ICLR 2027 Submission 48006

You are Claude Code, running on **Baylor Kodiak (OpenPBS)**, in the same working copy that produced
the paper. Read `rebuttal/CLAUDE_REBUTTAL.md`, this file, then `rebuttal/STATE.md`, then
`README.md` sections 4 to 6. Update `rebuttal/STATE.md` after every meaningful step, not at the end
of a phase.

## 0. Context, priorities and dates

Three reviews: Reviewer 1 (weak reject), mjT5 (weak reject, will raise the score if the concerns are
addressed), 87FH (clear reject, mostly presentation). The public discussion closes on
**Nov 18, 2026**. A revised 10-page PDF is allowed until then. Results must be final by
**Nov 14** so the authors can write the responses and revise the paper.

Work in this order. If time runs out, stop after the last finished experiment. Never skip a gate.

| ID | Answers | Compute | Due |
|---|---|---|---|
| R0 | Kodiak sanity check (GATE) | one small `batch` job | Nov 8 |
| Q  | mjT5 Q2: how targets are chosen | none, code reading | Nov 8 |
| R1 | mjT5 main concern: M2/M3 leakage growth vs. contractivity | a few CPU core-hours | Nov 10 |
| R2 | Reviewer 1 W1, mjT5 Q1: projection ablation, out-of-distribution public data | small CPU | Nov 11 |
| R3 | mjT5 units, Reviewer 1 W2: client-level privacy as the number of clients grows | small CPU | Nov 11 |
| R4 | Reviewer 1 W2: accuracy vs. examples per class | small CPU | Nov 12 |
| R5 | Reviewer 1 W3, paper limitation: DP-FedAvg baselines at matched epsilon | moderate CPU | Nov 13 |
| R6 | Results summary for the authors | none | Nov 14 |

No experiment needs a GPU. Features already exist in `features/`.

## 1. Non-negotiable rules

1. Never modify, overwrite or delete existing files in `results/`, `figs/`, `tables/`, `features/`
   or `shadows/`. Every new output is `results/rb_<name>.csv` with a `.meta.json` sidecar written by
   `provenance.py`, plus one line in `results/RUN_LOG.jsonl`. New per-shadow files go under
   `shadows_rb/`.
2. Hyperparameters, projections and clip norms are chosen only on public `ref` data
   (ref-train / ref-val), never on train or test. This is the paper's rule and it still holds.
3. Every protocol below is fixed before running. Do not change a protocol after seeing its result.
   If a stated prediction fails, report it exactly as it is.
4. Kodiak site rules: work under `/data`, never `/home`; no production runs on the login node. Every
   script that writes to `results/` runs inside a PBS job, because `provenance.py` expects
   `PBS_JOBID`.
5. Stop and ask the user at every GATE, and also when:
   - the sanity check in R0 fails;
   - a job or array would exceed 2,000 CPU core-hours;
   - a new result contradicts a number in the submitted paper;
   - you would need to change code that the paper's existing results depend on.
6. Add, do not edit. New code goes in new files (`code/scripts/rb_*.py`,
   `code/src/p3fcl/rebuttal/*.py`, new method files under `code/src/p3fcl/methods/`, new job
   templates `code/scripts/pbs/rb_*.pbs`). Do not edit existing source files, configs or templates.
7. Keep `make test` and `make verify` passing. Add a test for every new statistic or method.

## 2. Kodiak platform (verified 2026-09-14; re-check anything that fails)

- **Scheduler:** OpenPBS 23.06.06. Resources are requested as
  `-l select=1:ncpus=C:mem=Mgb`; `nodes=` and `ppn=` are rejected.
- **Arrays:** `-J 0-N%K`, with index in `$PBS_ARRAY_INDEX`. The `%K` throttle works.
- **Queues:** `batch` (default, 372 h), `gpu` (168 h), `interactive` (240 h). Use `batch` for
  everything here.
- **Core cap:** `batch` allows a user at most 512 concurrent cores. Size arrays so that
  ncpus times K stays at or below 480, as the paper's shadow arrays did (`ncpus=8`, `%60`).
- **Environment:** `module load python/3.10.4`, then `source envs/p3fcl/bin/activate`. There is no
  conda module.
- **Storage:** `/data` is GPFS and about 88% full, and the per-user quota is unknown. Before writing
  `shadows_rb/` (expected 1 to 2 GB), check free space and your usage with `df -h /data` and
  `du -sh` on the repo.
- **Templates:** copy the patterns in `code/scripts/pbs/shadow_v3.pbs` (manifest-driven arrays) and
  `run_job.pbs`. Keep `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1` and `OPENBLAS_NUM_THREADS=1` in every
  CPU job.
- **Monitoring:** `qstat -u islamm`; job logs go in `logs/`.

## 3. R0: sanity check (GATE)

1. **Confirm the working copy.** Confirm this is the working copy that produced the submission:
   - `features/` exists for cifar100, cub200 and imagenet_r (splits train, test, ref);
   - `envs/p3fcl` exists;
   - the `source_hash` in a recent results sidecar matches the current `code/src` and
     `code/configs`.

   Record whether the per-shadow files `shadows/<dataset>/<method>/shadow_*.npz` still exist for
   m2_target and m3_fot. R1 has an optional arm that uses them.
2. Copy the `rebuttal/` folder in if it is not there (the user copies it from their laptop). Do
   **not** copy anything else from the laptop's repository. That copy is an anonymised release copy,
   and it would overwrite provenance.
3. Run `make test`.
4. **Reproduction check** in one small `batch` job. Compare seed-matched rows with the committed
   CSVs:
   - M8 non-private final average accuracy on CIFAR-100, CUB-200 and ImageNet-R
     (`results/m9_sweep_<dataset>.csv`; paper means 0.883, 0.871, 0.635).
   - M9 under U1 with gamma = 1 and epsilon_0 = 4 on CIFAR-100, with the selected hyperparameters
     from `results/m9_hparam_selection.csv` (paper mean 0.810).
   - M0 final average accuracy on CIFAR-100 (`results/m9_sweep_cifar100.csv`, method `m0_fedavg`).

   These use the same code and features, so they should match to 1e-6. Write the outputs to
   `rebuttal/r0_check.csv`, not to `results/`.

   **GATE R0.** Write the comparison table into `rebuttal/STATE.md` and ask the user to approve
   before R1. If any check fails, stop and report.

## 4. Q: how the five targets are chosen (mjT5 Q2)

No compute. Read `shadow_runner.build_targets`, `_population_mask` and
`code/configs/attack_lira.yaml`, and write two plain sentences into `rebuttal/answers.md`.
Expected content, to be confirmed from the code: in every client shard of every task, 5 targets
are drawn uniformly at random without replacement by a seeded permutation; the pool is fixed
across shadows; and in every shadow the membership of every record, targets included, is
resampled independently with probability 0.5.

## 5. R1: is the M2/M3 leakage growth new information? (mjT5, main concern)

**Claim under test.** M2 and M3 read a task's data only during that task (accountant count 2,
constant). After the target's task, every later broadcast of M2 or M3 is a function of earlier
broadcasts and of data that is independent of the target. By the data-processing inequality, the
best possible attack on the transcript cannot gain anything after the target's task. So the growth
the paper reports must come from the logit-margin attack getting better at reading information that
was already released at the target's task.

At that task, M2 releases per-class Gaussian statistics (family F6), and M3 releases a task
covariance and the subspace computed from it (family F5). The paper's audit scored only the F1 logit
margin for both methods (`METHOD_REGISTRY` in `shadow_runner.py` maps both to `MODEL_DELTA`).

**Prediction, fixed in advance.** For each dataset and method, the TPR at 1% FPR of an attack on the
release alone at e = 0 is at least the TPR at 1% FPR of the paper's logit-margin transcript attack
at e = 6 (`results/a1_lira_fixedk_summary_<dataset>_<method>_full.csv`, column `tpr1`).

**Method.** This is the cheap path, with no federation training. The release at task k depends only
on which records of task k's shards are members, so it can be recomputed from the population mask.

1. Loop over dataset in {cifar100, cub200, imagenet_r}, method in {m2_target, m3_fot}, seed in
   {0, 1, 2} and shadow in 0..1023. For each, rebuild:
   - the same base stream (`stream.base_seed = seed`; 10 tasks, 10 clients, beta 0.5);
   - the same target pool (`build_targets(base_stream, 5, base_seed)`);
   - the same population mask (`_population_mask(dataset, method, shadow_id, n_total, 0.5)`).
     The method string must be the registry key the original shadows used, so the masks are
     identical.

   Then apply the mask.
2. For the target's task only, compute the released statistic from the member records. Call the
   method's own code for the statistic wherever possible, instead of re-implementing it.
   - **M2, broadcast view** (what every client receives): the count-weighted aggregated per-class
     mean and diagonal variance, exactly as `TARGET.fit_task` builds `_generators`. Score: Gaussian
     log-likelihood of the target's feature under the diagonal Gaussian of its class.
   - **M2, client view** (what the server sees without secure aggregation): the target's own
     client's per-class mean and variance. Same score.
   - **M3, broadcast view** (what clients receive): the top `subspace_rank` eigenvectors of the task
     covariance. These are the new directions clients use to project gradients. Use the same
     `subspace_rank` as the paper's M3 shadows; check the shadow configs or meta files (the paper
     says 8). Score: the fraction of the centred target feature's squared norm that lies in that
     subspace. The centre is the mean of the public `ref` features of that task's classes.
   - **M3, server view:** the full centred task covariance R (the GRAM record's payload). Score:
     leverage f^T (R + lambda I)^{-1} f, with the same centring and the same lambda convention as
     the existing GRAM scorer in `shadow_runner._score_target`.

   Each release happens once, so a target's score is constant from its task onward. Write the
   scores in the existing per-shadow npz schema (`scores_<view>` arrays, NaN before the target's
   task) under `shadows_rb/<dataset>/<method>/shadow_XXXXXX.npz`, so that the existing
   `run_lira_pertask.py` pipeline can score them.
3. **Equivalence test, which must pass before the array runs.** For 3 shadow indices per
   (dataset, method), run the real method's `fit_task` on the masked stream up to the target's task.
   Assert that the cheap path's statistics equal the method's own release (maximum absolute
   difference below 1e-10). Add this to the test suite.
4. **LiRA scoring with the existing pipeline:**
   - fixed target-task population K = {0, 1, 2, 3}, horizon E = 6, calibration fraction 0.8;
   - the same calibration/evaluation shadow split as the paper's `full` view for that
     (dataset, method), so the comparison with the e = 6 margin numbers is paired. Reuse the `full`
     calibration key, for example through `file_suffix`;
   - report TPR at 1% and at 0.1% FPR and AUC, with the same hierarchical bootstrap
     (2,000 resamples), at e = 0 and e = 6. The two should be equal.
5. Output `results/rb_release_audit.csv` with these columns:
   - `dataset`, `method`, `view`, `elapsed`;
   - `tpr1`, `tpr1_ci_lo`, `tpr1_ci_hi`, `tpr01`, `auc`;
   - `paper_margin_tpr1_e0` and `paper_margin_tpr1_e6`, copied from the paper's summary file;
   - `paper_source`;
   - `prediction_holds`.
6. **Optional, only if the original `shadows/` files for m2_target and m3_fot still exist.** Add a
   combined arm. For each shadow, sum the per-round log-likelihood ratios of the paper's margin score
   (`scores_full`) and of the release score, using the same calibration split. Report it at e = 0 to
   6 as `view = combined`. This shows the best of both attacks on the same shadows.

Expected cost: 18,432 shadows of pure statistics, a few CPU core-hours in total. Use one small array
with one task per (dataset, method, seed), with `ncpus=8` and at most `%20`.

If the prediction fails in some cell, change nothing and report it. The data-processing argument
still bounds the best possible attack; a failure would only mean that our e = 0 scores are weaker
than the margin attack at e = 6.

## 6. R2: projection ablation and out-of-distribution public data (Reviewer 1 W1, mjT5 Q1)

Accuracy only, no shadows.

Setup:
- All three datasets, 10 clients, seeds {0, 1, 2}, gamma = 1.
- Unit U1 with epsilon_0 in {1, 2, 4, 8, inf}.
- Unit U2 at epsilon_0 = 8 only, for completeness.

Arms:
- **A `pca_id`** (the paper's setting): PCA fitted on the dataset's own `ref` split; p and lambda
  selected on its own ref-val. Reuse `results/m9_hparam_selection.csv`.
- **B `no_proj`:** P = I (p = 768). Select lambda on the dataset's own ref-val with the paper's grid.
  Nothing is fitted for the projection, so public data only choose lambda.
- **C `pca_ood`:** PCA fitted on another dataset's `ref` split. Select p, lambda (and C for U2) on
  that other dataset's ref-val, using its own labels and stream. Pairs:
  - cifar100 uses imagenet_r;
  - cub200 uses cifar100;
  - imagenet_r uses cifar100.
- **D `random_proj`:** a data-independent orthonormal projection (seeded Gaussian matrix, then QR)
  with p in {32, 64, 128, 256}. Select p and lambda on the out-of-distribution ref-val, as in C.
  This arm uses no in-distribution public data at all.

Output `results/rb_projection_ablation.csv` with columns `dataset`, `arm`, `unit`, `eps`,
`selected_p`, `lambda`, `clip_C`, `seed`, `final_avg_acc` and `bwt`.

## 7. R3: client-level privacy as the number of clients grows (mjT5 units, Reviewer 1 W2)

**A problem in the paper to document first.** The paper's n_clients arm (`run_m9_sweep.py`,
n in {50, 100}) reused clip_C from 10 pseudo-clients on ref-train. A client's statistic grows with
its number of examples. With more clients each client is much smaller than C, so the noise, which
scales with C, did not shrink, and the arm could not show the expected gain. Write this into
`rebuttal/answers.md` in two plain sentences, with the file and line numbers.

**Protocol.**
- **C(n):** the 95th percentile of the pair norms of pseudo-clients whose size matches the real
  clients. The real expected client size is m(n) = (training examples per task) / n. Build each
  pseudo-client on ref-train: draw Dirichlet(beta = 0.5) class proportions over the task's classes,
  then sample m(n) examples with replacement from ref-train. Fit nothing on train or test.
- For each n, select (p, lambda) on ref-val with the paper's grid, using C(n).
- **Values of n:**
  - CIFAR-100: n in {10, 50, 100, 500, 1000, 2000}.
  - CUB-200 and ImageNet-R: n in {10, 50, 100}, because their tasks are small.
- **Runs:** U2 with epsilon_0 in {1, 4, 8}, gamma = 1, seeds {0, 1, 2}. As references, also run U1
  at the same values of n (U1 does not depend on C) and non-private M8.
- In the same script, also rerun the paper's old setting (fixed C, n in {50, 100}), so the table
  shows the old and the corrected calibration side by side.
- **Record per run:** the fraction of clients clipped, the noise standard deviation, and the
  Frobenius norm of the first task's noiseless aggregate.

Output `results/rb_client_scaling.csv`.

## 8. R4: accuracy vs. examples per class (Reviewer 1 W2)

**Hypothesis, fixed in advance.** Under U1 the noise has a fixed size, while the statistics grow with
the number of examples per class. So accuracy depends on examples per class, not on the dataset.

**Test inside one dataset.**
- Subsample the CIFAR-100 training set per class (seeded, stratified) to m in
  {25, 50, 120, 250, all (about 400)}.
- Run M9 under U1 with epsilon_0 in {1, 4, 8, inf}, gamma = 1, seeds {0, 1, 2}, and the paper's
  selected (p, lambda) for CIFAR-100.
- Also run non-private M8 at each m.

Expected: at m = 25, accuracy at finite epsilon_0 falls to CUB-like levels while M8 stays high.

Output `results/rb_examples_per_class.csv`.

## 9. R5: DP-FedAvg baselines at matched epsilon (Reviewer 1 W3, and a stated limitation)

Two new methods in new files, with honest ledger records:

- **`m0_dpsgd_u1` (example level):**
  - FedAvg as in M0: one round per task, 30 local full-batch epochs.
  - In every local step, the client clips each per-example gradient of the linear head to norm Cg,
    sums them, adds N(0, (z * Cg)^2 I), and divides by its number of examples.
  - Choose z with the repo's RDP accountant so that 30 steps at sampling rate 1 give
    (epsilon_0, 1e-5).
  - Each example is read only in its own task's round, so the lifelong guarantee under U1 equals
    the per-round guarantee (Theorem 2 with h = 1). Check with the ledger accountant that the count
    is 1 under U1.
- **`m0_dpfedavg_u2` (client level, McMahan et al., ICLR 2018):**
  - Clip each client's update to norm S.
  - The server sums the clipped updates, adds N(0, sigma^2 I) with sigma = sigma_AG(epsilon_0, 1e-5, S),
    and divides by the number of participating clients. Use the same analytic Gaussian routine
    as M9.

**Hyperparameters**, selected on ref-val:
- learning rate in {0.1, 0.5, 1.0};
- Cg or S from a 5-point grid around the median per-example gradient norm (or median client update
  norm), measured on ref-train.

**Runs:** epsilon_0 in {1, 4, 8, inf}, all three datasets, seeds {0, 1, 2}, 10 clients. If R3 shows
gains at large n, also run `m0_dpfedavg_u2` on CIFAR-100 at n in {100, 1000}, with matched selection.

**Correctness gate.** With no noise and no clipping, both methods must reproduce M0's accuracy within
0.01.

Output `results/rb_dp_fedavg.csv` with final average accuracy, BWT and newest-task accuracy.

## 10. R6: summary for the authors

Write `rebuttal/RESULTS_SUMMARY.md` containing:
- one short table per experiment (means over seeds, with 95% intervals over seeds);
- each stated prediction and whether it held;
- for every number, its source file and row filter.

Do not write rebuttal prose; the authors will do that. Then run `make test` and `make verify`, and
list any check that fails.

## 11. Out of scope

- Re-running the paper's main shadow campaigns.
- Tuning anything on test data, or after seeing a result.
- Training prompt or adapter methods with a trainable backbone.
