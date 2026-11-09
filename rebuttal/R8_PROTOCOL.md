# R8 protocol: symmetric re-selection (fixed 2026-11-09, before any R8 job)

Requested by the user on 2026-11-09.

**Provenance of this design (stated explicitly): the grid widening and the size-matched selection
federation below were motivated by the test-split sensitivity results of R7 and R7b**
(`results/rb_sensitivity.csv`, `results/rb_sensitivity_ext.csv`). Those showed that the R3/R5
selections were far from the test oracle, and that many oracle points lay outside or on the edge of
the R5 grid. R8 is therefore a test-informed re-design, and must be reported as such. Selection
inside R8 uses public data only (ref-train / ref-val). The test split is used only for the final runs.
Earlier outputs are not overwritten. R3/R5/R7/R7b stay as reported.

## 1. Methods and cells (ε0 ∈ {1, 4, 8} throughout; 72 cells)
- **U1, n = 10, all three datasets:** DP-SGD (`m0_dpsgd_u1`) and M9 (`ContractiveDPAnalytic`, U1).
  18 cells.
- **U2:** DP-FedAvg (`m0_dpfedavg_u2`, McMahan fixed-denominator mean) and M9 (U2, size-matched C):
  - CIFAR-100: n ∈ {10, 100, 500, 1000, 2000};
  - CUB-200 and ImageNet-R: n ∈ {10, 100}.
  54 cells.
- Every method keeps R3/R5's code paths and fixed settings: 10 tasks, β = 0.5, 30 local epochs,
  δ = 1e-5, γ = 1, B = 1.

## 2. Grids (identical for every cell; never extended after results are seen)
- **Baselines:**
  - lr ∈ {0.0005, 0.001, 0.002, 0.005, 0.01, 0.05, 0.1, 0.5};
  - clip multiplier ∈ {0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 1, 4};
  - 64 points. Clip = median × multiplier, with R5's public median definitions. DP-SGD uses the
    median per-example gradient norm at W = 0 on ref-train (lr-free). DP-FedAvg uses the median client
    update norm of non-private M0 at that lr on R5's ref-train federation (10 clients, seed 0). The job
    checks that the recomputed medians at lr ∈ {0.1, 0.5} equal R5's stored values.
- **M9:** p ∈ {32, 64, 128, 256, 768} × λ ∈ {0.01, 0.1, 1, 10, 100, 1000, 10000} (35 points).
  - **Selection basis:** PCA fitted on ref-train (as in R3's selection).
  - **Final basis:** PCA fitted on full ref (as in R3's finals).
- Tie-breaking: the first strict maximum in grid order (lr then multiplier ascending; p then λ ascending).
- **Edge flag:** a selected point is on an edge if any of its coordinates is the min or max of its
  grid axis. The axes touched are listed.

## 3. Size-matched selection federation (public data only; identical for all methods and cells of a dataset)
- **Task class sets:** `class_incremental_tasks(labels, 10, seed = 0)`. This depends only on the
  label set, so it is the same for ref-train and train.
- **For each task t:**
  - The target size N_t is the number of training examples in task t. These are counts of the public
    training split's labels, i.e. dataset metadata. No training features are used.
  - The class composition is ref-train's within the task: class c gets
    round(N_t · ref_c / Σ_{c' ∈ t} ref_c') examples.
  - Those examples are drawn with replacement from ref-train examples of class c, with
    rng = `p3fcl.rng.seeded("rb_r8::selection_federation::{dataset}", 0)`.
- **Federation:** `streams.build_stream(y_sel, arange, 10 tasks, n clients, β = 0.5, seed = 0)`, the
  paper's Dirichlet partition, with n = the cell's n.
- **Selection runs:** same unit and ε0 as the cell, noise seed 0, `sim.run` seed 0. The score is
  final average accuracy on ref-val (`task_eval_sets(y_ref_val, 10, seed = 0)`).
- **C(n, p) for M9 U2:** read from R3 (`results/rb_client_scaling_selection.csv`). It exists for
  every n and p used here.
- **ŵ(n) for DP-FedAvg:** R5's values (`rebuttal/r5_wcap.csv`, `r5_wcap_nscale.csv`). The missing
  values (CIFAR n = 500 and 2000; CUB and ImageNet-R n = 100) are computed by R5's amendment-1
  procedure in a first job (`rb_r8_reselect.py cap` → `rebuttal/r8_wcap.csv`). They are recorded
  below before the main array is submitted.
- M9 U1 does not use C. It is passed but has no effect (U1 sensitivity is sqrt(B⁴ + B²)).

## 4. Finals
The selected point is run on train/test with seeds {0, 1, 2} (stream seed = noise seed = run seed),
through the same code paths as R3/R5.

## 5. Output
- `results/rb_r8.csv`: one row per cell × seed, with the selected point, its ref-val score,
  final_avg_acc, bwt, newest_task_acc, `on_edge` and `edge_axes`.
- `results/rb_r8_summary.csv`: one row per cell with seed means and 95% t-intervals, plus comparison
  columns:
  - the R3/R5 (or paper, for M9 U1) selected mean, which is unchanged;
  - the R7/R7b test-oracle best, labelled "test-oracle upper bound, not used for selection", or n/a
    where R7/R7b did not cover the cell.
- `results/rb_r8_selection.csv`: every selection grid point's ref-val score.
- Intermediates go to `rebuttal/work/r8/`.

## 6. Cost and limits
Estimated about 40 core-hours, under 100 with a 2× margin. That is about 3,560 selection runs at
full training size plus 216 finals, scaled from R7's measured per-run times. This is below the
1,000 core-hour threshold. The array is 72 tasks × 8 cores, throttled to 55 concurrent tasks
(440 cores). Target finish: Nov 12.

## Record (2026-11-09, before the main array was submitted; values only, no protocol change)
- Missing ŵ(n) from `rebuttal/r8_wcap.csv` (job 269140, R5 amendment-1 procedure): CIFAR-100
  n = 500: 15.0; n = 2000: 4.0; CUB-200 n = 100: 9.0; ImageNet-R n = 100: 40.0. Reused from R5:
  CIFAR-100 n = 10 / 100 / 1000: 679.0 / 72.0 / 8.0; CUB-200 n = 10: 76.0; ImageNet-R n = 10: 377.05.
- Selection-federation sizes (label-only check on login): CIFAR-100 40,000 (train 40,000); CUB-200
  5,000 (train 4,994); ImageNet-R 23,920 (train 23,918). The small differences come from per-class
  rounding.
