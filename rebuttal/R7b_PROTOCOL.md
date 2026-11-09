# R7b protocol: extended test-split sensitivity grid for the DP baselines (fixed 2026-11-09, before any R7b job)

Requested by the user on 2026-11-09. This is a **sensitivity analysis, not selection**. No reported result
or selection changes. The best point is labelled **"test-oracle upper bound, not used for selection"**.

## Cells (18)
- DP-SGD U1 (`m0_dpsgd_u1`): cifar100, cub200, imagenet_r at n = 10; ε0 ∈ {1, 4, 8}. 9 cells.
- DP-FedAvg U2 (`m0_dpfedavg_u2`, McMahan fixed-denominator mean): cifar100 at n ∈ {10, 100, 1000};
  ε0 ∈ {1, 4, 8}. 9 cells.
- Seeds {0, 1, 2}. Code paths, ŵ and noise calibration are as in R5/R7 (`rb_r5_dp_fedavg._make` / `_run`).

## Base extended grid (round 0)
- lr ∈ {0.01, 0.02, 0.05} ∪ {0.1, 0.5, 1.0}.
- Clip multiplier ∈ {0.05, 0.1} ∪ {0.25, 0.5, 1, 2, 4}.
- Fully crossed: 6 × 7 = 42 points per cell.
- Clip = median × multiplier, with R5's public medians (ref-train federation, 10 clients, seed 0):
  - DP-SGD: the median per-example gradient norm at W = 0. It is independent of lr.
  - DP-FedAvg: the median client update norm of non-private M0 at that lr. For the new lrs this is
    computed by the same code. **Check:** the job recomputes the medians for the existing lrs and
    asserts they equal the stored `median_norm` values exactly.
- The 15 existing points are reused from `results/rb_sensitivity_grid.csv` (R7, same code and seeds,
  deterministic). **Check:** each cell reruns the R7 best point and asserts the result is bit-identical
  to the stored values before reusing them.

## Edge extension (fixed now)
- The best point is the highest seed-mean test final average accuracy. Ties go to the first point in
  grid order (lr then multiplier ascending). A NaN accuracy ranks last.
- If the best point lies on an edge of the current grid (its lr is the grid's min or max, or its
  multiplier is the min or max), the grid is extended by one step in each such direction. The new
  value is crossed with all current values of the other axis. This is repeated at most twice
  (rounds 1 and 2).
- Steps:
  - lr below: 0.005, then 0.002;
  - lr above: 2.0, then 5.0;
  - multiplier below: 0.02, then 0.01;
  - multiplier above: 8, then 16.
- After the last round: `interior` = 1 if the best point is on no edge of the final grid. Otherwise
  0, and the edges hit are listed.

## Output
- `results/rb_sensitivity_ext.csv`: one row per cell with the best point (lr, multiplier, clip),
  seed-mean and 95% Student-t interval, per-seed values, label, `interior`, `edges_hit`, number of
  extension rounds, final lr / multiplier grids, number of points, and for reference (unchanged) the
  reported selected value and the R7 15-point best.
- `results/rb_sensitivity_ext_grid.csv`: every point × seed, with its source (R7 reuse or R7b run).
Intermediate files go to `rebuttal/work/r7b/`. About 1,500–2,500 runs, well under 2,000 core-hours.
