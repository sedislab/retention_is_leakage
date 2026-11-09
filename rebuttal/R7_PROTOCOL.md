# R7 protocol: test-split sensitivity of the selection grids (fixed 2026-11-09, before any R7 job)

Requested by the user on 2026-11-09. This is a **sensitivity analysis, not selection**. Nothing chosen
here changes any reported result. The reported (selected) configurations stay exactly as in R3/R5.

## Cells (44)
- **DP-SGD U1** (`m0_dpsgd_u1`): cifar100, cub200, imagenet_r at n = 10; ε0 ∈ {1, 4, 8, ∞}. 12 cells.
- **DP-FedAvg U2** (`m0_dpfedavg_u2`, McMahan fixed-denominator mean, R5 amendment 1): cifar100 at
  n ∈ {10, 100, 1000}; cub200 and imagenet_r at n = 10; ε0 ∈ {1, 4, 8, ∞}. 20 cells.
- **M9 U2, size-matched C** (R3 `size_matched`): cifar100 at n ∈ {100, 500, 1000, 2000}; ε0 ∈ {1, 4, 8}.
  12 cells.

## Grid points: exactly the existing selection grids, with no new values
- DP-SGD and DP-FedAvg: lr ∈ {0.1, 0.5, 1.0} × clip multiplier ∈ {0.25, 0.5, 1, 2, 4} (15 points).
  The clip value (Cg or S) is copied from the `clip` column of `results/rb_dp_fedavg_selection.csv`
  (n = 10) or `results/rb_dp_fedavg_nscale_selection.csv` (n ∈ {100, 1000}). ŵ comes from
  `rebuttal/r5_wcap.csv` and `rebuttal/r5_wcap_nscale.csv`.
- M9: p ∈ {32, 64, 128, 256, 768} × λ ∈ {0.01, 0.1, 1, 10, 100} (25 points). C(n, p) is copied from
  `results/rb_client_scaling_selection.csv`.

## Runs
Every grid point is run on train/test with seeds {0, 1, 2}, with the same code paths that produced the
reported results: `rb_r5_dp_fedavg._make` / `_run` for the baselines, and `rb_r3_client_scaling._run`
with the `ContractiveDPAnalytic` config of R3 for M9 (PCA basis from the full ref). Noise seed = run
seed. That gives 2,340 runs in total.

## Per cell (metric: final average accuracy, mean over the 3 seeds; 95% Student-t interval over seeds)
- **selected**: the reported configuration and its reported per-seed values, read unchanged from
  `results/rb_dp_fedavg.csv`, `results/rb_dp_fedavg_nscale.csv` or `results/rb_client_scaling.csv`.
  **Check:** the R7 rerun of that grid point must equal the reported value bit for bit for every seed.
  Otherwise the merge fails and writes nothing to `results/`.
- **best**: the grid point with the highest seed-mean test accuracy. Labelled **"test-oracle upper
  bound, not used for selection"**.
- **worst**: the grid point with the lowest seed-mean test accuracy.
- Ties are broken by the first point in grid order (lr then multiplier ascending; p then λ ascending).
  The number of tied points is recorded.
- Also recorded: the selected point's rank among the grid points (1 = best), and the gap
  best − selected. Both are oracle quantities, labelled as such.

## Output
- `results/rb_sensitivity.csv`: one row per (cell, role ∈ {selected, best, worst}) with the grid point,
  mean, interval, per-seed values, label, rank and tie count.
- `results/rb_sensitivity_grid.csv`: every grid point × seed (final_avg_acc, bwt, newest_task_acc
  where available).
Intermediate files go to `rebuttal/work/r7/`. Estimated compute is well under 2,000 core-hours
(44 array tasks × 8 cores, each expected < 1 h).
