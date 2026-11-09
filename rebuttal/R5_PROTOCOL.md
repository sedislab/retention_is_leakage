# R5 protocol: DP-FedAvg baselines at matched ε (fixed 2026-11-06, before any R5 job)

This adds detail to `REBUTTAL_EXPERIMENTS.md` §9. It is not changed after any result is seen.

## Methods (new files only)
- `code/src/p3fcl/methods/m0_dpsgd_u1.py` `DPSGDFedAvgU1`: M0 with per-example clipping to Cg,
  noise N(0, (z·Cg)² I) on the clipped sum, divided by the client's n, in each of the 30 local
  full-batch steps. z = `noise_multiplier_for(ε0, 30, 1e-5)`: bisection on the repo's
  `eps_gaussian_composed` (RDP, sampling rate 1). Add/remove neighbouring with sensitivity Cg on the
  clipped sum, the same add/remove convention as M9's U1. n is treated as public, as in M0's FedAvg
  weights. Per-client ledger records, the same as M0's.
- `code/src/p3fcl/methods/m0_dpfedavg_u2.py` `DPFedAvgU2`: M0's local training, clip each update to
  Frobenius norm S, sum, add N(0, σ² I) with σ = `analytic_gaussian_sigma(ε0, 1e-5, S)`, then divide by
  the number of participating clients (uniform, as §9 specifies). One `client=-1` record per round
  (the noisy average), as for M9.
- Tests (`tests/test_rb_dp_fedavg.py`): with no clip and no noise, DP-SGD equals M0 exactly. DP-FedAvg
  in its test-only `weighted` mode equals M0 exactly. The ledger count is 1 per U1 example
  (DP-SGD) and per U2 shard (DP-FedAvg).

## Correctness gate (§9), run first as its own job
On train/test, all three datasets, seeds {0, 1, 2}, 10 clients, lr = 0.5, 30 epochs (M0's config):
M0, DP-SGD with Cg = ∞ and z = 0, and DP-FedAvg with S = ∞ and σ = 0 (uniform mean, the method as
specified). Pass if |acc − M0 acc| ≤ 0.01 for every (dataset, seed, method). The M0 rerun must also
equal `results/m9_sweep_<ds>.csv` M0 rows bit for bit.
Expected risk, stated in advance: uniform averaging differs from M0's size-weighted FedAvg under
Dirichlet(0.5) client sizes, so DP-FedAvg may miss 0.01 for an algorithmic reason, not a bug. The
gate also runs DP-FedAvg in `weighted` mode, which must equal M0 exactly. If the uniform gate fails,
the job stops and the user decides.

## Selection (public ref only; §9)
- The paper's ref split (`_ref_train_val_split(seed 0)`). The selection federation is ref-train with
  10 tasks × 10 clients, β = 0.5, seed 0, scored by ref-val final average accuracy (as in
  `_run_and_eval`). Noise seed 0.
- Medians on ref-train:
  - DP-SGD: the median per-example gradient norm of the linear head at W = 0 (M0's initial model):
    ||x_i|| · ||softmax(0) − y_i||.
  - DP-FedAvg: for each lr, the median client update norm ||Δ||_F over all (task, client) pairs of
    non-private M0 on the selection federation.
- Grid: lr ∈ {0.1, 0.5, 1.0} × clip = median × {0.25, 0.5, 1, 2, 4}. Per ε0 the winner is the first
  strict maximum in (lr ascending, multiplier ascending) order. At ε0 = ∞ there is no noise, but
  clipping still applies.

## Runs
ε0 ∈ {1, 4, 8, ∞}, all three datasets, seeds {0, 1, 2}, 10 clients. Noise seed = run seed. Recorded:
`final_avg_acc`, `bwt`, `newest_task_acc` (`acc_matrix[T−1, T−1]`), `avg_incremental_acc`, the clipped
fraction, z or σ, and the ledger's maximum m_T over T under the method's unit. The lifelong ε equals
ε0 when max m_T = 1.
The optional n ∈ {100, 1000} CIFAR-100 DP-FedAvg arm is decided after R3, and only if R3 shows gains
at large n.

## Output
`results/rb_dp_fedavg.csv`, `results/rb_dp_fedavg_selection.csv`; gate: `rebuttal/r5_gate.csv`.

## Amendment 1 (2026-11-06, user decision after the first correctness gate failed; fixed before any R5 run)
The first gate (job 263212) failed in 3/9 cells, for DP-FedAvg with the uniform mean only. DP-SGD
and the weighted test mode matched M0 exactly. On the user's instruction, DP-FedAvg now uses
**McMahan et al. (ICLR 2018)'s fixed-denominator weighted mean**:
- w_c = min(n_c / ŵ, 1), so no client weight exceeds 1.
- Δ = (Σ_c w_c · clip_S(Δ_c) + N(0, σ² I)) / W, where W = Σ_c w_c over the task's base-stream
  clients is treated as a public constant. Each weighted clipped update has norm ≤ S, so the sum has
  add/remove-one-client sensitivity S, and σ = `analytic_gaussian_sigma(ε0, 1e-5, S)`.
- **Cap ŵ, from public data only, fixed before the main run.** It is the 95th percentile of client
  sizes in size-matched pseudo-federations. Each class gets round(N_train · its ref-train
  frequency) pseudo-examples (N_train is the public dataset size). The task classes come from
  `class_incremental_tasks(y_ref_train, 10, s)`. The paper's `dirichlet_partition` splits each task
  across 10 clients with β = 0.5, for seeds s ∈ {1000, …, 1019} (disjoint from the run seeds). This
  differs from R3's pseudo-clients, which all have the same size m(n) by construction, so their 95th
  percentile would just be m(n) and carry no information about size variation. The gate job computes
  ŵ and writes it to `rebuttal/r5_wcap.csv`. The values are copied below before the main run is
  submitted.
- **Gate (rerun).** No noise, no clipping, ŵ = the largest client size of the run's stream (no cap
  binds). Must match M0 within 0.01 in all 9 (dataset, seed) cells. DP-SGD (unchanged) and the exact
  weighted test mode are rerun alongside it.
- Selection, grids, runs and outputs are otherwise as above. The S grid is still the median
  unweighted client update norm × {0.25, …, 4}. Rows also record `w_cap` and `frac_capped`. The
  uniform-mean numbers are kept in `notes/2026-11-06_rb_r5_uniform_mean_gate.md` only, not as a
  reported baseline.
- **Fixed ŵ (copied from `rebuttal/r5_wcap.csv`, job 263741, before the main run was submitted):**
  cifar100 679.0 (median pseudo-client size 383, max 1,344); cub200 76.0 (median 49, max 110);
  imagenet_r 377.05 (median 229, max 565). 2,000 pseudo-clients each.
- **Gate attempt 2 (job 263741): PASSED 36/36.** Fixed-denominator DP-FedAvg (no noise, no clip,
  no binding cap), DP-SGD (no clip, no noise) and the weighted test mode all equal M0's final average
  accuracy exactly (|diff| = 0) in all 9 cells. M0 equals the paper's sweep rows in all 9 cells.

## Amendment 2 (2026-11-06): the conditional n-scaling arm (§9), triggered by R3
R3 (`results/rb_client_scaling.csv`) shows gains at large n under the size-matched C (CIFAR-100 U2
ε0 = 4: 0.015 at n = 10, 0.284 at n = 500, 0.511 at n = 2,000). So §9's arm runs:
`m0_dpfedavg_u2` on CIFAR-100 at n ∈ {100, 1000}, ε0 ∈ {1, 4, 8, ∞}, seeds {0, 1, 2}.
- "Matched selection" follows R3's convention. The selection procedure is unchanged (paper ref-train
  federation, 10 clients, same lr × S grid, S median from that federation). Only the n-dependent
  quantity changes: ŵ(n) is computed with the amendment-1 public procedure at n clients.
- ŵ(n) is computed first by its own job (`nscale_cap` → `rebuttal/r5_wcap_nscale.csv`) and copied
  here before the run job is submitted.
- Output: `results/rb_dp_fedavg_nscale.csv`, `results/rb_dp_fedavg_nscale_selection.csv`.
- **Fixed ŵ(n) (from `rebuttal/r5_wcap_nscale.csv`, job 263786, before the run job was submitted):**
  CIFAR-100 n = 100: 72.0 (median pseudo-client size 37, max 147); n = 1000: 8.0 (median 4, max 18).
