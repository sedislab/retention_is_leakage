# R3 protocol: client-level privacy as n grows (fixed 2026-11-06, before any R3 job)

This adds detail to `REBUTTAL_EXPERIMENTS.md` §7. It is not changed after any result is seen.

## Size-matched clip norm C(n, p)
- m(n) = round(N_train / (10 · n)), the mean real client size (N_train / 10 examples per task),
  with a floor of 1.
- Pseudo-clients use ref-train only (the paper's `_ref_train_val_split(seed=0, frac=0.5)`) and the
  paper's pseudo-client task class sets (`class_incremental_tasks(y_ref_train, 10, seed=0)`, as inside
  `_pseudo_client_norms`). For each task: 50 pseudo-clients. Each draws class proportions
  ~ Dirichlet(0.5 · 1) over the task's classes and per-class counts ~ Multinomial(m(n), proportions),
  then samples those counts with replacement from ref-train examples of each class. This gives 500
  pseudo-clients per (dataset, n). Their ids come from `rng.seeded("rb_r3::pseudo::{dataset}::{n}", 0)`
  and are shared by every p.
- Pair norm = Frobenius norm of `(G_c, H_c)` from M9's `project_and_cap` (B = 1) and `client_stats`.
- C(n, p) = 95th percentile of the 500 norms (the paper's percentile).

## Selection (public data only)
For each (dataset, n, ε0): the paper's grid (p ∈ {32, 64, 128, 256, 768}, λ ∈ {0.01, …, 100}) and
the paper's ref-val scoring `_run_and_eval` (ref-train federation, 10 clients, seed 0), with
`clip_C = C(n, p)`. The winner is the first strict maximum (the paper's rule). Only C changes relative
to the paper's selection, so the comparison isolates the calibration.

## Runs (10 tasks, β = 0.5, γ = 1, seeds {0, 1, 2})
n ∈ {10, 50, 100, 500, 1000, 2000} for CIFAR-100, and n ∈ {10, 50, 100} for CUB-200 and ImageNet-R.
- `size_matched`: U2, ε0 ∈ {1, 4, 8}, the selected (p, λ, C(n, p)).
- `old_fixed_C`: U2, ε0 ∈ {1, 4, 8}, the paper's selected U2 (p, λ, C) from
  `results/m9_hparam_selection.csv`, which is independent of n. This is the paper's setting (its arm
  was CIFAR at n ∈ {50, 100}). It is also run at every n above, to show the two side by side.
  Check: CIFAR rows at n ∈ {10, 50, 100} must equal the paper's `m9_sweep_cifar100.csv` rows bit for
  bit (n = 10 against the main sweep's U2, γ = 1 rows).
- `u1_reference`: U1, ε0 ∈ {1, 4, 8}, the paper's selected U1 (p, λ). U1 does not depend on C.
- `m8_reference`: non-private M8 (ridge 1.0). M8 adds ridge λ·I per client, so its accuracy can
  depend on n. It runs through a wrapper whose `fit_task` returns no ledger records: at large n the
  per-client 768 × 768 payloads would need tens of GB, and accuracy does not read the ledger.
  Check: at n = 10 it must equal the paper's M8 rows bit for bit.
All M9 runs go through `sim.run` with the same construction as `run_m9_sweep._run_m9`. They are
checked against `_run_m9` at n = 10.

## Recorded per run
`frac_clipped` (U2: fraction of all (task, client) pairs with pair norm > C), `noise_std` (the
ledger's `meta["sigma"]`, task 0), and `agg0_frob` (Frobenius norm of task 0's noiseless aggregate
(G_sum, H_sum) after clipping). These are computed with M9's own `project_and_cap` / `client_stats` /
`clip_pair`. They are n/a for M8.

## Output
`results/rb_client_scaling.csv`: `dataset, method, unit, calibration, n_clients, m_n, eps, seed, p,
lambda, clip_C, frac_clipped, noise_std, agg0_frob, final_avg_acc, bwt, avg_incremental_acc`.
`results/rb_client_scaling_selection.csv`: C(n, p) and the ref-val selection grid.
No prediction is stated in §7. The expectation that larger n helps under the size-matched C is
reported as observed.
