# R2 protocol: projection ablation and out-of-distribution public data (fixed 2026-11-06, before any R2 job)

This adds detail to `REBUTTAL_EXPERIMENTS.md` §6. It is not changed after any result is seen.

## Common (all as in the paper unless stated)
- M9 (`ContractiveDPAnalytic`), γ = 1, δ = 1e-5, B = 1, 10 tasks × 10 clients, β = 0.5, seeds {0,1,2}
  on train/test via `run_m9_sweep._run_m9` (the code R0 verified bit for bit).
- Cells: U1 with ε0 ∈ {1, 2, 4, 8, ∞}; U2 with ε0 = 8.
- Selection uses public data only, through the paper's own procedure (`run_m9_hparam_selection`):
  - ref split `_ref_train_val_split(seed=0, frac=0.5)`;
  - for each p: C = 95th percentile of `_pseudo_client_norms` on ref-train under that projection;
  - for each λ ∈ {0.01, 0.1, 1, 10, 100}: ref-val final average accuracy from `_run_and_eval`
    (seed 0, the selection dataset's own labels and stream);
  - the winner is the first strict maximum over p ascending, then λ ascending (the paper's tie rule).
- Final runs use the selected (P, λ, C). For PCA arms the final basis is refit on the full `ref` of
  the source dataset, as in `run_m9_sweep`.

## Arms
- **A `pca_id`**: the paper's selection, read from `results/m9_hparam_selection.csv` (`selected = 1`).
  The basis is the PCA of the dataset's own full ref.
  Check: the job reruns the selection with this script's code for the A cells and asserts that it
  reproduces the CSV's (p, λ, C). This validates the selection code that arms B–D use.
- **B `no_proj`**: P = I (p = 768). λ is selected on the dataset's own ref-val. C (U2) comes from its
  own ref-train pseudo-clients with P = I.
- **C `pca_ood`**: PCA fitted on the source dataset's ref (ref-train for selection, full ref for
  the final runs). p ∈ {32, 64, 128, 256, 768}, λ and C are selected on the source dataset's ref
  split. Pairs (target ← source): cifar100 ← imagenet_r, cub200 ← cifar100, imagenet_r ← cifar100.
- **D `random_proj`**: Q from the QR of a d × p standard Gaussian matrix, with
  rng = `p3fcl.rng.seeded("rb_r2::random_proj::{p}", 0)` and p ∈ {32, 64, 128, 256}. This basis is
  data-independent and identical for selection and final runs. p, λ and C are selected on the same
  source dataset as C.

## Output
`results/rb_projection_ablation.csv`: `dataset, arm, unit, eps, selection_dataset, selected_p,
lambda, clip_C, seed, final_avg_acc, bwt, avg_incremental_acc`.
`results/rb_projection_ablation_selection.csv`: the full selection grid (ref-val accuracies).
No prediction is stated in §6, so none is evaluated.

## Amendment 1 (2026-11-06, after job 263206 failed in the arm-A reselection check; nothing else changed)
The reselection reproduced the paper's selected p and λ exactly. Its C for cub200 U1 ε0 = 1 was
12.690162942130655 against the paper's 12.690162952890445, a relative difference of 8.5e-10. This
is consistent with floating-point rounding in the SVD/percentile pipeline (the paper's selection ran
under an earlier job and environment). The check now requires p and λ equal and C within relative
1e-6. Arm A still uses the paper's own (p, λ, C) values. Arms B–D are unaffected, because they use
their own computed C. No R2 output was written by the failed job.
