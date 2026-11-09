# R4 protocol: accuracy vs. examples per class (fixed 2026-11-06, before any R4 job)

This adds detail to `REBUTTAL_EXPERIMENTS.md` §8. It is not changed after any result is seen.

## Hypothesis (from §8)
Under U1 the noise has a fixed size, while the statistics grow with the number of examples per class.
So accuracy depends on examples per class (m), not on the dataset.

## Design
- CIFAR-100 train, m ∈ {25, 50, 120, 250, all}. "all" is the full train set (400 per class,
  `np.arange`), the paper's exact input. For m < 400 and seed s: per class, the first m ids of
  `rng.seeded("rb_r4::subsample::{class}", s).permutation(class ids)`, then sorted. The subsets are
  nested in m for a fixed seed.
- The stream is `build_stream(y, subset_ids, 10 tasks, 10 clients, β = 0.5, seed = s)`. The task class
  sets are the same as the paper's, because they depend only on the labels. The test sets are the
  paper's (full test, `task_eval_sets`).
- M9 U1, ε0 ∈ {1, 4, 8, ∞}, γ = 1, with the paper's selected CIFAR-100 U1 (p, λ) per ε0 from
  `results/m9_hparam_selection.csv` and the PCA basis from the full ref (as in `run_m9_sweep`).
  Seeds {0, 1, 2}.
- Non-private M8 (ridge 1.0) at each m.
- Check: the m = all rows must equal the paper's `m9_sweep_cifar100.csv` rows bit for bit.

## Prediction test (fixed now)
Reference: CUB-200's paper M9 U1, γ = 1, n = 10 means (CUB has about 25 train examples per class):
ε0 = 1 → 0.0175, 4 → 0.0883, 8 → 0.1612, read from `results/m9_sweep_cub200.csv` at merge time.
For each finite ε0, `holds_eps` = 1 if the CIFAR m = 25 mean is closer to the CUB mean than to the
CIFAR m = all mean. `m8_stays_high` = 1 if the M8 m = 25 mean ≥ M8 m = all mean − 0.10. The
prediction holds if every `holds_eps` = 1 and `m8_stays_high` = 1. CUB has 200 classes and CIFAR has
100, so "CUB-like" is approximate. The raw means are reported alongside.

## Output
`results/rb_examples_per_class.csv`: `method, unit, m_per_class, eps, seed, p, lambda,
final_avg_acc, bwt, avg_incremental_acc, paper_check`.
`results/rb_examples_per_class_prediction.csv`: one row per finite ε0 with the means, `holds_eps`,
`m8_stays_high` and `prediction_holds`.
