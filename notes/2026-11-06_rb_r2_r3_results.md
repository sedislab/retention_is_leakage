# 2026-11-06: Rebuttal R2 (projection ablation) and R3 (client scaling) (EVIDENCE)

## R2: `results/rb_projection_ablation.csv` (protocol `rebuttal/R2_PROTOCOL.md` + amendment 1)
M9, γ = 1, final average accuracy, mean over 3 seeds (selected p in parentheses). Arm A's reselection
reproduces the paper's p and λ exactly.

| dataset | arm | U1 ε1 | U1 ε2 | U1 ε4 | U1 ε8 | U1 ∞ | U2 ε8 |
|---|---|---|---|---|---|---|---|
| cifar100 | pca_id | 0.596 | 0.690 | 0.810 | 0.818 | 0.884 | 0.017 |
| cifar100 | no_proj | 0.564 | 0.751 | 0.818 | 0.843 | 0.884 | 0.015 |
| cifar100 | pca_ood (IN-R ref) | 0.482 | 0.668 | 0.738 | 0.760 | 0.883 | 0.014 |
| cifar100 | random_proj | 0.462 | 0.680 | 0.760 | 0.530 (p32) | 0.845 | 0.019 |
| cub200 | pca_id | 0.017 | 0.050 | 0.088 | 0.161 | 0.868 | 0.007 |
| cub200 | no_proj | 0.014 | 0.032 | 0.053 | 0.154 | 0.868 | 0.008 |
| cub200 | pca_ood (CIFAR ref) | 0.010 | 0.015 | 0.047 | 0.150 | 0.872 | 0.007 |
| cub200 | random_proj | 0.015 | 0.035 | 0.116 | 0.339 | 0.837 | 0.005 |
| imagenet_r | pca_id | 0.082 | 0.203 | 0.324 | 0.385 | 0.634 | 0.006 |
| imagenet_r | no_proj | 0.053 | 0.146 | 0.309 | 0.434 | 0.634 | 0.008 |
| imagenet_r | pca_ood (CIFAR ref) | 0.068 | 0.159 | 0.279 | 0.341 | 0.633 | 0.006 |
| imagenet_r | random_proj | 0.040 | 0.128 | 0.281 | 0.392 | 0.519 | 0.006 |

Observed: in-distribution PCA is not required for M9's accuracy. No projection is within a few
points (better at ε ≥ 2 on CIFAR, worse at ε = 1), and OOD PCA loses about 2–8 points. The
data-independent random projection is competitive. On CUB it is better at ε0 = 4 and 8
(0.116 / 0.339 vs 0.088 / 0.161). On CIFAR at ε0 = 8 it drops to 0.530 because OOD selection picked
p = 32. All of this is reported as observed, without retuning.

## R3: `results/rb_client_scaling.csv` (protocol `rebuttal/R3_PROTOCOL.md`); 81/81 paper rows reproduce bit for bit
U2, γ = 1, mean over 3 seeds. CIFAR-100, n = 10 / 50 / 100 / 500 / 1000 / 2000:
- old fixed C, ε0 = 4: 0.016 / 0.019 / 0.018 / 0.018 / 0.018 / 0.018 (clipped fraction 1.00 / 0.46 /
  0.03 / 0 / 0 / 0)
- size-matched C, ε0 = 4: 0.015 / 0.020 / 0.039 / 0.284 / 0.460 / 0.511 (clipped fraction
  0.18–0.28)
- size-matched C, ε0 = 8: 0.016 / 0.035 / 0.089 / 0.620 / 0.519 / 0.537
- U1 references are constant in n (0.596 / 0.810 / 0.818 at ε0 = 1 / 4 / 8): U1 sums are
  partition-invariant. M8 is 0.883 → 0.882.

CUB / ImageNet-R at n ≤ 100: size-matched C reaches only 0.035 / 0.031 at ε0 = 8, n = 100.

Reading:
1. At n = 10 the size-matched C does not help (0.015 vs 0.016). The paper's poor U2 numbers at
   10 clients are not an artifact of the pseudo-client size mismatch noted in `rebuttal/answers.md`,
   even though that calibration clipped 100% of clients.
2. The paper's n_clients arm (fixed C) could not show gains: it plateaus at about 0.048. With a
   size-matched C, client-level DP becomes usable on CIFAR-100 at n ≥ 500. No paper number changes,
   but the paper's statement about the n_clients arm should be revised by the authors.
3. ε0 = 8 is not monotone in n (0.620 at n = 500 > 0.519 at n = 1000). The selection picked
   different (p, λ) per n; see `results/rb_client_scaling_selection.csv`. Reported as observed.
