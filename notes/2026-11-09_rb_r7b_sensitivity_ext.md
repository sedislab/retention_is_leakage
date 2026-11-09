# 2026-11-09: Rebuttal R7b, extended DP-baseline sensitivity grid (EVIDENCE: results/rb_sensitivity_ext.csv)

This is a sensitivity analysis, NOT selection. "Best" = test-oracle upper bound, not used for
selection. No reported result changed. Protocol: `rebuttal/R7b_PROTOCOL.md`. In all 18 cells the
recomputed medians equal the stored ones and R7's best point reruns bit-identically.

**Main finding: the R5 baselines are badly under-tuned by the plan's grid (lr ∈ {0.1, 0.5, 1.0}).**
Much smaller learning rates (0.002–0.01), with small clip multipliers, give far higher test accuracy:

| cell | selected (reported) | R7 15-pt best | R7b best (oracle) | interior |
|---|---|---|---|---|
| DP-SGD CIFAR ε0=4 | 0.145 | 0.145 | 0.611 | no (lr_low, mult_low) |
| DP-SGD CIFAR ε0=8 | 0.167 | 0.167 | 0.683 | no (lr_low, mult_low) |
| DP-FedAvg CIFAR n=1000 ε0=4 | 0.194 | 0.210 | 0.622 | no (lr_low) |
| DP-FedAvg CIFAR n=1000 ε0=8 | 0.193 | 0.230 | 0.696 | no (lr_low) |
| DP-FedAvg CIFAR n=100 ε0=8 | 0.172 | 0.172 | 0.315 | yes |
| DP-FedAvg CIFAR n=10, all ε0 | 0.012–0.020 | | 0.012–0.023 | mixed |
| DP-SGD CUB / IN-R ε0=8 | 0.019 / 0.040 | | 0.046 / 0.127 | yes / no |

- 9/18 best points are still on an edge after 2 extensions (mostly lr_low). Those oracle values are
  therefore not the grid optimum, which may be higher.
- Forgetting drops sharply at small lr: DP-SGD CIFAR ε0 = 4 BWT is −0.18 (best) vs −0.63
  (selected); DP-FedAvg n = 1000 ε0 = 4 is −0.15 vs −0.77.
- NOT EVIDENCE (hypothesis): with tiny lr and 30 full-batch steps from W ≈ 0, each client update is
  roughly lr·30·X^T(Y − P0), a class-sum statistic. FedAvg then accumulates class sums across tasks
  and is not overwritten, so tiny-lr DP-FedAvg/DP-SGD behaves like a noisy sufficient-statistic
  accumulator, close to M8/M9's mechanism. That would explain both the accuracy and the BWT.

Consequences for the rebuttal (the authors decide):
1. "DP-FedAvg baselines collapse at matched ε" (R5 as reported) is a statement about the plan's
   grid, not about the methods. Against oracle bests, M9 still leads on CIFAR U1 (0.810 selected vs
   DP-SGD 0.611 oracle at ε0 = 4), but at large n (U2) the picture is close: n = 1000, ε0 = 4 gives
   M9 0.460 selected / 0.700 oracle vs DP-FedAvg 0.194 selected / 0.622 oracle.
2. Any re-selection over a wider grid would now be informed by test results. A new grid chosen after
   seeing R7b is test-influenced, and must be labelled as such if it is run at all.
