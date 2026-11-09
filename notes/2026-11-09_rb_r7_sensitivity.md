# 2026-11-09: Rebuttal R7, test-split sensitivity of the selection grids (EVIDENCE: results/rb_sensitivity.csv)

This is a sensitivity analysis, NOT selection. "Best" = test-oracle upper bound, not used for
selection. All 44 selected reruns equal the reported values bit for bit, so no reported result changed.
Protocol: `rebuttal/R7_PROTOCOL.md`.

Reading (seed-mean final average accuracy):
1. **The DP baselines' weakness is not a selection artifact.** At finite ε the selected point is the
   oracle best in 19 of 27 cells. The largest best − selected gap is CIFAR DP-FedAvg n = 1000,
   ε0 = 8: 0.193 vs 0.230. Even the oracle best stays low: DP-SGD CIFAR ε0 = 4 is 0.145; DP-FedAvg
   n = 10 is 0.016 (CIFAR), 0.007 (CUB), 0.006 (IN-R).
2. **M9 U2 with size-matched C is strongly under-selected at large n.** The selected point ranks 4–24
   of 25. The oracle best is much higher: n = 1000 ε0 = 4 0.460 vs 0.700; n = 2000 ε0 = 8 0.537 vs
   0.826; n = 2000 ε0 = 1 0.057 vs 0.393. The oracle-best points are always p ∈ {128, 256}, λ = 100.
   The selection federation (ref-train, 10 clients) does not resemble the large-n federation. The
   reported R3 numbers are therefore conservative for M9. This must not be read as M9's accuracy,
   because the oracle number is an upper bound.
3. The non-monotonicity in R3 at ε0 = 8 (0.620 at n = 500 > 0.519 at n = 1000) is a selection effect.
   The oracle bests are monotone in n: 0.709 / 0.799 / 0.826.
4. Fairness of the R5 vs R3 comparison at large n: M9 loses more to selection than DP-FedAvg does.
   At n = 1000, ε0 = 4, the oracle bests are M9 0.700 vs DP-FedAvg 0.210; the selected values are
   0.460 vs 0.194.
