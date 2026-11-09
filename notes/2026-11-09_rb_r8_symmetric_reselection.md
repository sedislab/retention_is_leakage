# 2026-11-09: Rebuttal R8, symmetric re-selection (EVIDENCE, TEST-INFORMED DESIGN: results/rb_r8_summary.csv)

The grids and the size-matched public selection federation were motivated by R7/R7b's test-split
results (stated in `rebuttal/R8_PROTOCOL.md`). Selection used ref-train / ref-val only. 72 cells,
3 seeds, about 36 core-hours. The first merge crashed writing the selection CSV (mixed columns). It
was fixed and rerun (job 269604); no cell was rerun.

Findings (final average accuracy, seed mean; M9 − baseline under the same unit):
1. **U1 (n = 10): M9 leads in every cell.** CIFAR +0.47 / +0.22 / +0.16 (ε0 = 1 / 4 / 8),
   CUB +0.02 / +0.15 / +0.36, IN-R +0.11 / +0.24 / +0.32. DP-SGD is now much stronger than in R5
   (CIFAR ε0 = 4: 0.611 vs 0.145 reported). It matches its R7b oracle, so the size-matched public
   federation selects about as well as the oracle did.
2. **U2, CIFAR-100: M9 leads for n ≥ 500** (+0.07 to +0.14, every interval separated except
   ε0 = 1 at n = 500, which is a tie). **DP-FedAvg leads at n = 100** (ε0 = 4: 0.132 vs 0.095;
   ε0 = 8: 0.309 vs 0.262, intervals separated). At n = 10 both are at chance.
3. **U2, CUB-200 and IN-R (n ≤ 100): both methods are at or near chance** (≤ 0.054). There is no
   separable difference.
4. M9 also gains over its own reported numbers, from λ up to 1,000–10,000 and the size-matched
   selection. U1 CIFAR ε0 = 1: 0.694 vs paper 0.596; CUB ε0 = 8: 0.405 vs 0.161. U2 n = 2000
   ε0 = 1: 0.528 vs R3 0.057. Several R8 M9 U2 values exceed R7's oracle, because R7's grid stopped
   at λ = 100. These are test-informed numbers. The paper's numbers are unchanged and remain what
   the paper reports. Whether to add R8 to the revision is the authors' decision.
5. Edges: 26/72 selected points are on a grid edge. In informative cells these are mostly baseline
   lr = 0.0005 (CIFAR DP-FedAvg n = 1000 ε0 = 1/4, n = 2000 ε0 = 8; IN-R DP-SGD), M9 p = 768 (U1
   CIFAR ε0 = 8) and λ = 10,000 (U2 n = 100 ε0 = 4; IN-R n = 100 ε0 = 8). In those cells the grid may
   not contain the optimum. Edge selections in chance-level cells are uninformative (ties near chance).
