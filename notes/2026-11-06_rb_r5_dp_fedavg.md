# 2026-11-06: Rebuttal R5, DP-FedAvg baselines at matched ε (EVIDENCE: results/rb_dp_fedavg*.csv)

Protocol: `rebuttal/R5_PROTOCOL.md` + amendments 1–2. The gate passed 36/36 with all diffs 0. Ledger
max m_T = 1 in every checked run, so the lifelong ε equals ε0. Final average accuracy, mean over 3 seeds:

| | ε0=1 | 4 | 8 | ∞ | M9 same unit, ε0=4 (paper / R3) |
|---|---|---|---|---|---|
| CIFAR DP-SGD (U1) | 0.068 | 0.145 | 0.167 | 0.333 | 0.810 (U1) |
| CIFAR DP-FedAvg (U2) n=10 | 0.012 | 0.015 | 0.020 | 0.331 | ≈0.016–0.018 (U2) |
| CIFAR DP-FedAvg (U2) n=100 | 0.025 | 0.100 | 0.172 | 0.283 | 0.039 (U2, size-matched C) |
| CIFAR DP-FedAvg (U2) n=1000 | 0.139 | 0.194 | 0.193 | 0.240 | 0.460 (U2, size-matched C) |
| CUB DP-SGD | 0.004 | 0.011 | 0.019 | 0.362 | 0.088 |
| CUB DP-FedAvg n=10 | 0.007 | 0.007 | 0.009 | 0.370 | |
| IN-R DP-SGD | 0.011 | 0.026 | 0.040 | 0.196 | 0.324 |
| IN-R DP-FedAvg n=10 | 0.005 | 0.006 | 0.008 | 0.228 | |

Caveats:
- Selection: in 24/24 finite-ε cells the best ref-val accuracy is < 0.06. The selection federation
  (ref-train, 10 × 10) is about 8× smaller than the real one, so selection is close to arbitrary,
  and ties resolve to the first grid point (lr 0.1, multiplier 0.25).
- DP-FedAvg at n = 100 beats size-matched M9 U2 (0.100 vs 0.039 at ε0 = 4). At n = 1000, M9 is
  better (0.460 vs 0.194). The two were selected under different procedures, so this comparison is
  not tuned to be fair at every n.
- Like M0, the baselines forget heavily (BWT −0.35 to −0.79 on CIFAR). Their newest-task accuracy
  at ε0 = 8 is 0.75 (DP-SGD) and 0.91 (DP-FedAvg n = 1000), so the DP models do learn each task
  and the low final average reflects forgetting.
