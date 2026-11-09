# 2026-11-06: R5 first correctness gate, uniform-mean DP-FedAvg (NOT A REPORTED BASELINE)

Job 263212 (`rebuttal/r5_gate_attempt1_uniform_263212.csv`). No clip, no noise, lr 0.5, 30 epochs,
10 clients, vs. M0. Tolerance 0.01. DP-FedAvg divided by the number of clients, as the plan
first specified.

| dataset | seed | uniform DP-FedAvg | M0 | |diff| | gate |
|---|---|---|---|---|---|
| cifar100 | 0 | 0.311000 | 0.309700 | 0.0013 | pass |
| cifar100 | 1 | 0.298400 | 0.324900 | **0.0265** | FAIL |
| cifar100 | 2 | 0.353700 | 0.363000 | 0.0093 | pass |
| cub200 | 0 | 0.418611 | 0.410492 | 0.0081 | pass |
| cub200 | 1 | 0.380943 | 0.431468 | **0.0505** | FAIL |
| cub200 | 2 | 0.305817 | 0.316731 | **0.0109** | FAIL |
| imagenet_r | 0 | 0.258795 | 0.253186 | 0.0056 | pass |
| imagenet_r | 1 | 0.226010 | 0.223825 | 0.0022 | pass |
| imagenet_r | 2 | 0.270792 | 0.266993 | 0.0038 | pass |

In the same job, DP-SGD (no clip, no noise) and the size-weighted test mode equalled M0 exactly in
all 9 cells (diff 0). So the gap is an aggregation-weight effect under Dirichlet(0.5) client sizes,
not an implementation error. The user chose McMahan et al.'s fixed-denominator weighted mean instead
(R5_PROTOCOL.md amendment 1).
