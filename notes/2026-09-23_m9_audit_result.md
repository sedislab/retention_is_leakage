# 2026-09-23 — M9 LiRA audit (FIG10): Theorem 11 passes, with a working non-private control

## What this checks

`08_FIX_PLAN.md` §8's empirical check of Theorem 11 (R12: "there is no M9... needs an empirical
privacy-utility result"). Online LiRA against M9 (`ContractiveDPAnalytic`) on CIFAR-100, unit U1
(per-example), eps in {1, 4, inf}, gamma=1.0, scored by leverage on the `global` running state
(the actual matrix the server inverts to predict) — 1024 shadows, 1 seed, elapsed=0, trajectory
ablation. Compares the empirical (FPR, TPR) curve against the analytic Gaussian mechanism's DP bound
`TPR <= e^eps * FPR + delta`.

## Result

**The audit passes at both finite eps values, with a real, functioning non-private control confirming
the attack has power:**

| eps | TPR@1%FPR (95% CI hi) | DP bound @1%FPR | TPR@0.1%FPR (95% CI hi) | DP bound @0.1%FPR | AUC |
|---|---|---|---|---|---|
| 1 | 0.0095 (0.0104) | 0.0272 | 0.00084 (0.00113) | 0.00273 | 0.4997 |
| 4 | 0.0087 (0.0095) | 0.5460 | 0.00080 (0.00108) | 0.0546 | 0.4986 |
| inf (non-private control) | 0.8711 (0.8740) | 1.0 (no bound) | 0.7556 (0.7593) | 1.0 | 0.9926 |

At eps=1 and eps=4, the empirical curve sits at chance (AUC ~0.499) and its 95% CI stays comfortably
under the analytic bound at both headline FPR points. At eps=inf, the SAME attack pipeline, SAME
shadow count, SAME target population — with DP noise switched off — cleanly detects membership
(AUC=0.9926, TPR@1%FPR=0.87). This is the control that makes the finite-eps passes meaningful: the
attack demonstrably has real power against this exact release mechanism, and DP noise is what
suppresses it, not an underpowered attack that would pass against anything.

## Why this matters for the paper

This is the first real empirical evidence for M9/Algorithm 3's DP guarantee (Theorem 11) holding up
under an actual membership-inference attack, not just the analytic sensitivity calculation. Combined
with FX5's core-sweep utility numbers (TAB08/FIG08: U1 gives a smooth privacy-utility tradeoff,
U2 struggles at low client counts), this completes the P1 half of C4's evidence — the P2 stretch
(DP-FedAvg/DP-M5 baselines, FIG09 T=50 panel) remains open but is lower priority.

## Scope and honest limits

- CIFAR-100 only, U1 only, gamma=1.0 only (recovers M8's unbounded running sum — this audit is about
  whether the per-release analytic-Gaussian mechanism holds up, not the contractive C3/C4 retention
  story), 1 seed (per the plan's own spec: "1024 shadows x 1 seed"). Extending to U2, other datasets,
  gamma<1, or more seeds is a real follow-up, not claimed here.
- `full` view is structurally undefined for M9 (it only ever releases one `client=-1`
  post-secure-aggregation record per task, never a per-client one) — `global` is the only meaningful
  view and is what the audit uses throughout.
- The pass/fail check is at exactly two FPR points (1%, 0.1%, per CLAUDE.md non-negotiable #5's
  mandatory reporting pair), using `metrics.membership_report`'s exact Clopper-Pearson CI — not a
  claim that the bound holds at every conceivable FPR, though the full log-log curve
  (`figs/fig10_m9_audit.pdf`) shows the same pattern (empirical curve under the bound) continuously
  across the whole FPR range for eps=1/4.

## Where the numbers live

- `results/fig10_m9_audit.csv` — the full 201-point log-log ROC curve per eps, for the FIG10 plot.
- `results/m9_audit_summary.csv` — the exact headline numbers per eps (this table), each with its own
  Clopper-Pearson CI, `audit_passed` flag.
- `results/paper_numbers.csv` — both cited under `m9_audit_cifar100_eps{1,4,inf}_*` keys.
- `figs/fig10_m9_audit.{pdf,png}` — the 3-panel log-log ROC plot.

## Code changed to make this possible

See `notes/2026-09-23_m9_audit_design_notes.md` for the original 3 subtleties scoped before
implementation (PCA-space dimension mismatch, gamma-decay generalization, fitting PCA once not per
shadow) and a 4th found while implementing (DP noise seed defaulting to a fixed value across all
shadows in one `run_shadow_range` call, which would have made every shadow draw identical noise —
fixed by defaulting `noise_seed` to the shadow's own id in `shadow_runner._run_one_shadow`, verified
with a test that fails without the fix and passes with it). `shadow_runner.py`, `run_m9_audit.py`,
`test_shadow_runner.py` (+9 tests), `test_run_m9_audit.py` (4 new tests, including a full
synthetic-data end-to-end run of `--shadows` then `--score` that caught a real flaw in the original
pass/fail check before it ever touched real data).
