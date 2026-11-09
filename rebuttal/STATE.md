# Rebuttal STATE (maintained by Claude Code, read first in every session)

Resume file. When a session starts, read this, then `rebuttal/CLAUDE_REBUTTAL.md` and
`rebuttal/REBUTTAL_EXPERIMENTS.md`, then `qstat -u islamm`, and continue from "Next action" without
redoing finished work. Rewrite the sections in place; keep it under a page.

## Current phase
ALL DONE: R0–R7 (R7 added 2026-11-09 on user request). Awaiting the authors.

## Status (2026-11-06)
- Working copy confirmed. features/ has train/test/ref for cifar100, cub200, imagenet_r (content-
  addressed `features/<hash>.npz`, both backbones). envs/p3fcl exists. Current source_hash
  `31831aaa…1496` = newest results sidecars and last RUN_LOG lines (job 184680, 2026-09-24).
  Sidecars were NOT anonymised in place (0/575 contain `<repo>`/`anonymous`).
- `m9_sweep_*.csv` were written under older hash `fab2e48b…` (2026-09-23, pre-FX9), so R0 also
  checks that FX9 left M0/M8/M9 code paths unchanged.
- Per-shadow files exist (R1 optional combined arm is possible). Paper-current sources:
  M2 = `shadows_v3/<ds>/m2_target/seed{0,1,2}/` (FX9 V3 wave, 1024 each);
  M3 = `shadows_v2/<ds>/m3_fot/seed{0,1,2}/` (preserved, 1024 each). `shadows/` (v1, 4096×5 seeds)
  is pre-fix. Confirm in R1 against `run_lira_pertask.py` paths before use.
- `make test`: 318 passed, 2 warnings (35 s); RUN_LOG untouched.
- New files: `code/scripts/rb_r0_check.py`, `code/scripts/pbs/rb_r0.pbs`.
- /data: 431T free (80% used). shadows_rb/ expected ~4.5 GB on GPFS (~250 MB per seed dir).

## R0 comparison (job 263120.bcm11, exit 0, source_hash 31831aaa…, `rebuttal/r0_check.csv`)
45 values compared (final_avg_acc, bwt, avg_incremental_acc × 15 seed-matched runs); 45/45 pass,
max |diff| = 0 (bit-identical). Means over seeds 0–2 of final_avg_acc:

| Check | Committed (`results/m9_sweep_<ds>.csv`) | Rerun | Paper |
|---|---|---|---|
| M8 CIFAR-100 | 0.883300 | 0.883300 | 0.883 |
| M8 CUB-200 | 0.871392 | 0.871392 | 0.871 |
| M8 ImageNet-R | 0.635125 | 0.635125 | 0.635 |
| M9 U1 γ=1 ε0=4 CIFAR-100 (p=128, λ=100, C=43.14) | 0.810500 | 0.810500 | 0.810 |
| M0 CIFAR-100 | 0.332533 | 0.332533 | — |

Hyperparameters used equal the committed rows' (hparams_match=1 for all 3 M9 seeds).
Note: rb_r0_check.py builds its manifest at the end, so its sidecar wall_seconds (10.5 s) is
not the job's wall time. Later rb_ scripts create the manifest at start.

Q done: `rebuttal/answers.md` (expected content confirmed, plus one caveat: the population mask is
not seeded by the run seed, so shadow i shares its IN/OUT vector across seeds 0–2).

`make verify` baseline: PASS (2026-11-06).

## R1 status (2026-11-06)
Protocol fixed before any run: `rebuttal/R1_PROTOCOL.md`. It covers the choices §5 leaves open:
M3 server λ = 1.0 (the paper's GRAM payload ridge), ref-mean centring for both M3 views, NaN for a
missing statistic, trajectory ablation for the prediction, and two combined variants.
Code (new files only): `code/src/p3fcl/rebuttal/{__init__,common,release_audit}.py`,
`code/tests/test_rb_release_audit.py` (9 tests), `test_rb_m9_select.py` (2), `test_rb_client_scaling.py` (4), `test_rb_r4_subsample.py` (1), `test_rb_dp_fedavg.py` (10); new methods `methods/m0_dpsgd_u1.py`, `methods/m0_dpfedavg_u2.py`, `code/scripts/rb_r1_{equivalence,release_shadows,score,merge}.py`,
`code/scripts/pbs/rb_r1_{equiv,shadows,score,merge}.pbs`. `make test`: 344 passed.
Source hash is now 54c194f5… (rebuttal files added). The manifests also record `paper_source_hash`
(the hash excluding rebuttal/), which = 31831aaa…, the submission hash.
Findings so far: (a) M2's aggregated broadcast generator has no ledger record, and its per-class
counts are not in the GENERATIVE meta. The broadcast view is therefore computed with the method's
formula and checked against `_generators`. (b) M3's GRAM R is singular for CUB (about 250 members
< d = 768) without a ridge.

## Next action
R2 protocol fixed: `rebuttal/R2_PROTOCOL.md`; R2 job 263206 runs alongside the R1 chain. R3 code-reading paragraph is
written (`answers.md`). It found that the paper's pseudo-clients are 8–15× smaller than the real clients even at n = 10.
Wait for the chains. Then check that
the merge's `margin_repro` rows match the paper exactly and `rb_release_audit.csv` exists. Then write
the R1 result table here and report it to the user (R1 is not a gate, but a failed prediction is
reported as is). R6 done: RESULTS_SUMMARY.md regenerated after R5. `make test` 348 passed; `make verify` PASS (2026-11-06). No failing checks. Nothing queued. Next: only what the authors ask for. R1–R4 are done; notes in notes/2026-11-06_rb_*.md.

## Jobs in flight
| Job id | What | Depends | Status |
|---|---|---|---|
| 268708[].bcm11 | R7 cells, 44×8 cores → rebuttal/work/r7/cell_XX.csv | — | DONE, no errors |
| 268709.bcm11 | R7 merge → results/rb_sensitivity.csv (+_grid). Fails if any selected rerun ≠ reported | afterok 268708[].bcm11 | DONE: 44/44 selected reruns = reported bit for bit; 132 rows + 2,340 grid rows |
| 263120 | R0 reproduction | — | DONE exit 0 |
| 263193 | R1 equivalence, attempt 1 | — | FAILED (float32 span check only, amendment 1). Chain auto-cancelled. Its RUN_LOG line says exit 0 and points to the renamed `rebuttal/r1_equivalence_attempt1_263193.csv` |
| 263202 | R1 equivalence, attempt 2 | — | PASSED 180/180 (exact statistics diff = 0) |
| 263203[] | R1 release shadows | afterok 263202 | DONE 18/18, all paired with the paper. 5.1 GB. M2 client NaN: CUB ≈ 11%, IN-R ≈ 1.5%, CIFAR < 0.5% (all OUT targets) |
| 263204[] | R1 LiRA+bootstrap 30×2 | afterok 263203[] | DONE |
| 263205 | R1 merge → results/rb_release_audit.csv | afterok 263204[] | held |
| 263206 | R2 projection ablation | — | FAILED: own arm-A reselection check (C rel diff 8.5e-10; p, λ exact). Amendment 1 in R2_PROTOCOL. No output |
| 263214 | R2 projection ablation, attempt 2 | — | DONE: 216 rows, reselect check PASS → results/rb_projection_ablation*.csv |
| 263208[] | R3 client scaling 12×2 | — | DONE |
| 263209 | R3 merge → results/rb_client_scaling.csv | afterok 263208[] | DONE: 360 rows, 81/81 paper checks bit for bit |
| 263210 | R4 examples per class | — | DONE exit 0. results/rb_examples_per_class*.csv. 15/15 m=all rows = paper. PREDICTION HOLDS |
| 263212 | R5 correctness gate → rebuttal/r5_gate.csv | — | GATE FAILED 33/36. DP-SGD = M0 exactly (9/9); DP-FedAvg weighted = M0 exactly (9/9); DP-FedAvg uniform misses 0.01 in 3/9 (cub s1 0.0505, cub s2 0.0109, cifar s1 0.0265). **User decided (b), 2026-11-06: McMahan fixed-denominator weighted mean (R5_PROTOCOL amendment 1). Deviation and the 3 failed uniform cells (CIFAR s1 0.0265, CUB s1 0.0505, CUB s2 0.0109) are logged in answers.md and notes/2026-11-06_rb_r5_uniform_mean_gate.md. The uniform mean is NOT a reported baseline. Attempt-1 file renamed `rebuttal/r5_gate_attempt1_uniform_263212.csv` (its RUN_LOG line points to the old name) |
| 263213 | R5 selection + finals | afterok 263212 | auto-cancelled (gate failed) |
| 263741.bcm11 | R5 gate, attempt 2 (McMahan fixed-denominator mean; also computes ŵ → rebuttal/r5_wcap.csv) | — | PASSED 36/36 (all diffs 0). ŵ = 679.0 / 76.0 / 377.05 (CIFAR/CUB/IN-R), recorded in R5_PROTOCOL |
| 263767.bcm11 | R5 selection + finals → results/rb_dp_fedavg.csv | — | DONE 72 rows (started 14:57:25 CST, before the 14:58 script edit that added the nscale modes; main logic identical) |
| 263786.bcm11 | R5 nscale ŵ(n) for CIFAR n ∈ {100, 1000} → rebuttal/r5_wcap_nscale.csv | — | DONE: ŵ(100) = 72.0, ŵ(1000) = 8.0, recorded in amendment 2 |
| 263787.bcm11 | R5 nscale DP-FedAvg CIFAR n ∈ {100, 1000} → results/rb_dp_fedavg_nscale.csv | — | DONE 24 rows |
| 263789/263790 | R6 summary trial runs (generator `code/scripts/rb_r6_summary.py`, `rb_r6.pbs`) | — | DONE. Rerun after R5 finishes |

## Gates
| Gate | Passed (UTC) | User approved? |
|---|---|---|
| R0 reproduction | 2026-11-06 18:28 | YES (2026-11-06) |

## Results ready for the authors
| ID | File | Prediction held? |
|---|---|---|
| R1 | results/rb_release_audit.csv | YES, 12/12 cells (release-only TPR@1%FPR at e=0 ≥ paper margin at e=6). See notes/2026-11-06_rb_r1_release_audit.md |
| R5 | results/rb_dp_fedavg.csv, rb_dp_fedavg_nscale.csv (+_selection) | no prediction. Gate 36/36 after amendment 1. Ledger max m_T = 1. Selection near chance at finite ε (24/24 cells < 0.06). notes/2026-11-06_rb_r5_dp_fedavg.md |
| R7 | results/rb_sensitivity.csv (+_grid) | sensitivity only (best = test-oracle upper bound, not used for selection). notes/2026-11-09_rb_r7_sensitivity.md |
| R6 | rebuttal/RESULTS_SUMMARY.md (generated by code/scripts/rb_r6_summary.py) | — |
| R2 | results/rb_projection_ablation.csv (+_selection) | no prediction stated |
| R3 | results/rb_client_scaling.csv (+_selection) | no prediction. n=10: size-matched C does not help (paper U2 at n=10 stands). Large n on CIFAR with size-matched C: big gains (ε4: 0.015→0.511 at n=2000). Old fixed C plateaus ≈0.048 |
| R4 | results/rb_examples_per_class.csv, _prediction.csv | YES (all 3 finite ε0; M8 m=25 0.854 vs all 0.883) |

## Open problems
- `README.md` has no sections 4–6 (it is the original scaffold readme); nothing to read there.
- `results/a1_lira_fixedk_summary_<ds>_m3_fot_full.csv` (R1's M3 comparison source) has no
  `.meta.json` and no RUN_LOG line; it is byte-identical to `archive/2026-09-23b_pre_fx9/`. Its
  per-seed inputs `a1_lira_pertask_<ds>_m3_fot_seed*_full.*` do have sidecars. Flag in R1/R6.

- 2026-11-09: R7 done. RESULTS_SUMMARY regenerated with an R7 section. make test 350 passed; make verify PASS.
