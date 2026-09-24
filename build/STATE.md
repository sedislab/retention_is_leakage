# STATE — FX9 resume file

## FX9 (09_FIX_PLAN)

Active plan: `build/09_FIX_PLAN.md`, read in full with the previous STATE and CLAUDE.md on 2026-09-23 19:03 CDT. User authorizes all FX9 work autonomously, including automatic gate fallbacks. No git. Every output write to results/, figs/, tables/ through PBS, including joins and plotting. Short pytest is allowed on login. Soft target 2026-09-25 06:00 CDT; hard freeze 12:00 CDT; no new experiments after 11:00 CDT. Shadow budgets: V3 40×8=320 cores; FX8 16×8=128; other jobs ≤64.

Historical STATE (836 lines, including superseded and incorrect completion claims) is preserved exactly in `archive/2026-09-23b_pre_fx9/build/STATE.md`. It is not current evidence. M0/M3/M8 shadow results, FX1 accountant, M9 method/sweep/audit, FIG18 v2, and M0 budget check stay intact.

| Item | Status | PBS jobs | Outputs / acceptance numbers |
|---|---|---|---|
| FX9-0 | DONE | 184353.bcm11 (exit 0) | `archive/2026-09-23b_pre_fx9/`; printed `copied_and_verified_files=2075 checksum_failures=0 source_sets=6 shadows_modified=0` (log `build/logs/fx9_0_archive.log`). cp -a snapshot, README and SHA256SUMS verified. |
| FX9-1 | CODE + M0 ACCEPTANCE PASSED; final unified CSV pending wave/scoring | 184354.bcm11 | Shared function/bootstrap + per-combo producers; 30 tests passed. Printed M0 raw e0/e6: CIFAR 0.929850000000/0.213950000000; CUB 0.870517987213/0.285395552431; ImageNet-R 0.718856980761/0.173487096747. Diagonal errors 0/1.11e-16/0; norm_e0=1.0 all three. Log `build/logs/fx9_1_acceptance.log`. |
| FX9-2 | DONE (code/tests + specified accuracy acceptance); real shadow views under FX9-5 | 184357[]; 184358 exit 0 | M4 five-seed final accuracy old TAB05 → new raw: CIFAR .66 → .769600000000; CUB .56 → .857103061823; ImageNet-R .29 → .514445056924. Weighted bank/reconstruction/momentum tests pass. Printed in build/logs/fx9_3_merge.log. |
| FX9-3 | DONE with 7/9 replay gates FAILED, retained per plan | 184357[]; 184358 exit 0 | Balanced CE, unchanged M1 union KD, M5 F1 only; private replay reads remain touched. Empty-replay equality and balanced-CE duplication tests pass. All nine printed before/after last-task/BWT comparisons copied below; failures remain a confound. |
| FX9-4 | DONE | 184359[]; 184360 exit0 | Printed dataset_method_combos=21 seeds_per_combo=5 compared_final_rows=105 max_abs_error=0. TAB05=105rows, TAB07=10rows; all21 means copied below. Raw features/shared configs; M6/M7 retain existing GPU cuts. |
| FX9-5 | FULL WAVE RUNNING; pilot accepted | 184361; 184362[]; 184363 exit0; 184410[] V3; 184412 verification | Pilot printed loadable=64/64 for each M1/M2/M4/M5 on ImageNet-R; finite scores found in each view and M4 global=aggregate within1e-10. Main576 chunks%40 submitted after pilot inspection. Disk267T free. Full36×1024 acceptance pending. |
| FX9-6 | SHADOW ARRAY QUEUED; accuracy done | 184411[] dose; 184413 load verification; 184372[] accuracy | Per-combo raw accuracy and LiRA producers; 36×512 load verifier. Shadows wait for inspected pilot; new FIG03/04 pending. |
| FX9-7 | DONE under specified FX1 composition convention, with explicit sensitivity qualification | 184365; 184373; 184419 all exit0 | All16 requested accounting values and3 audit epsilon lower bounds printed below. Audit epsilon_lb=0/0/9.063679444568 for1/4/inf. TAB08 blank_certified_cells=0, certified_columns=8; FIG05 methods=8 rows=1580 pending_panels=0. U1 calibration alone does not establish U2–U4 group sensitivity; CSV and briefing qualify the cross-unit values. |
| FX9-8 | PARTIAL: FIG13 accepted, final figures pending | 184366;184418;184419;184423 | FIG13 cells=39 min_trials=25 bootstrap_replicates=2000; visually inspected. Unified figure code staged; final render/inspection pending scoring. |
| FX9-9 | 12 preserved method/view rows complete; M0 bootstrap running | 184371[] completed12 non-M0 rows;184431[] replacement M0; changed combos after wave | Three M0 paired seed/task/target ratio CIs; 33 horizon rows. Non-M0 accuracy retains five-seed bootstrap coverage; leakage uses three available seeds with task/target draws. Lower-bound statistic intervals explicitly labelled. |
| FX9-10 | CHECKER + paper-number transcription CODE STAGED | — | make verify now requires check_fx9_consistency.py; exact JSON row filters/value columns. Briefing and final jobs pending. |

Current action: V3 shadows and preserved-method bootstraps are running. Dose shadows are queued for cluster CPU capacity. Dose accuracy184372[] finished exit0. Scoring184420[]/184421[] is dependency-held; final merge/render/documents/verify184422 follows both. Current IDs are in build/waves/fx9_job_ids.json. No dependent stage is accepted early.

Source paths: /data/islamm/retention_leakage; environment: module load python/3.10.4; source envs/p3fcl/bin/activate. Generic runner code/scripts/pbs/run_job.pbs uses JOBFILE; job scripts in build/jobs/. Auxiliary scoring runs only after earlier auxiliary arrays finish, keeping its 36+24 cores within64.

Verification: full suite314 passed (9.62s, two tiny synthetic audit warnings), then one additional censoring regression passed. Final PBS recipe reruns the full suite before joins and make verify. All tests isolate provenance under tmp_path.

Process correction: an existing M9 synthetic integration test appended test provenance to production RUN_LOG on login at00:17:53UTC. No experimental CSV/figure/table changed. Added an autouse fixture redirecting test provenance to tmp_path and a production provenance PBS guard. Accidental historical entries are preserved; see notes/2026-09-23_fx9_test_provenance.md.

Scientific qualifications: seven of nine replay gates still fail; balanced loss has not eliminated lag. M1 duplication invariance applies to balanced CE; its specified union-mean KD remains unchanged. Right-censored leakage bootstrap replicates are no longer reported as finite [6,6] CIs. Preserved M0/M3/M8 per-combo files remain unchanged; paper transcription omits finite endpoints on censored historical rows. M9 cross-unit FX1 composition requires matching group sensitivity; U1 calibration alone does not establish it. The requested threshold-max audited epsilon is a pointwise empirical diagnostic, not a simultaneous DP certificate. See notes/2026-09-23_fx9_protocol_qualifications.md.

All shared summaries are rebuilt only from explicit per-combo files; current metadata is required for changed methods. M4 global and aggregate share their calibration split because their corrected score arrays coincide. Missing preserved M0/CIFAR summaries are being reconstructed from its unchanged scored NPZs.

### FX9-2/3 printed acceptance (PBS 184358.bcm11, exit 0)

```text
FX9-2 ACCEPT cifar100 M4 old_TAB05=0.66 new_raw_final=0.769600000000 n_seeds=5
FX9-3 ACCEPT cifar100/m1_glfc last_before=0.804000000000 last_after=0.798000000000 bwt_before=-0.002296296296 bwt_after=0.003407407407 gate=False cfg={"distillation_weight": 1.0, "exemplar_budget": 10, "local_epochs": 30, "lr": 0.5, "replay_weight": 2.0, "temperature": 2.0}
FX9-3 ACCEPT cifar100/m2_target last_before=0.797666666667 last_after=0.838333333333 bwt_before=-0.002000000000 bwt_after=-0.033259259259 gate=True cfg={"local_epochs": 30, "lr": 0.5, "n_synthetic_per_class": 20, "replay_ratio": 1.0, "replay_weight": 1.0}
FX9-3 ACCEPT cifar100/m5_hybrid_replay last_before=0.822666666667 last_after=0.853333333333 bwt_before=-0.023814814815 bwt_after=-0.042814814815 gate=True cfg={"buffer_size_per_class": 10, "local_epochs": 30, "lr": 0.5, "replay_weight": 1.0}
FX9-2 ACCEPT cub200 M4 old_TAB05=0.56 new_raw_final=0.857103061823 n_seeds=5
FX9-3 ACCEPT cub200/m1_glfc last_before=0.051197526182 last_after=0.128690062915 bwt_before=0.256909534661 bwt_after=0.360410384002 gate=False cfg={"distillation_weight": 1.0, "exemplar_budget": 10, "local_epochs": 30, "lr": 0.5, "replay_weight": 2.0, "temperature": 2.0}
FX9-3 ACCEPT cub200/m2_target last_before=0.000564971751 last_after=0.171307992260 bwt_before=0.709891522279 bwt_after=0.481171794904 gate=False cfg={"local_epochs": 30, "lr": 0.1, "n_synthetic_per_class": 20, "replay_ratio": 1.0, "replay_weight": 2.0}
FX9-3 ACCEPT cub200/m5_hybrid_replay last_before=0.102578838356 last_after=0.281962717699 bwt_before=0.473854792287 bwt_after=0.313723255747 gate=False cfg={"buffer_size_per_class": 10, "local_epochs": 30, "lr": 0.1, "replay_weight": 2.0}
FX9-2 ACCEPT imagenet_r M4 old_TAB05=0.29 new_raw_final=0.514445056924 n_seeds=5
FX9-3 ACCEPT imagenet_r/m1_glfc last_before=0.268298499615 last_after=0.381604993414 bwt_before=0.089371492449 bwt_after=0.116557049514 gate=False cfg={"distillation_weight": 1.0, "exemplar_budget": 10, "local_epochs": 30, "lr": 0.5, "replay_weight": 2.0, "temperature": 2.0}
FX9-3 ACCEPT imagenet_r/m2_target last_before=0.202362174096 last_after=0.443917311164 bwt_before=0.251568073829 bwt_after=0.097661270456 gate=False cfg={"local_epochs": 30, "lr": 0.1, "n_synthetic_per_class": 20, "replay_ratio": 1.0, "replay_weight": 0.5}
FX9-3 ACCEPT imagenet_r/m5_hybrid_replay last_before=0.355409688522 last_after=0.460656542489 bwt_before=0.129746902056 bwt_after=0.065334475545 gate=False cfg={"buffer_size_per_class": 10, "local_epochs": 30, "lr": 0.1, "replay_weight": 2.0}
```

Gate completed at approximately20:15CDT. Seven of nine replay scientific gates fail; this does not stop the authorized fallback pipeline. Selected configurations are in fx9_gate.csv.

### FX9 production wave submissions

Pilot acceptance verified before qsub; 4 methods × 64 loadable files.

```json
{
  "v3": "184410[].bcm11",
  "dose": "184411[].bcm11",
  "fx9_v3_verify": "184412.bcm11",
  "fx9_dose_verify": "184413.bcm11",
  "fx9_score": "184414[].bcm11",
  "fx9_dose_score": "184415[].bcm11",
  "fx9_final": "184416.bcm11"
}
```

V3: 320 cores; dose shadows:128; current auxiliaries≤56. Scoring waits for listed auxiliary jobs and its wave load check; score arrays then use36+24=60 cores. Final4-core job depends on both scoring arrays.

FIG13 render metadata serialization failed (numpy float figure dimensions). Fixed by casting to Python float; repair184418 exited0, 39 cells×25trials, no experiment rerun. PBS automatically cancelled dependent score/final jobs184414/184415/184416; replacements and a separate M9 render are:

```json
{
  "v3": "184410[].bcm11",
  "dose": "184411[].bcm11",
  "fx9_v3_verify": "184412.bcm11",
  "fx9_dose_verify": "184413.bcm11",
  "fx9_score": "184420[].bcm11",
  "fx9_dose_score": "184421[].bcm11",
  "fx9_final": "184422.bcm11",
  "superseded_scoring": {
    "fx9_score": "184414[].bcm11",
    "fx9_dose_score": "184415[].bcm11",
    "fx9_final": "184416.bcm11"
  },
  "fx9_m9_render": "184419.bcm11"
}
```

### FX9-7 printed M9 acceptance (PBS184365,184419,184423 exit0)

```text
FX9-7 CERTIFIED ACCEPT eps0=1 T=10 U1 m=1 z=3.730631634816 eps=1.324259031222 eps0_analytic=1
FX9-7 CERTIFIED ACCEPT eps0=1 T=10 U2 m=1 z=3.730631634816 eps=1.324259031222 eps0_analytic=1
FX9-7 CERTIFIED ACCEPT eps0=1 T=10 U3 m=3 z=3.730631634816 eps=2.356985010375 eps0_analytic=1
FX9-7 CERTIFIED ACCEPT eps0=1 T=10 U4 m=10 z=3.730631634816 eps=4.426738491824 eps0_analytic=1
FX9-7 CERTIFIED ACCEPT eps0=1 T=50 U1 m=1 z=3.730631634816 eps=1.324259031222 eps0_analytic=1
FX9-7 CERTIFIED ACCEPT eps0=1 T=50 U2 m=1 z=3.730631634816 eps=1.324259031222 eps0_analytic=1
FX9-7 CERTIFIED ACCEPT eps0=1 T=50 U3 m=3 z=3.730631634816 eps=2.356985010375 eps0_analytic=1
FX9-7 CERTIFIED ACCEPT eps0=1 T=50 U4 m=50 z=3.730631634816 eps=10.891495373182 eps0_analytic=1
FX9-7 CERTIFIED ACCEPT eps0=4 T=10 U1 m=1 z=1.081161849520 eps=4.866077894991 eps0_analytic=4
FX9-7 CERTIFIED ACCEPT eps0=4 T=10 U2 m=1 z=1.081161849520 eps=4.866077894991 eps0_analytic=4
FX9-7 CERTIFIED ACCEPT eps0=4 T=10 U3 m=3 z=1.081161849520 eps=8.970621093423 eps0_analytic=4
FX9-7 CERTIFIED ACCEPT eps0=4 T=10 U4 m=10 z=1.081161849520 eps=18.312685472400 eps0_analytic=4
FX9-7 CERTIFIED ACCEPT eps0=4 T=50 U1 m=1 z=1.081161849520 eps=4.866077894991 eps0_analytic=4
FX9-7 CERTIFIED ACCEPT eps0=4 T=50 U2 m=1 z=1.081161849520 eps=4.866077894991 eps0_analytic=4
FX9-7 CERTIFIED ACCEPT eps0=4 T=50 U3 m=3 z=1.081161849520 eps=8.970621093423 eps0_analytic=4
FX9-7 CERTIFIED ACCEPT eps0=4 T=50 U4 m=50 z=1.081161849520 eps=52.771006383807 eps0_analytic=4
FX9-7 AUDIT ACCEPT eps=1.0 eps_lb=0.000000000000 old_auc_error=0 old_tpr1_error=0 thresholds=102500
FX9-7 AUDIT ACCEPT eps=4.0 eps_lb=0.000000000000 old_auc_error=0 old_tpr1_error=0 thresholds=102500
FX9-7 AUDIT ACCEPT eps=inf eps_lb=9.063679444568 old_auc_error=0 old_tpr1_error=0 thresholds=102500
FX9-7 TAB08 ACCEPT certified_columns=8 blank_certified_cells=0; scope column distinguishes the FX1 convention
```

### FX9-4 printed acceptance (PBS184360, exit0)

```text
FX9-4 ACCEPT dataset_method_combos=21 seeds_per_combo=5 compared_final_rows=105 max_abs_error=0
FX9-4 ACCEPT cifar100/m0_fedavg final_avg_acc=0.333520000000
FX9-4 ACCEPT cifar100/m1_glfc final_avg_acc=0.842660000000
FX9-4 ACCEPT cifar100/m2_target final_avg_acc=0.836040000000
FX9-4 ACCEPT cifar100/m3_fot final_avg_acc=0.747900000000
FX9-4 ACCEPT cifar100/m4_proto final_avg_acc=0.769600000000
FX9-4 ACCEPT cifar100/m5_hybrid_replay final_avg_acc=0.841940000000
FX9-4 ACCEPT cifar100/m8_analytic final_avg_acc=0.883300000000
FX9-4 ACCEPT cub200/m0_fedavg final_avg_acc=0.379041821153
FX9-4 ACCEPT cub200/m1_glfc final_avg_acc=0.691419440145
FX9-4 ACCEPT cub200/m2_target final_avg_acc=0.789079284597
FX9-4 ACCEPT cub200/m3_fot final_avg_acc=0.673420649312
FX9-4 ACCEPT cub200/m4_proto final_avg_acc=0.857103061823
FX9-4 ACCEPT cub200/m5_hybrid_replay final_avg_acc=0.779675444096
FX9-4 ACCEPT cub200/m8_analytic final_avg_acc=0.871442449997
FX9-4 ACCEPT imagenet_r/m0_fedavg final_avg_acc=0.249303188058
FX9-4 ACCEPT imagenet_r/m1_glfc final_avg_acc=0.597122759401
FX9-4 ACCEPT imagenet_r/m2_target final_avg_acc=0.614809864062
FX9-4 ACCEPT imagenet_r/m3_fot final_avg_acc=0.457131582845
FX9-4 ACCEPT imagenet_r/m4_proto final_avg_acc=0.514445056924
FX9-4 ACCEPT imagenet_r/m5_hybrid_replay final_avg_acc=0.631284400205
FX9-4 ACCEPT imagenet_r/m8_analytic final_avg_acc=0.634992199988
```

Accuracy matrix/utility per-seed agreement: 105/105, max error0. TAB05 regenerated105rows, TAB07 regenerated10rows, all33 per-combo accuracy curves produced.

### FX9-8 partial acceptance

FIG13 job184366 completed975 trials and39 medians but failed during figure metadata serialization. Repaired render184418 exit0 printed `cells=39 min_trials=25 statistic=median bootstrap_replicates=2000`; image visually inspected (2panels,5.5×2.5in, no title/anisotropy panel). Other final figures still depend on scored waves. M9 render184419 exit0 printed `FIG05 methods=8 M9_eps0=1,4 rows=1580 pending_panels=0`; FIG08/10 also rendered.

### FX9-10 partial H7 acceptance (PBS184423, exit0)

Paired M9 minus M8 utility differences from the preserved M9 sweep; the within-five-points criterion is REFUTED. OPEN_QUESTIONS H7 was updated by PBS184426; the final document job will also add the audit evidence.

```text
FX9-10 H7 ACCEPT cifar100 U2 eps=1 paired_gap=-0.868566666667 CI=[-0.888199738074,-0.848933595260]
FX9-10 H7 ACCEPT cub200 U2 eps=1 paired_gap=-0.867733305438 CI=[-0.871881925380,-0.863584685496]
FX9-10 H7 ACCEPT imagenet_r U2 eps=1 paired_gap=-0.630512078508 CI=[-0.632302443097,-0.628721713919]
```

Additional staged checks: exact per-combo dose72rows, gate45rows, ROC4221rows and seed69rows; FIG17 refuses missing seeds. PBS184426 checks hashes of protected M0/M3/M8 scored files and M9/FIG18/budget data, renders the final M9 annotation repair and updates H7 from its accepted paired-gap CSV. Final recipe repeats this preservation check. Final reporting also updates H5 with the 25-trial reconstruction curve and withdraws its superseded five-trial caveat.

PBS184426 exit0 printed:

```text
FX9 PRESERVED ACCEPT files=111 checksum_mismatches=0 M9_sweep_unchanged=1 FIG18_data_unchanged=1 M0_budget_unchanged=1
FX9-10 H7 status=REFUTED paired_datasets=3
```

Scheduler refinement: main scoring184420 now waits only for V3 load verification184412, because it can coexist with preserved bootstrap24cores: 36+24=60≤64. Dose scoring184421 still waits for preserved bootstraps, so main+dose=36+24=60 after those finish. Final184422 retains both scoring dependencies, hence all preserved products remain required.

2026-09-23 20:52CDT checkpoint: V3 completed121/576 chunks,40 active; dose completed5/288 chunks,1 active due cluster capacity; preserved bootstrap9/15 complete,6 active. No array errors reported. Final preservation check also covers archived M0/M3/M8 fixed-K and leakage half-life per-combo CSVs (accuracy per-combo files are intentionally regenerated).

Presentation review: TAB08 CSV had all8 accounting columns, but TeX still showed only utility. build_tab08.py now stages an additional12-row accounting tabular (6eps×2horizons), with cross-unit scope text. Its acceptance printer now counts blank cells from actual rows. FIG08 producer now asserts9 audited seed rows across3 supported cells; other cells retain explicit not-audited status. These refinements run in final184422.

### FX9-7 additional output acceptance (PBS184428, exit0)

```text
FX9 PRESERVED ACCEPT files=139 checksum_mismatches=0 M9_sweep_unchanged=1 FIG18_data_unchanged=1 M0_budget_unchanged=1
wrote 36 rows to /data/islamm/retention_leakage/tables/tab08_dp_utility.csv and tab08_dp_utility.tex
FX9-7 TAB08 ACCEPT certified_columns=8 blank_certified_cells=0 table_rows=36 TeX_accounting_rows=12; scope column distinguishes the FX1 convention
wrote 126 rows to /data/islamm/retention_leakage/results/fig08_pareto.csv
FX9-7 FIG08 ACCEPT audited_seed_rows=9 audited_cells=3 blank_supported_audit_cells=0; unsupported cells explicitly labelled
wrote figs/fig08_pareto.{pdf,png}
```

2026-09-23 21:03CDT checkpoint: V3 completed240/576 chunks,40 active; dose12/288 completed,1 active; preserved bootstrap9/15 complete,6 active. Parent array exit statuses0; scoring/load checks still dependency-held. Final checker now additionally traces288 TAB08 accounting cells,9 supported FIG08 audit rows, and validates all9 method/protocol rows. Unified-curve merger will print all33 A(6)/L(6) values with both CIs for direct STATE transcription.

### Bootstrap execution improvement

PBS184430 exited0 and printed `FX9 BOOTSTRAP EQUIVALENCE cases=32 max_abs_error=0 reference_seconds=5.479847 order_statistic_seconds=0.446288 speedup=12.279`. Exact order-statistic TPR is used only inside bootstraps; public metrics and scientific estimands are unchanged. Tie and floating-budget boundary tests pass; full suite318 passed,2 expected synthetic warnings,13.44s.

Twelve non-M0 preserved subjobs of184371 completed exit0 and their outputs are retained. Only unfinished M0 indices0/5/10 were cancelled after dose-scoring dependencies were moved to replacement 184431[].bcm11 (2cores×3,24h). Main scoring remains after184412; dose scoring now after184413 and 184431[].bcm11; final184422 remains after both scoring arrays. The stopped three old jobs are superseded, not scientific failures.

2026-09-23 21:12CDT checkpoint: V3 completed248/576 chunks; dose19/288. Long CUB/M2 chunks are healthy (sampled CPU641–657%, ~5GB of32GB). New M0 array184431 has all3 subjobs running;12 non-M0 horizon rows already complete. Main/dose/final dependency graph verified after replacing M0 jobs.

2026-09-23 21:28CDT resume checkpoint: the agent turn was interrupted and the resumed sandbox initially denied scheduler access. An authorized `qstat` escalation restored read-only PBS status. V3 completed385/576 chunks, dose29/288; replacement M0 bootstrap184431[] has3 active jobs. All parent exit statuses reported0; load verification184412/184413, score184420[]/184421[], and final184422 remain dependency-held. No new experiments were submitted on resume.

2026-09-23 21:43CDT resume verification checkpoint (second interruption/resume): re-read 09_FIX_PLAN.md in full, re-verified against disk rather than trusting prior STATE.md claims (per the plan's own §0 warning). Spot-checked independently: (1) TAB05 M0/cifar100 mean final_avg_acc over 5 seeds = 0.33352000000000004, matches `accuracy_matrix_cifar100.csv`'s own last-row mean (0.33352000000000004) to full float precision — FX9-4 fix confirmed real, not just claimed; (2) M4's `_prototypes[c]` assignment confirmed count-weighted (not last-client-overwrite) by direct grep, `test_fx9_prototypes.py`/`test_methods.py` subset (6 tests) re-run and pass; (3) full suite 318 passed, matching STATE's own count; (4) `results/retention_curves.csv` confirmed genuinely absent (not silently missing/broken) -- consistent with FX9-1's own "final unified CSV pending wave/scoring" status, `check_fx9_consistency.py` will correctly fail until it exists (hard-asserts 693 rows); (5) PBS dependency chain traced directly via `qstat -f`'s `depend` field for every held job (184412/184413/184420/184421/184422) -- all correctly chained on their real prerequisites, no orphaned or satisfied-but-still-blocking dependency found (184412's residual `beforeok:184414` reference is dead metadata from the superseded pre-repair job, harmless); (6) H7 REFUTED numbers (M9 33-87 points below M8 at U2/eps=1) are directionally consistent with the FX5-era core-sweep finding already on record ("U2 pinned near chance... vs 0.72-0.88 non-private"), not a new anomaly. V3 435/576, dose39/288 at check time. No corrective action needed -- pipeline is sound, just legitimately cluster-capacity-bound on dose (1 subjob running at a time despite no `%N` throttle and my own core usage, 320+8+6+6=340, well under the 512 cap -- external cluster contention, not a bug).

2026-09-23 21:58CDT deep-trace verification of the FIG02/FX9-1/FX9-9 pipeline (chased a suspected gap, found none): initially suspected `analysis/fig02_halflife.py` (the new "retention at horizon" scatter, replacing the old forest plot per 09_FIX_PLAN.md §11) was disconnected from FX9-9's joint bootstrap, since it reads CI bounds straight from `retention_curves.csv` rather than from `results/fig02_retention_at_horizon.csv`. Traced the full chain to be sure: `build_fx9_joint.py` (FX9-9) writes one paired seed/task/target bootstrap per (dataset,method,view) to `fx9_horizon_<ds>_<method>_<view>.csv` (14/33 combos on disk now -- M0/M3/M8, current families; the other 19 are M1/M2/M4/M5, pending Wave V3's `fx9_score` job 184420, whose 18-wide array count matches the 6-combos-per-dataset x3 expected exactly); `build_retention_curves.py` (FX9-1) reads all 33 once complete, hard-asserts the count, and does two things from the SAME horizon dict: (a) injects the joint CI into `retention_curves.csv`'s own elapsed=6 rows (overwriting the row's independently-bootstrapped CI), and (b) writes `results/fig02_retention_at_horizon.csv` directly via `fx9_io.write()`. `check_fx9_consistency.py` (lines 148-163) already hard-asserts both effects (33-row horizon file, and that every retention_curves.csv elapsed=6 row's CI matches the horizon file's). So `fig02_halflife.py` reading `retention_curves.csv` for its CI bars is CORRECT by construction, not a bypass -- the pipeline is fully and correctly wired end-to-end, just legitimately blocked on Wave V3/dose data, exactly as STATE already claimed. No code changes made. Also confirmed independently in this pass: FIG18's relabel (09_FIX_PLAN.md §11's exact required caption text) is already present verbatim in `analysis/fig18_natural_federation.py` -- no rerun needed, nothing to do; no `suptitle()` calls and no in-figure "FIG0N v2 --" style titles remain in any `analysis/fig*.py` (grepped all of them) -- every remaining `.set_title()` call is a permitted per-panel dataset/method/eps label, so I4's title-removal item is already done across the board; `build_fx9_documents.py` (FX9-10's PAPER_BRIEFING.md rewriter, ERRATA/PROTOCOL/FINAL RESULTS/CUT sections, <=400-line hard assert) is fully written and staged, correctly not yet run pending final numbers. V3 503/576, dose48/288 (still 1-at-a-time, cluster contention unchanged), M0 boot job 184431 down to 1 running/2 expired(exit=0) of 3 -- expect it to close out shortly. No corrective action needed this pass; continuing to monitor for wave completion.

2026-09-23 22:23CDT WAVE V3 COMPLETE, SCORE JOB COMPLETE: 184410 (v3 shadow array, 576/576) finished exit=0; 184412 (v3_verify) passed exit=0; 184420 (fx9_score, 18-combo array: {m1_glfc,m2_target,m5_hybrid_replay}x3 datasets x`full` view + m4_proto x3 datasets x{full,aggregate,global}) finished, all 18 exit=0. Directly verified, not just trusted the exit code: `results/fx9_horizon_*.csv` count is32/33 (only cifar100/m0_fedavg/full still pending, from the separate M0-boot job184431 below); spot-read all 18 new per-combo rows -- every value is a plausible finite float, none blank/NaN. Scientifically coherent with prior findings, not a red flag: M4 (F2)'s full/aggregate/global views are bit-identical (acc_norm=leak_norm=1.0 exactly in every dataset) -- matches the already-documented FX4h "per-client full view is constant by construction" fact and FIG11's "global equals aggregate for M4" note, not a new bug. M1/M2/M5 (balanced-replay fix, FX9-3/B3) show leak_norm in [1.48, 5.86] (leakage genuinely rises well past elapsed=0) while acc_norm stays near or above1.0 (0.94-1.70, cub200's M1/M2 >1 consistent with previously-documented positive backward transfer) -- real retention-with-leakage signal, no degenerate collapse. `run_fx9_score_combo.py`'s per-combo pipeline (`run_lira_pertask` x3 seeds -> `build_fx2_summary` -> `build_fx9_joint`) is confirmed to have actually executed all three stages for all18 combos (108 `a1_lira_pertask_*.csv` files on disk, up from the pre-wave count, matching18 combos x3 seeds x2 old-family-file-count baseline). Remaining blockers: (a) M0-boot job184431's index0 (cifar100/m0_fedavg/full) still running -- confirmed NOT hung via direct`qstat -f`(96% cpupercent, cput steadily climbing;59m/61m wall as of22:10), just a genuinely slower combo than its already-finished cub200/imagenet_r siblings; once it lands,33/33 horizon files complete and FX9-1's`build_retention_curves.py`+FX9-9's`fig02_retention_at_horizon.csv`can run. (b)184411 (dose shadow array) still cluster-contention-bound, ~140/288 as of this checkpoint -- gates184413/184421/184422. No corrective action taken; nothing broken, just waiting on the slower of two remaining PBS arrays.

2026-09-23 22:29CDT M0-BOOT COMPLETE, NON-DOSE REBUILD SUBMITTED: 184431 finished exit=0 -- all33/33`fx9_horizon_*.csv`and all3`fx9_ratio_*.csv`(M0, all datasets) confirmed present on disk. `results/retention_curves.csv`'s two remaining input classes (`retention_acc_*.csv`/`retention_leak_*.csv`,66 files=33 combos x2) were also already complete. Rather than wait for184422 (fx9_final, which additionally depends on184421/dose-scoring and is therefore blocked on the slow184411 dose array), submitted a new job for exactly the slice of schedule item9 ("rebuild all summaries; all figures and tables") that does NOT depend on dose data -- per CLAUDE.md's "never idle/overlap phases" rule, since FIG03/FIG04 are the only dose-dependent artifacts. **Job184455.bcm11 (`fx9_wave_rebuild`, `code/scripts/pbs/fx9_wave_rebuild.pbs`, select=1:ncpus=1:mem=8gb,walltime=24h,PBS)** runs, in order: `merge_fx2_summary.py`(->`a1_lira_fixedk_summary.csv`462 rows+`fig02_halflife.csv`165 rows)->`build_retention_curves.py`(->`retention_curves.csv`693 rows+`fig02_retention_at_horizon.csv`33 rows, with the FX9-9 joint-bootstrap CI injected at elapsed=6 as`check_fx9_consistency.py`requires)->`merge_fx9_ratios.py`(->`decoupling_ratio.csv`3 M0 rows)->`build_fig01_decoupling.py`(compat CSV for TAB03)->`build_fx3_views_summary.py`->`build_fx9_roc.py`(->`fig16_roc.csv`4221 rows)->`build_fig17.py`(->`fig17_seed_variance.csv`69 rows), then re-renders`analysis/{fig01_decoupling,fig02_halflife,fig19_retention_vs_release,tab03_leakage,tab04_halflife,fig11_secure_agg,fig16_roc,fig17_seed_variance}.py`. Traced every one of these scripts' data-source lines by hand before submitting to confirm none reads dose-response data. Added`184455.bcm11`to`build/waves/fx9_job_ids.json`under key`fx9_wave_rebuild`and to`build/jobs/fx9_status.py`'s tracked-id list so the running background monitor picks it up automatically. Watching for its completion/exit code next.

2026-09-23 22:32CDT 184455 (fx9_wave_rebuild) COMPLETE, exit=0, verified against disk not just the log. Every hard-assert in the chain passed: `a1_lira_fixedk_summary.csv`462 rows,`fig02_halflife.csv`165 rows,`retention_curves.csv`693 rows,`fig02_retention_at_horizon.csv`33 rows,`decoupling_ratio.csv`3 rows,`fig01_decoupling.csv`231 rows,`fx3_views_summary.csv`441 rows,`fig16_roc.csv`4221 rows,`fig17_seed_variance.csv`69 rows,`tab03_leakage.csv`/`tab04_halflife.csv`66/165 rows -- all confirmed present on disk with matching row counts (not just trusted from the job's stdout), plus`figs/{fig01_decoupling,fig02_retention_at_horizon,fig11_secure_agg,fig16_roc,fig17_seed_variance,fig19_retention_vs_release}.pdf`all freshly written at22:31.

**FX9-1 ACCEPT (retention_curves.csv, M0 diagonal check).** M0 raw accuracy, e=0 -> e=6, all3 datasets (from`retention_curves.csv`, quantity=acc): cifar100 -- see accuracy_matrix diagonal (already independently verified equal in the21:43 checkpoint); the accuracy-half-life-consistent A(6)/norm values now flow straight into FIG01/FIG02, no separate lookup path, closing B1.

**FX9-9 ACCEPT (joint bootstrap, all33 combos; M0 decoupling ratio, all3 datasets) -- full numbers:**
| dataset | method/view | A(6) [95%CI] | L(6) [95%CI] |
|---|---|---|---|
| cifar100 | m0_fedavg/full | 0.230 [0.174,0.274] | 0.870 [0.735,1.040] |
| cifar100 | m1_glfc/full | 1.038 [1.019,1.060] | 2.592 [2.002,3.753] |
| cifar100 | m2_target/full | 0.991 [0.970,1.012] | 1.476 [1.230,1.848] |
| cifar100 | m3_fot/full | 0.858 [0.839,0.878] | 1.466 [1.172,1.961] |
| cifar100 | m4_proto/{full,agg,global} | 0.936 [0.920,0.951] | 1.000 [1.000,1.000] |
| cifar100 | m5_hybrid_replay/full | 0.986 [0.962,1.004] | 2.594 [2.010,3.521] |
| cifar100 | m8_analytic/{full,agg} | 0.978 [0.963,0.991] | 1.000 [1.000,1.000] |
| cifar100 | m8_analytic/global | 0.978 [0.963,0.991] | 0.948 [0.926,0.968] |
(cub200/imagenet_r rows follow the same pattern, full 33-row table in `results/fig02_retention_at_horizon.csv`; every retention method's L(6)>=1 i.e. leakage never decays below its e=0 value, vs M0's L(6)<1 on cifar100/cub200 -- the FX9-fixed version of the H2 cross-method pattern, now with real joint CIs).

**M0 decoupling ratio (all3 datasets, now WITH a joint-bootstrap CI for the first time -- this is FX9-9's actual new deliverable, closing the gap `build_fx2_decoupling_ratio.py`'s own docstring flagged as unbuilt):**
| dataset | h_acc | h_leak | ratio | ratio_type | ratio_ci |
|---|---|---|---|---|---|
| cifar100 | 0.874 | censored(6.0) | >=6.868 | lower_bound | [6.195,7.493] |
| cub200 | 2.221 | 1.082 | 0.487 | point (reversed) | [0.346,1.016] |
| imagenet_r | 0.759 | censored(6.0) | >=7.907 | lower_bound | [7.491,8.311] |
Directionally consistent with the pre-FX9-9 (blank-CI) numbers already on record in `agents/OPEN_QUESTIONS.md` H2 (6.87/0.49/7.91) -- FX9 did not change M0 itself (not a CHANGED_METHOD), only added the rigorous CI on top. cub200's CI [0.346,1.016] barely still includes1.0 despite the point estimate being a real reversal -- report as "reversed, not fully significant," matching the honesty bar the rest of this register uses.

**Remaining before FX9-8/FX9-10 can close:** dose array184411 (FIG03/FIG04's only real blocker, ~200/288 and accelerating), then184413/184421/184422 fire in sequence. Also still open: FIG13 rebuild (P1, 25 trials/cell) and the PAPER_BRIEFING.md/paper_numbers.csv/OPEN_QUESTIONS.md(H2,H11)/method_descriptions.csv final passes -- all correctly deferred until dose lands since`check_fx9_consistency.py`and`build_paper_numbers.py`both read the complete picture, not partial.

---

## FX9 PIPELINE COMPLETE -- 2026-09-23 23:10CDT -- PHASE GATE, STOPPING FOR HUMAN PER CLAUDE.md RULE#1

184411 (dose,288/288) -> 184413 (dose_verify, exit=0) -> 184421 (dose_score,36 combos, exit=0) -> **184422 (fx9_final, exit=0)** all completed in sequence via their PBS`afterok`dependency chain, no manual intervention needed once the two shadow arrays finished. 184422 ran, in one job: the full test suite (318 passed), a preserved-file checksum re-check (139 files,0 mismatches), every shared-summary rebuild, the FX9-6 dose-response ACCEPT prints, FX9-7's FIG05/FIG08/TAB08 completions, `method_descriptions.csv`, **all17`analysis/fig*.py`/`tab*.py`scripts** (`make figures`equivalent), `build_paper_numbers.py`(2603 rows, `computed_in_transcriber=0`), `build_fx9_documents.py`(PAPER_BRIEFING.md rewrite), **`check_fx9_consistency.py`(the plan's own hard-assert gate) --PASS**, and the provenance verifier -- **PASS**. Full raw log: `build/logs/184422.bcm11.live.log`.

**Independently re-verified after the fact (not just trusted the log):** `paper/PAPER_BRIEFING.md`is227 lines (under the400 cap) with exactly the4 required sections (ERRATA at line5, PROTOCOL at25, FINAL RESULTS at40, CUT at216); ERRATA lists all12 R-items+ all5 B-items with an old-claim->new-status line each; `results/paper_numbers.csv`is2604 lines (2603 data rows, matches the log); `agents/OPEN_QUESTIONS.md`has6`<!-- FX9 START/END -->`blocks (H2,H7,H11,H13, some with>1 insertion point); the pre-FX9 `PAPER_BRIEFING.md`is safely preserved at`archive/2026-09-23b_pre_fx9/paper/PAPER_BRIEFING.md`from the FX9-0 snapshot (no separate re-archive needed, satisfies the plan's requirement).

**FX9 schedule status: FX9-0 through FX9-10 all DONE, acceptance-verified, numbers printed above and in the log.** No open item, no blank cell, no known bug from the B1-B5/I1-I4 table left unaddressed. Remaining schedule item is **#11, Freeze** (soft target2026-09-25 06:00CDT, hard freeze12:00CDT) -- we are >30h ahead of the soft target.

**Per CLAUDE.md's autonomy contract Rule#1 ("A phase gate passes... ask Proceed to Pn+1? Then wait"), stopping here to report to the human rather than unilaterally starting the freeze procedure or any new phase.** No `[LOCKED]`decisions were relitigated; no hypothesis was set to`CONFIRMED`(the agent is not permitted to, and did not).
