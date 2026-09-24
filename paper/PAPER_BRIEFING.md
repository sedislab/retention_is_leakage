# PAPER BRIEFING — FX9 final evidence

This file supersedes the archived pre-FX9 briefing. Experimental numbers below are formatted from the named final CSVs; exact cells and row filters are in `results/paper_numbers.csv`. Intervals are 95% unless explicitly qualified. No hypothesis is marked CONFIRMED by the agent.

## ERRATA

- R1: training accuracy presented as generalization → withdrawn; utility and accuracy matrices now use the held-out test split, with separate acc_train columns.
- R2: late-task replay learning described as retention → withdrawn; client-private buffers, old-class KD, balanced losses and the FX9 gate comparison below replace it.
- R3: privacy touched sets used as FedAvg weights → withdrawn; released aggregation weights reconstruct the actual model; touched still records replay reads.
- R4: M4/M8 full-view persistence called a retention half-life → withdrawn; per-client releases are constant by construction. Aggregate and global state are separate views.
- R5: old lifelong-accountant curves → withdrawn; current FIG05 uses the unchanged, corrected FX1 record-composition accountant on real ledgers. See its sensitivity qualification below.
- R6: extrapolated exponential half-lives → withdrawn; current TAB04 uses fixed K, chance floors, observed first crossing and explicit censoring at E.
- R7: independent-pair headline leakage CIs → withdrawn; current leakage curves use hierarchical seed/target bootstrap, with paired seed/task/target horizon intervals.
- R8: unmatched-client FIG18 comparison → withdrawn; preserved v2 has matched client counts and slide groups within hospital tasks, with limited scope.
- R9: blanket secure-aggregation privacy claim → withdrawn; F1 model-based attacks are identical by construction, while F2/F5 views are measured separately.
- R10: original dose pilot and inert M4 knob → withdrawn; balanced M5/M2 sweeps replace them, with independent replay weight and sample-count knobs.
- R11: old forest plot, small fonts and causal anisotropy interpretation → withdrawn; horizon scatter and consistent styling replace them; FIG13 uses medians and omits the anisotropy panel. A4 remains inconclusive; A6 is explicit count leakage.
- R12: empirical M9 results absent while C4 was asserted → withdrawn; M9 sweep/audit exist, with DP-baseline superiority still untested.
- B1: FIG01/FIG19 elapsed/task-index mix-up → corrected through accuracy_curve and retention_curves.csv; diagonal checks pass for every method.
- B2: last-client prototype overwrite → count-weighted server bank; momentum applies once per round. Global and aggregate scores and leakage now agree in class-incremental streams. The old TAB05 comparison also differs in feature preprocessing, so it is not an isolated bank-only ablation.
- B3: single union-mean replay loss → balanced current and replay means, with unchanged M1 union KD. Gate failures below remain failures; persistent last-task lag confounds retention interpretations on those combinations.
- B4: TAB05 standardized features → raw attacked pipeline, identical configurations and independent per-seed accuracy checks.
- B5: M5 raw-buffer F8 release → removed; only F1 is released, with private buffer IDs still counted in touched.

## PROTOCOL

- Frozen ViT-B/16 (IN21k-pretrained) feature caches; raw features, no standardization.
- One FedAvg round per task; 30 local epochs; default learning rate 0.5. Train-validation gate overrides in fx9_gate.csv take precedence and are never selected on test utility.
- 10 class-incremental tasks × 10 clients, Dirichlet β=0.5; test-split predictions restricted to seen classes.
- Main changed-method LiRA: 1024 shadows × 3 seeds; calibration_frac=0.8, K={0,1,2,3}, E=6, trajectory and last-round ablations. M0 preserves its existing larger five-seed budget; see a1_m0_budget_check.csv.
- Accuracy uses 5 seeds. Main leakage uses 3 seeds except M0 (5). Two thousand bootstrap replicates; accuracy resamples seeds/tasks, leakage seeds/targets; horizon leakage adds task resampling.
- M0 ratio uses a joint seed/task/target draw for both half-lives. Lower-bound ratio intervals describe the lower-bound statistic, not a finite upper confidence bound on the uncensored ratio.
- Non-M0 horizon accuracy CIs retain all 5 accuracy seeds; leakage intervals use the available 3 shadow seeds. Do not claim a 5-seed joint privacy/accuracy covariance for these rows.
- FX1 accountant baseline noise multiplier σ=2, δ=1e-5, W=3; multiplicity counts released records. M9 instead uses actual analytic-Gaussian σ/Δ and records exact single-release eps0_analytic.
- M9 hyperparameters and PCA are selected/fitted on the public ref split; sweep and audit remain unchanged. Accounting ledgers use γ=1 at 10 and 50 tasks.
- M9 cross-unit FX1 numbers are conditional on the corresponding group sensitivity. A U1-calibrated ledger alone does not establish a U2–U4 group guarantee; do not cite those conditional values as unconditional certificates.
- Audited epsilon uses the requested maximum of forward/mirror one-sided Clopper–Pearson bounds over thresholds. These are pointwise empirical bounds on dependent shadow/target observations, not a simultaneous formal DP certificate.
- M2 is a Gaussian feature-replay approximation; M4 is prototypes only; M1/M5 train frozen-feature heads. Full backbone/prompt/LoRA reproductions were not delivered.

## FINAL RESULTS

**FIG01_DECOUPLING — `figs/fig01_decoupling.pdf`; source `results/retention_curves.csv`.** Normalized accuracy and leakage by dataset. Full per-client M4/M8 releases are dashed and constant by construction; global state is distinct. Exact horizon values follow below.

**FIG02_HALFLIFE — `figs/fig02_halflife.pdf`; source `results/retention_curves.csv`.** Compatibility filename for the same horizon scatter as FIG02; it no longer plots half-lives.

**FIG02_RETENTION_AT_HORIZON — `figs/fig02_retention_at_horizon.pdf`; source `results/retention_curves.csv`.** A(6) versus L(6), with intervals on both axes and reference lines y=1 and y=x. It replaces the old portrait forest plot; half-lives remain in TAB04.

**FIG03_DOSE_RESPONSE — `figs/fig03_dose_response.pdf`; source `results/fx9_dose_summary.csv`.** Balanced replay sample-count dose response for M5 and M2 on CIFAR. The utility axis is forgetting (−BWT); curve shape is reported without imposing monotonicity.

**FIG04_SEMANTIC_VS_INDIVIDUAL — `figs/fig04_semantic_vs_individual.pdf`; source `results/fx9_dose_summary.csv`.** Forgetting versus leakage for individual and semantic replay, with dose labels and both intervals. All points come from the same dose CSV as FIG03.

**FIG05_EPS_OF_T — `figs/fig05_eps_of_T.pdf`; source `results/fig05_eps_of_T.csv`.** FX1 record-composition values by privacy unit; solid segments are observed and dashed segments extrapolated. M9 uses actual noise, with both calibration epsilons and the cross-unit sensitivity qualification above.

**FIG08_PARETO — `figs/fig08_pareto.pdf`; source `results/fig08_pareto.csv`.** Preserved M9 utility sweep at gamma=1, with non-private M0/M8 references and utility intervals. Audited epsilon is populated only for the actual CIFAR/U1/epsilon audit cells; unsupported cells are explicitly labelled, not inferred.

**FIG10_M9_AUDIT — `figs/fig10_m9_audit.pdf`; source `results/fig10_m9_audit.csv`.** Preserved empirical ROC versus analytic single-release DP envelope. A weak attack supports an audit result, not proof that a mechanism is private.

**FIG11_SECURE_AGG — `figs/fig11_secure_agg.pdf`; source `results/fx3_views_summary.csv`.** Full, aggregate and global TPR at e=0 and e=6 on CIFAR. F1 bars coincide by construction; M4 aggregate/global coincide after its bank and calibration fixes.

**FIG11_SECURE_AGG_APPENDIX_CUB200 — `figs/fig11_secure_agg_appendix_cub200.pdf`; source `results/fx3_views_summary.csv`.** Same three-view comparison on CUB, retaining M4 aggregate/global equality.

**FIG11_SECURE_AGG_APPENDIX_IMAGENET_R — `figs/fig11_secure_agg_appendix_imagenet_r.pdf`; source `results/fx3_views_summary.csv`.** Same three-view comparison on ImageNet-R, retaining M4 aggregate/global equality.

**FIG13_GRAM_INVERSION — `figs/fig13_gram_inversion.pdf`; source `results/fig13_gram_inversion_summary.csv`.** Median of trial-mean reconstruction cosines with bootstrap intervals, 25 trials per feasible cell. No anisotropy panel or causal anisotropy claim; infeasible large CUB shards are absent.

**FIG16_ROC — `figs/fig16_roc.pdf`; source `results/fig16_roc.csv`.** Log-log ROC from the exact calibrated fixed-K score sidecars, e=0, full view, representative seed 0. This appendix visualization does not replace pooled multi-seed headline estimates.

**FIG17_SEED_VARIANCE — `figs/fig17_seed_variance.pdf`; source `results/fig17_seed_variance.csv`.** Per-seed e=0 full-view TPR. M0 has five seeds; the other methods have three.

**FIG18_NATURAL_FEDERATION — `figs/fig18_natural_federation.pdf`; source `results/fig18_natural_federation.csv`.** Preserved v2 appendix experiment: clients are slide groups within each hospital task versus matched Dirichlet groups. This is not a hospital-as-client federation and cannot settle that design objection.

**FIG19_RETENTION_VS_RELEASE — `figs/fig19_retention_vs_release.pdf`; source `results/retention_curves.csv`.** M4/M8 full-release and global-state leakage alongside accuracy, all from the unified CIFAR curves. Release persistence must not be interpreted as an uncensored accuracy/leakage half-life ratio.

**FIG_A6_PROPERTY_INFERENCE — `figs/fig_a6_property_inference.pdf`; source `results/a6_property_inference.csv`.** Count-based property inference, retained as an explicit released-count disclosure demonstration. It is not evidence of subtle inferred leakage.

**Horizon numbers — `results/fig02_retention_at_horizon.csv` (dataset / method / view: A(6); L(6)).**
- cifar100 / M0 FedAvg / full: 0.230102 [0.174239, 0.273655]; 0.869585 [0.734997, 1.03972].
- cifar100 / M1 GLFC / full: 1.03799 [1.01939, 1.06032]; 2.5916 [2.00163, 3.75341].
- cifar100 / M2 Gaussian replay (semantic) / full: 0.990746 [0.969869, 1.01184]; 1.47626 [1.22954, 1.84766].
- cifar100 / M3 FOT / full: 0.857631 [0.839289, 0.878121]; 1.46599 [1.1718, 1.96093].
- cifar100 / M4 prototypes / aggregate: 0.93615 [0.919926, 0.950994]; 1 [1, 1].
- cifar100 / M4 prototypes / full: 0.93615 [0.919926, 0.950994]; 1 [1, 1].
- cifar100 / M4 prototypes / global: 0.93615 [0.919926, 0.950994]; 1 [1, 1].
- cifar100 / M5 exemplar replay (individual) / full: 0.98599 [0.961894, 1.00422]; 2.59396 [2.01035, 3.52146].
- cifar100 / M8 analytic / aggregate: 0.977618 [0.962708, 0.990752]; 1 [1, 1].
- cifar100 / M8 analytic / full: 0.977618 [0.962708, 0.990752]; 1 [1, 1].
- cifar100 / M8 analytic / global: 0.977618 [0.962708, 0.990752]; 0.948202 [0.925648, 0.967619].
- cub200 / M0 FedAvg / full: 0.330866 [0.289135, 0.379125]; 0.320462 [0.253363, 0.404142].
- cub200 / M1 GLFC / full: 1.48912 [1.31196, 1.7373]; 3.9714 [2.92498, 5.85763].
- cub200 / M2 Gaussian replay (semantic) / full: 1.69503 [1.45083, 2.07094]; 4.85644 [4.12201, 5.87386].
- cub200 / M3 FOT / full: 1.10573 [0.983388, 1.28395]; 1.53961 [1.11452, 1.8952].
- cub200 / M4 prototypes / aggregate: 0.942644 [0.931792, 0.954475]; 1 [1, 1].
- cub200 / M4 prototypes / full: 0.942644 [0.931792, 0.954475]; 1 [1, 1].
- cub200 / M4 prototypes / global: 0.942644 [0.931792, 0.954475]; 1 [1, 1].
- cub200 / M5 exemplar replay (individual) / full: 1.31858 [1.21817, 1.45742]; 4.69204 [4.11175, 5.62604].
- cub200 / M8 analytic / aggregate: 0.957093 [0.943463, 0.969178]; 1 [1, 1].
- cub200 / M8 analytic / full: 0.957093 [0.943463, 0.969178]; 1 [1, 1].
- cub200 / M8 analytic / global: 0.957093 [0.943463, 0.969178]; 1 [1, 1].
- imagenet_r / M0 FedAvg / full: 0.241767 [0.212854, 0.267444]; 0.968084 [0.814498, 1.12463].
- imagenet_r / M1 GLFC / full: 1.22505 [1.14909, 1.31403]; 3.44978 [2.7495, 4.53865].
- imagenet_r / M2 Gaussian replay (semantic) / full: 1.09801 [1.04086, 1.16261]; 2.66561 [2.31575, 3.09246].
- imagenet_r / M3 FOT / full: 0.663176 [0.634221, 0.696557]; 1.16405 [0.86431, 1.43198].
- imagenet_r / M4 prototypes / aggregate: 0.813451 [0.790096, 0.833376]; 1 [1, 1].
- imagenet_r / M4 prototypes / full: 0.813451 [0.790096, 0.833376]; 1 [1, 1].
- imagenet_r / M4 prototypes / global: 0.813451 [0.790096, 0.833376]; 1 [1, 1].
- imagenet_r / M5 exemplar replay (individual) / full: 1.04895 [1.01248, 1.09056]; 3.68995 [3.10428, 4.61352].
- imagenet_r / M8 analytic / aggregate: 0.857577 [0.841328, 0.875427]; 1 [1, 1].
- imagenet_r / M8 analytic / full: 0.857577 [0.841328, 0.875427]; 1 [1, 1].
- imagenet_r / M8 analytic / global: 0.857577 [0.841328, 0.875427]; 0.988339 [0.982363, 0.995589].

**M0 ratio — `results/decoupling_ratio.csv`.**
- cifar100: ratio 6.86834 [6.19464, 7.49258], lower_bound; interval for lower-bound statistic; not an upper bound on true ratio.
- cub200: ratio 0.486871 [0.345576, 1.01581], point; interval for lower-bound statistic; not an upper bound on true ratio.
- imagenet_r: ratio 7.90731 [7.49079, 8.31064], lower_bound; interval for lower-bound statistic; not an upper bound on true ratio.

**Gate comparison — `results/fx9_gate_comparison.csv`: last-task accuracy before → after; BWT before → after.**
- cifar100/m1_glfc: 0.804 → 0.798; -0.0022963 → 0.00340741; gate=False.
- cifar100/m2_target: 0.797667 → 0.838333; -0.002 → -0.0332593; gate=True.
- cifar100/m5_hybrid_replay: 0.822667 → 0.853333; -0.0238148 → -0.0428148; gate=True.
- cub200/m1_glfc: 0.0511975 → 0.12869; 0.25691 → 0.36041; gate=False.
- cub200/m2_target: 0.000564972 → 0.171308; 0.709892 → 0.481172; gate=False.
- cub200/m5_hybrid_replay: 0.102579 → 0.281963; 0.473855 → 0.313723; gate=False.
- imagenet_r/m1_glfc: 0.268298 → 0.381605; 0.0893715 → 0.116557; gate=False.
- imagenet_r/m2_target: 0.202362 → 0.443917; 0.251568 → 0.0976613; gate=False.
- imagenet_r/m5_hybrid_replay: 0.35541 → 0.460657; 0.129747 → 0.0653345; gate=False.

**Dose response — `results/fx9_dose_summary.csv`: sample count; forgetting; TPR@1%FPR at e=6.**
- M2 Gaussian replay (semantic), 1: 0.135556 [0.108608, 0.162503]; 0.0245446 [0.0198782, 0.0292111].
- M2 Gaussian replay (semantic), 2: 0.102407 [0.0906182, 0.114197]; 0.0295567 [0.0228128, 0.0363006].
- M2 Gaussian replay (semantic), 5: 0.073037 [0.0619953, 0.0840788]; 0.0340619 [0.0311379, 0.0369859].
- M2 Gaussian replay (semantic), 10: 0.048963 [0.038311, 0.0596149]; 0.0372511 [0.030296, 0.0442062].
- M2 Gaussian replay (semantic), 20: 0.0332593 [0.0249406, 0.041578]; 0.0425093 [0.0381747, 0.046844].
- M2 Gaussian replay (semantic), 50: 0.0174074 [0.0083773, 0.0264375]; 0.0459488 [0.0401979, 0.0516997].
- M5 exemplar replay (individual), 1: 0.152815 [0.124848, 0.180781]; 0.0375091 [0.028433, 0.0465853].
- M5 exemplar replay (individual), 2: 0.11263 [0.0905927, 0.134667]; 0.0469373 [0.0378557, 0.056019].
- M5 exemplar replay (individual), 5: 0.0712222 [0.0582671, 0.0841773]; 0.0560525 [0.0513764, 0.0607287].
- M5 exemplar replay (individual), 10: 0.0428148 [0.0355, 0.0501297]; 0.0615549 [0.0526308, 0.070479].
- M5 exemplar replay (individual), 20: 0.025 [0.022367, 0.027633]; 0.0752536 [0.0528518, 0.0976554].
- M5 exemplar replay (individual), 50: 0.00907407 [0.00122177, 0.0169264]; 0.0832856 [0.0599698, 0.106601].

**TAB02 — `tables/tab02_units.csv/.tex`.** Privacy units and neighboring relations; definitional table, no empirical interval.
**TAB03 — `tables/tab03_leakage.csv/.tex`.** TPR@1% and @0.1% FPR plus AUC at e=0/e=6, from current fixed-K summaries. Rates are reported with hierarchical intervals; full M4/M8 transcript persistence is structural.
**TAB04 — `tables/tab04_halflife.csv/.tex`.** Accuracy/leakage half-lives and explicit ok/censored/no_signal status. Blank finite CIs for censored half-lives are intentional, not missing estimates.
**TAB05 — `results/tab05_utility_baselines.csv`; means/CIs in `results/fx9_utility_summary.csv`.** Raw attacked configurations; 10 tasks, five seeds. Every final per-seed value agrees with its accuracy matrix within 1e-9.
- cifar100/M0 FedAvg: final average accuracy 0.33352 [0.304452, 0.362588]; BWT -0.667533 [-0.702349, -0.632717].
- cifar100/M1 GLFC: final average accuracy 0.84266 [0.840457, 0.844863]; BWT 0.00351111 [0.00111636, 0.00590587].
- cifar100/M2 Gaussian replay (semantic): final average accuracy 0.83604 [0.831327, 0.840753]; BWT -0.0330444 [-0.0374556, -0.0286333].
- cifar100/M3 FOT: final average accuracy 0.7479 [0.732793, 0.763007]; BWT -0.176133 [-0.196518, -0.155748].
- cifar100/M4 prototypes: final average accuracy 0.7696 [0.7696, 0.7696]; BWT -0.0664667 [-0.0706289, -0.0623044].
- cifar100/M5 exemplar replay (individual): final average accuracy 0.84194 [0.834237, 0.849643]; BWT -0.0397778 [-0.0462082, -0.0333473].
- cifar100/M8 analytic: final average accuracy 0.8833 [0.8833, 0.8833]; BWT -0.0450222 [-0.0482861, -0.0417583].
- cub200/M0 FedAvg: final average accuracy 0.379042 [0.32367, 0.434414]; BWT -0.548906 [-0.619907, -0.477906].
- cub200/M1 GLFC: final average accuracy 0.691419 [0.680702, 0.702137]; BWT 0.352311 [0.319599, 0.385023].
- cub200/M2 Gaussian replay (semantic): final average accuracy 0.789079 [0.781851, 0.796307]; BWT 0.486105 [0.474131, 0.498079].
- cub200/M3 FOT: final average accuracy 0.673421 [0.661438, 0.685404]; BWT -0.0236682 [-0.105397, 0.0580604].
- cub200/M4 prototypes: final average accuracy 0.857103 [0.856895, 0.857311]; BWT -0.0547953 [-0.0604606, -0.04913].
- cub200/M5 exemplar replay (individual): final average accuracy 0.779675 [0.763913, 0.795438]; BWT 0.320529 [0.291349, 0.349709].
- cub200/M8 analytic: final average accuracy 0.871442 [0.87117, 0.871715]; BWT -0.0481755 [-0.0527972, -0.0435538].
- imagenet_r/M0 FedAvg: final average accuracy 0.249303 [0.223227, 0.275379]; BWT -0.528166 [-0.564722, -0.49161].
- imagenet_r/M1 GLFC: final average accuracy 0.597123 [0.593683, 0.600563]; BWT 0.122256 [0.108201, 0.136311].
- imagenet_r/M2 Gaussian replay (semantic): final average accuracy 0.61481 [0.609943, 0.619676]; BWT 0.0966424 [0.0926111, 0.100674].
- imagenet_r/M3 FOT: final average accuracy 0.457132 [0.442687, 0.471577]; BWT -0.259055 [-0.287376, -0.230734].
- imagenet_r/M4 prototypes: final average accuracy 0.514445 [0.512362, 0.516528]; BWT -0.0874197 [-0.0934704, -0.081369].
- imagenet_r/M5 exemplar replay (individual): final average accuracy 0.631284 [0.621146, 0.641423]; BWT 0.0664861 [0.0607498, 0.0722223].
- imagenet_r/M8 analytic: final average accuracy 0.634992 [0.634338, 0.635646]; BWT -0.0889096 [-0.0935653, -0.084254].
**TAB07 — `results/tab07_reproduction_gap.csv`.** Updated raw-feature utility versus published protocols. Backbone, federation and algorithm differences prevent treating these gaps as faithful reproduction rankings.
**TAB08 — `tables/tab08_dp_utility.csv/.tex`.** Preserved M9 utility with eight FX1 accounting columns at T=10/50. The columns follow the explicit sensitivity qualification above; infinity denotes the non-private mechanism.
**M9 accounting — `results/m9_certified.csv`: calibration epsilon / T / unit: FX1 epsilon (deterministic, no sampling CI).**
- 1.0 / 10 / U1: 1.32426; exact single-release epsilon=1.0.
- 1.0 / 10 / U2: 1.32426; exact single-release epsilon=1.0.
- 1.0 / 10 / U3: 2.35699; exact single-release epsilon=1.0.
- 1.0 / 10 / U4: 4.42674; exact single-release epsilon=1.0.
- 1.0 / 50 / U1: 1.32426; exact single-release epsilon=1.0.
- 1.0 / 50 / U2: 1.32426; exact single-release epsilon=1.0.
- 1.0 / 50 / U3: 2.35699; exact single-release epsilon=1.0.
- 1.0 / 50 / U4: 10.8915; exact single-release epsilon=1.0.
- 4.0 / 10 / U1: 4.86608; exact single-release epsilon=4.0.
- 4.0 / 10 / U2: 4.86608; exact single-release epsilon=4.0.
- 4.0 / 10 / U3: 8.97062; exact single-release epsilon=4.0.
- 4.0 / 10 / U4: 18.3127; exact single-release epsilon=4.0.
- 4.0 / 50 / U1: 4.86608; exact single-release epsilon=4.0.
- 4.0 / 50 / U2: 4.86608; exact single-release epsilon=4.0.
- 4.0 / 50 / U3: 8.97062; exact single-release epsilon=4.0.
- 4.0 / 50 / U4: 52.771; exact single-release epsilon=4.0.
- CIFAR/U1 epsilon=1.0: empirical epsilon_lb=0; TPR@1%FPR=0.009547 (upper endpoint 0.0104263).
- CIFAR/U1 epsilon=4.0: empirical epsilon_lb=0; TPR@1%FPR=0.00870412 (upper endpoint 0.00954563).
- CIFAR/U1 epsilon=inf: empirical epsilon_lb=9.06368; TPR@1%FPR=0.871143 (upper endpoint 0.874032).
**TAB10 — `tables/tab10_taxonomy.csv/.tex`.** Taxonomy and proof-scope table; definitions rather than new measurements.

**FIG13 selected cells — `results/fig13_gram_inversion_summary.csv`.**
- cifar100, n=1, ref=clean_full: median cosine 1 [1, 1], trials=25.
- cifar100, n=4, ref=clean_full: median cosine 0.600158 [0.507064, 0.667171], trials=25.
- cifar100, n=16, ref=clean_full: median cosine 0.463877 [0.447378, 0.519931], trials=25.
- cifar100, n=64, ref=clean_full: median cosine 0.469909 [0.426691, 0.482513], trials=25.
- cub200, n=1, ref=clean_full: median cosine 1 [1, 1], trials=25.
- cub200, n=4, ref=clean_full: median cosine 0.761096 [0.733299, 0.821914], trials=25.
- cub200, n=16, ref=clean_full: median cosine 0.676209 [0.632925, 0.696079], trials=25.
- FIG18 natural, e=0: TPR@1%FPR 0.0214851 [0.0116117, 0.0313585]; normalized accuracy 1; clients=5.
- FIG18 natural, e=4: TPR@1%FPR 0.0343842 [0.0217884, 0.04698]; normalized accuracy 1.04683; clients=5.
- FIG18 dirichlet, e=0: TPR@1%FPR 0.0230728 [0.0110203, 0.0351253]; normalized accuracy 1; clients=5.
- FIG18 dirichlet, e=4: TPR@1%FPR 0.048681 [0.0180992, 0.0792629]; normalized accuracy 0.988107; clients=5.

**FIG17 representative seed values — `results/fig17_seed_variance.csv`, CIFAR e=0 full-view TPR@1%FPR.** The strip plot shows individual seeds; pooled intervals are in TAB03.
- M0 FedAvg: seed 0=0.0288345; seed 1=0.0303471; seed 2=0.0281835; seed 3=0.0276624; seed 4=0.0236189.
- M1 GLFC: seed 0=0.02725; seed 1=0.0275046; seed 2=0.0259257.
- M2 Gaussian replay (semantic): seed 0=0.0316413; seed 1=0.0345618; seed 2=0.0286218.
- M3 FOT: seed 0=0.0244764; seed 1=0.0320227; seed 2=0.028913.
- M4 prototypes: seed 0=0.324499; seed 1=0.361238; seed 2=0.342092.
- M5 exemplar replay (individual): seed 0=0.0316245; seed 1=0.0301243; seed 2=0.0306605.
- M8 analytic: seed 0=1; seed 1=1; seed 2=1.
**A6 — `results/a6_property_inference.csv`.** Released counts directly disclose the property; this is a disclosure check, with no multi-seed uncertainty estimate.
- elapsed=-1: balanced accuracy 0.5.
- elapsed=0: balanced accuracy 1.
- elapsed=6: balanced accuracy 1.

## CUT

- DP-FedAvg / DP linear-probe baselines: not delivered; no equal-epsilon superiority claim. H8 is CUT.
- FIG09 / privacy-induced participation drift: cut from this repair schedule to prioritize the mandatory corrected experiments; no deciding simulation.
- 20-task TAB05 and Camelyon TAB05 rows: removed from the current attacked-protocol table. FIG18 appendix retains its own separately scoped accuracy.
- FIG18 hospital-as-client design: not delivered; v2 uses slide groups within hospital tasks on the 5k-patch subsample.
- M6/M7 prompt/LoRA methods and M4 LoRA half: GPU reproduction cuts; main evidence covers seven CPU methods plus the separately scoped M9.
- Full-trajectory empirical M9 epsilon certification: not claimed by the one-release audit. The threshold maximum is not a simultaneous calibrated certificate.
- Unconditional U2–U4 certificates from U1-calibrated M9: not established by the unchanged FX1 composition calculation; sensitivity matching is required.
- Unobserved/undefined retention ratios outside M0: omitted; no infinity-as-point-estimate or exponential extrapolation.
- FIG06/FIG07/FIG12/FIG14/FIG15 and TAB01/TAB06/TAB09: no separate final artifact in this delivery; related evidence is scoped to the listed existing figures/tables.
- C2 universality / superiority to original trainable-backbone methods: not established by these feature-space experiments.
