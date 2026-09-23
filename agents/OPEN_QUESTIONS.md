# Open questions register

**This file is the project's state.** Debates that do not update it did not happen. Only the human
(Mohaimanul) may set a claim to `CONFIRMED`. Refuted claims are never deleted — the record of what was
tried and failed is as valuable as the record of what worked.

Statuses: `OPEN` · `SUPPORTED` (survived attack, deciding test named, not yet run) · `CONFIRMED`
(test ran, human-set) · `REFUTED` · `SPECULATIVE` (unfalsifiable as stated) · `PARKED`

Last updated: 2026-08-25 (initial draft — nothing has been debated yet; one toy-scale observation
logged against H5, see `notes/2026-08-25_preliminary.md`)

---

## Hypotheses

### H1 — Vacuity of existing lifelong claims
*For every published FCL method, standard DP-FL accounting under unit U4 (client-level, unbounded)
gives ε growing without bound in T; hence no published FCL method makes a non-trivial lifelong
client-level privacy claim.*
- Status: `OPEN`
- Deciding test: paper exercise — enumerate the release schedule of M0–M8 and account each. No GPU.
- Owner: DP Theorist · Blast radius: **High** (this is Paper B's motivating claim)
- Notes: likely easy to establish; the risk is that it is *too* easy and reads as trivial. Ask
  Reviewer Sim persona 2 early.

### H2 — Retention–leakage trade-off  ⭐ FLAGSHIP
*Across FCL methods, anti-forgetting strength is positively associated with cross-task membership
leakage: `−BWT` on task 1 at time T correlates with TPR@1%FPR for MIA on task-1 data from round-T
artifacts. Target: Spearman ρ ≥ 0.5, p < 0.05, ≥8 methods × ≥3 datasets.*
- Status: `OPEN`
- Deciding test: the full audit (§5). **Empiricist's proposed upgrade: within-method dose–response
  over ~6 retention-strength levels on 2–3 methods** — adopt this; it is stronger and cheaper.
- **Real A1 numbers now exist for 7/8+ methods on CIFAR-100** (`notes/2026-09-16_p3_a1_lira_results.md`,
  `notes/2026-09-16_p3_a1_m0_h3.md`, `notes/2026-09-16_p3_a1_dose_response.md`), and the cross-method
  shape is exactly H2's predicted pattern:

  | Method | Retention mechanism | AUC(e=0) | AUC(peak) | AUC(e=9) | Shape |
  |---|---|---|---|---|---|
  | M0 FedAvgSequential | none | 0.560 | 0.573 | 0.555 | rises briefly, then **decays** |
  | M5 HybridReplay | exemplar replay | 0.566 | 0.581 | 0.567 | rises, flat/mild decline |
  | M2 TARGET | generative replay | 0.567 | 0.600 | 0.595 | rises, plateaus high |
  | M1 GLFC | distillation + exemplar | 0.523 | 0.548 | 0.548 | rises, plateaus |
  | M3 FOT | orthogonal subspace projection | 0.592 | 0.612 | 0.603 | rises fast, flat/high |
  | M4 PrototypeFCL | class-mean carry-forward | 0.883 | — | 0.869 | flat from the start, high |
  | M8 AnalyticFCL | exact running Gram sum | 1.000 | — | 1.000 | flat, saturated |

  **This is a real, quantitative, single-dataset preview of the exact H2 relationship** — the only
  method with zero anti-forgetting mechanism is the only one whose leakage curve turns over and decays;
  every method with some retention mechanism holds or grows its leakage instead. Not yet the deciding
  test (needs ≥8 methods × ≥3 datasets and an actual −BWT-vs-TPR Spearman ρ with CI), so status stays
  `OPEN` — but this is strong motivating evidence, obtained essentially for free by extending A1 to
  methods P2 had already built and validated.
- **P4/claim C1's decoupling ratio is now real, quantitative, and gate-passing (2026-09-17)** —
  distinct from H2's own correlation criterion, but the same underlying thesis, formalized as
  `h_leak / h_acc` (half-life ratio) rather than a cross-method correlation. Real 5-seed run, CIFAR-100,
  M0 vs M8 (`results/fig01_decoupling.csv`, `fig02_halflife.csv`, `decoupling_ratio.csv`,
  `figs/fig01_decoupling.pdf`, `figs/fig02_halflife.pdf`, `notes/2026-09-17_p4_fig01_decoupling.md`):

  | Method | Accuracy half-life | Leakage half-life | Decoupling ratio |
  |---|---|---|---|
  | M0 (no retention) | 4.68 [3.24, 5.78] | 17.58 [12.00, 33.95] | **3.75 [2.60, 6.87]** |
  | M8 (exact retention) | 53.42 [46.55, 59.65] | no decay detected | **infinite** |

  **Gate P4's literal criterion is met**: M0's decoupling-ratio CI excludes 1.0 entirely (a real,
  paired bootstrap over seeds, not a forced fit). M8's leakage shows zero decay over the full observed
  horizon even as its accuracy decay, though very slow, is real and well-fit (R²=0.999) — reported
  honestly as censored/lower-bound-only, per `00_BUILD_PLAN.md`'s explicit instruction, not forced to
  a fabricated number.
- **Extended to all 7 A1-validated methods, same dataset/seeds (2026-09-18,
  `notes/2026-09-18_p4_fig01_all_methods.md`) — decoupling holds for 6/7**:

  | Method | Decoupling ratio (h_leak / h_acc) | 95% CI | Excludes 1.0? |
  |---|---|---|---|
  | M0 (none) | 3.75 | [2.60, 6.87] | yes |
  | M1 (distillation+exemplar) | ∞ (no leak decay) | — | yes (trivially) |
  | M2 (generative replay) | ∞ (no leak decay) | — | yes (trivially) |
  | M3 (orthogonal projection) | 2.80 | [0.72, 26.12] | **no** |
  | M4 (class-mean carry-forward) | 3.71 | [1.81, 176.57] | yes (wide) |
  | M5 (exemplar replay) | ∞ (no leak decay) | — | yes (trivially) |
  | M8 (exact Gram sum) | ∞ (no leak decay) | — | yes (trivially) |

  **Only M3 is inconclusive** — point estimate still >1 but the 5-seed CI is wide enough to include
  1.0; reported as "directionally consistent, not yet significant," not forced either way. **A real
  complication found and reported precisely, not hidden**: M1's accuracy curve isn't monotonic (rises
  to 2.46x its initial value by elapsed=6 before declining — real positive backward transfer,
  consistent with M1's own BWT=+0.27 to +0.37, plausibly from active distillation-driven consolidation,
  not just forgetting-prevention), so its exponential-half-life fit is a poor description (R²=0.043) —
  its `ratio=inf` finding still stands (driven by the leakage side being censored, independent of the
  shaky accuracy-side number), but the specific accuracy-half-life value for M1 should not be
  reported as reliable.
- **Extended to a second dataset, CUB-200, same 7 methods/5 seeds (2026-09-19,
  `notes/2026-09-19_p4_second_dataset_cub200.md`) — decoupling holds on both datasets for 6/7 methods,
  with one real, honestly-reported counter-example**:

  | Method | CIFAR-100 ratio | Excludes 1.0? | CUB-200 ratio | Excludes 1.0? |
  |---|---|---|---|---|
  | M0 | 3.75 [2.55, 6.89] | yes | 1.53 [1.05, 2.10] | yes (barely) |
  | M1 | ∞ | yes | ∞ | yes |
  | M2 | ∞ | yes | ∞ | yes |
  | M3 | 2.80 [0.74, 18.71] | no | **~0 (reversed)** | n/a |
  | M4 | 3.71 [1.81, 34.71] | yes | 134.49 [47.15, 2034.32] | yes |
  | M5 | ∞ | yes | ∞ | yes |
  | M8 | ∞ | yes | ∞ | yes |

  **M4's decoupling is far more extreme on CUB-200** (ratio jumps from 3.71 to 134) — consistent with
  H5's already-established finding that CUB-200's small-per-class-n regime amplifies leakage.
  **M3 on CUB-200 is a genuine counter-example, not noise**: its accuracy shows *no detectable decay
  at all* (censored) while its leakage *does* decay (halflife≈11, R²=0.446) — the reverse of every
  other method/dataset pairing. Reported precisely per CLAUDE.md non-negotiable #7 (negative results
  are results); does not undermine the 6/7 pattern but should not be smoothed over in the paper.
  **Two real bugs found and fixed while assembling this**: `build_fig02.py`'s ratio logic crashed on
  the acc-censored/leak-finite case (never seen before M3/CUB-200 — fixed to report ratio→0 for this
  case explicitly); `fig01_decoupling.py` grouped plot lines by method name alone, so with 2 datasets
  the same method's two datasets' worth of points got connected into one wrong zigzagging line —
  caught by inspecting the actual rendered figure before trusting it, fixed with a proper
  dataset-column subplot grid.
- **Extended to a third dataset, ImageNet-R, same 7 methods/5 seeds (2026-09-21,
  `notes/2026-09-21_p4_third_dataset_imagenet_r.md`) — H2's own ≥3-dataset bar is now met.
  ImageNet-R gives the cleanest result of the three: decoupling holds for 7/7 methods, not 6/7**:

  | Method | CIFAR-100 ratio | CUB-200 ratio | ImageNet-R ratio | Excludes 1.0 on ImageNet-R? |
  |---|---|---|---|---|
  | M0 | 3.75 [2.55, 6.89] | 1.53 [1.05, 2.10] | **109.34 [3.70, 161.91]** | yes |
  | M1 | ∞ | ∞ | ∞ | yes |
  | M2 | ∞ | ∞ | ∞ | yes |
  | M3 | 2.80 [0.74, 18.71] (incl. 1) | **~0 (reversed)** | **∞** | yes |
  | M4 | 3.71 [1.81, 34.71] | 134.49 [47.15, 2034.32] | **∞** | yes |
  | M5 | ∞ | ∞ | ∞ | yes |
  | M8 | ∞ | ∞ | ∞ | yes |

  **M3 now has three genuinely different pictures across three datasets** (inconclusive on
  CIFAR-100, reversed on CUB-200, infinite-ratio on ImageNet-R) — report as an honest, unresolved,
  dataset-dependent complication specific to this method, not a single narrative. M0's ImageNet-R
  ratio has a very shallow leakage half-life fit (R²=0.007, extrapolated well beyond the observed
  9-task horizon) — real, converged, but should carry that caveat when quoted. A real operational
  finding: `run_lira.py` was 60-100x slower per (method,seed) combo on ImageNet-R's shadow store than
  on CIFAR-100/CUB-200 for identically-shaped data (root cause not fully confirmed — likely
  compression-ratio-dependent `np.savez_compressed` decode cost — worked around by parallelizing the
  35 combos 12-at-a-time on the login node rather than running sequentially).
- **CORRECTION (2026-09-23, `08_FIX_PLAN.md` FX2): every decoupling-ratio number above this line is
  built on an invalid half-life extrapolation and should not be cited.** The post-review audit (R1-R12)
  that produced `08_FIX_PLAN.md` found the pre-fix half-life fit extrapolated an exponential curve far
  beyond the observed horizon whenever leakage hadn't visibly decayed yet, and reported the result as
  literal `∞` — exactly the practice Definition 10 (the plan's replacement) explicitly forbids: "the
  first e ≥ 1 at which the normalised value is ≤ 1/2... censored at E=6 if no crossing occurs, reported
  as a lower bound `> 6`, never extrapolated, never reported as infinite." Every `∞` cell in the tables
  above is a symptom of that bug, not a real finding.
  **Rebuilt on wave V2 (post-FX4 fixed methods, real ledgers) with Definition 10's non-parametric
  first-crossing + hierarchical bootstrap (2000 replicates, `code/scripts/build_fx2_summary.py` /
  `build_fx2_accuracy_summary.py` / `build_fx2_decoupling_ratio.py`), fixed-k population K={0,1,2,3},
  E_MAX=6, all 3 datasets complete — **the honest result is much less clean than the pre-fix tables
  suggested**:

  | Method (cifar100) | h_acc | h_leak (tpr1, trajectory) | ratio_type |
  |---|---|---|---|
  | M0 (none) | 0.87 [ok] | censored (>6) | **lower_bound, rho >= 6.9** |
  | M1, M2, M3, M5 | censored (>6) | censored (>6), except M1/cub200 h_leak=4.47 [ok] | **undefined** |
  | M4 (full view) | censored (>6) | censored (>6) | **by_construction** (per-client full view is constant by construction, FX4h) |
  | M4 (aggregate/global) | censored (>6) | censored (>6) | **undefined** |
  | M8 (full view) | censored (>6) | censored (>6) | **by_construction** |
  | M8 (aggregate/global) | censored (>6) | censored (>6) | **undefined** |

  **M0 across all 3 datasets is itself not one consistent story**:

  | dataset | h_acc | h_leak | ratio | ratio_type |
  |---|---|---|---|---|
  | cifar100 | 0.87 [ok] | censored (>6) | >= 6.87 | lower_bound |
  | cub200 | 2.22 [ok] | 1.08 [ok] | **0.49** | **point (reversed!)** |
  | imagenet_r | 0.76 [ok] | censored (>6) | >= 7.91 | lower_bound |

  On CUB-200, M0's leakage decays *faster* than its accuracy (both are real, non-censored crossings,
  checked against the raw per-e curves) — the opposite direction from the other two datasets. This
  echoes the pre-fix M3/CUB-200 reversal already on record just below ("the reverse of every other
  method/dataset pairing") — a second, independent reversal on the same dataset is worth treating as
  a real pattern (CUB-200's small-per-class-n, fine-grained regime), not a coincidence.

  **Why**: every actual retention method's accuracy is now SO effective at resisting forgetting
  (post-FX4 fixes) that its accuracy half-life doesn't cross within the E=6 observed horizon either —
  not just the leakage side. A ratio needs BOTH sides to cross to be well-defined (or the numerator
  alone, for a `lower_bound`); with the denominator also censored, `decoupling_ratio()` correctly
  reports `undefined` rather than fabricating a number, per its own explicit contract
  (`code/src/p3fcl/halflife.py`). **This is not a weaker version of the same finding — it is a
  different finding.** M0 (no retention) now has the ONE well-defined, non-extrapolated decoupling
  signal in the whole table (leakage persists past the point where accuracy has already halved, by at
  least a factor of ~6.9) — arguably the cleanest possible statement of the paper's thesis (retention
  keeps leakage alive after forgetting would have erased it), but it is a statement about the BASELINE,
  not about the retention methods the paper is actually about. For the retention methods themselves,
  the honest post-fix statement is "leakage persists for at least as long as our 6-task observation
  window, same as accuracy does" — real, but not the dramatic 3.71x-134x multiplier the pre-fix numbers
  claimed.
  **What would resolve this**: a longer observation horizon (bigger `E_MAX`, which needs a bigger
  `K_SET`/`n_tasks` than this phase's fixed 10-task streams support) would let a genuinely slower
  accuracy decay eventually cross 50%, at which point the ratio would become defined again. Out of
  scope for this fix phase (`E_MAX=6` and `n_tasks=10` are `08_FIX_PLAN.md`'s own fixed design, not
  something to change mid-phase); flagged here for whoever scopes a follow-up.
  **Status stays `OPEN`** (this correction neither confirms nor refutes H2 — it corrects the
  evidentiary record and narrows what can currently be claimed). Full data:
  `results/decoupling_ratio.csv`, `results/fig02_halflife.csv`; writeup:
  `notes/2026-09-23_fx2_decoupling_ratio_post_fix.md`.
- **A small, explicitly-scoped-down within-method dose-response pilot ran 2026-09-21** (under a hard
  external deadline that ruled out the full ≥6-level x ≥3-method x ≥2-dataset sweep this entry's
  "Empiricist's proposed upgrade" line calls for): M5 (HybridReplay) on CIFAR-100, 3 levels of its
  `buffer_size_per_class` knob (0, 10, 20), 3 seeds, reduced 1,024-shadow budget
  (`results/fig03_dose_response_pilot.csv`, `figs/fig03_dose_response_pilot.pdf`,
  `notes/2026-09-21_p5_dose_response_pilot.md`). **Real result — a clean, monotonic dose-response with
  no crossovers across any seed at any level**:

  | `buffer_size_per_class` | mean -BWT (retention) | mean TPR@1%FPR (leakage) |
  |---|---|---|
  | 0  | -0.671 | 0.024 |
  | 10 | -0.047 | 0.043 |
  | 20 | -0.009 | 0.049 |

  This is real, motivating evidence for H2's causal direction *within one method* — but status stayed
  `OPEN`, not `SUPPORTED`, because it was 1 method x 1 dataset x 3 levels. **Superseded by the FX8
  rerun below — kept here as an honest historical record, not to be cited over the real sweep.**
- **CORRECTION (2026-09-23, `08_FIX_PLAN.md` FX8) — the real dose-response sweep**: CIFAR-100, 2
  methods x 6 levels {1,2,5,10,20,50} x 3 seeds x 512 shadows/level (18,432 shadows), leakage pooled
  over K=[0,1,2,3] at elapsed=6 (`results/fig03_dose_response.csv`, `figs/fig03_dose_response.pdf`,
  `notes/2026-09-23_fx8_dose_response_result.md`). **Retention (-BWT) rises identically for both
  methods as the knob increases** (both ~0.25 at level=1 down to ~-0.05 at level=50 — matched by
  construction). **Leakage does NOT rise identically**: M5 (individual/exemplar, F8) climbs steeply,
  TPR@1%FPR ~0.03 -> ~0.09 (mean; individual seeds over 0.10-0.15) across the same range; M2
  (semantic/Gaussian, F6) stays much flatter, ~0.03 -> ~0.045, turning back down slightly at the
  highest level. At the top retention-strength level, M5's leakage is roughly double M2's despite
  matched retention benefit. This is real, quantified, CI-supported evidence for H2's causal claim
  *and* directly answers the semantic/individual objection below in the same sweep. Still **short of
  H2's own ≥8-methods x ≥3-datasets bar** (this clears the "upgrade" line's ≥6-levels-on-2-3-methods
  bar, not H2's full deciding test) — **status stays `OPEN`**, not `SUPPORTED`. Do not cite the
  2026-09-21 pilot's numbers above over this rerun.
- **M3's outsized signal vs. M1/M2's more modest ones is itself relevant to the Red Team's semantic-
  vs-individual objection below** — subspace projection (M3) plausibly protects individual gradient
  directions more directly than exemplar/generative *summaries* (M1/M2) do. Worth a direct comparison
  when the formal dose-response sweep is scoped.
- Owner: Empiricist · Blast radius: **Very high** (Paper A's framing; §7.2 is the fallback)
- Known strongest objection (Red Team): retention is semantic, membership is individual; a method can
  retain class means perfectly and leak nothing example-specific. A preliminary answer from
  2026-09-21 used M4 (semantic/prototype) vs. M5 (individual/exemplar)
  (`notes/2026-09-21_p5_dose_response_pilot_m4_semantic_arm.md`) and found M4 completely flat on both
  axes — traced to a confirmed mechanistic cause: M4's momentum-blending branch in `m4_proto.py` only
  fires when a `(client, class)` key recurs across tasks, and `streams.py::build_stream`'s default
  class-incremental split means every class is assigned to exactly one task, so that branch is
  provably dead code under every dataset this project uses. **`08_FIX_PLAN.md` §10 dropped the M4 arm
  for exactly this reason ("its knob was dead code") and substituted M2 (semantic/Gaussian replay) as
  the semantic-retention arm instead — see the FX8 correction above for the real result**: M2's
  leakage rises much less steeply than M5's individual-retention leakage across the same 6-level
  sweep, real positive evidence for the semantic/individual split the objection predicts. Also caught
  while wiring this: `m2_target.py`'s own `MethodSpec.retention_type` was mislabeled `"individual"`
  (never previously exercised by any FIG04 pipeline) — fixed to `"semantic"`, matching this framing
  and the mechanism's actual per-class-distributional-summary nature. Still **not the full sweep** the
  objection ultimately needs (≥3 methods x ≥2 datasets to be conclusive) — but this is real,
  quantified, mechanistically-grounded evidence, not an "unanswered" gap.
- Note: the project must be designed so H2 is upside, not load-bearing. Verify that in Round 1.

### H3 — Accumulation leakage
*TPR@1%FPR under `V_full` substantially exceeds `V_final`, growing with the number of observed rounds,
at least ~Θ(√(#observations)) in the low-signal regime.*
- Status: `SUPPORTED` (qualitative claim; the specific `Θ(√(#observations))` growth rate is not — see
  below) — 2026-09-16
- Deciding test: attack A1 with observation set as the ablation axis.
- **A1 on M4 (F2) + M8 (F5) gave a real but degenerate null first** (both release a target's evidence
  in exactly one round, so `trajectory`/`last_round` are algebraically identical — not a bug, a scope
  gap, `results/a1_lira_cifar100_{m4_proto,m8_analytic}.csv`, `notes/2026-09-16_p3_a1_lira_results.md`).
- **Extended A1 to M0 (F1, FedAvg-sequential) specifically to get a real test — and it's positive.**
  `_reconstruct_running_w` (`shadow_runner.py`) replays M0's own FedAvg aggregation from ledger-visible
  `MODEL_DELTA` records alone, so a target's logit-margin score genuinely varies round to round. Real
  run, 4,096 shadows, CIFAR-100 (`results/a1_lira_cifar100_m0_fedavg.csv`,
  `notes/2026-09-16_p3_a1_m0_h3.md`): the `trajectory` vs `last_round` AUC gap opens immediately
  (elapsed=1: 0.570 vs 0.539) and **stays open** at a roughly flat ~0.03-0.04 AUC advantage out to
  elapsed=9 (well-powered, n≈20k-205k per point) — real, positive evidence that the full transcript
  retains more membership signal than the latest checkpoint alone. **The advantage plateaus rather than
  growing** like the stated `Θ(√(#observations))` — that stronger quantitative form is not confirmed at
  this horizon (only 10 tasks) and should not be cited as established; the qualitative "transcript >
  checkpoint" claim is.
- **Unplanned bonus, directly relevant to H2/C1**: both arms *decay* toward chance as elapsed grows for
  M0 (the no-retention baseline) — the opposite of M4/M8's completely flat curves. A method with no
  anti-forgetting mechanism shows decaying individual-level leakage; methods with retention mechanisms
  don't. A real, cheap preview of the H2 correlation using an actual calibrated attack, not just A6's
  direct statistic read.
- Owner: Empiricist · Blast radius: Medium
- Note: partially precedented (CS-MIA used confidence series within a task); differentiation is the
  cross-task, cross-artifact-family extension. Librarian to draft the differentiation sentence.

### H4 — Task-onset inference
*Task onset localizable to ±2 rounds with precision > 0.8 for prototype- and statistics-based methods,
with no auxiliary data.*
- Status: `SUPPORTED` (F1/gradient-based methods only — 2026-09-17; **not** confirmed, and not
  mechanistically expected, for the prototype-/statistics-based methods H4 as originally worded
  predicted)
- Deciding test: attack A4 (CUSUM change-point detection on the aggregate, secure-agg-visible update
  norm) vs. random and vs. an update-norm-spike heuristic baseline —
  `src/p3fcl/attacks/onset_inference.py`, `code/scripts/run_onset_inference.py`,
  `results/a4_onset_inference.csv`, `notes/2026-09-17_p3_a4_onset_h4.md`. Real run, CIFAR-100, a
  deliberately skewed stream (`n_clients=50, beta=0.02` — verified first that the standard
  `n_clients=10, beta=0.5` config gives zero genuine arrival events to detect), 20 calibration + 20
  evaluation seeds, tolerance ±2 rounds.
- **Real result, client-arrival onset**: M0 (F1/gradient delta) — CUSUM precision 0.716 [0.593, 0.820]
  (CI upper bound just touches the stated >0.8 bar; point estimate falls short), recall 0.981, clearly
  above both baselines (norm_spike 0.675, random 0.625). M4 (F2/prototype) and M8 (F5/Gram) — CUSUM
  barely separates from random (0.591/0.601 vs 0.550/0.588, within CI overlap) despite perfect recall
  (achieved only via a much larger, low-precision flag budget).
- **A real mechanistic correction to H4's original framing, not a failure to find the effect**: F1's
  released quantity is a gradient, whose magnitude genuinely reflects how unfamiliar new data is
  (bigger loss on brand-new classes -> bigger gradient -> a real, detectable norm spike when a new
  client's fresh contribution enters the aggregate sum). F2/F5's released quantities are closed-form
  statistics (a mean, a Gram matrix) with no such "surprise" signature — a new client's contribution is
  the same *kind* of object as an existing one's. The norm-of-update-spike signal this attack (and its
  baseline) depends on is a property of gradient-based releases specifically, not of prototype-/
  statistics-based ones as H4 originally guessed.
- **A real, general bug found and fixed along the way**: `Ledger.aggregate_view()` crashed with
  `KeyError` on any per-class dict payload (F2's prototype) where clients hold different class subsets
  — common at low beta / high client count, first exercised by this attack's deliberately skewed
  stream. Fixed (union of keys, sum only over records that have each key); checked no earlier reported
  number was silently affected (none used `aggregate_view()` under skewed-key conditions before this).
- Owner: Threat Modeler + Empiricist · Blast radius: Medium-high (best real-harm story in the project)

### H5 — Analytic Gram inversion  ⭐ CHEAPEST DECISIVE TEST
*For analytic FCL with class-conditional Gram statistics and n ≤ 64 per class on frozen ViT-B/16,
feature reconstruction reaches cosine similarity > 0.8 for a majority of samples; image-space
reconstructions are visually identifiable.*
- Status: `SUPPORTED` (re-specified curve — 2026-09-16; **original numeric threshold NOT met**)
- Deciding test: run — `code/scripts/run_gram_inversion.py` on real CUB-200 + CIFAR-100 features
  (both `vit_base_patch16_224.augreg_in21k`), 195 rows, `results/fig13_gram_inversion.csv` +
  `results/fig13_anisotropy.csv`. Full writeup: `notes/2026-09-16_p3_fig13_h5.md`.
- **Real result**: n=1 is exact everywhere (cosine 1.000, confirms the algebra on real features, not
  just synthetic). Beyond n=1, cosine drops fast and plateaus around 0.4-0.6 (CIFAR-100) / 0.65-0.70
  (CUB-200) out to the largest n tested (up to 128 where the dataset has enough per-class samples) —
  a real, non-chance, non-trivial leakage signal that **never reaches the originally-hypothesized
  >0.8 bar beyond n~2-4**. CUB-200 (the fine-grained, small-per-class-n dataset) retains more than
  CIFAR-100 at matched n, exactly as `02_DATASETS.md` predicted before any data existed. Measured
  anisotropy: CIFAR-100 86.4, CUB-200 34.0 (both far above an isotropic ~1.0 baseline, confirming
  real ViT features are strongly anisotropic — but that alone isn't enough to push reference-based
  reconstruction above 0.8 at realistic n).
- **Honest framing**: the original ">0.8 for a majority of samples, n<=64" claim is refuted as
  literally stated. The re-specified claim (RESEARCH_PLAN's own register note: "the curve is the
  deliverable regardless of where it lands") is fully answered — F5 leaks a real, measurable,
  bounded amount at every practical n, well short of near-perfect reconstruction. This is the finding,
  not a failure to find one.
- **Caveat**: only 5 trials/cell. The n-dependence above is robust at that count; the
  reference-quality (noise/pool-size) axis specifically shows some non-monotonic cells likely due to
  small-sample variance and should not yet be trusted — `notes/2026-09-16_p3_fig13_h5.md` recommends
  re-running with 20-30 trials/cell (cheap, CPU-only) before this axis goes in the paper.
- Owner: Proposer · Blast radius: High · Scoop risk: **Highest in the project**

### H6 — Only F5 and single-pass F2 admit task-disjointness
*Of the eight artifact families, only F5 (analytic Gram) and F2-with-single-pass prototypes admit a
task-disjoint implementation without accuracy collapse; F3, F4, F6, F8 provably cannot without
abandoning their anti-forgetting mechanism.*
- Status: `SUPPORTED` (formal half only — 2026-09-15)
- Deciding test: formal — apply the V1–V5 checker to each family; "without accuracy collapse" needs a
  small empirical component.
- **Formal half done**: `results/disjointness_report.csv` (`code/scripts/build_disjointness_report.py`,
  `notes/2026-09-15_p2_disjointness.md`) runs M0/M1/M2/M3/M4/M5/M8 with their real anti-forgetting
  mechanism switched on and off. With the mechanism **on**: M8 (F5) and M4-class-IL (F2 single-pass)
  certify `task_disjoint=True`; M0 (F1, multi-epoch), M1 (F1+F7+F8), M2 (F1+F6), M3 (F1+subspace),
  M5 (F8+F1) all certify `False` with V2/V3 (or V5b for M0's within-task multi-pass). Exactly matches
  the hypothesis for the families tested. F3/F4 not yet tested (M6/M7/M4-LoRA not built — GPU-heavy,
  deferred).
- **Empirical half (accuracy collapse) still open** — needs real per-dataset accuracy, blocked on P1
  feature extraction (GPU queue contention as of 2026-09-15, see `build/STATE.md`).
- Owner: DP Theorist · Blast radius: High (justifies §4.4's architecture choice)
- Note: "provably cannot" is strong. Weaken to "cannot under any implementation we could construct"
  unless a real impossibility argument materializes. Flagged for honesty. The formal result above is
  "every implementation we constructed," not a proof of impossibility — same caveat applies.

### H7 — DP-Analytic-FCL utility
*At client-task-level (U2) ε = 1, final average accuracy within 5 points of the non-private analytic
baseline and above every DP-SGD-based FCL baseline at equal ε, on CIFAR-100/10 and ImageNet-R/10 with
frozen ViT-B/16.*
- Status: `OPEN`
- Deciding test: §5 sweeps, ~120 GPU-h.
- Owner: Proposer · Blast radius: High (Paper B's constructive half)
- Known objection (Red Team): the *non-private* gap between analytic FCL and prompt SOTA may already
  exceed 5 points, making "best at equal ε" a hollow win. **Measure the non-private gap first.**

### H8 — The advantage widens with T
*At T = 50 tasks, DP-SGD-based FCL under sequential U2 accounting is at chance while DP-Analytic-FCL
is flat in T.*
- Status: `OPEN`
- Deciding test: long-horizon 50-task stream (§5.2), ~160 GPU-h.
- Owner: Proposer · Blast radius: Medium-high (**this is the money plot for Paper B**)

### H9 — Privacy-induced participation drift
*Per-client privacy filters exhaust heterogeneously (data-rich, high-drift clients first), converting
a privacy mechanism into a participation-bias mechanism that measurably degrades accuracy on those
clients' distributions over time.*
- Status: `OPEN`
- Deciding test: filter simulation over the long-horizon stream; measure per-client accuracy trajectory.
- Owner: DP Theorist + Empiricist · Blast radius: Medium (candidate third paper; bridges to P8)

### H10 — Leakage half-life
*Property-inference accuracy about client c's task-k data decays with (T − k) at a rate that differs
systematically by artifact family, with F5/F8 showing no decay.*
- Status: `SUPPORTED` (with a scope caveat — 2026-09-16; family-dependent decay itself not yet shown)
- Deciding test: attack A6 over the task index — `code/scripts/run_property_inference.py`,
  `attacks/property_inference.py` (F2+F7 / M4 only; F5/F1-based property inference is a stated scope
  cut, not silent). Real run on CIFAR-100 (10 tasks, 10 clients, 926 true (client,class,task)
  positives, 300 negatives/elapsed), `results/a6_property_inference.csv` (13 rows). Full writeup:
  `notes/2026-09-16_p3_a6_h10.md`.
- **Real result**: a clean step function, not a decay curve — `balanced_accuracy = 0.500` (chance) at
  `elapsed < 0` (before the task happened), jumps to exactly `1.000` at `elapsed = 0` and **stays at
  1.000 out to elapsed = +10** (the full observed horizon). Zero decay.
- **Honest framing**: this is *not* the family-dependent decay H10 predicts (F2/F7 decaying while
  F5/F8 doesn't) — F2/F7 shows exactly the "no decay" pattern H10 reserved for F5/F8 only. The
  mechanism is structural, not a coincidence: under a persistent-observer (`view=full`) adversary and
  every single-pass, class-incremental method built so far, a (client, class) pair's evidence is
  released exactly once and never retracted or overwritten, so there is nothing for a `view=full`
  adversary to lose track of, regardless of family. This is a real, reportable finding in its own
  right (`00_BUILD_PLAN.md` explicitly sanctions "no decay detected" as a strong result, not a null
  to hide) — but it does not yet establish H10's *family-dependent* claim.
- **What would actually test the family-dependent claim** (not yet run): a *weaker* adversary
  (`ledger.view("task")` or `view="final"`, which does not retain full history) or a method that
  actively dilutes old per-class evidence (M4 at `prototype_momentum > 0`, or M1's
  distillation-driven drift) — either is a cheap, already-supported rerun of the same script with a
  different adversary view / config, not new machinery.
- Owner: Empiricist · Blast radius: Medium
- Note: this is the cleanest *quantitative* operationalization of the whole "persistence" idea, and
  may end up being a better headline metric than H2's correlation — but the headline number is "no
  decay under `view=full`," not a decay rate, until the weaker-adversary variant is run.

### H11 — Secure aggregation does not help
*Every attack in the suite that operates on `V_full` retains its effectiveness under secure
aggregation, because it uses the aggregate broadcast rather than individual client updates.*
- Status: `SUPPORTED, with an important split by artifact family` (2026-09-21) — the hypothesis as
  literally stated ("every attack... retains its effectiveness") is **refuted for F2/F5** and
  **supported for F1**; the honest finding is more interesting than either blanket answer.
- **A3/A5/A6 already have real secure-agg findings** (A3/A6 survive against `ledger.aggregate_view()`,
  A5 correctly does not — client attribution needs per-client records that secure aggregation removes
  by construction). **A4 (built 2026-09-17) is secure-agg-native by design** — it was built against
  `ledger.aggregate_view()` from the start (the aggregate update-norm and participant-count series),
  never a per-client record, and its real result (H4 above, `notes/2026-09-17_p3_a4_onset_h4.md`) is
  therefore *already* H11-relevant evidence that onset inference survives secure aggregation for F1
  methods.
- **A1's secure-agg variant, the flagship attack behind claim C1, now has a real, complete deciding
  test (2026-09-21, `notes/2026-09-21_secure_agg_a1.md`, `code/scripts/check_secure_agg_a1.py`,
  `figs/fig11_secure_agg.pdf`)** — and the answer splits cleanly by family, exactly the kind of
  mechanistically-grounded result this register rewards:
  - **F2/F5 (M4, M8 — the two methods with the most extreme leakage findings anywhere in this
    project) become completely unattackable via A1 under secure aggregation.** `_score_target`'s
    PROTOTYPE/GRAM branch needs a specific client's own record (`r.client == target["client"]`);
    `aggregate_view()` sets `client=-1` on every record by construction (it is a sum over clients),
    so this filter can never match — the score is provably NaN for every target, every round, 100% of
    the time. A deterministic structural fact, verified on real CIFAR-100 federations and captured as
    a permanent regression test (`test_prototype_and_gram_scores_undefined_under_secure_agg`).
  - **F1 (M0, empirically re-tested; the same mechanism applies to M1/M2/M3/M5 by construction, not
    separately re-verified) shows *no detectable protection*.** A 3-seed, 1,024-shadow pilot
    (`shadow_runner.py`'s new `adversary_view=secure_agg` support) gives TPR@1%FPR essentially
    identical to the full-ledger attack at every elapsed value tested (e.g. elapsed=0: full=0.0275
    [0.0223,0.0326] vs. secure_agg=0.0287 [0.0240,0.0333] — CIs almost fully overlapping). The attack
    only ever needed the round-by-round *global* model, which any real FL deployment broadcasts to
    every client every round regardless of aggregation method — secure aggregation was never designed
    to hide that.
- Owner: Threat Modeler · Blast radius: **High for reviewer reception** — this is the claim that
  pre-empts "just use secure aggregation" as a dismissal, and it now has a real, nuanced answer rather
  than an open gap.

### H13 — "Your clients are a Dirichlet artifact" (FIG18, Camelyon17)
*The retention-leakage decoupling effect (C1/H2) is a synthetic-client-partition artifact and would
not appear (or would be weaker) under a real, natural federation.*
- Status: `OPEN` (2026-09-23, post-FIG18-v2 matched-5-client redesign) — **the original `REFUTED`
  verdict (2026-09-21) was itself built on a confounded comparison and must not be cited.** The pilot's
  "natural" arm used `n_clients=1` (the whole hospital as one client) against the "dirichlet" arm's
  `n_clients=10` — two variables changed at once (partition STRATEGY and client COUNT), so the observed
  "natural leaks more" effect could not be attributed to either alone. `08_FIX_PLAN.md` itself flagged
  this (§11: "H13 → OPEN (confounded; the natural arm had 1 client). Update it after FIG18 v2.").
- **CORRECTION (2026-09-23, FIG18 v2, the matched-5-client redesign)**: reran with BOTH arms at
  `n_clients=5` (matching the 5 real hospitals) — **natural** now groups by physical microscopy slide
  (`p3fcl.get_data.prepare_camelyon17()` records `slide` per sample from `Camelyon17Dataset`'s own
  metadata; `streams.natural_chunked_partition` distributes whole slides across 5 clients via a seeded
  shuffle + round-robin, never splitting one slide's patches across clients) instead of the trivial
  "whole hospital = 1 client" the pilot used; **dirichlet** is the same per-class Dirichlet(0.5) split
  as before, just also at `n_clients=5`. Same M0, domain-incremental stream, 3 seeds, 1,024-shadow
  budget, same ~5,000-image subsample (`results/fig18_natural_federation.csv`,
  `figs/fig18_natural_federation.pdf`).
  - **Leakage (the actual H13 question): the two arms' TPR@1%FPR curves now overlap almost entirely
    within their 95% CIs at every elapsed value** (e.g. elapsed=4: natural 0.034 [0.022, 0.047] vs.
    dirichlet 0.049 [0.018, 0.079] — point estimates close, CIs heavily overlapping, dirichlet's point
    estimate if anything slightly *higher*, the opposite direction from the pilot). **No significant
    natural-vs-synthetic leakage difference survives once client count is controlled** — the pilot's
    "natural leaks more" finding does not replicate at matched client count, at this 3-seed pilot power.
  - **Accuracy/BWT (a separate axis, checked along the way)**: `results/accuracy_matrix_camelyon17.csv`
    also converged once client count was matched — final_avg_acc natural=[0.92,0.92,0.89] vs.
    dirichlet=[0.93,0.93,0.89], BWT natural=[+0.012,+0.017,-0.024] vs. dirichlet=[-0.001,-0.005,-0.034]
    — much closer than the pilot's dramatic BWT gap (natural pinned at ~-0.02 constant vs. dirichlet's
    +0.001 to +0.030). Consistent with the leakage-side finding: the pilot's apparent "natural vs.
    synthetic" effect looks like it was substantially a client-COUNT artifact (fewer, larger clients),
    not a real-structure-vs-random-split effect.
  - **Honest interpretation**: this does NOT mean "natural federations are safe" or "Dirichlet
    understates nothing" — it means the SPECIFIC comparison this pilot ran, once properly controlled,
    shows no clear signal at 3-seed power. A real difference smaller than this pilot's CI width could
    still exist. Report as a null/underpowered result, not as a second REFUTED-in-the-other-direction
    claim — do not overcorrect.
- **A real data-acquisition obstacle, worth noting for reproducibility** (unchanged from the pilot):
  the official `wilds.get_dataset(dataset="camelyon17", download=True)` path is unreachable from this
  cluster (`worksheets.codalab.org` times out at the TCP level; general internet access is otherwise
  fine) — worked around with a verified community re-hosting of the same CC0 public-domain data
  (`wltjr1007/Camelyon17-WILDS` on Hugging Face Hub), reconstructed into the exact on-disk layout the
  `wilds` package expects so `p3fcl.get_data.prepare_camelyon17()` ran unmodified against it.
- Owner: Threat Modeler · Blast radius: **High for reviewer reception** — this was `00_BUILD_PLAN.md`'s
  last remaining "never cut" item; getting the comparison honestly null (rather than leaving a
  confounded REFUTED standing) is itself the responsible outcome here, not a setback.
- **Honest scope limit**: M0 only, 3 seeds, pilot-scale subsample — not yet the full 7-method,
  full-dataset-scale headline treatment `02_DATASETS.md §5` originally specified. Extending to more
  methods and more seeds is the natural next increment if this question needs a sharper answer than
  "no clear difference detected at this power."

### H12 — Reproduction attrition
*At least 3 of M1–M7 will not reproduce within 2 accuracy points using public code in a reasonable
effort budget.*
- Status: `SUPPORTED` (2026-09-15, with an important caveat — read before citing)
- **TAB07** (`results/tab07_reproduction_gap.csv`, `notes/2026-09-15_p2_tab05_tab07.md`): of the 5
  methods (M1, M2, M4, M5; M3's published number was unobtainable this session, M0/M8/M9 have no
  independent public number to compare against) with a real, cited published number, **4 of 5 miss
  by far more than 2 points** — M1 by 14-19 pts, M2 by 14-49 pts (three papers, three different
  published numbers for the SAME method — a second, independent instance of reproduction
  attrition), M4 by -13 pts, M5 by 19 pts. Literally satisfies the hypothesis's stated threshold.
- **The caveat that matters**: the mechanism is not "we tried to match their setup and failed" —
  it's a deliberate, stated protocol difference (every method here runs on a frozen IN21k-pretrained
  ViT-B/16; the original papers train a ResNet18 substantially from scratch). M1/M2/M5 score *higher*
  than published because a strong frozen backbone is doing a lot of work regardless of the
  anti-forgetting mechanism on top of it; M4 scores *lower* because we omit PILoRA's LoRA half,
  which is the part that would let the backbone adapt at all. That's real, informative, and worth
  a full paragraph in the paper — but it is a different flavor of "reproduction attrition" than the
  hypothesis's original framing (broken/unmaintained code, hyperparameter sensitivity, etc.)
  presumably anticipated. Human should decide whether H12 as literally worded is the right claim to
  carry into the paper, or whether the backbone-confound framing deserves its own numbered finding.
- Deciding test: months 3–6 (superseded — actually run and cited numbers now exist ahead of that
  schedule). Owner: Systems · Blast radius: Medium (schedule risk)
- Note: a *finding*, not just a risk — report it, as the NeurIPS'25 resource-constraint paper did.

---

## Theorem obligations

| ID | Statement | Status | Owner |
|---|---|---|---|
| T1-fwd | Task-disjoint FCL ⇒ lifelong DP at `max_k ε_k` under U2 | `ASPIRATIONAL` | DP Theorist |
| T1-conv | No task-disjoint mechanism with `ε_k ≥ ε₀` for infinitely many k is lifelong-DP under U4 | `ASPIRATIONAL` | DP Theorist |
| T2 | Renewal model + individual accounting ⇒ person-level (U5) ε bounded independent of T | `ASPIRATIONAL` | DP Theorist |
| Q-T3 | Optimal/near-optimal MF for block-structured unbounded streams | `OPEN QUESTION` (8-week timebox) | DP Theorist |

Honesty rule: `ASPIRATIONAL` → `SKETCHED` requires a written proof sketch another agent has read.
`SKETCHED` → `PROVED` requires a full proof and a human sign-off.

---

## Contested design decisions

| ID | Question | Current position | Challenger |
|---|---|---|---|
| C1 | Is `V_full` a realistic observation set? | Yes for cross-silo; and it survives secure aggregation (H11) | Red Team |
| C2 | Is DP-Analytic-FCL too simple for a main-track ML venue? | The contribution is the taxonomy + impossibility, not the mechanism | Reviewer Sim P2 |
| C3 | Is T1's converse trivial? | Unresolved — needs a real technical difficulty identified | DP Theorist to answer |
| C4 | Is the renewal assumption (T2) realistic for healthcare? | Partly; report the guarantee as a function of tenure `m` and measure the long-tenure tail honestly | Red Team |
| C5 | Attack-first vs theory-first sequencing | Attack-first `[LOCKED]` — shared codebase, de-risked, stronger motivation for Paper B | — |
| C6 | Should we audit methods nobody deploys? | Unanswered. Red Team must be given one full Round on this | Red Team |

---

## Parking lot

Good ideas, out of scope. One line each; do not let these into the plan without a human decision.
- Federated continual **unlearning** verification via our audit machinery → that is P7.
- Fairness of leakage across client subgroups → that is P8; H9 and A5 feed it.
- Leakage under *natural* temporal drift rather than synthetic splits → needs P1's benchmark.
- Cross-modal / VLM FCL leakage (CLIP prompts) → P9 territory; revisit after Paper A.
- LLM federated continual instruction tuning leakage (FCIT benchmark) → P4 territory; the memorization
  literature there is much more mature and the bar is higher.
- Backdoor persistence across tasks (the security dual of persistence leakage) → genuinely interesting,
  entirely separate paper.
