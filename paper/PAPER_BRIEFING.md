# Briefing for the paper-writing session — read this whole file before writing anything

**Who this is for**: a separate Claude session (Opus) tasked with writing the actual manuscript.
**What this is**: a complete, honest account of what was built, what was found, what is real vs.
preliminary, and where every number lives. **This document is not the paper.** Do not copy its prose
into the paper verbatim — read it, verify the claims against the cited files, and write the paper in
your own words, in whatever structure and register is right for the target venue.

**Deadline context**: this project was built under a hard time constraint (target: paper draft ready
2026-09-23, hard deadline 2026-09-25). That constraint is *why* claims C2 and C4 below are pilots/
theoretical arguments rather than full empirical validations, and it is why this briefing exists at
all instead of the builder session writing the paper itself. Write the paper honestly around what is
actually here — do not imply more empirical depth than exists for C2/C4.

**A real LaTeX scaffold now exists at `paper/main.tex`** (built 2026-09-21) — start from it rather
than a blank file. It has the section structure, correct `\input{}`/`\includegraphics{}` wiring for
every table/figure that exists, and `% TODO(Opus)` comments pointing at the exact section of this
briefing to read for each one. It contains no argumentative prose — write directly into it. Two
build commands now work end to end (previously neither existed): `make paper` (4-pass
pdflatex+bibtex — `latexmk`/`biber` are not installed on this system, checked directly) and
`make verify` (the CLAUDE.md non-negotiable #3 provenance check). **Building this scaffold caught a
real bug**: the table-generation scripts wrote method names like `m4_proto` into `.tex` cells
unescaped, which does not compile (LaTeX reads `_` as a subscript outside math mode) — fixed via a
new `p3fcl.plotting.latex_escape()` helper applied everywhere a free-text field lands in a `.tex`
table. If you add a new table script, use it.

---

## 0. Where to look first, and in what order

1. **`RESEARCH_PLAN.md`** (repo root) — the full research plan: the thesis, the four claims (C1-C4),
   the hypothesis register's origin, `[LOCKED]` decisions that must not be relitigated, and the
   Part A / Part B (Paper A / Paper B) framing. Read this in full before writing anything — it has
   the motivating narrative, related-work framing, and the precise mathematical setup (§3.4 especially,
   the analytic-FCL inversion argument) that the paper needs.
2. **`CLAUDE.md`** (repo root) — the standing rules this entire project was built under. Section
   "Non-negotiables" (12 items) is directly relevant to how the paper must cite evidence: no bare AUC,
   TPR@1%/0.1%FPR are primary, every number must trace to a CSV, seeds/CIs discipline, negative results
   are results. **The paper should honor these same standards** — e.g. never hand-type a number into
   the text that isn't backed by one of the files listed in §3 below.
3. **`agents/OPEN_QUESTIONS.md`** — the hypothesis register (H1-H12, T1-fwd, T1-conv, T2). This is the
   single source of truth for which hypotheses are `SUPPORTED`, `OPEN`, or inconclusive, and exactly
   what evidence backs each status. Read the H2, H3, H4, H5, H10, H11 entries in full — they contain
   the precise, hedged language this project uses for its own findings, which the paper should match
   in spirit (not necessarily verbatim).
4. **`build/03_RESULTS_SPEC.md`** — the original figure/table specification (FIG01-FIG22, TAB00-TAB10).
   Only FIG01, FIG02, FIG03(pilot), TAB02-TAB04, TAB10 (plus FIG05, FIG13 as CSVs) actually exist with
   real data — see §2 for the full, current list.
5. **The dated files in `notes/`** — one file per major finding, each with full methodological detail,
   bugs found and fixed, and honest caveats. §4 below indexes exactly which ones matter for which claim.

---

## 1. The one-paragraph thesis (for the abstract/intro)

In federated continual learning (FCL), every published anti-forgetting mechanism works by *retaining*
some artifact of past data — a replay buffer, a distilled model, a class prototype, an accumulated
statistic. This project's central claim is that **the mechanism which prevents forgetting is the same
mechanism which prevents a privacy attacker's evidence from decaying**: methods that fight forgetting
well also leak information about old clients' data for far longer (sometimes indefinitely) than their
own accuracy on that data lasts. This inverts the usual intuition that catastrophic forgetting is
*purely* a cost — under an adversary that observes the full training transcript (not just the final
model), forgetting is also, incidentally, a privacy mechanism, and *preventing* it undoes that
protection. The project calls this the **stability-plasticity-privacy trilemma**.

---

## 2. What exists as real, generated output right now

Everything below regenerates from `results/*.csv` via `make figures` (see `code/Makefile`) — this is
the literal CLAUDE.md non-negotiable #3 ("no number reaches the paper except through a CSV"). Cite
numbers from these files, not from this briefing's prose.

| File | What it is | Status |
|---|---|---|
| `figs/fig01_decoupling.pdf`/`.png` | Accuracy decay vs. A1 membership-leakage decay, 3 datasets x 7 methods, 5 seeds | **Real, done** |
| `figs/fig02_halflife.pdf`/`.png` | Half-life forest plot, log-x, 42 rows (3 datasets x 7 methods x {acc,leak}) | **Real, done** |
| `tables/tab03_leakage.csv`/`.tex` | TPR@1%FPR/TPR@0.1%FPR/AUC at elapsed=0 and elapsed=max, per method/dataset | **Real, done** |
| `tables/tab04_halflife.csv`/`.tex` | Half-life + CI + R² + censoring flag per method/dataset/quantity | **Real, done** |
| `results/decoupling_ratio.csv` | The single most important number per method/dataset: h_leak/h_acc with a paired-bootstrap CI | **Real, done** |
| `results/fig05_eps_of_T.csv` + `figs/fig05_eps_of_T.pdf` | ε(T) under 5 privacy units (U1-U5), all 7 methods, real ledgers | **Real, done** (see §4, claim C3/C4 — includes a sharp finding that only U2 gives task-disjoint methods a flat ε) |
| `results/fig13_gram_inversion.csv` + `fig13_anisotropy.csv` | H5's analytic-inversion cosine-similarity curves, CIFAR-100 + CUB-200 | **Real, done** |
| `results/a1_lira_<dataset>_<method>[_seed<N>].csv` | Raw A1 (LiRA) membership reports, per method/dataset/seed | **Real, done**, 105 files (7 methods x 5 seeds x 3 datasets) |
| `results/a3_prototype_diff.csv` | A3 prototype-difference attribution attack | **Real, done** |
| `results/a4_onset_inference.csv` | A4 onset-inference attack (CUSUM change-point) | **Real, done** |
| `results/a6_property_inference.csv` | A6 property-inference attack | **Real, done** |
| `results/tab05_utility_baselines.csv`, `tab07_reproduction_gap.csv` | Method utility + reproduction-gap-vs-published-numbers | **Real, done** |
| `tables/tab02_units.csv`/`.tex` | Units-of-privacy table (U1-U5), generated directly from `units.py`'s own definitions | **Real, done** (descriptive, no new experiments) |
| `tables/tab10_taxonomy.csv`/`.tex` | Artifact family taxonomy F1-F8 | **Real, done** (descriptive, transcribed from `RESEARCH_PLAN.md §2.2` + this project's `cacheable` determination) |
| `results/fig03_dose_response_pilot.csv` + `figs/fig03_dose_response_pilot.pdf` | **Small-scale pilot only** — see §4, claim C2 | **Real, done** (3 rows, all 9 (level,seed) shadow stores hit their full 1,024/1,024 budget) |
| `results/accuracy_matrix_imagenet_r.csv`, `a1_lira_imagenet_r_*.csv` | 3rd dataset for claim C1 | **Real, done** (35/35 method x seed combos, full 4,096-shadow budget; folded into `fig01_decoupling.csv`/`fig02_halflife.csv`/`decoupling_ratio.csv` — C1 is a full 3-dataset result now, see §4) |
| `results/fig16_roc.csv` + `figs/fig16_roc.pdf` | Log-log ROC, all 21 (dataset,method) combos, elapsed=0, seed=0 | **Real, done** (built 2026-09-21 — this is CLAUDE.md non-negotiable #5's mandatory appendix requirement, previously missing entirely from the project; use it) |
| `results/fig17_seed_variance.csv` + `figs/fig17_seed_variance.pdf` | Per-seed spread of the headline TPR@1%FPR (elapsed=0), all 105 (dataset,method,seed) combos | **Real, done** (built 2026-09-21) |
| `results/fig04_semantic_vs_individual.csv` + `figs/fig04_semantic_vs_individual.pdf` | Semantic (M4) vs. individual (M5) retention pilot — the Red Team's objection | **Real, done** (built 2026-09-21 — still a pilot scale, but a real, mechanistically-explained qualitative split; see §4, claim C2) |
| `results/fig13_gram_inversion_summary.csv` + `figs/fig13_gram_inversion.pdf` | H5's Gram-inversion cosine-similarity-vs-n curve + feature anisotropy, CIFAR-100 + CUB-200 | **Real, done** (built 2026-09-21 — the raw per-trial data existed since P3 but had no plot script; the numbers already cited under "H5" below were always real, now there's a figure to back them) |
| `figs/fig_a6_property_inference.pdf` | A6 property-inference balanced accuracy over time (H10), M4/CIFAR-100 — the step-function finding | **Real, done** (built 2026-09-21, from pre-existing `results/a6_property_inference.csv`). Standalone, not folded into FIG01's grid, because A6 is scoped to M4/CIFAR-100 only while FIG01 covers all 7 methods x 3 datasets — `03_RESULTS_SPEC.md`'s original FIG01 spec wanted this as a middle panel, but that would misrepresent scope, so it gets its own figure instead |
| `results/secure_agg_pilot_m0.csv` + `figs/fig11_secure_agg.pdf` | H11: does A1 (the flagship attack) survive secure aggregation? | **Real, done** (built 2026-09-21 — a complete, family-split answer: no protection for F1/M0, complete/structural protection for F2-F5/M4-M8. See the dedicated write-up below and §4's C1 section) |
| `results/accuracy_matrix_camelyon17.csv` + `results/fig18_natural_federation.csv` + `figs/fig18_natural_federation.pdf` | FIG18: Camelyon17 real hospital federation vs. synthetic Dirichlet split (the "never cut" item) | **Real, done** (built 2026-09-21, M0-only PILOT — 3 seeds, 1,024-shadow budget, ~5,000-image subsample of the full 455,954-patch release; see dedicated write-up below) |

**Figures/tables that do NOT exist and should not be referenced as if they do**: the full-spec FIG04
(≥6 levels x ≥3 methods x ≥2 datasets — only the 3-level, 2-method, 1-dataset pilot version exists,
see above), FIG06-FIG09, FIG12, FIG14-FIG15, FIG19-FIG22, TAB00-TAB01, TAB06, TAB08-TAB09. If the paper's
structure calls for one of these, either build it from what exists or omit it and say so in a
limitations paragraph. (TAB02 and TAB10 *do* now exist — see the table above; they were built
directly from `units.py` and `RESEARCH_PLAN.md §2.2` since they need no new experiments.)

---

## 3. The system that produced all of this (for the Method section)

A full experimental harness was built from scratch in `code/src/p3fcl/`:
- **The artifact ledger** (`artifacts.py`): the single interface between methods and attacks. A method
  declares what it releases as `ArtifactRecord`s (round, task, client, artifact family, payload, and a
  `touched` field — the set of raw data ids that influenced this release, honestly over-reported per
  CLAUDE.md non-negotiable #2). Eight artifact families are defined (F1 model-delta, F2 prototype, F3
  prompt, F4 low-rank, F5 Gram/analytic statistics, F6 generative, F7 counts, F8 exemplar) — see
  `RESEARCH_PLAN.md §2.2` for the taxonomy and `build/03_RESULTS_SPEC.md`'s TAB10 row for how to
  present it.
- **Seven methods**, one per file under `methods/`, spanning the retention spectrum: **M0**
  (FedAvgSequential, F1, no anti-forgetting mechanism at all — the floor), **M1** (GLFC, F1+F7+F8,
  exemplar replay + knowledge distillation), **M2** (TARGET, F1+F6, per-class generative replay),
  **M3** (FOT, F1, orthogonal-subspace gradient projection), **M4** (PrototypeFCL, F2+F7, class-mean
  prototypes), **M5** (HybridReplay, F1+F8, latent exemplar buffer), **M8** (AnalyticFCL, F5, exact
  closed-form ridge regression over accumulated Gram statistics — the strongest, most literal
  retention mechanism in the zoo). M6 (prompt-pool ViT) and M7 (LoRA) exist in the plan but were not
  built to completion (GPU-only, out of scope given the timeline).
- **The DP accountant** (`dp/accountant.py`, `dp/mechanisms.py`): computes ε(T) under 5 different
  "units of privacy" (U1 example-level through U5 individual/renewal-level), correctly routing
  task-disjoint methods to parallel composition (flat ε) and accumulating methods to sequential
  composition (growing ε) — see §5.
- **The attack suite** (`attacks/`): **A1** cross-task LiRA (shadow-model membership inference,
  extended to a *trajectory* over the transcript, not just the final checkpoint — this is the main
  engine behind claim C1), **A2** analytic Gram inversion (closed-form feature reconstruction for F5),
  **A3** prototype-difference attribution, **A4** onset inference (CUSUM change-point detection on the
  aggregate, secure-aggregation-visible update norm), **A5** client attribution, **A6** property
  inference over time (the direct test for a "leakage half-life" idea before A1 generalized it).
- **The shadow-federation runner** (`shadow_runner.py`): the compute engine behind A1. For each of
  ~500-4,096 "shadow" federations, resamples the *entire* candidate population (not just a few tracked
  examples — an earlier, wrong design that resampled only tracked targets gave a fake, degenerate
  perfect-AUC result; fixed to match standard LiRA practice, see `notes/2026-09-16_p3_a1_population_resampling.md`),
  reruns the method, and records each tracked target's per-round score trajectory plus its true
  membership label. A calibration/evaluation split over shadows (never the same shadows used to
  calibrate and to report) gives real TPR@1%FPR/TPR@0.1%FPR/AUC with exact Clopper-Pearson CIs.

**179 unit tests pass** (`code/tests/`), `ruff` clean, confirmed as of the final 3-dataset FIG01/FIG02
rebuild (2026-09-21). Every real experimental run is
logged in `results/RUN_LOG.jsonl` with a source-code hash, config hash, seed, and wall time — "a run
not in the log did not happen" is a literal project rule, and it was followed.

---

## 4. Claim-by-claim status — the core of this briefing

### Claim C1 — Leakage half-life differs by artifact family and decouples from accuracy decay

**Status: real, strong, multi-seed, multi-dataset. This should be the paper's flagship result.**

The protocol: for each (dataset, method, seed), run the real federation to get an accuracy matrix, and
run A1's shadow-calibrated LiRA attack to get a TPR@1%FPR trajectory over `elapsed = T - k` (T = the
round an adversary observes from, k = the task the target data belongs to). Fit an exponential decay
to both the accuracy-decay curve and the leakage-decay curve; take the ratio of their half-lives
(`h_leak / h_acc`) with a paired bootstrap CI over seeds. A ratio credibly above 1 means leakage
outlives accuracy — decoupling. **`00_BUILD_PLAN.md`'s own instruction, followed exactly: "fitting a
half-life to a flat curve is a reporting error" — a curve that never decays is reported as `censored`
with "no decay detected, lower bound only," which for several methods below means the *true* ratio is
literally infinite, not just large.**

Real numbers (`results/decoupling_ratio.csv`, 5 seeds each, all 3 datasets now complete):

| Method | Retention mechanism | CIFAR-100 ratio [95% CI] | CUB-200 ratio [95% CI] | ImageNet-R ratio [95% CI] |
|---|---|---|---|---|
| M0 | none | 3.75 [2.55, 6.89] | 1.53 [1.05, 2.10] | **109.34 [3.70, 161.91]** |
| M1 | distillation + exemplar | ∞ (no leak decay) | ∞ (no leak decay) | ∞ (no leak decay) |
| M2 | generative replay | ∞ (no leak decay) | ∞ (no leak decay) | ∞ (no leak decay) |
| M3 | orthogonal projection | 2.80 [0.74, 18.71] (CI includes 1 — inconclusive) | **~0 — reversed** (see below) | ∞ (no leak decay) |
| M4 | class-mean carry-forward | 3.71 [1.81, 34.71] | 134.49 [47.15, 2034.32] | ∞ (no leak decay) |
| M5 | exemplar replay | ∞ (no leak decay) | ∞ (no leak decay) | ∞ (no leak decay) |
| M8 | exact Gram/ridge accumulation | ∞ (no leak decay) | ∞ (no leak decay) | ∞ (no leak decay) |

**The headline sentence this table supports**: across 7 methods and all 3 datasets, the one method
with *no* anti-forgetting mechanism (M0) is the *only* one whose leakage curve reliably decays
alongside its accuracy — and even M0's own decoupling ratio excludes 1.0 on every dataset; every
method with a real retention mechanism holds its leakage flat or lets it grow, often for the entire
observed horizon. **ImageNet-R is the cleanest of the three — 7/7 methods show decoupling (CI
excludes 1 or trivially infinite), not 6/7** — the strongest single-dataset instance of this claim
found in the whole project. Lead with ImageNet-R's clean result, but report all three datasets; do
not cherry-pick the best one alone (CLAUDE.md non-negotiable #4/#7).

**Things the paper must report honestly, not smooth over** (this project's own standard, and it
matters for reviewer credibility):
1. **M3's picture is genuinely three different stories across the three datasets, not a single
   pattern.** CIFAR-100: negative BWT, decoupling ratio CI includes 1 (inconclusive). CUB-200: a real
   counter-example — accuracy shows *no detectable decay at all* while leakage *does* decay (half-life
   ≈ 11 tasks, R²=0.45), the reverse of every other pairing. ImageNet-R: negative BWT again, but now
   an *infinite* decoupling ratio (leakage never decays, like most other methods). Report this as a
   real, unresolved, dataset-dependent complication specific to M3 — do not force one narrative to
   cover all three; a plausible but unconfirmed story is that orthogonal-projection retention's
   effect on the accuracy/leakage relationship depends on how fine-grained/small the dataset is, but
   this project did not test that further.
2. **M1's accuracy curve on CIFAR-100 is not monotonic** (it *rises* to 2.46x its initial value by
   elapsed=6 before declining — real positive backward transfer, consistent with M1's own BWT of
   +0.27 to +0.37, plausibly from active distillation-driven consolidation). Its accuracy
   half-life number (26.20, R²=0.043) is not a reliable fit to a non-monotonic curve and should not be
   quoted as if it were — but its `ratio=∞` finding is untouched by this, since it comes entirely from
   the leakage side being censored.
3. **M4's decoupling is far more extreme on the small-per-class-n dataset (CUB-200: ratio 134 vs.
   CIFAR-100's 3.71), and infinite on ImageNet-R** — broadly consistent with, and can be cited
   alongside, claim H5's independent finding (§ below) that small per-class sample counts amplify
   leakage in general, though the CIFAR-100-vs-CUB-200-vs-ImageNet-R ordering is not a clean monotonic
   function of any single dataset statistic this project measured — say so rather than implying a
   tighter relationship than the data shows.
4. **M0's ImageNet-R ratio (109.34) has a very shallow, noisy leakage half-life fit** (R²=0.007 on a
   long, ~606-task estimated half-life against a horizon of 9) — the fit converged to a real finite
   number rather than being flagged censored, but a near-zero R² on a fit extrapolated many multiples
   beyond the observed horizon should be reported with that caveat attached, not quoted as a precise
   number.

**Files**: `figs/fig01_decoupling.pdf` (2-row x 3-column grid, one column per dataset), `figs/fig02_halflife.pdf`
(log-x forest plot, 42 rows), `tables/tab03_leakage.csv/.tex`, `tables/tab04_halflife.csv/.tex`,
`results/decoupling_ratio.csv` (21 rows). Notes, in order: `notes/2026-09-17_p4_fig01_decoupling.md`
(first pass, 2 methods), `notes/2026-09-18_p4_fig01_all_methods.md` (extended to 7 methods, M1's
non-monotonic curve found), `notes/2026-09-19_p4_second_dataset_cub200.md` (CUB-200 added, M3
counter-example found, two real plotting/analysis bugs found and fixed), `notes/2026-09-21_p4_third_dataset_imagenet_r.md`
(ImageNet-R added, cleanest 7/7 result, M3's three-way dataset-dependent picture, a real operational
finding about `run_lira.py`'s per-dataset performance variance) — the bug-catching pattern across all
three passes is worth a sentence in a "robustness of methodology" part of the paper if there's room,
since catching your own bugs before publishing is a genuine strength to claim, not just an
implementation detail.

**Related, supporting findings for C1's narrative**:
- **H5 (analytic Gram inversion, A2)**: `notes/2026-09-16_p3_fig13_h5.md`, figure
  `figs/fig13_gram_inversion.pdf` (built 2026-09-21 from the pre-existing raw data via
  `code/scripts/build_fig13.py` + `analysis/fig13_gram_inversion.py`). At n=1 sample per class,
  reconstruction is exact (cosine similarity = 1.000) on real CIFAR-100/CUB-200 ViT-B/16 features.
  Beyond n=1, cosine similarity drops and plateaus around 0.4-0.6 (CIFAR-100) / 0.65-0.70 (CUB-200) —
  never reaching the originally-hypothesized >0.8 bar past n~2-4, but still a real, substantial,
  non-chance leakage signal at every practical n. CUB-200 retains more than CIFAR-100 at matched n, as
  predicted from feature anisotropy (measured: CIFAR-100 86.4, CUB-200 34.0, both visible as a bar
  panel in the figure). **This directly motivates why M4/M8 (which are exactly the methods that release
  class-conditional statistics of this kind) show the most extreme decoupling in the table above.**
- **H10 (property inference over time, A6)**: `notes/2026-09-16_p3_a6_h10.md`, figure
  `figs/fig_a6_property_inference.pdf` (built 2026-09-21; standalone rather than folded into FIG01
  since A6 is scoped to M4/CIFAR-100 only). A cheaper, coarser attack than A1 — asks only "did client c
  hold class y at task k?" using the direct COUNTS/prototype join key. Under a persistent-observer
  adversary, this shows a clean step function: balanced accuracy exactly 0.5 (chance) at
  elapsed∈{-2,-1}, exactly 1.0 (perfect, never decaying) at every elapsed∈{0,...,10}. This is
  consistent with, and a simpler complement to, the A1 finding above, and can be cited as convergent
  evidence from an independent, much simpler attack.
- **H3 (transcript vs. checkpoint, A1's trajectory ablation)**: `notes/2026-09-16_p3_a1_m0_h3.md`.
  For M0 specifically, observing the *full* transcript gives a real, persistent AUC advantage
  (~0.03-0.04) over observing only the latest checkpoint — direct evidence that publishing intermediate
  rounds (which every real FCL deployment does) is its own, additional privacy cost beyond publishing
  a final model.
- **H4 (onset inference, A4)**: `notes/2026-09-17_p3_a4_onset_h4.md`. A different kind of harm
  (inferring *when* a new client's data entered the system, no auxiliary knowledge needed) — real for
  gradient-based methods (M0: CUSUM precision 0.716 vs. random's 0.625), not for closed-form-statistics
  methods (M4/M8) — a real, mechanistically-explained split (gradients spike on unfamiliar data;
  closed-form statistics don't), not a failure to find the effect.

**H11 (does secure aggregation save you? A1's secure-agg variant, `figs/fig11_secure_agg.pdf`,
`notes/2026-09-21_secure_agg_a1.md`) — the single highest-priority remaining item this project's own
hypothesis register had flagged ("Blast radius: High for reviewer reception... Prioritize it"), now
answered, and it directly pre-empts the most predictable reviewer dismissal of this entire paper.**
This deserves real space in the paper, not a footnote — it is a complete, mechanistically-grounded,
family-split answer:

- **F2/F5 (M4, M8 — exactly the two methods with the most extreme decoupling ratios and AUC≈1.0000
  findings in the whole C1 table above) become completely unattackable via A1 once the adversary is
  restricted to `Ledger.aggregate_view()` (the formal secure-aggregation view this project already
  built for A3/A4/A6).** Not weaker — provably undefined: the attack's PROTOTYPE/GRAM scoring needs
  one specific client's own released record, and secure aggregation's whole guarantee is that no
  individual client's contribution survives a sum. Verified directly on real CIFAR-100 federations
  (`code/scripts/check_secure_agg_a1.py`) and locked in as a permanent regression test. **This is real
  protection, and it should be stated plainly: the most extreme leakage findings in this paper are
  exactly the ones a standard, already-deployed FL privacy mechanism eliminates.**
- **F1 (M0, empirically re-tested at a 3-seed/1,024-shadow pilot scale; the same mechanism applies to
  M1/M2/M3/M5 by construction — not separately re-verified, say so) shows *no detectable
  protection*.** TPR@1%FPR under the secure-aggregation-restricted reconstruction is statistically
  indistinguishable from the full-ledger attack at every elapsed value tested (e.g. elapsed=0:
  full=0.0275 [0.0223,0.0326] vs. secure_agg=0.0287 [0.0240,0.0333]). The mechanistic reason is
  clean and worth stating in the paper's own words: **A1's model-delta attack never needed per-client
  visibility — it only ever used the round-by-round global model, which any real federated learning
  deployment broadcasts to every client every round by design, regardless of whether secure
  aggregation was used to compute it.** Secure aggregation protects the *inputs* to that computation,
  not its *output* — and the output is exactly what this attack consumes.
- **Write this as a genuinely two-sided, nuanced finding, not a win or a loss for either side of the
  "does privacy engineering work" question.** The paper's strongest, most reviewer-resistant framing:
  *secure aggregation is not a blanket defense against this project's findings; whether it helps
  depends entirely on which artifact family is being released, and this dependency is exactly the
  same fault line (per-client-keyed statistics vs. globally-shared models) that drives claim C1's
  decoupling results in the first place.*
- **Honest scope limits, state explicitly**: only M0 was empirically re-tested for F1 (not M1/M2/M3/M5
  individually); pilot-scale (3 seeds, 1,024 shadows, not the 5-seed/4,096-shadow headline budget);
  the "participation counts are public" modeling assumption behind treating the existing per-client
  reconstruction as secure-agg-legitimate for F1 is a standard, realistic choice (this is how real
  SecAgg+FedAvg deployments work) but is a stated modeling choice, not verified against a specific
  real system.

**FIG18 (is this an artifact of synthetic client partitions? Camelyon17, `figs/fig18_natural_federation.pdf`,
`notes/2026-09-21_fig18_camelyon17.md`) — the last of `00_BUILD_PLAN.md`'s "never cut" items, now
real.** This directly answers "your clients are a Dirichlet artifact" — the standard reviewer
challenge to every synthetic-client-partition FL study — using Camelyon17-WILDS's 5 real hospitals
as clients, compared against the same data under a synthetic Dirichlet split.

- **A real data-acquisition obstacle had to be worked around first, and this is worth a sentence in
  the paper's reproducibility appendix**: the official `wilds` package's Camelyon17 download URL
  (`worksheets.codalab.org`) is unreachable from this project's compute environment (a network-level
  connection timeout, not a code problem — verified directly, general internet access is otherwise
  fine). A verified community re-hosting of the identical, CC0-licensed public-domain data
  (`wltjr1007/Camelyon17-WILDS` on Hugging Face Hub — schema and per-hospital structure checked
  directly against the official dataset's description before use) was used instead, reconstructed
  into the exact on-disk layout the `wilds` package itself expects, so the project's own
  already-written `prepare_camelyon17()` function ran completely unmodified against it.
- **Scope**: M0 only (pilot), 3 seeds, 1,024-shadow budget, a ~5,000-image class+hospital-stratified
  subsample (both arms use the identical subsample — the comparison is apples-to-apples even though
  absolute numbers would shift at the full 455,954-patch scale). Domain-incremental stream (one task
  per hospital — Camelyon17 is binary-label, so the class-incremental split every other dataset here
  uses does not apply). Two client-partition arms via the *same* already-generic `streams.build_stream`
  (no new partitioning code needed, only a new `stream.use_domain_field` config flag in
  `shadow_runner.py`): **natural** (`n_clients=1`, hospital = sole client, no Dirichlet anywhere) and
  **dirichlet** (`n_clients=10`, matching every other dataset's client count).
- **The real result, and it is the opposite of what the objection predicts**: the natural federation
  shows a real, rising TPR@1%FPR trend (~0.025 at elapsed=0 to ~0.055 at elapsed=4, pooled over
  seeds); the Dirichlet-subpartitioned arm stays comparatively flat (~0.012–0.018 across the same
  range). **Synthetic Dirichlet partitioning does not inflate this project's leakage findings — if
  anything, it understates them relative to a real federation's actual client structure.** Write this
  plainly: this is a *stronger* answer to the objection than "no difference" would have been.
- **Honest caveats to state explicitly**: M0 only, not the full 7-method zoo; 3 seeds, and the
  natural arm's CI widens substantially at elapsed=4 (one seed reached TPR=0.118, pulling the band up)
  — a real consequence of pilot-scale seed count, not smoothed over; the mechanistic explanation
  offered (fewer, larger real clients release more concentrated, more attributable signal per update
  than many small synthetic clients) is a plausible hypothesis consistent with the direction of the
  result, not independently verified further this session.

### Claim C2 — Within a method, retention strength causally drives leakage (dose-response)

**Status: A SMALL, EXPLICITLY PRELIMINARY PILOT IN SCALE — but it now includes a real, mechanistically-
confirmed qualitative finding (the semantic-vs-individual split) that directly engages this claim's
strongest objection. Do not present the scale as validated (still 1 dataset, 3 levels, 3 seeds per
arm) — but do not undersell the semantic/individual contrast either, since it is a real result, not a
hedge.**

The original plan (`00_BUILD_PLAN.md` P5) called for ≥6 retention-strength levels x ≥3 methods x ≥2
datasets x 3 seeds — a large sweep, since every single data point requires its own ~1,000-4,000-shadow
A1 run. Given the timeline, the full sweep was not built. What exists instead: **1 method (M5 HybridReplay) x 1
dataset (CIFAR-100) x 3 levels of its `buffer_size_per_class` knob (0, 10, 20) x 3 seeds x a reduced
1,024-shadow budget**, complete and real — `results/fig03_dose_response_pilot.csv` (3 rows) and
`figs/fig03_dose_response_pilot.pdf`.

**The real numbers** (leakage = TPR@1%FPR from calibrated LiRA at `elapsed=5`; retention = -BWT from
a real, non-shadow federation; n=3 seeds per level; full detail and per-seed values in
`notes/2026-09-21_p5_dose_response_pilot.md`):

| `buffer_size_per_class` | mean -BWT (retention) | mean TPR@1%FPR (leakage) |
|---|---|---|
| 0  | -0.671 (weakest retention) | 0.024 |
| 10 | -0.047                     | 0.043 |
| 20 | -0.009 (strongest retention) | 0.049 |

Both quantities move monotonically together across all 3 levels with **no crossovers across any of
the 3 seeds at any level** — a clean dose-response pattern, exactly the shape H2/C2 predicts. Report
this as a real trend, not a null result — but see the scope caveats below before treating it as
validating the general hypothesis.

**How to write this section honestly**: present it explicitly as a pilot, in a subsection or
paragraph clearly marked as preliminary (e.g. "As a preliminary probe of the dose-response
relationship within a single method, we swept..."), report the real monotonic trend above with its
(wide, 3-seed) CIs from the CSV, and **explicitly state in a limitations paragraph** that the full
dose-response sweep across methods and datasets (≥6 levels, ≥3 methods, ≥2 datasets per the original
P5 spec) is future work. Do not claim this validates H2's causal dose-response hypothesis in general
— it is a real, clean trend within one method, at most suggestive of a broader causal claim.

**Complete (2026-09-21): the semantic-vs-individual split is now real, and it is the split the plan
predicted would be "a better paper than a uniform correlation."** A second pilot arm ran M4
(`PrototypeFCL`, `MethodSpec.retention_type="semantic"`) with its `prototype_momentum` knob at the
same 3-level ({0.0, 0.5, 0.9}) / 3-seed / reduced-budget scale as the M5 arm, `release_counts=False`
for both arms (the "counts off" condition the plan asks for, isolating the prototype/F2 channel).
Full numbers: `results/fig04_semantic_arm_m4.csv`, merged CSV `results/fig04_semantic_vs_individual.csv`,
figure `figs/fig04_semantic_vs_individual.pdf`, full writeup
`notes/2026-09-21_p5_dose_response_pilot_m4_semantic_arm.md`.

**The result**: M5 (individual/exemplar retention) shows the strong monotonic crossover already
described above. **M4 (semantic/prototype retention) is completely flat on both axes** — `-BWT` is
*bit-for-bit identical* across all 3 momentum levels for every seed (0.0693/0.0609/0.0660 for
seeds 0/1/2, unchanged at every level), and TPR@1%FPR varies only within shadow-sampling noise
(~0.34-0.36, no monotonic trend with the knob).

**This is not a null result or a weaker version of the M5 finding — it has a confirmed mechanistic
cause, verified directly against the code, and it is a *sharper* answer to the Red Team's objection
than an ambiguous flat trend would be**: `m4_proto.py`'s momentum-blending branch
(`if self.momentum > 0 and key in self._proto_touched`) only fires when the same `(client, class)` key
recurs across tasks. `streams.py::build_stream` is class-incremental by default — every class is
assigned to exactly one task — so that key is only ever seen once, and the blending branch is
**provably dead code under this project's entire experimental design**, regardless of the momentum
value. Prototypes, `touched` sets, accuracy, and the shadow-generation ledgers are therefore
bit-identical across all 3 levels by construction. Write this as: *under a class-incremental stream
(what every dataset in this project uses), M4's semantic retention mechanism cannot carry influence
across tasks at all — not weakly, but structurally — which is precisely why no dose-response can
appear, and stands in sharp contrast to M5's individual/exemplar retention, where the same 3-level
sweep produces a large, clean, monotonic effect.* This is a genuinely informative answer to "does
retention strength drive leakage," not an inconclusive one: the answer depends on *what kind* of
retention it is, exactly as the Red Team predicted.

**Honest limitation to state explicitly**: this pilot cannot speak to whether M4 would show a
dose-response under a *domain-incremental* stream (same classes recur across tasks — e.g. Camelyon17,
not used anywhere in this project), where M4's own code comment states the momentum-blending branch
*would* fire and `dp.accountant.check_disjointness` would correctly flag a disjointness violation.
That is a well-scoped, named future-work question, not a vague gap.

### Claim C3 — Contractive retention admits a T-independent lifelong ε; accumulating retention does not

**Status: real, already computed, no new work needed. Should get a full, confident treatment.**

`results/fig05_eps_of_T.csv` (245 rows, all 7 methods, 5 privacy units U1-U5, T ∈ {1,...,1000},
σ=2.0, δ=1e-5) already contains the exact evidence for this claim:

- **M4 and M8** (both task-disjoint by construction under class-incremental streams) hold **exactly
  flat ε = 2.529 from T=1 through T=1000** under unit U2 (`regime = "parallel-composition
  (task-disjoint, U2)"`, `task_disjoint = True`).
- **M0** (no retention, accumulates via sequential composition) grows from **ε=16.89 at T=1 to
  ε=4165.57 at T=1000** under the *same* unit U2 (`regime = "sequential-composition (U2, 30
  touches/task)"`, `task_disjoint = False`, disjointness violation `V5b`).

This is a direct, quantitative, already-real demonstration of the paper's Part-B claim: **a
task-disjoint retention mechanism (bounded influence per datum) gets a lifelong privacy guarantee that
does not grow with the number of tasks seen; an accumulating one does not.** This maps directly onto
the theorem obligation T1-fwd in `agents/OPEN_QUESTIONS.md` ("Task-disjoint FCL ⇒ lifelong DP at
max_k ε_k under U2") — the empirical result above is exactly what that theorem predicts, and can be
cited as an empirical confirmation of the theorem's practical content (the theorem itself is listed as
`ASPIRATIONAL` in the register — not proved, but the empirical pattern matching its prediction is real
and worth stating as such).

**A sharper, previously-undocumented point that the FIG05 plot surfaced (2026-09-21) and that
strengthens this claim's honesty and precision**: task-disjointness does **not** give a flat ε under
every unit — only under U2. Checked directly against `dp/accountant.py::account`'s own logic
(`if report.task_disjoint and unit == Unit.TASK: ...` is the *only* branch that routes to parallel
composition): for M4 and M8, **U1, U3, U4, and U5 all still show unbounded growth with T, identical
in shape to M0's accumulating pattern** — only U2 is flat. This is not a limitation to hide; it is
arguably the single most important point in Part B's argument, and directly motivates why the "units
of privacy" taxonomy (TAB02, `RESEARCH_PLAN.md §2.3`) is a real contribution rather than a formality:
**a task-disjoint mechanism only earns a bounded lifelong privacy guarantee under the specific unit
whose granularity matches its disjointness structure; report a mismatched unit (e.g. the client-level
or individual-level unit a real deployment would likely be asked about) and even M4/M8 look exactly as
unbounded as M0.** This should be stated explicitly in the paper, ideally as its own sentence or
short paragraph, not left implicit in the figure. See `figs/fig05_eps_of_T.png` — each panel now
annotates which units are numerically coincident for that method (most units collapse onto each
other; only U2 is genuinely different for M4/M8).

**TAB02 and TAB10 are already built** (`tables/tab02_units.csv/.tex`, `tables/tab10_taxonomy.csv/.tex`
— generated directly from `units.py` and `RESEARCH_PLAN.md §2.2`, no new experiments needed). Use them
directly in this section.

**Files**: `results/fig05_eps_of_T.csv`, `figs/fig05_eps_of_T.pdf` (built 2026-09-21 by
`analysis/fig05_eps_of_T.py` — one panel per method, log-log axes, one line per unit U1-U5, distinct
linestyle/marker per unit since several units are numerically identical for a given method and would
otherwise hide behind each other), `tables/tab02_units.csv`, `tables/tab10_taxonomy.csv`. No note file
exists specifically for the original FIG05 computation — the accounting-bug fix behind it is
documented in `notes/2026-09-16_p3_fig05_accounting_fix.md` (a real bug: the accountant originally
ignored `passes_over_data`, silently giving multi-epoch and single-epoch methods the same ε(T) curve —
fixed, regression-tested).

### Claim C4 — A bounded-influence retention operator keeps ε flat in T at competitive accuracy

**Status: THEORETICAL/ANALYTICAL ARGUMENT ONLY, using C3's existing data. No M9, no DP-SGD baselines,
no accuracy-at-matched-ε comparison exists. Do not claim this was empirically validated.**

The original plan called for a *new*, constructive method (M9, "DP-Analytic-FCL": M8's closed-form
ridge mechanism plus calibrated noise) compared against DP-SGD-based FCL baselines at matched ε across
multiple horizons (FIG08, FIG09, TAB08). None of that was built.

**What can be honestly said instead, using what already exists**: M8 (AnalyticFCL) is *already* a
real, working instance of a "bounded-influence retention operator" in the sense that matters for this
claim — each release is an exact, one-time function of that round's data only (task-disjoint by
construction, confirmed by the disjointness checker in `dp/accountant.py::check_disjointness`, see
`results/disjointness_report.csv` from the P2 phase), and C3's data above shows this gives a flat ε(T)
for free, at M8's real, measured accuracy (final average accuracy 0.891 on CIFAR-100, 0.974 on
CUB-200, 0.736 on ImageNet-R — `results/accuracy_matrix_*.csv`, seed-averaged). **The step this project
did not take is adding calibrated DP noise to M8's released `(R, Q)` statistics and re-measuring both
ε (via the accountant, already built) and accuracy (via `sim.run`, already built) at a few noise
levels.** That is a well-scoped, concrete "future work" sentence, not vague hand-waving — say so
explicitly, and consider naming it as the natural next experiment rather than implying it doesn't
exist as an idea.

**How to write this section honestly**: frame it as a theoretical argument grounded in C3's real data
("M8's task-disjoint structure already demonstrates the *mechanism* C4 depends on; realizing it as a
formal (ε,δ)-DP mechanism by adding calibrated noise to its released statistics, and validating the
resulting accuracy-privacy tradeoff against DP-SGD baselines, is direct future work") rather than as an
empirical result section. Do not create a FIG08/FIG09-shaped figure with fabricated or hoped-for
numbers.

---

## 5. Non-negotiable writing standards (carried over from CLAUDE.md, apply them here too)

1. **No number in the paper without a traceable CSV.** Every number above cites a specific file — use
   those files, don't approximate from this briefing's prose (round differently, re-derive a mean, or
   generalize past what's actually reported).
2. **Never report AUC alone for a membership-inference result.** TPR@1%FPR and TPR@0.1%FPR are
   primary; a log-log ROC curve is mandatory in the appendix — this now exists (`figs/fig16_roc.pdf`,
   all 21 dataset/method combos), use it rather than describing it as future work.
3. **Report exact intervals**, not just point estimates, whenever a CI exists in the source CSV
   (Clopper-Pearson for rates, the seed-bootstrap CIs used throughout P4).
4. **State claim scope precisely.** "7/7 methods on ImageNet-R, 6/7 on CIFAR-100 and CUB-200" is a
   stronger and more honest sentence than "methods with retention leak more" — match the hedging level
   already used in `agents/OPEN_QUESTIONS.md` and the `notes/` files, don't round it up to a universal
   claim.
5. **Report negative/inconclusive results as results**, not omissions: M3's CIFAR-100 CI including 1,
   M3's CUB-200 reversal, M1's unreliable accuracy fit, the pilot-only status of C2, the
   theoretical-only status of C4. A reviewer who finds an inconsistency the paper hid is far more
   damaging than the paper stating it first.
6. **Distinguish real findings from a standing methodological heuristic this project used repeatedly**:
   several real bugs were caught specifically because a result looked "too good" (a fake perfect AUC
   from a shadow-design flaw, a dead gradient in a prompt-pool method, a shape bug in a Procrustes
   alignment, an accounting gap that hid multi-epoch effects, a plotting bug that silently mixed two
   datasets into one line). This is worth one sentence in a methodology/robustness paragraph — it is a
   real, creditable practice, not an implementation detail to hide.
7. **Ethics framing is `[LOCKED]` in `RESEARCH_PLAN.md §9` — non-negotiable, read it in full.** The
   most important line for how to *write* this paper: **"Frame as a systemic finding, not as an
   accusation. The point is that the field's shared assumption is wrong, not that any one group was
   careless. Write it that way; it is also true."** M1/M2/M3/M4/M5/M8 are published, credited methods
   reimplemented faithfully in this project's harness (per non-negotiable #8 above) — the paper's
   argument is that *retention itself*, as a mechanism, has this cost, independent of which paper
   introduced a given retention strategy. Do not write sentences that read as "method X's authors
   missed this" — write "any method in this family inherits this property, including X." Also from
   `RESEARCH_PLAN.md §9`: no real-person re-identification claims (this project only ever used public
   benchmark data, so this is satisfied by construction, but say so explicitly if the paper discusses
   threat models); coordinated disclosure to the audited methods' authors is called for ≥60 days before
   *submission* — given this project's compressed timeline, flag this to the human as a process item
   separate from the writing itself, since a 2-day-to-draft schedule cannot satisfy a 60-day
   pre-submission disclosure window on its own.

---

## 6. Suggested paper structure given the *real* evidence balance

This is a suggestion, not a requirement — adapt to the actual venue/format:

1. **Abstract/Intro**: the trilemma thesis (§1). Lead with C1's headline number (decoupling ratios,
   several literally infinite).
2. **Related work**: FCL anti-forgetting methods, federated MIA (LiRA/Carlini et al.), the Jagielski et
   al. (ICLR 2023) "forgetting protects privacy" result this project's thesis inverts.
3. **Threat model & the artifact ledger abstraction** (§3 above) — this is a real systems contribution
   worth its own section (a unified interface across 7 heterogeneous methods' releases, used by 6
   different attacks without touching method internals).
4. **Claim C1 (the flagship)**: full treatment, §4/C1 above. This should be the longest, most detailed
   results section.
5. **Claim C3**: a shorter, confident section — the accountant + FIG05 data. Real theoretical content
   via T1-fwd's practical confirmation.
6. **Claims C2 and C4**: a shorter "preliminary evidence and future directions" section, explicitly
   scoped as such — do not pad this to look like C1's depth.
7. **Limitations**: the open items in §2/§4 above, plus anything from `build/STATE.md`'s "Open
   problems" section (read it for the current, complete list — it changes as work continues, so check
   it fresh rather than trusting a snapshot).
8. **Discussion**: the "just use secure aggregation" pre-emption — this now has real teeth, not just a
   gesture: A1's own secure-agg finding (§4/C1's H11 write-up above) shows a complete, family-split
   answer (no protection for F1, complete structural protection for F2/F5), and A3/A4/A5/A6 have their
   own consistent secure-agg findings (`agents/OPEN_QUESTIONS.md`'s H11 entry has the full picture).
   This is strong enough material to deserve its own subsection, not just a paragraph, if there's
   room — it directly answers the single most predictable reviewer objection. Also the
   backbone-confound finding from reproduction attempts (H12, `notes/2026-09-15_p2_tab05_tab07.md`) if
   there's room after that.

---

## 7. If you need something this briefing doesn't cover

Read `build/STATE.md` (the project's own running handover note — most current picture of what's done),
`agents/OPEN_QUESTIONS.md` in full (every hypothesis, its exact status and evidence), and the specific
dated `notes/*.md` file for whatever claim you're writing about — they are the actual lab notebook and
contain far more methodological detail (bugs found, design decisions, exact reasoning) than this
summary does. When in doubt about whether a number is real, find its source CSV and check it directly
rather than trusting any single document's restatement of it — including this one.
