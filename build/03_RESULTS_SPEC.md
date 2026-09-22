# 03 — RESULTS SPEC

Every figure and table the paper can contain, with its data contract. **Required** items block
submission; **appendix** items are expected but can be trimmed; **stretch** items are upside.

> **Naming convention — read this once.** `F1`–`F8` are **artifact families** (model deltas,
> prototypes, prompts, LoRA, Gram, generative, counts, exemplars), fixed by `RESEARCH_PLAN.md §2.2`.
> Figures are `FIG01`…`FIG22`; tables are `TAB00`…`TAB10`. Never write a figure as "F01" — it reads
> as a family, and this spec has already been bitten by exactly that collision once.

## 0. The contract

For every figure `FNN` and table `TNN`:

```
results/fNN_<slug>.csv          tidy/long data — one observation per row, never wide
results/figNN_<slug>.meta.json  source hash, config hash(es), seeds, n_runs, wall time, host, UTC
analysis/fNN_<slug>.py          reads ONLY the CSV, writes figs/fNN_<slug>.pdf and .png
figs/fNN_<slug>.pdf             vector, the version that goes in the paper
tables/tNN_<slug>.csv + .tex    booktabs, \input{} into the paper, never pasted
```

Rules:
- Plot scripts **do not compute**. If a number is not in the CSV it does not appear in the figure.
  This is what makes `make figures` a real regeneration rather than a re-run.
- Tidy format. `dataset, method, family, task_k, task_T, seed, metric, value, ci_lo, ci_hi, n`
  is the shape most of these want. Wide CSVs are a bug; reshape in the plot script if needed.
- Every rate carries `ci_lo`/`ci_hi` from `metrics.clopper_pearson` and the `n` it was computed from.
  A rate without its `n` is not reportable.
- `seed` is a real column with real per-seed rows. Do not pre-average into the CSV — aggregate in the
  plot script so the seed spread is always recoverable.
- Filenames are stable. If a figure is renumbered, `git mv` everything and update the paper; never
  leave two CSVs describing the same experiment.

## 0.1 Plot standards

- Vector PDF, `matplotlib`, no seaborn styling defaults, `figure.dpi=200` for the PNG.
- Colour by **artifact family** and keep that mapping identical in every figure in the paper. One
  legend the reader learns once. Colour-blind-safe; also vary marker shape so the figures survive
  greyscale printing.
- Confidence intervals always drawn — bands for curves, caps for points. A point estimate with no
  interval does not go in this paper.
- Axis labels carry units and the metric definition (`TPR @ 1% FPR`, not `TPR`).
- Font size ≥ 7pt **at final printed width**. Check by rendering at the column width, not by zooming.
- Log scale where the data is multiplicative (ε, FPR). Never truncate a y-axis to exaggerate an effect.
- Every figure must be readable without the caption, and the caption must state the takeaway in one
  sentence, the setting in the second, and nothing else.

---

## 1. Required figures

### FIG01 — Retention/leakage decoupling · **the headline** · claim C1, H10
**Three** aligned panels on a shared x-axis of *elapsed tasks* (T − k), each normalised to its value
at k: (top) accuracy on task k; (middle) **A6 property-inference** balanced accuracy about client c's
task-k data; (bottom) **A1 membership** TPR@1%FPR on task-k data. One line per artifact family,
bands = 95% CI over seeds.

The middle panel is what H10 is actually about, and the register's own note says it "may end up being
a better headline metric than H2's correlation" — so plot both, and do not pick one after seeing the
results. The story is visual: the top panel decays, the lower two do not, for append-only families.

**If the two leakage panels disagree — property leakage persists while membership leakage decays —
that is a finding, not a problem.** It would say the federation keeps *semantic* information about a
task long after it stops keeping *individual* information, which is a sharper and far more defensible
claim than a uniform one, and it is also the honest answer to the Red Team's standing objection.
Write the paper around whichever way it lands.

`results/fig01_decoupling.csv`: `dataset, method, family, task_k, task_T, elapsed, seed, acc, prop_balacc, prop_baseline, tpr1, tpr01, auc, ci_lo, ci_hi, n_pos, n_neg`

### FIG02 — Leakage half-life by family · C1
Forest plot: half-life estimate per artifact family × dataset, with bootstrap CI, plus a reference
line at the accuracy half-life. Families with no detected decay are drawn as a right-pointing arrow
with the lower bound, **not** as a fitted number.
`results/fig02_halflife.csv`: `dataset, method, family, quantity{acc|leak}, halflife, ci_lo, ci_hi, r2, fit_ok, horizon_tasks, censored`

### FIG03 — Dose–response · claim C2, H2
x = retention strength (normalised per method), y = TPR@1%FPR, colour = method, marker size or a
twin axis = −BWT. Six levels minimum per method.
`results/fig03_dose_response.csv`: `dataset, method, knob_name, knob_value, retention_bwt, tpr1, tpr01, final_acc, seed, ci_lo, ci_hi`

### FIG04 — Semantic vs individual retention · C2, the Red Team's objection
The same dose–response split into methods whose retention is *semantic* (prototypes, class means)
and *individual* (exemplar replay, Gram). If the effect holds only for the individual group, that is
the result — draw it that way and say so.
`results/fig04_semantic_vs_individual.csv`: `dataset, method, retention_type{semantic|individual}, knob_value, tpr1, retention_bwt, seed, ci_lo, ci_hi`

### FIG05 — ε(T) by unit of privacy · claim C3, H1
Log-y ε against T ∈ {1 … 1000}, one line per unit U1–U5, computed by `dp.accountant.account` on
**real ledgers** from P2 at matched σ. The flat U2 line against the diverging U4 line is the figure.
`results/fig05_eps_of_T.csv`: `method, ledger_hash, unit, sigma, delta, T, eps, regime, task_disjoint, violations`

### FIG06 — Long horizon, T = 50 · C1, C4
Accuracy and leakage on task 1 as the stream runs to 50 tasks, CIFAR-100 (+ ImageNet-R if it lands).
`results/fig06_long_horizon.csv`: `dataset, method, family, task_T, acc_task1, tpr1_task1, eps_U2, eps_U4, seed, ci_lo, ci_hi`

### FIG07 — Adversary view ablation · H3
TPR@1%FPR against number of observed rounds, for `view ∈ {final, task, full}`. Tests whether the
transcript is worth more than the checkpoint, with the √(#observations) reference curve overlaid.
`results/fig07_view_ablation.csv`: `dataset, method, family, view, n_observations, tpr1, tpr01, auc, seed, ci_lo, ci_hi`

### FIG08 — Privacy–utility Pareto · claim C4, H7
Final average accuracy against **empirical** ε (one-run audited lower bound) with the analytical ε as
a second marker. Our M9 against DP-SGD-FedAvg, DP federated linear probe, and the non-private
reference line. Panels for T = 10 and T = 50.
`results/fig08_pareto.csv`: `dataset, method, T, eps_target, eps_analytical, eps_audited_lb, eps_audited_ci_lo, eps_audited_ci_hi, final_acc, bwt, seed, ci_lo, ci_hi`

### FIG09 — The advantage widens with T · **the money plot for C4** · H8
Accuracy at fixed ε as T grows, ours flat, DP-SGD baselines decaying toward chance.
`results/fig09_advantage_vs_T.csv`: `dataset, method, eps_target, T, final_acc, chance_level, seed, ci_lo, ci_hi`

### FIG10 — Audit tightness
Analytical ε against one-run empirical ε lower bound, across mechanisms and σ. The 45° line is the
reference; the gap is the result either way, and a large gap is itself worth reporting.
`results/fig10_audit_tightness.csv`: `method, mechanism, sigma, delta, eps_analytical, eps_audited_lb, ci_lo, ci_hi, n_canaries, k_guesses, correct, seed`

### FIG11 — Secure aggregation does not help · H11 · **do not cut**
Grouped bars: TPR@1%FPR per attack, with and without secure aggregation. Every attack that consumes
the aggregate broadcast should be unchanged; those needing per-client records should drop to chance.
That contrast *is* the figure, and it pre-empts the most common reviewer dismissal.
`results/fig11_secure_agg.csv`: `dataset, method, attack, secure_agg{0|1}, metric, value, ci_lo, ci_hi, n, seed`

### FIG12 — Task-onset inference · H4
Detection precision and mean absolute error in rounds against tolerance, versus two baselines
(random, and an update-norm-spike heuristic). Plus one timeline panel: true boundaries against
detected change points for a single client, because it makes the harm legible.
`results/fig12_onset.csv`: `dataset, method, client, stat, tolerance, precision, recall, mae_rounds, tp, fp, n_boundaries, seed`
`results/fig12_onset_timeline.csv`: `dataset, method, client, round, statistic_value, is_true_boundary, is_detected`

### FIG13 — Gram inversion n-curve · H5 · re-specified, see `notes/2026-08-25_preliminary.md`
Cosine similarity against per-class n, one line per reference quality, one panel per dataset. **Not**
a single threshold — a curve. Second panel: measured feature anisotropy (eigenvalue spectrum decay)
per dataset, because the hypothesis is that anisotropy is what moves the curve. Optional qualitative
montage of decoded reconstructions for the CUB case.
`results/fig13_gram_inversion.csv`: `dataset, backbone, n_per_class, ref_quality, anisotropy_ratio, mean_cos, median_cos, frac_above_0.8, ci_lo, ci_hi, n_records, seed`

### FIG18 — Natural federation vs Dirichlet · **do not cut**
Camelyon17 with real hospital clients against the same data under a Dirichlet partition. Answers
"is your effect an artifact of synthetic client splits?" before it is asked.
`results/fig18_natural_federation.csv`: `partition{natural|dirichlet}, beta, method, family, elapsed, acc, tpr1, seed, ci_lo, ci_hi`

---

## 2. Appendix figures

- **FIG14 — non-IID sensitivity.** β ∈ {0.1, 0.5, 1.0, ∞} on the headline experiment.
  `results/fig14_beta_sensitivity.csv`: `dataset, method, beta, elapsed, acc, tpr1, seed, ci_lo, ci_hi`
- **FIG15 — backbone robustness.** FIG01 repeated on DINOv2 features. If the effect is backbone-dependent,
  state it as a limitation.
  `results/fig15_backbone.csv`: same schema as FIG01 plus `backbone`.
- **FIG16 — log-log ROC.** Mandatory for every headline attack; the community standard, and its absence
  is read as hiding a weak low-FPR regime.
  `results/fig16_roc.csv`: `dataset, method, family, attack, fpr, tpr, seed`
- **FIG17 — seed variance.** Per-seed values of the headline number as a strip/box plot.
  `results/fig17_seed_variance.csv`: `dataset, method, metric, seed, value`
- **FIG19 — schematic** of the release channel, the artifact families F1–F8 and the three adversary
  views (`final` / `task` / `full`), plus the secure-aggregation view. **Generate it in code** — a
  standalone `analysis/fig19_schematic.py` drawing it with matplotlib patches, or emitting TikZ that
  the paper `\input`s. Do not hand-draw it in an external editor: it has to regenerate with
  `make figures` like everything else, and it will need revising as the framing tightens.
  Its "data" is a small hand-written `results/fig19_schematic.csv` listing the families, what each
  releases, and which views expose it — so even the diagram's content is traceable.
  This is the figure that makes the framing land in ten seconds. Budget real time for it; it is
  worth more than another ablation.

## 3. Stretch figures

- **FIG20** leakage fairness across clients (per-client TPR dispersion; do data-rich clients leak more?) — H9 territory.
- **FIG21** privacy-filter exhaustion timeline (which clients drop out first under a per-client filter, and what it does to their accuracy) — H9.
- **FIG22** communication-vs-leakage frontier (truncated-SVD rank against reconstruction quality — does the bandwidth trick buy privacy? The prediction is no).

---

## 4. Tables

| ID | Content | Status |
|---|---|---|
| **TAB00** | Datasets: n, classes, tasks, partition, natural clients, license, role | required |
| **TAB01** | Feature-cache sanity: linear-probe top-1 per dataset × backbone | required |
| **TAB02** | **Unit-of-privacy taxonomy**: each method and cited paper × which unit U1–U5 its claim implies, with the quoted sentence | required, a contribution |
| **TAB03** | Main leakage table: method × dataset × TPR@1%FPR, TPR@0.1%FPR, AUC at elapsed = 0 and elapsed = max, ±CI | required |
| **TAB04** | Half-life estimates per family with CI, R², censoring flag | required |
| **TAB05** | Utility baselines: final avg acc, BWT, avg incremental acc, per method × dataset × {10,20} tasks | required |
| **TAB06** | Dose–response summary: slope, Spearman ρ, p, per method, with the semantic/individual split | required |
| **TAB07** | Reproduction gap: our reimplementation vs each paper's published number, tolerance achieved — this is H12 and it is a finding | required |
| **TAB08** | DP utility at matched ε ∈ {0.5,1,2,4,8} × T ∈ {10,50}, ours vs baselines vs non-private | required |
| **TAB09** | Cost: GPU-hours, CPU-node-hours, bytes released per round per method, shadow count | required |
| **TAB10** | Artifact family taxonomy F1–F8: what is released, sufficient-statistic content, which published methods ship it, cacheable, a-priori risk | required, conceptual |

`TAB02` and `TAB10` are partly written by hand from the literature rather than computed — that is fine,
but they still live as CSVs under `tables/` and are rendered to `.tex` by a script, so that the paper
never contains a hand-typed table body.

---

## 5. The statistical bar

- **Seeds.** ≥3 everywhere, ≥5 for FIG01, FIG03, FIG08, FIG09. Seeds are `{0,1,2,3,4}`, fixed, never
  cherry-picked. If a seed fails to run, report it as failed; do not replace it with seed 5.
- **Intervals.** 95% throughout. Clopper–Pearson for rates; percentile bootstrap (≥2,000 resamples)
  for half-lives, slopes and ratios. State which was used in every caption.
- **Multiplicity.** The paper makes many comparisons. For the family-level claims in FIG02, apply
  Holm–Bonferroni across families and report both raw and adjusted p. Do not report a bare p < 0.05
  from a sweep of twenty configurations.
- **Effect sizes, not just significance.** A half-life ratio with a CI says more than a p-value, and
  reviewers at this venue prefer it.
- **Pre-registration.** The hypotheses are already written in `agents/OPEN_QUESTIONS.md` with
  deciding tests. Do not edit a hypothesis after seeing its result — add a new line dated and
  labelled `post-hoc`, and mark any post-hoc analysis as such in the paper.
- **The negative-result protocol.** If an attack comes back at chance: report the TPR with its CI,
  report the audited ε lower bound from A7, and write the sentence "we could not distinguish this
  from chance at n = …; this bounds rather than establishes the leakage." That is a publishable
  sentence. A quietly dropped attack is not.

## 6. The provenance check, concretely

`make verify` must fail on all of these:
- a number in `paper/**/*.tex` that appears in no `results/*.csv` and no generated `tables/*.tex`;
- a figure in `figs/` with no sibling CSV, or whose CSV is newer than the PDF;
- a CSV with no `.meta.json`, or whose `.meta.json` records a source hash that no longer matches
  any code state the run log knows about;
- a row in a headline CSV with fewer than the required number of seeds.

Wire it into a pre-submission checklist in `notes/SUBMISSION_CHECKLIST.md` and run it before every
paper build, not once at the end.
