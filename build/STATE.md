# STATE — maintained by Claude Code, read first on every session

**This is the resume file.** A run this long will have its context compacted or reset. When a new
session starts, read this, then `results/RUN_LOG.jsonl` and `qstat -u islamm`, and pick up from
"Next action" without redoing finished work. Update it after every meaningful step — not at the end
of a phase, but as you go.

---

## CRITICAL CONTEXT — deadline pivot (2026-09-21)

The human set a hard external deadline: **paper draft ready 2026-09-23, hard deadline 2026-09-25**.
"Ready" = **a full written paper draft covering all 4 claims (C1-C4)**. Division of labor: **Claude
Code (this session) does not write the paper's prose** — a separate Claude Opus session writes the
manuscript from `paper/PAPER_BRIEFING.md`, which this session must keep complete and accurate.
**`paper/PAPER_BRIEFING.md` is this project's most important deliverable alongside the results
themselves.**

Scope, agreed with the human under the deadline:
- **C1**: full treatment, real, 3 datasets — **complete** (flagship claim).
- **C2**: a small, explicitly-labeled empirical **pilot** (not the full ≥6-level x ≥3-method x
  ≥2-dataset P5 sweep) — **complete**, clean monotonic trend.
- **C3**: full treatment, real — **complete** (predates the deadline conversation).
- **C4**: **theoretical/analytical argument only**, grounded in C3's data — no new M9 method, no
  DP-SGD baseline. Explicitly future work, stated as such.

## Current phase

`P4 (claim C1) and P5-pilot (claim C2) are both complete, plus 6 extra appendix/robustness figures
added post-completion (FIG04 semantic arm, FIG05 plot, FIG11 secure-agg, FIG13 Gram-inversion, FIG16
ROC, FIG17 seed variance, A6 property-inference standalone) after the human asked "what more can we
do" twice.` Remaining work is a documentation/QA pass on `paper/PAPER_BRIEFING.md`, not new
experiments — see "Next action."

## Status — all real experimental work for this session is done

**Claim C1 (`decoupling_ratio.csv`, 21 rows, 3 datasets x 7 methods, 5 seeds each)**:

| Method | CIFAR-100 ratio | CUB-200 ratio | ImageNet-R ratio |
|---|---|---|---|
| M0 (none) | 3.75 [2.55, 6.89] | 1.53 [1.05, 2.10] | **109.34 [3.70, 161.91]** |
| M1 (distillation+exemplar) | ∞ | ∞ | ∞ |
| M2 (generative replay) | ∞ | ∞ | ∞ |
| M3 (orthogonal projection) | 2.80 [0.74, 18.71] (CI incl. 1) | **~0 (reversed)** | **∞** |
| M4 (class-mean carry-forward) | 3.71 [1.81, 34.71] | 134.49 [47.15, 2034.32] | **∞** |
| M5 (exemplar replay) | ∞ | ∞ | ∞ |
| M8 (exact Gram sum) | ∞ | ∞ | ∞ |

ImageNet-R is the cleanest of the three (7/7 methods decouple, not 6/7). M3 shows three genuinely
different pictures across datasets (inconclusive / reversed / infinite) — a real, unresolved,
dataset-dependent complication, not a single clean story. FIG01/FIG02 regenerated and visually
inspected (3-column grid, 42-row forest plot, no artifacts). All 105 (method, dataset, seed) LiRA
reports and all shadow stores (7 methods x 5 seeds x 3 datasets, 4,096 shadows each) verified
complete by direct file count.

**Claim C2 pilot (`fig03_dose_response_pilot.csv`, M5/CIFAR-100, `buffer_size_per_class` in
{0,10,20}, 3 seeds, 1,024-shadow budget)**: clean, monotonic, no crossovers —

| buffer_size_per_class | mean -BWT (retention) | mean TPR@1%FPR (leakage) |
|---|---|---|
| 0  | -0.671 | 0.024 |
| 10 | -0.047 | 0.043 |
| 20 | -0.009 | 0.049 |

Built a reusable `method_config_override` mechanism in `shadow_runner.py` for this (not a one-off
hack — usable for a future full-scale P5 sweep).

**A second pilot arm, added the same day, directly answers the Red Team's "semantic vs. individual"
objection** (`results/fig04_semantic_arm_m4.csv` + `fig03_dose_response_pilot.csv` merged into
`results/fig04_semantic_vs_individual.csv`, `figs/fig04_semantic_vs_individual.pdf`): M4
(`prototype_momentum`, semantic retention) at the same 3-level/3-seed scale shows **-BWT and
TPR@1%FPR completely flat across all 3 levels** — a mechanistically confirmed finding, not noise:
M4's momentum-blending branch only fires when a `(client,class)` key recurs across tasks, and
`streams.py`'s default class-incremental split means every class is assigned to exactly one task, so
that branch is provably dead code under every dataset this project uses. Sharp contrast with M5's
strong monotonic response — exactly the qualitative split `00_BUILD_PLAN.md` P5 asked for.

**Five more figures added the same day from already-computed or cheaply-derived data, no new shadow
generation beyond the M4 arm above**: **FIG05** (`analysis/fig05_eps_of_T.py`, plotting
`results/fig05_eps_of_T.csv` which previously had no script) surfaced a real, previously-undocumented
finding — only unit U2 ever triggers parallel composition in the accountant, so M4/M8 show a flat ε
*only* under U2, and grow unboundedly under every other unit (U1/U3/U4/U5), exactly like M0; **FIG16**
(log-log ROC, all 21 dataset/method combos) satisfies CLAUDE.md non-negotiable #5, previously
unaddressed anywhere in the project; **FIG17** (seed-variance strip plot, all 105 combos) shows the
headline TPR@1%FPR is stable across seeds; **FIG13** (Gram-inversion n-curve, H5 — required by spec,
not just appendix) aggregates pre-existing raw per-trial data via a new `build_fig13.py`, confirms
every number already cited in the briefing; **A6 property-inference standalone figure** (H10) — a
step function, scoped to M4/CIFAR-100 only (not folded into FIG01, which covers all 7 methods x 3
datasets). See `notes/2026-09-21_cheap_wins_fig05_fig16_fig17.md`,
`notes/2026-09-21_fig05_unit_specificity_of_flat_eps.md`, `notes/2026-09-21_fig13_and_a6_figures.md`.

**H11 (secure aggregation, the single highest-priority remaining item the hypothesis register had
flagged) now has a real, complete, family-split answer** (`notes/2026-09-21_secure_agg_a1.md`,
`figs/fig11_secure_agg.pdf`): **F2/F5 (M4, M8 — the two most extreme leakers in the whole project)
become completely unattackable via A1 under secure aggregation** (the attack needs a specific client's
own record; `Ledger.aggregate_view()` structurally cannot preserve client identity — provably NaN for
every target, verified on real federations, locked in as a regression test). **F1 (M0, 3-seed/
1,024-shadow pilot) shows no detectable protection** — TPR@1%FPR statistically indistinguishable with
vs. without the secure-agg restriction, because the attack only ever needed the round-by-round global
model, which any FL deployment broadcasts regardless of aggregation method. New `shadow_runner.py`
support: `adversary_view=secure_agg` config flag, threaded exactly like `method_config_override`.

`paper/PAPER_BRIEFING.md` and `agents/OPEN_QUESTIONS.md`'s H2 and H11 entries are all updated with
everything above. 181 tests pass, `ruff` clean; `make figures` runs clean from an empty
`figs/`/`tables/` (10 figures, 4 tables).

## Prior phases (P0-P3), closed out 2026-09-15 through 09-17 with documented cuts

A1 covers 7/9 cacheable methods (M6/M7 need a separate offline/GPU variant, not built — GPU-only,
out of scope); A1 has no secure-aggregation variant (A4 does, natively); A7 (one-run audit CLI
wiring) not started. Real results, one line each — full detail in the dated `notes/` files:
- **A2 (Gram inversion) — H5 `SUPPORTED`.** `notes/2026-09-16_p3_fig13_h5.md`.
- **FIG05 (ε(T) accountant)** — caught and fixed a real `passes_over_data` accounting bug.
- **A3 (prototype-difference attribution)** — self-corrected a within-class-similarity confound.
- **A6 (property inference) + A5 — H10 `SUPPORTED`** with a scope caveat.
- **A1 (cross-task LiRA), 7 methods** — caught a shadow-design flaw giving a fake `AUC=1.0000`
  (fixed: resample the whole population, not just tracked targets). **H3 `SUPPORTED`**.
- **A4 (onset inference) — H4 `SUPPORTED`, F1/gradient-based methods only** (gradients spike on
  unfamiliar data, closed-form statistics don't). Found and fixed a real bug in
  `Ledger.aggregate_view()` (crashed on per-class dict payloads with per-client-varying keys).

**All four of `00_BUILD_PLAN.md`'s "never cut" items are now done**: secure-aggregation ablation
(FIG11), log-log ROCs (FIG16), seed variance (FIG17), `make verify` (previously didn't exist as a
Makefile target at all, despite the script being written — fixed, see below), and **FIG18 (Camelyon17
natural federation)**, completed after the human asked "ok start FIG18" despite the earlier
out-of-scope assessment.

**FIG18 hit a real external blocker and was worked around, not skipped**: the official
`wilds.get_dataset(download=True)` path is unreachable from this cluster (`worksheets.codalab.org`
times out at the TCP level — general internet access is otherwise fine). Used a verified community
re-hosting of the same CC0 public-domain data (`wltjr1007/Camelyon17-WILDS` on Hugging Face Hub),
reconstructed into the exact on-disk layout the `wilds` package expects (new script:
`code/scripts/build_camelyon17_from_hf_mirror.py`) so `p3fcl.get_data.prepare_camelyon17()` — already
written, never previously exercised against real data — ran completely unmodified. Pilot scope: M0
only, 3 seeds, 1,024-shadow budget, ~5,000-image class+hospital-stratified subsample. `shadow_runner.py`
gained one new additive config flag, `stream.use_domain_field`, for the domain-incremental (one task
per hospital) stream Camelyon17 needs instead of class-incremental (it's binary-label).

**Real result (H13 in `agents/OPEN_QUESTIONS.md`, now `REFUTED`)**: the natural hospital federation
shows *more*, not less, leakage than the synthetic Dirichlet split (TPR@1%FPR ~0.025→0.055 vs. a flat
~0.012-0.018) — the opposite of what "your clients are a Dirichlet artifact" predicts, and a stronger
answer than "no difference" would have been. See `notes/2026-09-21_fig18_camelyon17.md`.

**`make paper`/`make verify` did not exist as Makefile targets until 2026-09-21** (`analysis/verify_provenance.py`
was written but never wired up; `paper/main.tex` did not exist at all, contrary to CLAUDE.md's Layout
section claiming it was "already scaffolded in `llm/paper/`" — that directory does not exist). Built
both: a real LaTeX scaffold (`paper/main.tex`, structure only, no prose — `% TODO(Opus)` comments
throughout) with working `\input{}`/`\includegraphics{}` wiring for every table/figure, plus the two
Makefile targets (`paper` uses 4-pass pdflatex+bibtex, since `latexmk`/`biber` are not installed on
this system). **Running `make paper` for the first time ever caught a real bug**: all 4
table-generation scripts wrote method names (`m4_proto` etc.) into `.tex` cells unescaped, which does
not compile (`_` reads as a subscript outside math mode) — fixed via a new
`p3fcl.plotting.latex_escape()` helper, applied everywhere. `make setup && make figures && make paper`
(the literal P9 gate) now runs end to end from empty `figs/`/`tables/`/`paper/main.pdf`, producing a
real 10-page PDF; `make verify` passes cleanly. See
`notes/2026-09-21_make_paper_make_verify_and_latex_escaping_bug.md`.

## Next action

**All experimental work is done, including all 5 of the plan's "never cut" items** (secure-agg,
log-log ROC, seed variance, `make verify`, and now FIG18/Camelyon17). What remains is a final
documentation/QA pass:
1. One clean top-to-bottom re-read of `paper/PAPER_BRIEFING.md` for accuracy against the final file
   state (it was updated incrementally through this session).
2. Stop unless the human asks for more. The deadline is for a written draft (produced by a separate
   Opus session reading `paper/PAPER_BRIEFING.md` and starting from `paper/main.tex`), not further
   experiments — do not start new experimental scope past this point without checking in.

## Jobs in flight

None.

Note: ImageNet-R's seed-0 shadows live directly under `shadows/imagenet_r/<method>/` with no
`seed0/` subdirectory, same convention as CIFAR-100/CUB-200 — not a bug if a fresh session notices
this asymmetry. Also note: `run_lira.py` ran 60-100x slower per (method,seed) combo on ImageNet-R's
shadow store than on CIFAR-100/CUB-200 for identically-shaped data (root cause not fully confirmed,
likely `np.savez_compressed` decode-cost variance) — worked around by parallelizing 12-at-a-time on
the login node rather than root-caused, given the deadline. See
`notes/2026-09-21_p4_third_dataset_imagenet_r.md` if this recurs on a future dataset.

## Gates passed

| Phase | Passed (UTC) | Human approved next phase? |
|---|---|---|
| P0 | 2026-09-15 | Yes — standing authorization given in advance |
| P1 | 2026-09-15 | Yes — explicit "yes" after the P1 gate report |
| P2 | 2026-09-16 | Yes — explicit choice to stop with documented cuts, move to P3 |
| P3 | 2026-09-17 | Yes — chose to close out with documented cuts, move to P4 |
| P4 | 2026-09-17 (first pass), 09-18 (7 methods), 09-19 (2 datasets), 09-21 (3rd dataset, complete) | Superseded by the deadline instruction — the human's remaining checkpoint is the paper draft itself, not a phase gate |
| P5 (C2) | 2026-09-21, scoped to a pilot, complete | Yes — human chose "small pilot for both (C2/C4)" over the full spec, given the deadline |

## Open problems

- FIG13's reference-quality axis needs more trials (5 -> 20-30) before trusting it in the paper.
- No full TAB05-shape sweep exists yet for M6; M7 not built (both deliberate, approved cuts).
- FOT's (M3) published CIFAR-100 number still unresolved (`notes/2026-09-15_p2_tab05_tab07.md`).
- A1's `_reconstruct_running_w` FedAvg weight is exact for M0, an approximation for M1/M2/M3/M5
  (their `touched` is honestly inflated beyond the raw shard) — worth revisiting if absolute AUC
  values (not just cross-method ordering) need to be load-bearing later.
- M1's accuracy-half-life number is unreliable on CIFAR-100 (R²=0.043, non-monotonic curve from
  positive backward transfer) — its qualitative "no leak decay" finding is unaffected.
- M3 shows three genuinely different pictures across three datasets: CIFAR-100 (inconclusive),
  CUB-200 (reversed), ImageNet-R (infinite) — worth a dedicated paragraph in the paper.
- M0's ImageNet-R decoupling ratio (109.34) rests on a very shallow leakage half-life fit (R²=0.007,
  extrapolated well past the 9-task horizon) — real and converged, but should carry that caveat.
- C2 is still pilot-scale (1 dataset, 3 levels, 3 seeds per arm, 2 of 7 methods) despite now including
  a real semantic-vs-individual finding; C4 is theoretical/analytical only, no new DP mechanism built.
  Both explicitly scoped this way under the deadline, not silently cut.
- M4's flat dose-response is confirmed only under class-incremental streams; whether it would show a
  real effect under a domain-incremental stream (where its momentum-blending branch actually fires) is
  untested — a well-scoped future-work question, not evaluated this session.
- FIG16's ROC curves use seed=0 only per (dataset,method), not all 5 seeds — a reasonable follow-up
  if there's more time before the draft is due.
- Silent array-task deaths under high concurrent submission load are a real, recurring Kodiak
  reliability pattern at every large sweep this session (consistently needs 2-3 backfill rounds) —
  always fully recoverable via resumability + resubmission. Verify final output counts on every large
  sweep; never assume a cleared queue means success.

## Decisions made inside a phase

- 2026-09-14/15: repo-root-is-data-root; `module load python/3.10.4` gotcha; CUDA ceiling 13.0;
  no-git provenance (`source_hash()` only); PBS batch/gpu jobs for everything, never login-node
  compute; GPU pool is genuinely mixed P100+V100.
- 2026-09-15: M5 releases both F8 and F1; TAB07's cross-paper gap is a backbone confound, stated
  explicitly in the paper's framing, not just the CSV.
- 2026-09-15/16: **standing heuristic** — a "too good" result, a trainable parameter that never
  changes, or a metric that can't move given a supposed ablation are all signals to check for a
  broken gradient path or an under-used/under-reported field before trusting the result. Confirmed
  repeatedly, most recently as "always inspect the rendered figure, don't trust exit code 0 alone."
- 2026-09-16: A2/H5 deliberately reports only the reference-based (realistic-adversary) curve, never
  an "oracle" curve against ground truth. A3's F7/counts ablation uses scaled L2 error, not cosine.
  A1's shadow design resamples the *entire* candidate pool per shadow (standard LiRA practice).
- 2026-09-17: A4 is built secure-agg-native from the start (reads only `ledger.aggregate_view()`).
  P4's half-life CI is a bootstrap *over seeds*; the decoupling ratio uses a *paired* bootstrap.
- 2026-09-18: FIG02 uses a log-x axis (half-life is multiplicative; the reciprocal-slope bootstrap CI
  can span orders of magnitude for a shallow fit).
- 2026-09-19: `build_fig01.py`/`build_fig02.py`/`run_accuracy_matrix.py` are dataset-generic
  (`DATASETS` list / CLI arg), not hardcoded to `cifar100`.
- 2026-09-21: **Deadline pivot.** Human set hard external deadline (draft 09-23, hard 09-25) and
  clarified: (a) "ready" = full written draft covering all 4 claims, (b) Claude Code does not write
  paper prose — a separate Opus session does, from `paper/PAPER_BRIEFING.md`, (c) C2/C4 get
  scoped-down treatments rather than being skipped or built to full spec.
- 2026-09-21: Built a general-purpose `method_config_override` mechanism in `shadow_runner.py` for
  the C2 pilot — reusable for a future full-scale P5 sweep, not a throwaway hack. Discovered
  `run_lira.py` is much slower on ImageNet-R's shadow store than the other two datasets for
  identically-shaped data — worked around via parallelization rather than root-caused.
- 2026-09-21: After the human asked what else could be done in 1-2 days, added 4 more real
  figures/findings beyond the original scope-cut plan: the M4 semantic-arm pilot (answers the Red
  Team's objection with a mechanistically-confirmed flat result), FIG05's plotting script (surfaced
  the U2-only flat-ε finding), FIG16 (log-log ROC, closes a CLAUDE.md non-negotiable gap), FIG17
  (seed variance). `build_fig16.py` and `run_dose_response_pilot_m4.py`/`build_fig04.py` follow the
  same parallelize-across-the-login-node pattern established for the ImageNet-R LiRA backfill when a
  sequential run would have been too slow.

---

### How to update this file

Rewrite the sections above in place; do not append a changelog. Keep it under a page — it is a
handover note, not a history. Anything worth keeping longer goes in `notes/<date>_<topic>.md`, and
anything that changes a hypothesis goes in `agents/OPEN_QUESTIONS.md`.
