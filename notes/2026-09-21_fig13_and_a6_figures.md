# 2026-09-21 — Two more real figures from already-computed data: FIG13, A6 property inference

Continuation of the "what more can we do in 1-2 days" pass (see
`notes/2026-09-21_cheap_wins_fig05_fig16_fig17.md`). Found by inventorying `results/*.csv` against
`analysis/*.py` and looking for real, already-computed data with no plot script — the same pattern
that produced FIG05.

## FIG13 — Gram inversion n-curve (H5)

`results/fig13_gram_inversion.csv` (195 raw per-trial records: 2 datasets x up to 8 `n_per_class`
levels x 3 `ref_quality` levels x 5 trials each) and `results/fig13_anisotropy.csv` have existed
since P3 (`notes/2026-09-16_p3_fig13_h5.md`) and are the source of exact numbers already cited
throughout `paper/PAPER_BRIEFING.md`'s H5 discussion — but there was no plot script. Unlike FIG05
(which could be plotted directly), this raw CSV has per-trial rows, not a mean/CI summary, so a
computation step was actually needed: `code/scripts/build_fig13.py` aggregates the 5 trials per
(dataset, n_per_class, ref_quality) into mean + t-distribution CI via `metrics.seed_ci` (reused
as-is; a "trial" is statistically the same kind of small-sample point-estimate collection as a
"seed"), writing `results/fig13_gram_inversion_summary.csv` (39 rows). `analysis/fig13_gram_inversion.py`
then plots it: one panel per dataset (cosine similarity vs. `n_per_class`, log2-x, one line per
`ref_quality`, 95% CI bands), plus a third panel for the anisotropy bar chart.

**The rendered figure confirms every number already cited in the briefing**: n=1 reconstruction
exact (cosine=1.000) on both datasets; curves decay and plateau around 0.4-0.6 (CIFAR-100) vs.
0.65-0.70 (CUB-200); CUB-200's curves sit visibly above CIFAR-100's at matched n, consistent with its
lower anisotropy ratio (34.0 vs. 86.4, shown directly in the third panel). No surprises — this is a
confirmation exercise, not a new finding, but it closes a real gap: a heavily-cited result with no
figure to back it.

## A6 property inference over time (H10) — standalone, not folded into FIG01

`03_RESULTS_SPEC.md`'s *original* FIG01 spec called for three panels (accuracy / A6 property-inference
balanced accuracy / A1 membership TPR), but the FIG01 actually built this session only has two —
documented as a deliberate, stated cut (`analysis/fig01_decoupling.py`'s own docstring: "No
property-inference (A6) panel: A6 is scoped to F2/M4 only"). The reason a middle panel was never
added: A6's real data (`results/a6_property_inference.csv`, from `notes/2026-09-16_p3_a6_h10.md`) only
covers **M4 on CIFAR-100** — folding it into FIG01's grid (7 methods x 3 datasets) would silently
misrepresent its actual scope.

Built `analysis/fig_a6_property_inference.py` as a small, honestly-scoped standalone figure instead —
just reads the existing CSV, no computation. Confirms the already-cited step function exactly:
balanced accuracy = 0.5 (chance) at elapsed ∈ {-2,-1} (before the client's class is introduced),
jumps to and stays at 1.0 (perfect, never decaying) for every elapsed ∈ {0,...,10} after.

## Status

Both added to `code/Makefile`'s `figures` target. `make figures` from an empty `figs/`/`tables/` now
produces 9 figures + 4 tables, all real. 179 tests pass, `ruff` clean.
