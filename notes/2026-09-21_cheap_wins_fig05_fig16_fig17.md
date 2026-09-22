# 2026-09-21 — Three cheap post-deadline-pivot additions: FIG05 plot, FIG16 (ROC), FIG17 (seed variance)

With the P4/C1 3-dataset extension and the C2 pilot both complete (see the other 2026-09-21 notes
files), the human asked what more could be done in the remaining 1-2 days. These three were the
identified "cheap, do them regardless" tier — no new experiments, just wrapping existing or
cheaply-derivable data into a real figure.

## FIG05 plotting script (`analysis/fig05_eps_of_T.py`)

`results/fig05_eps_of_T.csv` existed since P3 but had no plotting script. Building it and inspecting
the render (per this project's standing "always inspect the rendered figure" heuristic) surfaced a
real, previously-undocumented finding, not a bug: only unit U2 ever triggers parallel composition in
`dp/accountant.py::account` (`if report.task_disjoint and unit == Unit.TASK`), so for M4/M8 every
other unit (U1, U3, U4, U5) still shows unbounded ε growth with T, identical in shape to M0's. See
`notes/2026-09-21_fig05_unit_specificity_of_flat_eps.md` for the full writeup — this is now folded
into `paper/PAPER_BRIEFING.md`'s C3 section as a sharper, more important statement of the claim than
"task-disjoint implies flat ε" alone.

## FIG16 — log-log ROC (`code/scripts/build_fig16.py` + `analysis/fig16_roc.py`)

This is CLAUDE.md non-negotiable #5's mandatory appendix requirement ("log-log ROC is mandatory in
the appendix... its absence is read as hiding a weak low-FPR regime") — the one piece of that rule
this entire project had not yet satisfied, despite reporting TPR@1%/0.1%FPR everywhere. Built by
recomputing the full ROC curve from the already-generated shadow stores (reusing `run_lira.py`'s own
`load_shadow_store`/`compute_log_lr_surfaces` and the *same* seeded calibration split, so this curve
is consistent with the already-reported TPR numbers, not an independent re-derivation that could
silently disagree), downsampled onto a 200-point log-spaced FPR grid via monotonic interpolation of
the exact empirical step function.

**Scope**: one curve per (dataset, method) = 21 combos, **seed=0 only**, `elapsed=0` (the standard,
most-immediate MIA setting), trajectory ablation. Not all 5 seeds per curve — a reasonable follow-up
if there's more time, but seed=0 is the same "primary" seed every other per-seed CSV in this project
already treats as the default.

**A real operational note**: running all 21 combos sequentially in one process was projected to take
over an hour (ImageNet-R's shadow-store loading is far slower than CIFAR-100/CUB-200's for
identically-shaped data — the same issue documented in
`notes/2026-09-21_p4_third_dataset_imagenet_r.md`). Refactored `build_fig16.py` to take optional
`<dataset> <method>` CLI args and process a single combo, then ran all 21 in parallel (`xargs -P 12`)
on the login node, same pattern as the ImageNet-R LiRA-report backfill — finished in a few minutes
instead. Per-combo CSVs were merged into the final `results/fig16_roc.csv` (4,221 rows = 21 x 201
points) and the intermediates deleted.

**What the figure shows** (real, matches every other finding this session): M8 sits at ~TPR=1.0
across the entire FPR range on all 3 datasets (consistent with M8's previously-known `AUC≈1.0000`);
M4 is elevated (TPR≈0.1-0.9 even at FPR=1e-4) but below M8; the F1-family methods (M0/M1/M2/M3/M5)
cluster together, generally above the chance diagonal even at FPR=1e-4, confirming real (if smaller)
low-FPR signal for the standard baseline methods too.

## FIG17 — seed variance (`code/scripts/build_fig17.py` + `analysis/fig17_seed_variance.py`)

Per-seed strip plot of the headline TPR@1%FPR (elapsed=0, trajectory) for all 105 (dataset, method,
seed) combos, reading directly from the already-computed `results/a1_lira_*.csv` files (no new
computation beyond extracting one row per file). Real result: seed-to-seed spread at elapsed=0 is
visibly tight for every method/dataset in this session's data — the headline leakage numbers are
stable across seeds, which is itself worth a sentence in the paper (it's evidence the 5-seed CIs
elsewhere in the project aren't hiding a wildly unstable point estimate).

## Status

All three added to `code/Makefile`'s `figures` target, `make figures` regenerates all 5 figures + 4
tables from a clean run, 179 tests pass, `ruff` clean. `paper/PAPER_BRIEFING.md` updated with all
three as real, done entries.
