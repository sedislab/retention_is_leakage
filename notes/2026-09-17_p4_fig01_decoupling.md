# 2026-09-17 — P4: FIG01/FIG02/TAB03/TAB04 built and run for real — Gate P4 passed

## What was built

The entire P4 analysis pipeline, from scratch (`analysis/` had only `verify_provenance.py` before this):
- `src/p3fcl/metrics.py`: `seed_ci` (t-distribution CI across a small number of seed-level point
  estimates — distinct from `clopper_pearson`, which is for a single seed's rate on a finite test
  set) and `fit_exponential_halflife` (log-linear regression fit of `y = a*exp(-x/tau)`;
  `fit_ok=False`/`halflife=inf` whenever the fitted trend is flat or increasing, per
  `00_BUILD_PLAN.md`'s explicit instruction that "fitting a half-life to a flat curve is a reporting
  error"). 8 new tests.
- Extended A1's infrastructure to support multiple seeds: `shadow_array.pbs`/`run_lira.py` now accept
  a `SEED`/`stream_seed` parameter (seed 0 keeps the original directory layout so nothing already
  computed needed to move; seeds 1+ get their own subdirectory and a seed-suffixed results CSV).
- `code/scripts/run_accuracy_matrix.py`: the real (non-shadow, full-population) accuracy matrix per
  method x seed — the accuracy half of FIG01.
- `code/scripts/build_fig01.py` / `build_fig02.py`: pure post-processing assembly scripts (read
  already-computed CSVs, write `results/fig01_decoupling.csv` / `results/fig02_halflife.csv` +
  `results/decoupling_ratio.csv`, no computation happens later) — CLAUDE.md non-negotiable #3.
- `analysis/fig01_decoupling.py`, `analysis/fig02_halflife.py`, `analysis/tab03_leakage.py`,
  `analysis/tab04_halflife.py`: the actual figure/table generators, reading only the CSVs above.
  `make figures` (new Makefile target) regenerates all four from a clean `figs/`/`tables/`.

**Scope of this first pass, stated precisely, not silently**: CIFAR-100 only (H2's own stated bar
needs >=3 datasets for its correlation claim — a deliberate cut); M0 (F1, no retention mechanism) and
M8 (F5, exact retention) only, the starkest and cheapest contrast for the decoupling story, not all 7
methods A1 already covers. 5 seeds each (`{0,1,2,3,4}`, the project's stated convention), the required
minimum for a headline number (CLAUDE.md non-negotiable #4). No A6 property-inference panel: A6 is
scoped to F2/M4 only, which isn't part of this first pass.

## A real, general bug found while extending to 5 seeds

`Ledger.aggregate_view()` was already fixed once this session (A4's `KeyError` on mismatched dict
keys, `notes/2026-09-17_p3_a4_onset_h4.md`) — not encountered again here since M0/M8's payloads are
plain arrays / fixed-key dicts. No new bug from `aggregate_view()` this time; noted only because the
earlier fix mattered for this pass to run cleanly (M0/M8's `aggregate_view()` calls, if any existed in
this pipeline, benefit from it — in practice this pipeline doesn't call `aggregate_view()` at all,
since A1 reads per-client shadow scores directly, but the fix was general and already merged).

**A recurring infra pattern continued**: extending to 5 seeds meant 10 new 4,096-shadow shadow stores
(2 methods x 4 new seeds, on top of the existing seed-0 stores). Silent array-task deaths recurred
across most of them (as documented for the earlier single-seed runs) — all resolved by the same
resumability-based backfill-and-reverify loop, with one case (`m0_fedavg` seed 2, indices 10-11)
needing three rounds of backfill before succeeding in isolation, matching the earlier-established
"isolation breaks the pattern" diagnostic (`notes/2026-09-16_p3_a1_population_resampling.md`). One
process error on my part, caught immediately: a stale copy of the "bigmem" PBS template (created
before the seed-support code existed) briefly wrote to the wrong (seed-0) output directory — harmless
in practice (every shadow index it touched already existed there, so it was a costless no-op), but
worth naming so a future session regenerates ad hoc template copies from the *current* file rather
than reusing an old `/tmp` copy.

## The result: Gate P4's literal criterion is met

Real run, CIFAR-100, 5 seeds, `results/fig01_decoupling.csv`, `results/fig02_halflife.csv`,
`results/decoupling_ratio.csv`, `figs/fig01_decoupling.{pdf,png}`, `figs/fig02_halflife.{pdf,png}`,
`tables/tab03_leakage.{csv,tex}`, `tables/tab04_halflife.{csv,tex}`.

| Method | Accuracy half-life | Leakage half-life | Decoupling ratio (h_leak / h_acc) |
|---|---|---|---|
| M0 (FedAvgSequential, no retention) | 4.68 [3.24, 5.78] | 17.58 [12.00, 33.95] | **3.75 [2.60, 6.87]** |
| M8 (AnalyticFCL, exact retention) | 53.42 [46.55, 59.65] | no decay detected (censored) | **infinite** |

**M0's decoupling ratio CI excludes 1.0 entirely** (2.60–6.87) — Gate P4's literal, stated criterion
("the decoupling ratio has a CI that excludes 1.0 for at least one family") is satisfied with a real,
paired-bootstrap CI, not an artifact of forcing a fit. M8's leakage shows literally zero decay across
the full 10-task observed horizon even as its own (very slow) accuracy decay is real and well-fit
(R²=0.999) — reported as "no decay detected, lower bound only," per the build plan's own explicit
instruction, rather than a fabricated finite number. Both results tell the same story at different
intensities: **the mechanism that protects accuracy against forgetting is the same mechanism that
keeps leakage from decaying** — exactly the paper's central thesis, now demonstrated with a real,
quantitative, 5-seed, bootstrapped-CI result.

## What's left for P4

- Only 2/7 methods A1 already covers (M1/M2/M3/M5/M4 are a natural, cheap extension using identical
  infrastructure — the accuracy matrices for M4 are already computed as a side effect of
  `run_accuracy_matrix.py`, just not yet wired into `build_fig01.py`/`build_fig02.py`'s
  `METHOD_FAMILY` dict).
- Only 1 dataset — H2's own stated bar needs >=3 for the eventual retention-vs-leakage correlation.
- FIG06 (long-horizon T=50 variant) not attempted.
- No A6/property-inference panel in FIG01 yet (would need M4 wired into this same pipeline, since A6
  is scoped to M4).
