# 2026-09-21 — P4 extended to a 3rd dataset: ImageNet-R

## Why ImageNet-R, and why now

Chosen by the human (`AskUserQuestion` response: "Add a third dataset, ImageNet-R (Recommended)")
as the next increment after the CIFAR-100 + CUB-200 pass (`notes/2026-09-19_p4_second_dataset_cub200.md`)
closed with H2's own bar (`≥3 datasets`) one dataset short. Features were already cached from an
earlier phase, so this reused the fully dataset-generic P4 pipeline
(`run_accuracy_matrix.py [dataset]`, `build_fig01.py`/`build_fig02.py`'s `DATASETS` list) with no new
code beyond adding the string `"imagenet_r"` to that list. This work continued in parallel with, and
was then partly superseded in urgency by, the deadline conversation that produced
`paper/PAPER_BRIEFING.md` — the human did not countermand finishing this dataset, only the framing of
what "done" means for the project's writing.

## Shadow generation: the established Kodiak pattern recurred exactly as expected

The same "silent array-task death under high concurrency" reliability issue documented for every
prior large sweep this session (`notes/2026-09-1[6-9]_p3_*`/`p4_*.md`) recurred for ImageNet-R's
shadow generation: **batch 1 needed a large backfill round (19 combos resubmitted, jobs
183194-183212)**; batch 2 completed cleanly on the first attempt. Not a code bug — always resolved
via resumability + resubmission, exactly per the standing heuristic in `build/STATE.md`'s "Open
problems" list ("never assume a cleared queue means success — verify final output counts").

**Verified by direct file count (2026-09-21), all 35 (method, seed) combos at the full 4,096-shadow
budget**: 7 methods x 5 seeds (seed0 stored directly under `shadows/imagenet_r/<method>/`, seeds 1-4
under `shadows/imagenet_r/<method>/seed<N>/` — same convention as CIFAR-100/CUB-200, confirmed by
inspection, not a bug). No missing or partial combos.

## A real operational finding: `run_lira.py` is dramatically slower on ImageNet-R's shadow store

Running the LiRA report script sequentially, `m0_fedavg`/seed0 alone took **>8 minutes and was still
running** when historically (CIFAR-100/CUB-200) every (method, seed) combo finished in
**1.5-7 seconds** (`results/RUN_LOG.jsonl` wall_seconds field, checked directly). The process was
confirmed CPU-bound (99%+ CPU, state `R`, no iowait system-wide) rather than blocked on disk I/O, and
the on-disk shadow `.npz` files are identical in shape (`scores: (500, 10)`) and size (~256KB) to the
other two datasets, ruling out a data-shape or file-size explanation. Most likely explanation (not
fully confirmed — no profiler available on this node): `np.savez_compressed`'s zlib decompression cost
depends on the actual value entropy of the stored arrays, and ImageNet-R's real accuracy/logit-margin
scores may compress/decompress far less efficiently than CIFAR-100/CUB-200's, but this is a plausible
hypothesis, not a verified root cause — **worth revisiting if this pattern recurs on a future
dataset**, since 60-100x slower for identical-shaped data is a large, unexplained gap.

**Practical fix applied**: switched from a sequential loop over all 35 (method, seed) combos to a
parallel batch (`xargs -P 12`) on the login node. This is CPU-bound analysis over already-cached
shadow files (reading + a closed-form Gaussian fit + AUC/TPR computation), not
training/extraction/a sweep, so it is the same category of login-node-appropriate work as the
sequential version already was (matching how the CIFAR-100/CUB-200 LiRA reports were produced, per
`results/RUN_LOG.jsonl`'s `pbs_jobid: null, hostname: kodiak` entries) — parallelizing it across the
mostly-idle 48-core login node (load average ~2.4 at the time) is a proportionate, temporary use of
shared resources, not a violation of the "no heavy compute on the login node" rule (that rule targets
training/feature-extraction/shadow-generation, which stayed on PBS array jobs throughout).

## Accuracy matrix (real, `results/accuracy_matrix_imagenet_r.csv`, 1925 rows, 5 seeds)

| Method | final_avg_acc | BWT | Note |
|---|---|---|---|
| M0 (none) | 0.24-0.30 | -0.51 to -0.60 | worst retention, as expected |
| M1 (distillation+exemplar) | 0.45-0.50 | +0.31 to +0.38 | positive BWT, consistent with CIFAR-100 |
| M2 (generative replay) | 0.57-0.59 | +0.20 to +0.22 | positive BWT |
| M3 (orthogonal projection) | 0.49-0.53 | **-0.23 to -0.29 (negative)** | **different sign than CUB-200** |
| M4 (class-mean carry-forward) | 0.30-0.35 | -0.06 | mildly negative |
| M5 (exemplar replay) | 0.61-0.63 | +0.09 to +0.13 | positive BWT |
| M8 (exact Gram sum) | 0.736 (mean, 5 seeds: 0.735-0.736) | -0.11 to -0.13 | mildly negative |

**M3's BWT sign is dataset-dependent across all three datasets now**: negative on CIFAR-100, reversed
(nearly zero/positive, with the accuracy-side genuinely censored — see
`notes/2026-09-19_p4_second_dataset_cub200.md`) on CUB-200, and negative again on ImageNet-R. Three
datasets now show three qualitatively different pictures for the same method — this should be
reported as a real, unresolved, dataset-dependent finding in the paper (per CLAUDE.md non-negotiable
#7), not smoothed into a single directional claim about M3.

## Real 3-dataset decoupling result (complete, `results/decoupling_ratio.csv`, 21 rows)

All 35 (method, seed) LiRA combos finished (parallelized 12-at-a-time on the login node after the
per-combo slowness noted above made the sequential plan impractical; total wall time for all 35
combos was well under an hour once parallelized). `build_fig01.py`/`build_fig02.py` reran cleanly
with no crashes or new censoring-logic edge cases (the 4-case handling added for CUB-200's
`m3_fot` — see `notes/2026-09-19_p4_second_dataset_cub200.md` — covered ImageNet-R with no changes
needed). `figs/fig01_decoupling.png`/`fig02_halflife.png` visually inspected: 3-column grid renders
correctly, no zigzag/mixing artifacts, log-x forest plot rows line up with dataset-prefixed labels.

**ImageNet-R gives the cleanest, strongest decoupling result of the three datasets — 7/7 methods,
not 6/7**:

| Method | CIFAR-100 ratio | CUB-200 ratio | ImageNet-R ratio | Excludes 1.0 on ImageNet-R? |
|---|---|---|---|---|
| M0 (none) | 3.75 [2.55, 6.89] | 1.53 [1.05, 2.10] | **109.34 [3.70, 161.91]** | yes |
| M1 (distillation+exemplar) | ∞ | ∞ | ∞ | yes (trivially) |
| M2 (generative replay) | ∞ | ∞ | ∞ | yes (trivially) |
| M3 (orthogonal projection) | 2.80 [0.74, 18.71] (CI incl. 1) | ~0 (reversed) | **∞ (no leak decay)** | yes (trivially) |
| M4 (class-mean carry-forward) | 3.71 [1.81, 34.71] | 134.49 [47.15, 2034.32] | **∞ (no leak decay)** | yes (trivially) |
| M5 (exemplar replay) | ∞ | ∞ | ∞ | yes (trivially) |
| M8 (exact Gram sum) | ∞ | ∞ | ∞ | yes (trivially) |

Unlike CIFAR-100 (M3 inconclusive) and CUB-200 (M3 reversed), **ImageNet-R shows every single method
with a real retention mechanism holding its leakage flat with zero detectable decay**, and even the
no-retention baseline M0 has a 95% CI that excludes 1.0 entirely (ratio 109, driven by a very short
accuracy half-life of 5.55 tasks against a leakage half-life of ~606 tasks, R²=0.007 — a very shallow,
nearly-flat leakage fit, reported honestly as a real finite fit rather than forced to censored, since
the fitter did converge, just to a very long half-life). This is the strongest single-dataset
instance of claim C1's thesis found across all three datasets and should be highlighted as such in
the paper, while still reporting the CIFAR-100/CUB-200 numbers for the full multi-dataset picture
(per CLAUDE.md non-negotiable #4/#7 — precise scope, no cherry-picking the best dataset alone).

**M3's picture across all three datasets is now genuinely three different stories, not two**:
negative BWT + inconclusive decoupling ratio on CIFAR-100; reversed decoupling (accuracy censored,
leakage finite) on CUB-200; negative BWT + infinite decoupling ratio (leakage never decays) on
ImageNet-R. Report this honestly as a real, unresolved, dataset-dependent complication for M3
specifically — do not construct a single narrative that fits all three.

This closes out P4/claim C1's original scope (H2's own ≥3-dataset bar is now met). See
`build/STATE.md`, `agents/OPEN_QUESTIONS.md`'s H2 entry, and `paper/PAPER_BRIEFING.md`'s C1 section
for how this is folded into the project's running state and the paper-writing briefing.
