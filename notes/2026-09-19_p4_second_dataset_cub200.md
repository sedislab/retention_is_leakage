# 2026-09-19 — P4 extended to a second dataset (CUB-200) — H2's ≥2-dataset bar reached, one real counter-example found

## What was done

Extended P4 (`notes/2026-09-17_p4_fig01_decoupling.md`, `notes/2026-09-18_p4_fig01_all_methods.md`)
from CIFAR-100-only to CIFAR-100 + CUB-200, all 7 A1-validated methods, 5 seeds each. Parameterized
`run_accuracy_matrix.py` (now takes `[dataset]` as a CLI arg) and `build_fig01.py`/`build_fig02.py`
(now loop over a `DATASETS` list) rather than hardcoding `cifar100` — a real, useful generalization,
not just a one-off script edit, since a third dataset is now a one-line addition plus the usual
shadow-generation run.

CUB-200 (200 classes, ~25 samples/class, real feature cache already existed from A2/H5's earlier
FIG13 work) needed 35 new shadow stores (7 methods x 5 seeds, 4,096 shadows each) and their LiRA
reports — the same real infrastructure as CIFAR-100's extension, no new attack code. **The
silent-array-task-death pattern documented in every prior large sweep this session recurred here at
roughly the same rate as CIFAR-100's 20-array extension** (needed 3 backfill rounds to reach
35/35 complete stores) — fully expected at this point and handled the same way (resubmit, verify
counts, repeat). No data loss, no new infra surprises.

## A real bug found and fixed while assembling the two-dataset figures

**`build_fig02.py` crashed** (`IndexError` in `np.percentile` on an empty array) the first time a
method showed the *opposite* censoring pattern from every case seen so far: `cub200/m3_fot` has a
**censored accuracy curve** (no detectable decay at all over the horizon) paired with a **finite**
leakage half-life (10.94 tasks, R²=0.446). The ratio code only handled "leak censored, acc finite"
(→ ∞) and "both finite" (→ real ratio); it silently assumed the accuracy side would always be the one
with real bootstrap samples to divide by, and crashed when that assumption broke. Fixed by handling
all four cases explicitly: both finite (real paired-bootstrap ratio), leak-only censored (∞), **acc-only
censored (ratio → 0 — leakage decays away against an accuracy curve that shows no decay at all, a
real and different pattern, not an error)**, and both censored (undefined — nothing to compare).

**`analysis/fig01_decoupling.py` had a real, separate bug**, also only surfaced by adding a second
dataset: it grouped plot data by method name alone. With two datasets, the same method now has two
rows per elapsed value (one per dataset) that got sorted together by elapsed and plotted as one
zigzagging line jumping between the CIFAR-100 and CUB-200 values at every point — a visibly wrong
sawtooth pattern in the rendered figure, caught by looking at the actual output image before trusting
it (the same discipline applied throughout this session: always inspect a regenerated figure, don't
assume the pipeline ran correctly just because it exited 0). Fixed with a proper 2-row x N-dataset-
column subplot grid, grouping by `(dataset, method)`. `analysis/fig02_halflife.py`'s row labels had
the same latent bug (`"{method} / {quantity}"` would produce duplicate, ambiguous labels across
datasets) — fixed to include the dataset in the label before it caused visible confusion.

## The result: decoupling holds on both datasets, with one real, honestly-reported counter-example

`results/decoupling_ratio.csv`, `figs/fig01_decoupling.pdf`, `figs/fig02_halflife.pdf`:

| Method | CIFAR-100 ratio | CIFAR-100 excludes 1.0? | CUB-200 ratio | CUB-200 excludes 1.0? |
|---|---|---|---|---|
| M0 (none) | 3.75 [2.55, 6.89] | yes | 1.53 [1.05, 2.10] | yes (barely) |
| M1 (distillation+exemplar) | ∞ | yes | ∞ | yes |
| M2 (generative replay) | ∞ | yes | ∞ | yes |
| M3 (orthogonal projection) | 2.80 [0.74, 18.71] | no | **~0 (reversed!)** | n/a — different pattern |
| M4 (class-mean carry-forward) | 3.71 [1.81, 34.71] | yes | 134.49 [47.15, 2034.32] | yes |
| M5 (exemplar replay) | ∞ | yes | ∞ | yes |
| M8 (exact Gram sum) | ∞ | yes | ∞ | yes |

**6/7 methods (all but M3) show decoupling on both datasets**, several far more extreme on CUB-200
(M4's ratio jumps from 3.71 to 134 — CUB-200's small-per-class-n regime amplifying the effect, matching
the same pattern H5 already established for the analytic-inversion attack). M0's CUB-200 ratio (1.53,
CI [1.05, 2.10]) barely excludes 1.0 — real but much weaker decoupling than on CIFAR-100.

**M3 (FOT, orthogonal subspace projection) is a real, honestly-reported counter-example on CUB-200,
not a data quality issue**: its accuracy curve shows *no detectable decay at all* over the 10-task
horizon (censored, R² not computable — the fit never found a net-negative trend), while its leakage
signal *does* decay (halflife≈11 tasks, R²=0.446, a moderate but real fit). This is the reverse of
every other method's pattern: here, FOT's orthogonal-projection retention is protecting task-k accuracy
so completely on this small, fine-grained dataset that there's nothing left to decay, while the
individual-level membership signal genuinely fades for some other reason (plausibly numerical drift
in the projected subspace, or the leakage-relevant local update norm just naturally shrinking as the
subspace accumulates more directions — not investigated further this pass). **This does not undermine
the overall decoupling thesis** (6/7 methods still support it, on both datasets) but it is a real
finding that should go in the paper as exactly what it is: one method, one dataset, where the pattern
reverses, reported per CLAUDE.md non-negotiable #7 ("negative results are results... do not delete
failed branches").

## Status

No new hypothesis status change (H2/C1 already `SUPPORTED`-leaning from the CIFAR-100-only and
7-method passes) — this note adds a second dataset (partial progress toward H2's own ≥3-dataset bar)
and documents the M3/CUB-200 reversal precisely so it isn't lost or smoothed over later.
`agents/OPEN_QUESTIONS.md` updated with the 2-dataset table and the M3 counter-example.

## What's left for P4

- A third dataset (ImageNet-R's features are already cached, same backbone) would fully satisfy H2's
  own stated ≥3-dataset bar.
- M3's reversal on CUB-200 is itself worth a dedicated follow-up: does it replicate at more seeds, and
  is there a mechanistic explanation (checking the actual per-round subspace-projection update norms
  directly, rather than inferring from the aggregate half-life fit)?
- FIG06 (long-horizon T=50 variant) not attempted.
- No A6/property panel in FIG01 (A6 data exists for M4 on CIFAR-100 only; would need extending to
  CUB-200 too for a fair comparison).
- `build_fig01.py`/`build_fig02.py` are now dataset-generic; the remaining manual step for a new
  dataset is submitting the shadow-generation + accuracy-matrix jobs and backfilling to completion.
