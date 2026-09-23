# 2026-09-23 — FIG18 v2: the matched-5-client redesign resolves H13's confound

## Why this redesign happened

`08_FIX_PLAN.md`'s post-review audit (R8) flagged that the 2026-09-21 FIG18 pilot's "natural vs.
Dirichlet" comparison was confounded: the "natural" arm used `n_clients=1` (the whole hospital as one
client) while the "dirichlet" arm used `n_clients=10`. Two variables changed at once — partition
STRATEGY (real hospital identity vs. random Dirichlet split) and client COUNT (1 vs. 10) — so the
pilot's headline finding ("natural federation leaks more") could not be attributed to either cause
alone. `agents/OPEN_QUESTIONS.md`'s H13 entry was accordingly marked `REFUTED` prematurely; the plan
itself said to set it back to `OPEN` and rerun with matched client counts.

## The redesign

Both arms now use `n_clients=5` (matching the 5 real hospitals), varying only HOW a hospital's
task-data splits into those 5 clients:
- **natural**: grouped by physical microscopy slide (`Camelyon17Dataset.metadata_fields` exposes
  `slide` directly; added to `p3fcl.get_data.prepare_camelyon17()`'s per-sample record). Slides are a
  genuine, indivisible unit — new `streams.natural_chunked_partition` never splits one slide's patches
  across clients, distributing whole slides across 5 buckets via a seeded shuffle + round-robin (not a
  fixed sorted-order assignment, so seed-to-seed variance is real, unlike the pilot's `n_clients=1` arm
  which had none).
- **dirichlet**: unchanged, the existing per-class Dirichlet(0.5) split, just also at `n_clients=5`.

Same M0, domain-incremental stream (one task per hospital), 3 seeds, 1,024-shadow budget, same
~5,000-image class+hospital-stratified subsample as the pilot.

## Result

**Leakage (the actual H13 question): the two arms' TPR@1%FPR curves overlap almost entirely within
their 95% CIs at every elapsed value.** At elapsed=4 (the widest gap in the pilot): natural = 0.034
[0.022, 0.047], dirichlet = 0.049 [0.018, 0.079] — close point estimates, heavily overlapping CIs,
dirichlet if anything slightly *higher* (the opposite direction from the pilot's finding). The pilot's
"natural leaks more" result does not replicate once client count is controlled, at this 3-seed pilot
power.

**Accuracy/BWT converged too**, checked along the way: final_avg_acc natural=[0.92, 0.92, 0.89] vs.
dirichlet=[0.93, 0.93, 0.89]; BWT natural=[+0.012, +0.017, -0.024] vs. dirichlet=[-0.001, -0.005,
-0.034] — much closer than the pilot's dramatic BWT gap (natural pinned at ~-0.02 constant regardless
of seed vs. dirichlet's +0.001 to +0.030).

## Interpretation — read carefully before citing

This is a **null result at 3-seed pilot power**, not a second REFUTED-in-the-opposite-direction
finding. It means: the SPECIFIC comparison this pilot ran, once properly controlled for client count,
shows no clear leakage difference between a real slide-based partition and a synthetic Dirichlet split.
It does NOT mean natural federations are provably as safe as synthetic ones in general, and it does not
mean Dirichlet partitioning never understates real-world risk — a real effect smaller than this
pilot's CI width could still exist, and this was checked with M0 only (the retention-lower-bound
baseline), not the full 7-method zoo. `agents/OPEN_QUESTIONS.md`'s H13 status: `OPEN` (correctly
reflecting a genuine open question, not a confounded REFUTED masquerading as an answer).

## What changed in the code

- `p3fcl.get_data.prepare_camelyon17()`: added `slide` to each sample's record (from
  `Camelyon17Dataset.metadata_fields = ['hospital', 'slide', 'y']`, no new download needed).
- `p3fcl.streams`: new `natural_chunked_partition(field, idx, n_clients, seed)`; `build_stream(...,
  client_field=...)` uses it for the within-task client split when given, instead of
  `dirichlet_partition`. 5 new tests.
- `p3fcl.shadow_runner`: new `stream.client_field=<name>` config flag, mirroring the existing
  `stream.use_domain_field`.
- `run_accuracy_matrix_camelyon17.py`: rewritten for matched `n_clients=5` and (also fixed along the
  way) real test-split `eval_sets` — the pilot version used `acc_matrix_train` only and was explicitly
  marked "do not cite this CSV's numbers."
- `run_fig18_report.py` / `analysis/fig18_natural_federation.py`: repointed at `shadows_v2/`'s new
  `fig18_v2_<partition>` stores (the pilot's `shadows/camelyon17/.../fig18_natural|dirichlet` stores are
  superseded, not reused), added an `n_clients` column so the CSV itself shows the match.

## A self-caught process note, for transparency

While adding the `slide` field, an initial re-run of `prepare_camelyon17()` used its default
`subsample_target=60000` instead of the ~5000 the live pilot dataset actually uses, briefly replacing
`datasets/camelyon17/index.json`'s 4998-sample subsample with a different 60002-sample one. Caught
immediately by comparing the post-run `MANIFEST.json` against what was already on disk; fixed by
re-running with the correct `subsample_target=5000` (verified: exact same image/split counts restored,
and the result is self-consistent with the already-extracted feature caches). While double-checking
this, found — independently, not self-caused — that the OLD pilot's shadow store no longer matches
today's dataset regeneration (built against whatever `prepare_camelyon17` produced back on 2026-09-21,
before some other fix in this same phase apparently changed the sample-selection outcome). Not
blocking: this redesign replaces that store entirely.
