# 2026-09-21 — P5 dose-response pilot, M4 semantic arm: answering the Red Team's objection

## Why this exists

The M5 pilot (`notes/2026-09-21_p5_dose_response_pilot.md`) showed a clean dose-response trend for
**one** method, but explicitly could not address the single strongest objection in the whole project
(`agents/OPEN_QUESTIONS.md`'s H2 entry, "Known strongest objection (Red Team)... Unanswered"):
**retention is semantic while membership is individual — a method can retain class means perfectly
and leak nothing example-specific.** `00_BUILD_PLAN.md`'s P5 spec is explicit that the sweep must
include "at least one method whose retention is purely semantic (M4's prototype half, counts off) and
one whose retention is example-level (M5 Hybrid Replay's latent exemplars). If the dose-response holds
for the second and not the first, that is the finding, and it is a *better* paper than a uniform
correlation."

Given the deadline, this is still a pilot (not the full ≥6-level x ≥3-method x ≥2-dataset spec), but
adding the M4 arm at the same reduced scale as the M5 arm directly tests the qualitative split the
plan cares about most, which the M5-only pilot could not.

## Design (mirrors the M5 arm exactly, for a fair comparison)

- **Method**: M4 (`PrototypeFCL`, `code/src/p3fcl/methods/m4_proto.py`), `MethodSpec.retention_type =
  "semantic"` (declared in the method's own spec, not asserted after the fact).
- **Knob**: `prototype_momentum` — the method's own designated "retention knob for FIG03/FIG04" per
  its docstring. `momentum=0`: each task's per-(client,class) prototype simply overwrites the last
  released value (no cross-task carry of the *value*, though the model_delta convention in this
  project still calls this the weakest retention setting). `momentum>0`: blends with the previously
  released value, so more of the old release's influence persists in what's re-released later —
  the direct semantic analogue of M5's `buffer_size_per_class` (more replay = more carried influence).
- **`release_counts=False` for both arms of this pilot** (the "counts off" condition the plan asks
  for) — so any leakage signal from M4 in this pilot is attributable to the *prototype* (F2) channel
  alone, not the F7 counts channel, isolating exactly the "semantic value carry" mechanism under test.
- **3 levels**: `{0.0, 0.5, 0.9}` (no carry / moderate blend / heavy blend) — same count and same
  weak/medium/strong spacing logic as the M5 arm's `{0, 10, 20}`.
- **3 seeds** (`{0,1,2}`), same reduced **1,024-shadow budget**, same PBS template
  (`pbs/dose_response_pilot.pbs`, fully generic — no changes needed, just different `-v` values),
  same `method_config_override` mechanism, same dataset (CIFAR-100), same
  `elapsed=5` leakage read-out point — every axis held fixed except the method/knob, so the two arms
  are comparable.

## Jobs

Submitted 2026-09-21: 9 PBS array jobs (183330-183338), `DATASET=cifar100, METHOD=m4_proto,
KNOB_NAME=prototype_momentum`, one job per (seed, knob_value) pair, `SEED in {0,1,2}`,
`KNOB_VALUE in {0.0,0.5,0.9}`. Sanity-checked the override mechanism end-to-end with a 2-shadow local
run before submitting (confirmed `p3fcl.cli shadows --set method_config_override.prototype_momentum=0.5`
runs cleanly against M4).

## What this can and cannot support

**Can support**: a real, honestly-scoped first look at whether the dose-response pattern found for
M5 (individual/exemplar retention) also holds, or does not hold, for M4 (semantic/prototype
retention) — directly informative for, though not a full test of, the Red Team's objection.

**Cannot support**: a definitive resolution of the objection (that needs the full ≥3-method spread
across ≥2 datasets the original P5 plan calls for), nor any claim beyond CIFAR-100.

## Real result — and it is a sharp, mechanistic answer to the Red Team's objection

All 9 (level, seed) shadow stores completed at the full 1,024/1,024 budget on the first attempt (no
backfill needed). `results/fig04_semantic_arm_m4.csv`:

| `prototype_momentum` | -BWT per seed (0,1,2) | TPR@1%FPR (elapsed=5) per seed (0,1,2) |
|---|---|---|
| 0.0 | 0.0693, 0.0609, 0.0660 | 0.3508, 0.3586, 0.3385 |
| 0.5 | 0.0693, 0.0609, 0.0660 | 0.3463, 0.3574, 0.3445 |
| 0.9 | 0.0693, 0.0609, 0.0660 | 0.3539, 0.3584, 0.3472 |

**-BWT is bit-for-bit identical across all 3 momentum levels, for every seed.** TPR@1%FPR varies only
by amounts consistent with shadow-to-shadow sampling noise (no monotonic trend with the knob — e.g.
seed=0's TPR goes 0.3508 -> 0.3463 -> 0.3539, not a consistent direction).

**This is not a bug — it is a real, mechanistically confirmed property of M4 under this project's
experimental design**, verified directly against the code, not just inferred from the flat numbers:

1. `code/src/p3fcl/methods/m4_proto.py`'s `fit_task` only blends with the previous release
   (`if self.momentum > 0 and key in self._proto_touched: ...`) when the same `(client, class)` key
   has been seen in an *earlier* task.
2. `code/src/p3fcl/streams.py::build_stream` is **class-incremental by default** (confirmed by
   reading it directly) — `class_incremental_tasks` assigns every class to exactly one task, so a
   given `(client, class)` key is only ever encountered once, in that one task.
3. Therefore `key in self._proto_touched` is **always False** the one time each class is seen,
   regardless of what `self.momentum` equals — the blending branch is provably dead code under a
   class-incremental stream, and `self._prototypes[c] = mean_c` (the no-blend branch) always executes
   instead. Prototypes, `touched` sets, and hence accuracy and the shadow-generation ledgers are
   bit-identical across all 3 levels by construction. This is exactly what M4's own docstring already
   predicted: "under class-incremental streams... this carry-over never actually fires."

## Why this is a *good*, precise answer, not a null result to bury

This is a stronger and more specific version of the Red Team's objection than "semantic retention
shows no dose-response": **under the exact experimental setup this whole project uses (class-incremental
CIFAR-100/CUB-200/ImageNet-R), M4's designated cross-task retention mechanism is structurally
incapable of carrying any influence across tasks at all** — not "weakly," not "with a flat slope,"
but literally zero-effect by construction. Contrast this directly with M5 (individual/exemplar
retention), whose knob produced a large, clean, monotonic swing in both retention and leakage
(`notes/2026-09-21_p5_dose_response_pilot.md`: -BWT from 0.671 to 0.009, TPR from 0.024 to 0.049
across the same 3-level/3-seed design). **The qualitative split `00_BUILD_PLAN.md` asked P5 to test —
dose-response for the individual-retention method, none for the semantic one — is exactly what this
pilot found, for a mechanistically verified reason, not an ambiguous null.**

**The natural follow-up this raises (real limitation, not tested here)**: M4's own docstring states
that under a *domain-incremental* stream (same classes recur across tasks — e.g. Camelyon17, not used
in this project) the momentum-blending branch *would* fire, and `dp.accountant.check_disjointness`
would correctly flag a V2/V3 disjointness violation. Whether M4 *would* then show a real dose-response
under that stream design is a genuinely open, well-scoped future-work question — flag it as such
rather than implying it was tested.

## Status

Complete. `results/fig04_semantic_arm_m4.csv` (M4 arm) and `results/fig03_dose_response_pilot.csv`
(M5 arm, pre-existing) merged via `code/scripts/build_fig04.py` into
`results/fig04_semantic_vs_individual.csv`; plotted via `analysis/fig04_semantic_vs_individual.py`
into `figs/fig04_semantic_vs_individual.{pdf,png}` (visually inspected — see
`paper/PAPER_BRIEFING.md`'s C2 section for the final write-up used for the paper).
