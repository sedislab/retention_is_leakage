# 2026-09-23 — FX8 (C2 dose-response rerun): the real sweep, replacing the 2026-09-21 pilot

## Scope

CIFAR-100, 2 methods, 6 levels each, 3 seeds, 512 shadows/level (18,432 shadows total): M5
(HybridReplay, individual/exemplar retention, knob `buffer_size_per_class`) and M2 (Gaussian feature
replay, semantic/distributional retention, knob `n_synthetic_per_class`), levels {1, 2, 5, 10, 20, 50}
for both (chosen to match exactly, per `08_FIX_PLAN.md` §10's own design, so the x-axis needs no
per-method rescaling). Leakage measured pooled over K_SET=[0,1,2,3] at elapsed=6 (reusing FX2's own
fixed-population convention directly, not elapsed=0 — see the design note for why). Replaces the
2026-09-21 pilot (M5 only, 3 levels, plus a separate M4 semantic arm that turned out to be dead code).

## Result

**Retention (-BWT) rises identically for both methods as the knob increases** — both go from ~0.25
(level=1, heavy forgetting) down to ~-0.05 (level=50, near-perfect or slightly-positive backward
transfer), tracking each other almost exactly. This is expected: both knobs are calibrated to produce
matched retention strength by construction (same six levels, same general effect on how much
old-task information survives into later training).

**Leakage does NOT rise identically — this is the real finding.** M5 (individual/exemplar) climbs
steeply: TPR@1%FPR rises from ~0.03 at level=1 to ~0.09 (mean) at level=50, with a wide CI at the top
end (individual seeds reaching over 0.10-0.15). M2 (semantic/Gaussian) stays much flatter throughout —
~0.03 at level=1, peaking around ~0.045 at level=20, then actually turning back down slightly at
level=50 (~0.035-0.048 across seeds). At the highest retention-strength level, M5's leakage is roughly
double M2's, despite both methods achieving essentially the same retention benefit.

## Why this matters

This directly answers two things at once:
1. **C2 (dose-response)**: within a method, does retention strength causally drive leakage? Yes for
   both methods, but the SIZE of the effect differs sharply — a real, quantified answer, not just a
   qualitative "yes."
2. **The Red Team's strongest objection** ("retention is semantic, membership is individual; a method
   can retain class means perfectly and leak nothing example-specific"): the data supports this
   prediction. The individual-retention method's leakage tracks its retention strength much more
   tightly than the semantic-retention method's does. This is NOT "semantic retention leaks nothing" —
   M2's leakage does rise somewhat with retention strength too, just far less steeply than M5's, and it
   is not statistically distinguishable from flat at the lower/middle levels given the CI widths.

## Honest limits

- 2 methods, 1 dataset (CIFAR-100), 3 seeds — well short of H2's own stated bar (>=8 methods x >=3
  datasets) and the "Empiricist's proposed upgrade" bar (>=6 levels on 2-3 methods, which THIS sweep
  does clear). This is real, methodologically sound evidence for the causal direction and the
  semantic/individual split, not a claim that the deciding test for H2 as a whole is closed.
- M2's leakage curve is noisier and has wider relative CIs at the low end than M5's — the level=50
  downturn should not be over-read as a real non-monotonicity without more seeds; it is consistent
  with noise around a genuinely flatter underlying relationship, but a confirmatory rerun with more
  seeds would strengthen this specific claim if it matters for the paper's phrasing.
- The M4 "semantic" arm from the 2026-09-21 pilot is dropped here, per the plan ("Drop the M4 arm,
  because its knob was dead code" — confirmed mechanistically: `prototype_momentum` never fires under
  a class-incremental stream since every class belongs to exactly one task). M2 takes its place as the
  semantic-retention example in FIG04; M2's own mechanism (a per-class distributional summary) is
  arguably an even cleaner "semantic" case than M4's point-estimate prototype, since it retains a full
  distribution, not just a mean.

## A related, real bug fixed while wiring this up

`p3fcl.methods.m2_target.py`'s `MethodSpec.retention_type` was `"individual"` before this — a latent
mislabeling never exercised by any prior FIG04 pipeline (the 2026-09-21 pilot only used M4/M5 for the
semantic/individual split, never M2). Fixed to `"semantic"`, matching `08_FIX_PLAN.md` §10's own
explicit framing and the mechanism's actual nature. See `code/src/p3fcl/methods/m2_target.py`'s own
comment for the reasoning. This also surfaced a stale `results/disjointness_report.csv` (dated
2026-09-15, predating a real M2 fix made earlier in this same phase) — regenerated and confirmed the
corrected, deterministic result (M2 is task-disjoint at both `replay_ratio` levels tested, no
violations, unlike the stale report's spurious `V2/V3` flag on `replay_ratio=1`).

## Where the numbers live

- `results/dose_response_accuracy.csv`, `results/dose_response_lira.csv` — raw per-(method, level,
  seed) inputs.
- `results/fig03_dose_response.csv`, `results/fig04_semantic_vs_individual.csv` — the joined,
  schema-matching outputs (`03_RESULTS_SPEC.md`'s FIG03/FIG04 shapes).
- `figs/fig03_dose_response.{pdf,png}`, `figs/fig04_semantic_vs_individual.{pdf,png}`.

## A process note: `make figures` was silently broken until this pass caught it

Running the real `make figures` Makefile target (not a manual per-script loop) for the first time
revealed it would have aborted partway through: the now-fully-superseded `analysis/
fig03_dose_response_pilot.py` always exits 1 (its CSV was archived away by FX0 and never
regenerated), and the Makefile's loop uses `python "$f" || exit 1`. Deleted the obsolete script (its
historical output is preserved under `archive/2026-09-23_pre_fix/stale/`). Confirmed both
`make figures` and `make verify` now pass cleanly end to end — worth having found this today rather
than at the freeze.
