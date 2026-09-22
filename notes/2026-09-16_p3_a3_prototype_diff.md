# 2026-09-16 — P3: A3 prototype-difference attribution, and a methodological self-correction

## What happened

Built `attacks/prototype_diff.py` (exact sum recovery under the per-round-mean convention M4 uses
at `momentum=0`; EMA-inversion for the running-mean convention at `momentum>0`) and
`run_prototype_diff.py` (the real leakage-vs-increment-size sweep on CIFAR-100, varying client count
to vary realised per-shard size). All unit tests pass, including an exact-sum check against real
`PrototypeFCL` ledgers and an increment=1 exact-single-sample check.

## A self-caught methodological issue, before it became a wrong headline number

The first real sweep (job 153079) measured `best_match_cosine`: the cosine similarity between the
recovered direction and the closest true contributing sample. The raw curve looked alarming in a
subtle way — it dropped from 1.0 (increment=1) to ~0.82 by increment~5-10, but then **did not keep
decaying**, drifting back up to ~0.85 even at increment > 100. Read naively, that says "even huge
class-mean releases stay almost as revealing as tiny ones," which would be a striking and
paper-worthy claim.

It would also have been wrong to report without a control. CIFAR-100's ViT features are strongly
class-clustered (anisotropy ratio 86.4, measured in the same session for FIG13) — in a tight cluster,
*any* same-class sample is highly cosine-similar to *any other* same-class sample, regardless of
whether it specifically contributed to the release being attacked. A flat, elevated
`best_match_cosine` at large increment could therefore be pure generic-class-membership similarity,
carrying zero individual-level signal — exactly the semantic-vs-individual distinction
`04_METHODS_AND_ATTACKS.md`/FIG04 already treats as the paper's central methodological fault line.

Added a control: `baseline_best_match_cosine`, the same metric computed against an equally-sized
random same-class pool drawn from *outside* the true contributing shard. `attribution_premium =
best_match_cosine - baseline_best_match_cosine` isolates whatever signal is specific to having
actually contributed, over and above "being the same class."

## The corrected, honest result

`attribution_premium` decays fast and monotonically, and it is the real story:

| increment bucket | best_match_cosine | attribution_premium |
|---|---|---|
| 1 | 1.000 | 0.548 |
| 2 | 0.871 | 0.274 |
| 2-5 | 0.833 | 0.143 |
| 5-10 | 0.819 | 0.065 |
| 10-20 | 0.822 | 0.033 |
| 20-50 | 0.830 | 0.014 |
| 50-100 | 0.838 | 0.005 |
| 100+ | 0.850 | 0.010 |

The raw `best_match_cosine` staying elevated at large increment is almost entirely the within-class-
similarity artifact predicted above — once that's subtracted out, true individual-level attribution
signal is already below 0.1 by increment~10-20 and indistinguishable from noise by increment~50.
**This is a cleaner and more defensible finding than the raw curve**: A3 gives near-exact individual
reconstruction at small increments (as the spec predicted) and that advantage evaporates quickly,
consistent with — not contradicting — the semantic/individual split the project's own dose-response
design (P5, FIG04) is built around.

`results/a3_prototype_diff.csv` (7,529 rows) has both the raw and premium columns; report the premium,
not the raw `best_match_cosine`, as the leakage-vs-increment curve. The F7/counts ablation is
unaffected by any of this and stands as measured: at `increment=1`, `l2_err_with_counts` is exactly
0.0 across all 1,032 real samples (perfect reconstruction), while `l2_err_without_counts` averages
~1530 — turning counts off costs the adversary the ability to correctly scale a direction into a sum,
exactly as `04_METHODS_AND_ATTACKS.md` predicted, even though it costs nothing in cosine terms.

## What the next person needs to know

- **Standing methodological note for any future "closest match" style attack metric in this
  project**: always compute a same-class (or same-property) random baseline alongside a raw
  best-match number, especially on ViT features, which are known (now measured, FIG13) to be
  strongly class-clustered. A raw best-match number without this control risks conflating semantic
  (class-level) similarity with individual-level leakage — exactly the distinction the paper is
  trying to draw a line around elsewhere. This applies directly to A6 (property inference) and A1
  (LiRA) too, both still to be built.
- `run_prototype_diff.py` only exercises `momentum=0` (the exact, per-round-mean case). The
  `momentum>0` differencing path is unit-tested but not yet run at sweep scale on real data.
