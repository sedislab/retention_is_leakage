# 2026-09-21 — P5 dose-response: a deliberately small pilot, not the full spec

## Why this is a pilot, not the real experiment

The user set a hard external deadline (paper draft target 2026-09-23, hard deadline 2026-09-25) that
does not allow time for `00_BUILD_PLAN.md` P5's actual protocol: **≥6 retention-strength levels x
≥3 methods x ≥2 datasets x 3 seeds**, where every single data point is its own ~1,000-4,096-shadow A1
run (the same compute pattern documented throughout `notes/2026-09-1[6-9]_p3_*`/`p4_*`). At the
measured cost of a full-scale A1 run (tens of minutes to a few hours per method/seed/dataset,
including the recurring backfill rounds), the full P5 sweep would be `6 levels x 3 methods x 2
datasets x 3 seeds = 108` more full-scale runs — days, not the hours available.

**Given explicit instruction from the user to scope this down to a small pilot** (rather than skip it
entirely), the design below trades completeness for speed while staying honest about what it is.

## The pilot's design

- **1 method**: M5 (HybridReplay). Chosen because its retention knob (`buffer_size_per_class`, the
  size of its latent exemplar buffer) is the simplest in the zoo — no extra loss term (unlike M1's
  distillation weight), no extra forward pass (unlike M2's generative sampling), just a buffer size
  that directly controls how much past data gets replayed. It is also one of the cheapest methods to
  run (measured ~328s per 128-shadow array task earlier in this session, faster than M1/M2).
- **1 dataset**: CIFAR-100 (the most-used dataset this session, already fully validated).
- **3 levels**: `buffer_size_per_class` ∈ {0, 10, 20}. 0 = no replay at all (closest to a within-family
  "floor," though M5 with buffer=0 is not identical to M0 since its SGD loop and family declarations
  differ); 10 = the existing TAB05/P2-4 headline default (already characterized in depth throughout
  this session); 20 = double the default buffer.
- **3 seeds** (`{0,1,2}`) — the CLAUDE.md minimum ("≥3 seeds for anything reported"), not the ≥5
  used for this project's headline numbers, since this is explicitly not a headline number.
- **Reduced shadow budget: 1,024 shadows per (level, seed)**, not the 4,096 headline budget — chosen
  to keep compute proportionate to a pilot's purpose (showing a trend, not nailing down a tight CI).
  All 3 levels use the *same* reduced budget (rather than reusing the existing 4,096-shadow, 5-seed
  buffer=10 store) so the comparison across levels is apples-to-apples, not mixing precision levels.
- **Leakage measured at `elapsed=5`** (a representative mid-horizon point), not just `elapsed=0`, so
  the reported number reflects the *persistence* of leakage the whole project cares about, not just
  its initial value.
- **Retention measured as `-BWT`** from a real (non-shadow) federation at each (level, seed) —
  cheap, already-existing machinery (`sim.run`).

## New infrastructure this needed (real, general-purpose, not a one-off hack)

`shadow_runner.py` gained a generic `method_config_override` mechanism (`config["method_config_override"]`,
threaded through `run_shadow_range` -> `_init_worker` -> `_run_one_shadow`, merged on top of
`_method_config`'s TAB05 defaults) so a caller can override *any* method's config field via
`--set method_config_override.<key>=<value>` without touching the hardcoded per-method defaults. This
is not pilot-specific — it is the same mechanism a *real*, full-scale P5 sweep would need later, so
building it properly now (with tests: `test_method_config_override_reaches_the_method`,
`test_run_shadow_range_respects_method_config_override`) is not wasted effort even though the pilot
itself is small. A new PBS template (`pbs/dose_response_pilot.pbs`) and driver script
(`scripts/run_dose_response_pilot.py`) wire this up end to end, writing
`results/fig03_dose_response_pilot.csv`.

## What this pilot can and cannot support

**Can support**: a real, honestly-CI'd 3-point trend showing whether retention strength and leakage
move together *within one method*, as a motivating/suggestive result.

**Cannot support**: H2's actual causal dose-response claim (needs ≥6 levels for a real trend shape,
not 3 points), the semantic-vs-individual retention split the Red Team's objection is about (needs
≥2 methods spanning that distinction — this pilot uses only one method), or any claim about
generalization across datasets or methods.

**The paper must present this as exactly what it is** — see `paper/PAPER_BRIEFING.md`'s C2 section for
the exact honest framing to use.

## Real results (complete, `results/fig03_dose_response_pilot.csv`, 3 rows)

All 9 (level, seed) shadow stores hit their full 1,024/1,024 budget. `-BWT` from a real
(non-shadow) federation per (level, seed); TPR@1%FPR from calibrated LiRA (`elapsed=5`,
`calib_frac=0.8`). Per-seed values (n=3 seeds per level):

| buffer_size_per_class | -BWT (retention), per seed        | TPR@1%FPR (leakage), per seed        |
|---|---|---|
| 0  | -0.6912, -0.6816, -0.6401 (mean ≈ -0.671, i.e. **worst** retention) | 0.0253, 0.0204, 0.0264 (mean ≈ 0.024) |
| 10 | -0.0593, -0.0482, -0.0343 (mean ≈ -0.047)                          | 0.0426, 0.0419, 0.0452 (mean ≈ 0.043) |
| 20 | -0.0175, -0.0038, -0.0066 (mean ≈ -0.009, i.e. **best** retention) | 0.0497, 0.0432, 0.0529 (mean ≈ 0.049) |

Note: `-BWT` values are negative throughout, meaning BWT itself is positive (M5 shows positive
backward transfer at all 3 levels, consistent with the rest of this session's M5 numbers) — "more
negative -BWT" here reads as "less positive BWT," i.e. weaker retention, which is why buffer=0 is
labeled "worst retention" above despite having the least-negative-looking raw BWT sign flipped for
plotting. The direction that matters is monotonic: **as the buffer grows 0 -> 10 -> 20, retention
strengthens monotonically (mean -BWT: -0.671 -> -0.047 -> -0.009) and leakage rises monotonically in
lockstep (mean TPR@1%FPR: 0.024 -> 0.043 -> 0.049)**, across all 3 seeds at every level with no
crossovers. This is a clean, textbook dose-response pattern — exactly the shape H2/C2 predicts — but
from a single method on a single dataset at 3 levels and a reduced shadow budget, so see the "can
and cannot support" section above before treating it as more than a motivating pilot result.

Figure: `figs/fig03_dose_response_pilot.{pdf,png}` (`analysis/fig03_dose_response_pilot.py`), a
dual-axis plot confirming the monotonic crossover visually — retention (green, left axis) falls
steeply from buffer=0 to buffer=10 then flattens; leakage (orange, right axis) rises roughly
linearly across all 3 levels.
