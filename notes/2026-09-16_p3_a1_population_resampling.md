# 2026-09-16 — P3: A1's shadow design caught giving a fake AUC=1.0000, fixed before the real run

## What happened

Built the shadow-federation runner (`src/p3fcl/shadow_runner.py`) and the LiRA combination step
(`attacks/lira.py`) for A1, scoped to M4 (F2/prototype) and M8 (F5/Gram) per
`04_METHODS_AND_ATTACKS.md §4`'s ordering. First version's design: pick a small pool of "target"
datums (5 per client-task shard), and for each shadow, independently flip each *target's own*
membership with probability 0.5, leaving every other datum in the stream fixed across all 4000
shadows.

Ran a small (64-shadow) validation batch on real CIFAR-100 before committing to the full budget
(`build/00_BUILD_PLAN.md`'s "sharpest and cheapest first" -- also just good practice given a race
condition already turned up in the same validation pass, see below). `run_lira.py`'s report came back
**exactly** `AUC = 1.0000`, `TPR@1%FPR = 1.0000`, `TPR@0.1%FPR = 1.0000` -- for every one of the 10
`elapsed` buckets, for both the `trajectory` and `last_round` ablations. A perfect, uniform, exactly-
identical-to-four-decimals result across every condition is precisely the standing heuristic this
project has hit four times already (M5's `touched`, M6's dead gradient, `orthogonal_procrustes`'s
shape bug, FIG05's `passes_over_data` gap): *a "too good" result is a signal to check for a broken
gradient path or an under-reported field before trusting it.*

## Root cause

Checked the raw per-target IN/OUT score distributions directly (not just the pooled AUC). For every
target inspected, the "IN" scores across all shadows were **identical to four decimal places**, and
likewise for "OUT" -- not "tightly clustered," but exactly the same floating-point value. That is a
point mass, not a distribution, and it explains the perfect separation trivially: a two-point-mass
null needs no calibration at all.

Cause: `PrototypeFCL` releases one prototype **per class**, computed only from that class's members
within a client-task shard. `build_targets` samples targets from the *whole* shard (which spans every
class assigned to that task, not just one), so with 5 targets drawn from a shard covering ~10 classes,
a given target is very likely the **only** member of its own class that was ever a candidate for
resampling -- every other member of that class was fixed, present in literally every shadow. Flipping
the target's own bit is therefore the *only* source of randomness the released class-c prototype ever
sees; "IN" always means exactly the same set of contributors (the fixed majority plus this target),
and "OUT" always means exactly the fixed majority alone. Two fixed points, zero noise, and LiRA has
nothing to calibrate against except a difference that is *always* fully resolvable -- an artifact of
the experimental design's sparsity, not a genuine finding that F2 leaks perfectly for every example
regardless of shard size. (Confirmed range: shard-level class counts among the sampled targets ran
from 2 to 300 members, median 84 -- the perfect separation held at every size, which is the tell that
it isn't really about a small-N leave-one-out effect, since even N=300 gave a clean point mass.)

Standard LiRA (Carlini et al. 2022) avoids this by resampling roughly half of the *entire* candidate
pool per shadow model, not just a handful of tracked points -- every shadow's released statistic
reflects genuine population-level sampling noise, and detecting one target's membership means beating
that real noise floor, not a designed-to-be-noiseless two-point comparison.

## The fix

`shadow_runner._population_mask(dataset, method, shadow_id, n_total, p_in)`: one seeded draw over
*every* id in the base stream (not one `rng.seeded` call per id -- that would mean tens of thousands
of SHA-256 hashes per shadow; instead one `rng.seeded(...).random(n_total) < p_in` vector draw).
`_apply_population_mask` drops every id the mask excludes, from every shard, not just targets. A
target's recorded `in_out` label is now simply `keep_mask[target_id]` -- read off the same draw
everyone else's inclusion comes from, not a separate independent coin flip.

Added `test_out_shadows_have_real_between_shadow_noise` (`test_shadow_runner.py`) pinning the property
that broke: at least one target's OUT-shadow scores must show real variance (`std > 1e-9`) across
shadows, not a point mass. 156/156 tests pass, ruff clean, after the fix.

## What this means going forward

- **Deleted and is re-running the entire shadow store** (`shadows/cifar100/{m4_proto,m8_analytic}/`)
  under the corrected design -- the already-computed shadows were a different, easier experiment and
  cannot be reused or partially merged with the new ones.
- A real AUC under the corrected design will almost certainly be lower than 1.0 and will vary
  meaningfully with shard size -- that is the actual, trustworthy signal A1 is supposed to produce,
  and it's fine (expected, even) if it's messier than the first (wrong) result.
- This is now the fifth confirmed instance of the project's standing heuristic (a too-clean result
  flags a design or implementation problem before it flags a real effect) -- worth citing as-is in the
  paper's methodology section as evidence of how seriously this codebase treats "too good to be true."

## Validation after the fix: one real AUC, one still-perfect AUC that checks out

Reran a larger (512-shadow) validation batch on real CIFAR-100 under the corrected population-level
design (jobs 157149/157150), then `run_lira.py` for both methods.

- **M4 (F2/prototype): AUC ~0.87-0.89**, varying meaningfully across `elapsed` (0.884 -> 0.865 ->
  0.874, not monotonic, not uniform), TPR@1%FPR ~0.27-0.35, TPR@0.1%FPR ~0.14-0.20. This is exactly the
  shape of a *real*, noisy, non-degenerate result -- direct per-target inspection confirmed genuine
  within-group spread on both the IN and OUT sides now (e.g. target 0: `in_mean=-65.77, in_std=1.55,
  out_mean=-68.98, out_std=1.46`), not the previous point masses.
- **M8 (F5/Gram): still AUC=1.0000, TPR@1%=TPR@0.1%=1.0000, uniformly across every elapsed value.**
  Checked this before trusting it, same as the M4 fix above (the standing heuristic applies to a
  surviving-the-fix "too good" result just as much as a pre-fix one). Root cause this time is real
  math, not a design flaw: per-target inspection shows the IN-side score (self-leverage `x^T R^-1 x`)
  clusters extremely tightly near **0.9995-0.9996** (std ~0.0000-0.0003) while the OUT-side is far away
  at **600-2300** with real, substantial variance (std ~125-430) -- the OUT-side's nonzero variance
  confirms the population-resampling fix *is* working (it wasn't a point mass this time); the IN-side's
  tight clustering near 1.0 is a known, bounded property of ridge leverage scores for an *included*
  point (`h_in = x^T R_in^-1 x < 1` always), which saturates close to 1 whenever the per-class sample
  count is small relative to the feature dimension (`n << d = 768` here, median per-class shard size
  ~84 *before* the 50% population resampling roughly halves it again) -- exactly the same `n << d`
  regime `RESEARCH_PLAN.md §3.4` already identified as giving near-exact analytic recovery, and exactly
  what H5's real-data run (`notes/2026-09-16_p3_fig13_h5.md`) already found independently via direct
  Gram inversion rather than a calibrated attack. **This is a second, independent confirmation of the
  same underlying fact (F5's Gram statistics leak almost everything in the realistic n<<d regime)
  arrived at through a completely different methodology (shadow-calibrated LiRA vs. direct analytic
  inversion) -- a nice cross-check for the paper, not a residual bug.** Verified the Sherman-Morrison
  identity `h_out = h_in/(1-h_in)` directly against real matrices (see the shadow_runner.py inline
  comment and `attacks/lira.py::offline_log_lr`'s docstring) before accepting this.

Proceeding to the full `n_shadows: 4000` run for both methods on top of these already-valid 512
shadows (resumable -- the runner skips existing indices, so no compute is wasted).

## Also found and fixed in the same validation pass: a targets.json write race

A separate, unrelated bug surfaced in the same 64-shadow validation run: concurrent PBS array tasks
writing the shared `targets.json` convenience file via an identical `.tmp` name raced, and one array
task's `os.replace` deleted the file out from under another's, crashing with `FileNotFoundError` (job
157138, array index 2). Fixed by making the tmp filename unique per process
(`targets.json.tmp.{os.getpid()}`); `targets` itself is a pure function of the config, so whichever
writer's replace lands last is fine. Added
`test_concurrent_array_tasks_do_not_race_on_targets_json` -- note in that test's own commit message
that the race is timing-dependent and did not reliably reproduce against a local (fast) filesystem in
under five attempts, only against Kodiak's real `/data` (slower, networked) filesystem; the fix removes
the possibility of the collision unconditionally rather than narrowing the race window, so it does not
depend on reproducing the exact timing to be correct.
