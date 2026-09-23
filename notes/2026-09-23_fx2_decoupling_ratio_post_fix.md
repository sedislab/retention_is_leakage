# 2026-09-23 — Post-fix decoupling ratio: mostly `undefined`, not the pre-fix `∞` story

## What happened

Ran the complete FX2 leak pipeline (30-array job 183944, plus the standalone M0 job 183940) against
wave V2 data, merged (`merge_fx2_summary.py`), then computed `decoupling_ratio.csv` for every
(dataset, method, view) combination that has both a leak half-life and an accuracy half-life
available (cifar100 complete; cub200/imagenet_r complete except M0, still running as job 183978).

## The numbers

Every non-M0 method, on every dataset, for the primary leak metric (`leak_tpr1`, `trajectory`
ablation): `h_acc` status is `censored` (accuracy never drops to half its initial value within the
E=6 observed horizon). `h_leak` is `censored` too for nearly everything (one exception: M1 on
CUB-200, `h_leak=4.47`, a real crossing). Since `decoupling_ratio()` requires a non-censored
denominator to report anything other than `undefined` (`08_FIX_PLAN.md` §7c: a censored denominator
makes the ratio meaningless, not just a lower bound), essentially every retention method's
decoupling ratio comes back `undefined`.

M0 (no retention) is the one real exception, now complete across all 3 datasets (`imagenet_r`/`cub200`
needed a follow-on job, 183978, to fill in M0's shadow-scoring since it reuses the pre-fix `shadows/`
store per FX4i's design):

| dataset | h_acc | h_leak | ratio | ratio_type |
|---|---|---|---|---|
| cifar100 | 0.87 [ok] | censored (>6) | >= 6.87 | **lower_bound** |
| cub200 | 2.22 [ok] | 1.08 [ok] | **0.49** | **point** |
| imagenet_r | 0.76 [ok] | censored (>6) | >= 7.91 | **lower_bound** |

**cub200 is a real, dataset-dependent reversal, not a fluke.** Both `h_acc` and `h_leak` are genuine,
non-censored crossings there (checked directly against the raw per-e curves: leak tpr1 goes
0.058->0.025 over e=0..6, crossing 50%-of-base between e=1 and e=2; accuracy decays more slowly,
crossing between e=2 and e=3) — leakage decays *faster* than accuracy on this one dataset, the
opposite of the pattern on cifar100/imagenet_r. This echoes a pattern already on record in
`agents/OPEN_QUESTIONS.md` for a different method (pre-fix: "M3 on CUB-200 is a genuine
counter-example... the reverse of every other method/dataset pairing") — CUB-200's small-per-class-n,
fine-grained regime appears to produce real reversals more than once, for more than one method, which
is itself worth treating as a pattern rather than three unrelated coincidences.

## Why this is a real, important correction, not just "weaker data"

The pre-fix `agents/OPEN_QUESTIONS.md` entries for H2 reported finite decoupling ratios for every
method (M4: 3.71 on cifar100, 134.49 on cub200; M3: 2.80; etc.) and reported `∞` outright for
M1/M2/M5/M8. Both of these came from an EXPONENTIAL half-life FIT extrapolated arbitrarily far past
the actually-observed data whenever the leak curve hadn't visibly decayed within the observation
window — exactly the practice `08_FIX_PLAN.md`'s R-findings identified as invalid and Definition 10
was written to forbid ("never report infinity, never extrapolate; censor at E and report a lower
bound instead"). The pre-fix numbers were not a rougher version of the same finding; they were
constructed by a method the plan itself declared broken.

The honest, non-extrapolating replacement is *harder to turn into a clean headline number*: with
FX4's method fixes, retention methods now protect accuracy well enough that accuracy itself doesn't
cross the 50% mark within 6 further tasks — so there is no defined "half-life" to divide by, and no
defined ratio. This is not a failure of the pipeline; it is what Definition 10 is *supposed* to do
when the data doesn't support a point estimate: refuse to report one, rather than fabricate one via
extrapolation.

## What this means for the paper's C1 claim

The one number the current data actually supports without extrapolation is M0's: leakage measurably
outlives accuracy by a factor of at least ~6.9 for the no-retention baseline. That is a real,
defensible, and arguably still-relevant data point (it shows leakage does not track accuracy even in
the ABSENCE of any deliberate retention mechanism), but it is a claim about the baseline, not about
the retention methods the paper's thesis is actually about. For the retention methods themselves, the
honest current statement is narrower: "leakage persists for at least as long as our 6-task
observation window allows us to measure, same as accuracy does" — real, but not the dramatic
multiplier the pre-fix numbers implied.

Resolving this for the retention methods specifically needs a longer observation horizon than this
fix phase's `E_MAX=6`/`n_tasks=10` design supports (see `agents/OPEN_QUESTIONS.md`'s H2 entry for the
exact reasoning) — out of scope to change mid-phase, flagged for a follow-up.

## Status

Real, verified finding on real post-fix data (cifar100 complete, cub200/imagenet_r pending only M0's
remaining run). NOT a claim that H2 is refuted — the underlying retention-leakage association may
still be real and even strong; this is specifically about whether the *half-life ratio*
operationalization can currently produce a defined number for it, and mostly it cannot, honestly,
within this phase's fixed horizon. Logged in `agents/OPEN_QUESTIONS.md`'s H2 entry as a correction to
the evidentiary record, not a status change (H2 stays `OPEN`).
