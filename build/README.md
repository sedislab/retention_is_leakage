# build/ — instructions for Claude Code

Read in this order. `../CLAUDE.md` is loaded automatically by Claude Code and holds the standing
rules; these four files hold the plan.

| File | What it settles |
|---|---|
| [`00_BUILD_PLAN.md`](00_BUILD_PLAN.md) | **Start here.** Phases P0–P9, tasks, deliverables, gates, what to cut if short |
| [`01_KODIAK.md`](01_KODIAK.md) | Baylor Kodiak: PBS not Slurm, storage layout, job templates, the discovery protocol to run first |
| [`02_DATASETS.md`](02_DATASETS.md) | The four required + two stretch datasets, download rules, splits, the `ref` split trap |
| [`03_RESULTS_SPEC.md`](03_RESULTS_SPEC.md) | Every figure and table, its CSV schema, the plotting and statistics bar |
| [`04_METHODS_AND_ATTACKS.md`](04_METHODS_AND_ATTACKS.md) | The method zoo M0–M9, the attack suite A1–A7, the shadow runner, implementation order |
| [`05_RUNBOOK.md`](05_RUNBOOK.md) | **Running it on Kodiak**: bootstrap, the single master prompt, resume protocol, failure playbook |
| [`06_PACKAGE_SPEC.md`](06_PACKAGE_SPEC.md) | **What to build.** Module-by-module spec for `code/` — the ledger, accountant, metrics, the four claim tests |
| [`STATE.md`](STATE.md) | Maintained by Claude Code. The resume file — read it first after any context reset |

State lives in [`../agents/OPEN_QUESTIONS.md`](../agents/OPEN_QUESTIONS.md) (hypotheses H1–H12,
theorem obligations, contested decisions). Scientific ground truth lives in
[`../RESEARCH_PLAN.md`](../RESEARCH_PLAN.md); `[LOCKED]` sections there are not open for revision
without a human.

---

## Kickoff

See [`05_RUNBOOK.md`](05_RUNBOOK.md) — bootstrap, then one prompt. Claude Code runs P0–P9
autonomously and stops at each phase gate to ask whether to continue.

## Fast path to a first result

If you want evidence in week one rather than infrastructure in week one, do this in parallel with
P0/P1 — it needs no shadows, no GPU and no scheduler:

1. Extract ViT-B/16 features for **CUB-200** only (smallest dataset, smallest per-class *n*).
2. Run `attacks/inversion.py` across *n* ∈ {1, 2, 4, 8, 16, 32, 64, 128} × three reference
   qualities, and measure the feature anisotropy per class.
3. That is figure **FIG13** and it settles hypothesis **H5**, the project's sharpest and most
   scoopable claim. Two days of linear algebra.

Re-specify H5 first: the toy-scale check in `../notes/2026-08-25_preliminary.md` shows the "n ≤ 64"
threshold is too aggressive on isotropic synthetic features. The deliverable is an n-threshold
**curve** as a function of reference quality and anisotropy, wherever it lands.

---

## Audit log

**2026-09-14 — pre-experiment review.** Eight defects found and fixed before any code was written:

1. **ID collision (serious).** The method and attack IDs in `04_METHODS_AND_ATTACKS.md` did not match
   `RESEARCH_PLAN.md §3/§5`, which `agents/OPEN_QUESTIONS.md` cites. M2 meant TARGET in one file and
   "prototype" in the other; A5 meant client attribution in one and property inference in the other.
   Realigned to the plan's numbering — **M0–M8 and A1–A6 are the plan's, M9 and A7 are the only new IDs.**
2. **H10 was wired to the wrong attack.** Claim C1's deciding test is **A6** (property inference over
   time), not A1. FIG01 now plots accuracy, A6 property leakage and A1 membership leakage together.
3. **Figure IDs collided with artifact families.** `F01` (figure) vs `F1` (model-delta family).
   Figures are now `FIG01`…`FIG22`, tables `TAB00`…`TAB10`, with the convention stated in the spec.
4. **`CLAUDE.md`'s claim→artifact table was wrong** — it pointed C2 at TAB05 (utility baselines) and
   C3 at TAB07 (reproduction gap). Corrected.
5. **`results/t02_disjointness.csv` collided with table TAB02.** Renamed `disjointness_report.csv`.
6. **`#PBS` directives are not shell-expanded**, so `-o /data/$USER/...` would have written to a
   directory literally named `$USER`. Literal paths now.
7. **Torque vs PBS Pro portability**: resource syntax (`nodes/ppn` vs `select/ncpus`), array syntax
   (`-t`+`PBS_ARRAYID` vs `-J`+`PBS_ARRAY_INDEX`), and the `%N` throttle being Torque-only. Both
   forms documented; the scripts read either array variable.
8. **A3 was understated.** `methods/proto.py` releases a *per-round* mean, so mean × released count
   is the **exact sum of that shard's features** — no differencing needed. Spec now covers both
   release conventions and makes increment size the reported variable.

Also folded in from the Kodiak account form: the `/data`-not-`/home` rule, the no-production-on-login-node
rule (which changed the dataset download guidance), and the **Protected-data prohibition**.
