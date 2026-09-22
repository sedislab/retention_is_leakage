# 2026-09-21 — FIG05: flat ε only holds under U2, not universally across units

## What happened

Building `analysis/fig05_eps_of_T.py` (previously `results/fig05_eps_of_T.csv` existed as a CSV but
had no plotting script) and visually inspecting the rendered figure — per this project's standing
heuristic, always inspect the rendered image, don't trust exit code 0 — showed only 2 visually
distinct lines per panel instead of the expected 5 (one per unit U1-U5). This was not a plotting bug.

## The real finding

Checked directly against `code/src/p3fcl/dp/accountant.py::account`'s logic:

```python
if report.task_disjoint and unit == Unit.TASK:   # Unit.TASK == "U2"
    ...  # parallel composition, eps constant in T
```

This is the *only* branch that ever routes to parallel composition. Every other unit (U1 example-level,
U3 client-level, U4 individual-unbounded, U5 individual-renewal) always uses sequential composition,
**regardless of whether the method is task-disjoint**. Consequence, verified directly on
`results/fig05_eps_of_T.csv`:

- For M4 and M8 (both task-disjoint by construction): **U2 is flat (ε=2.529 for both, T=1..1000)**;
  **U1, U3, U4, U5 all show unbounded growth**, numerically close to or matching M0's growth shape.
- For M0/M1/M2/M3/M5 (not task-disjoint): all 5 units grow, with U1=U2=U3=U5 numerically identical
  to each other (differing only in per-task touch-count multiplier) and U4 following its own
  unbounded-client-level rate.

## Why this is real evidence, not a bug to fix

This is intentional accountant design, not an oversight — it directly encodes H1's own claim
(`agents/OPEN_QUESTIONS.md`): "standard DP-FL accounting under unit U4 ... gives ε growing without
bound in T; hence no published FCL method makes a non-trivial lifelong client-level privacy claim."
The accountant is correctly modeling that parallel composition is a property of *both* the mechanism's
disjointness structure *and* the unit's granularity matching that structure (RESEARCH_PLAN.md §4.3,
"L1 — When is parallel composition legal in FCL?") — task-disjointness at the *task* level (which is
what M4/M8 actually have) only legally justifies parallel composition under the *task-level* unit
(U2). It says nothing about client-level (U3) or individual-level (U4/U5) disjointness, so those units
correctly keep accumulating.

## Why this matters for the paper

This is a sharper, more honest, and arguably more important statement of claim C3/C4 than "task-disjoint
methods get a flat ε": **task-disjoint methods get a flat ε only under the unit whose granularity
matches their disjointness axis; under every other unit, they are exactly as unbounded as a
no-retention method.** This directly motivates why the "units of privacy" taxonomy (TAB02) is a real
contribution and not a formality — a real deployment asked "what's your client-level privacy
guarantee?" (U3) or "what's your per-individual guarantee?" (U4/U5) would get an unbounded answer from
M4/M8 just like from M0, even though M4/M8 are correctly described as "task-disjoint." Only the
task-level question (U2) has a good answer. This should be stated explicitly in the paper's C3/C4
sections — see `paper/PAPER_BRIEFING.md` for the exact language added.

## Fix applied

`analysis/fig05_eps_of_T.py` now uses a distinct linestyle + marker shape per unit (not just color) so
coincident lines remain visually distinguishable when overlapping, and annotates each panel with which
units are numerically identical (e.g. "U1=U2=U3=U5" for M0, "U1=U3=U5" for M4) so a reader isn't left
wondering whether a line is missing. `figs/fig05_eps_of_T.{pdf,png}` regenerated and inspected.
