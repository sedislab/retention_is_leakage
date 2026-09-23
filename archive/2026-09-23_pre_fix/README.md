# Pre-fix archive — 2026-09-23

Archived immediately before starting `build/08_FIX_PLAN.md`'s fix phase (FX0), per that plan's §3.

- **Archived (UTC):** 2026-09-22 (system clock date at archive time; the fix plan's own filename
  anticipates 2026-09-23 as the nominal start date — kept as-is for consistency with the plan's
  internal cross-references).
- **`source_hash()` at archive time:** `d79041744912c340d3d349629c37ddff30b4d3cac897064821e95241fdf609f1`
- **Contents:** verbatim copies (`cp -a`) of `results/`, `figs/`, `tables/`, `paper/PAPER_BRIEFING.md`,
  `agents/OPEN_QUESTIONS.md`, `build/STATE.md`, as they stood before any fix-phase code or data
  changes. `SHA256SUMS` in this directory covers every file here and was verified immediately after
  the copy (`sha256sum -c SHA256SUMS`: all OK).

**Why this archive exists**: `build/08_FIX_PLAN.md` §1 documents a post-review audit (R1–R12) that
found serious methodological problems in the pre-fix results — most importantly, accuracy was
measured on training examples rather than a held-out test split (R1), several methods' `touched`
sets were inflated in a way that broke the LiRA attack's model reconstruction (R3), the DP accountant
produced impossible orderings (R5), and the half-life/decoupling-ratio methodology extrapolated
exponential fits with no chance floor in a way that does not match the paper's own definitions (R6).
See `build/08_FIX_PLAN.md` §1 for the full finding-by-finding table and the fix each one maps to.

This archive is the historical record of what the project looked like before those fixes — kept for
comparison (e.g. `results/fx4_lag_diagnostic.csv` cites the pre-fix numbers directly) and so the fix
phase's own claims about what changed can be checked against a fixed reference point. Nothing here
was deleted from the live tree without first being copied here; see `08_FIX_PLAN.md` §3 for exactly
what was subsequently moved to `stale/` in the live tree versus what was left in place.
