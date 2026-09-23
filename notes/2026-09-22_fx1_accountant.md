# 2026-09-22 — FX1 accountant rewrite: `passes_over_data` moves from primary to appendix-only

## What happened

Rewriting `code/src/p3fcl/dp/accountant.py::account` per `build/08_FIX_PLAN.md` §6 (FX1), following
the plan's definition literally:

> `m_T(u)` is the number of released records `r` with `round(r) <= the end of task T-1` and
> `touched(r) & D(u) != empty`. ... Also write `m_T_passes`, the count weighted by
> `passes_over_data`, as a secondary appendix column only.

This is a **narrowing** relative to the pre-fix accountant. The old `_max_task_touch_multiplicity`
folded `max(1, r.passes_over_data)` directly into the composed count `k` that drove `eps` — added
2026-09-16 specifically as a fix for a real under-reporting bug (a method doing 30 local SGD epochs
per release was getting the identical `eps(T)` curve as one doing 1 epoch, silently ignoring real
sequential composition within a single release; see the old `test_account_multi_epoch_gives_higher_eps_than_single_epoch`,
now replaced). Under FX1's literal spec, that distinction no longer reaches `eps` at all — two
ledgers differing only in `passes_over_data` now get numerically identical `eps(T)` curves, and the
old bug's signature (`m_T_passes` differing) is demoted to a column nothing downstream reads by
default.

## Why I did not relitigate this

`08_FIX_PLAN.md` is the current authoritative, human-authored execution plan for this phase, and its
§6 spec is unambiguous and specific (it explicitly calls out `m_T_passes` as "secondary appendix
column only", which reads as a deliberate response to something, not an oversight — most likely a
judgment that composing over *releases* is the right primary quantity and per-release-internal
epoch composition is a separate, secondary concern for this accounting model). Per CLAUDE.md
non-negotiable #9, plan decisions are not mine to relitigate absent contradicting evidence; this is
also not a `[LOCKED]` `RESEARCH_PLAN.md` decision, just a specific implementation instruction I
should follow as written.

## Why this is still flagged, not silently absorbed

CLAUDE.md non-negotiable #2 ("over-report and note it" when unsure about privacy-critical
under-counting) makes this worth a flag regardless: this is a real, measurable step back from a
previously-fixed under-reporting bug, reintroduced by the new spec itself rather than by an
implementation mistake. `test_dp_accountant.py::test_account_m_T_passes_reflects_passes_but_eps_does_not`
pins the new behavior explicitly (same `eps`, different `m_T_passes`, for passes=1 vs passes=30) so
this is a documented, intentional design point that a future session can find and revisit — e.g. if
FX7's errata pass or a reviewer asks why `eps(T)` doesn't respond to `local_epochs`, the answer is
here, not a silent gap.

## Status

Not evidence of anything about the four claims (C1-C4) — this is an accounting-methodology note, not
an experimental result. Labelled here so it isn't lost across a context reset.
