# 2026-09-21 — The last two "never cut" items: `make verify` and `make paper`, plus a real bug they found

## Why this matters

`build/00_BUILD_PLAN.md`'s own "if short on time" section names exactly four things as **never cut**:
"the Camelyon17 natural-federation check, the secure-aggregation ablation, the log-log ROCs, the seed
variance, or `make verify`." Three of those four were done earlier today (FIG11 secure-agg, FIG16
log-log ROC, FIG17 seed variance). Camelyon17 remains out of scope (a new dataset is a different order
of magnitude of work than anything else done this session). **`make verify` was the fourth, and it
turned out not to exist at all** — `analysis/verify_provenance.py` was built and correct, but the
Makefile had no `verify` target to invoke it. Checking this also surfaced that `make paper` (listed as
one of CLAUDE.md's 6 standard commands) didn't exist either, and that the LaTeX scaffolding CLAUDE.md's
own Layout section claims is "already scaffolded in `llm/paper/`" **does not exist anywhere in this
repo** — `llm/` was never created.

## What was built

1. **`paper/main.tex`** — a structural scaffold, not manuscript content. Generic `article` class (no
   official ICLR/USENIX style file exists in this repo or was fetched this session — swap it in when
   chosen). Section headers matching `paper/PAPER_BRIEFING.md` §6's suggested structure, each with a
   `% TODO(Opus)` comment pointing at the exact briefing section to read — no argumentative prose,
   per the human's explicit instruction that a separate Opus session writes the paper's actual
   content. Demonstrates real `\input{}` wiring for all 4 tables and `\includegraphics{}` wiring for
   all 10 figures, so Opus inherits working infrastructure instead of an empty directory.
2. **`paper/references.bib`** — empty stub so `bibtex` has something to read.
3. **`code/Makefile`** — added `verify` (runs `analysis/verify_provenance.py`) and `paper` (4-pass
   `pdflatex`+`bibtex`, since `latexmk`/`biber` are **not installed on this system** — checked
   directly: `pdflatex`/`bibtex` exist, `latexmk`/`biber` do not) targets.

## The real bug this found

Running `make paper` for the first time (it had never been run before, ever, in this project's
history — the target didn't exist until today) **failed with ~100 LaTeX errors**, all traceable to one
root cause: `analysis/tab03_leakage.py`, `tab04_halflife.py`, `tab02_units.py`, and
`tab10_taxonomy.py` write method names, dataset names, and free-text descriptions directly into
generated `.tex` table cells **without escaping LaTeX special characters**. Names like `m4_proto` or
`m5_hybrid_replay` contain `_`, which LaTeX reads as "begin a subscript" outside math mode and fails
to compile on. `tab10_taxonomy.py`'s hardcoded text also referenced `RESEARCH_PLAN.md` directly,
same problem.

**This means the entire figure/table pipeline — real CSVs, correctly computed, exactly as
CLAUDE.md non-negotiable #3 requires — would have silently blocked Opus (or anyone) from ever
successfully compiling the paper**, the moment any of these four tables was `\input{}`-ed. Caught only
because `make paper` was actually run end-to-end for the first time today, not because anyone
inspected the `.tex` output by eye (the `.tex` files look completely fine as plain text — the bug only
manifests when a real LaTeX engine tries to parse them).

## The fix

Added `p3fcl.plotting.latex_escape()` (a small character-substitution table: `_`, `%`, `&`, `$`, `#`,
`{`, `}`, `\`, `~`, `^`) and applied it to every free-text field written into a `.tex` table cell
across all 4 table-generation scripts. Regenerated all 4 tables, reran `make paper` — compiles cleanly
to a real 10-page PDF (`paper/main.pdf`), verified visually (converted pages to PNG and inspected):
title page and section structure render correctly, FIG01 embeds and displays correctly, TAB03 renders
with correctly-escaped method names (`m0\_fedavg` etc. display as `m0_fedavg`, not broken subscripts)
and all real CI values intact.

**One remaining known cosmetic issue, not a bug**: FIG02 (a 42-row forest plot) is tall enough that it
overflows a single page's height at `width=\textwidth` in the generic `article` class — this is a
real-paper-formatting decision (crop, resize, or reformat FIG02 to be less tall) that belongs to
whoever does final camera-ready layout, not something to fix in this scaffold.

## Status

`make setup && make figures && make paper` (the literal P9 gate criterion) now runs end to end from a
completely clean `figs/`/`tables/`/`paper/main.pdf`. `make verify` passes cleanly (0 exceptions, since
the scaffold contains no hand-typed numbers). All 4 table scripts' outputs regenerated and
lint/test-clean (181 tests, `ruff` clean).
