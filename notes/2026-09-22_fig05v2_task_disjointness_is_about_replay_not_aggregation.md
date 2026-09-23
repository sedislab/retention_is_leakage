# 2026-09-22 — FIG05 v2: M0 is ALSO flat under U1/U2/U3; only M1/M5 (real buffers) accumulate

## What happened

Inspecting the regenerated `figs/fig05_eps_of_T.png` (job 183850, FX1's new accountant): every panel
annotates its per-unit regime (contractive/accumulating). Naive expectation going in was "M4/M8
(the structurally task-disjoint methods) are flat under U1-U3, M0/M1/M2/M3/M5 (the ones without a
special disjoint-aggregation structure) accumulate." That is **not** what the figure shows.

## The actual pattern

- **M0, M2, M3, M4, M8**: U1, U2, U3 all flat (contractive); only U4 (unbounded client history)
  accumulates.
- **M1, M5**: U1, U2, U3, U4 **all** accumulate.

M0 — plain sequential FedAvg, no retention mechanism of any kind — is just as flat under U1/U2/U3 as
M4/M8 are.

## Why, and why it's not a bug

Follows directly from Lemma 8 (`08_FIX_PLAN.md` §4b) once you track what "touched" actually means
post-FX4: a record's `touched` is exactly the private ids *read* to compute it that round; reading a
broadcast model or any earlier release is post-processing. M0's `MODEL_DELTA` record at round r has
`touched` = exactly that round's own shard — it never re-reads a client's past raw ids, because it
has nothing that would make it do so (no buffer, no distillation, no generative sampling). So U1/U2/U3
(which all ask "does anything ever re-touch this fixed pool of data") are trivially flat for M0 by
construction — not because M0 has any privacy-friendly structure, but because it never gets the
chance to re-touch anything. M2 (post-FX4e, honest per-client-then-aggregated Gaussian stats) and M3
(post-FX4b, its subspace is its own one-shot release) are the same way now — confirmed independently
already in `notes/` around FX4b/e as "M2 and M3 are now genuinely task-disjoint."

**M1 and M5 are the odd ones out because they are the only two methods with a literal, persistent,
later-read buffer** (FX4c/d: `self._buffer[(client, class)]`, read again in every later task's local
training). That buffer read IS a re-touch of old raw ids, every single task, which is exactly what
drives their U1/U2/U3 growth.

## Why this matters for the paper's C1 narrative

The clean story is **not** "M4/M8 have a special disjoint-aggregation structure that M0/M1/M2/M3/M5
lack." It is: **retention that operates on a once-computed summary statistic (Gram matrix, class
prototype, orthogonal subspace, Gaussian moments) never re-touches raw examples and is therefore
narrow-unit-private by construction, regardless of whether the method does anything for utility at
all; retention that operates by literally keeping and re-reading raw (or near-raw, e.g. exemplar)
examples is not, and cannot be, no matter how it's tuned.** M0 "passing" U1/U2/U3 is privacy by
default from doing nothing, not evidence of good design — the real, useful contrast for C1 is that
M4/M8 (and now M2/M3) get this same narrow-unit flatness *while also retaining real utility* (per
`results/fx4_gate.csv`, M3 passes the utility gate on all 3 datasets; M2/M4/M8 have real accuracy
gains over M0 in `results/accuracy_matrix_*.csv`), whereas M0's flatness comes with catastrophic
forgetting. This sharpens claim C1 rather than weakening it, but the framing should be corrected
wherever it currently says or implies "M4/M8 are special because they're task-disjoint" without the
"...and M0 trivially is too, and that's not the interesting part" caveat. Relevant to FX6 (figure
text/captions) and FX7 (errata / `paper/PAPER_BRIEFING.md` review) — flagging now so it isn't lost.

## Status

Not a new experimental result — a correct reading of an existing figure/mechanism, worth preserving
so a future session (or the paper-writing session reading `PAPER_BRIEFING.md`) doesn't restate the
naive version of the C1 story.
