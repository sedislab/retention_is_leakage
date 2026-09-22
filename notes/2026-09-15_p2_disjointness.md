# 2026-09-15 — P2: method zoo (7/9) + the H6 structural disjointness report

## What happened

Built M0 (FedAvg sequential), M1 (GLFC), M2 (TARGET), M3 (FOT), M5 (Hybrid Replay) — joining the
already-built M4 prototype-half and M8 analytic FCL from P0. That's 7 of 9; M6 (C²Prompt) and M7
(TPGP) need real backprop through the ViT on raw images and are deliberately deferred (GPU-heavy,
explicitly the lowest-priority pair per `00_BUILD_PLAN.md`'s cut order, and the GPU queue is under
heavy contention right now anyway — see `build/STATE.md`). M9 is P7's job.

Wrote `code/scripts/build_disjointness_report.py` and produced `results/disjointness_report.csv` —
the Gate P2 deliverable that is the empirical half of hypothesis H6. It runs every method (in both a
"mechanism off" and "mechanism on" configuration where that distinction is meaningful) through
`sim.run` on synthetic features and records what `dp.accountant.check_disjointness` certifies.

**This is evidence, not a toy check**: task-disjointness is a structural property of a method's
release schedule and `touched` bookkeeping, not of feature values, so a synthetic run and a real-ViT
run trigger identically for the same ids/rounds. The result:

| Method (mechanism on) | task_disjoint |
|---|---|
| M8 analytic FCL (F5) | **True** |
| M4 prototype, class-IL (F2, momentum=0.9) | **True** |
| M0 FedAvg, local_epochs=5 | False (V5b) |
| M1 GLFC, replay+distill | False (V2/V3) |
| M2 TARGET, replay on | False (V2/V3) |
| M3 FOT, projection on | False (V2/V3) |
| M5 Hybrid Replay, buffer on | False (V2/V3) |

Exactly F5 and F2-single-pass survive with their real anti-forgetting mechanism switched on; every
other family breaks disjointness the moment its mechanism does anything. That is H6's structural
claim, cleanly demonstrated. (The "without accuracy collapse" half of H6 still needs real per-dataset
accuracy — this only settles the formal half, honestly, per the script's own docstring.)

## A real bug this caught: M5 was silently mis-certifying disjoint (fixed, not evidence of anything)

First pass at M5 (Hybrid Replay) deliberately released **only** the F8 exemplar artifact, reasoning
that F1 leakage was already covered elsewhere and mixing signals would dilute FIG04's semantic-vs-
individual split. That was a mistake, not a defensible simplification: M5's `_local_train` already
retrains on task-data + buffered exemplars every round internally, so the resulting classifier update
genuinely *is* a function of old-task data via replay — but because that update was never released as
an artifact, `check_disjointness` had no evidence of the reuse and certified M5 **task-disjoint**,
directly contradicting H6 (F8 is hypothesized to NOT admit disjointness) and CLAUDE.md non-negotiable
#2 ("under-reporting `touched` silently converts a false privacy claim into a 'proved' one").

Fixed by releasing the replay-augmented delta too (family F1, alongside F8), with `touched` honestly
including every id currently in the buffer that the round's training drew on. Re-ran the disjointness
report: M5 now correctly shows V2/V3 once the buffer is in use. `tests/test_methods.py` has a test
(`test_m5_hybrid_replay_is_not_task_disjoint_once_buffer_is_reused`) specifically pinned to this so it
cannot regress silently.

**Labelled evidence**: the disjointness *pattern* across all 7 methods (F5/F2-single-pass survive,
everything else doesn't) is real support for H6, produced from the actual method code via the actual
checker. **Labelled not-evidence**: the `final_avg_acc_synthetic` column in the CSV — synthetic
Gaussian blobs, not a claim about any real dataset; do not cite it as a utility number.

## What the next person needs to know

- Two other subtleties the tests caught along the way, both correct-and-intentional rather than bugs:
  M0 at `local_epochs > 1` violates V5b (multiple SGD passes over the same shard within one round is
  real sequential composition, even with zero cross-task mechanism) — a sharper within-task check
  than I'd initially appreciated the spec asking for. And M1 with distillation *off* is still not
  disjoint if `exemplar_budget > 0`, because exemplar replay alone (independent of the KD loss term)
  already reuses old-task ids — the two anti-forgetting channels are separable and both need to be
  off for GLFC to degenerate to disjoint.
- If a future method's disjointness result looks "too good" (certifies disjoint) despite having a
  real anti-forgetting mechanism, treat that as a signal to check whether the release schedule is
  under-reporting `touched` before trusting it — this is now the second time in one session that
  pattern has caught a real bug (the first being this M5 fix).
