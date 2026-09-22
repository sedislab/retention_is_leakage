# Debate protocol

## The unit of work: a Round

A Round is about **exactly one** claim (a hypothesis `Hx`, a design decision, or a `[CONTESTED]`
section). Rounds are cheap; batching claims into one Round is the most common way this process fails.

```
1. FRAME      Moderator states the claim in one sentence + names its register ID.
2. STEELMAN   Proposer gives the strongest case FOR, in <=400 words, with citations.
3. SCOUT      Librarian reports prior work that already does this, or confirms none found,
              naming what was searched. This turn cannot be skipped.
4. ATTACK     Red Team + the relevant specialist (Threat Modeler / DP Theorist / Empiricist /
              Systems) each post ONE strongest objection. Objections must be specific and
              falsifiable-in-principle.
5. REBUT      Proposer answers each objection or concedes it explicitly. Partial concession is
              encouraged and is not a loss.
6. ADJUDICATE Moderator assigns a status and, if the claim survives, a DECIDING EXPERIMENT or
              PROOF OBLIGATION. A claim that survives with no deciding test is downgraded to
              SPECULATIVE, not SUPPORTED.
7. RECORD     Moderator writes the outcome + all dissents into OPEN_QUESTIONS.md.
```

## Statuses
- `OPEN` — stated, not yet debated.
- `SUPPORTED` — survived attack AND has a named deciding experiment/proof that has not yet run.
- `CONFIRMED` — the deciding experiment ran and agreed. **Only a human may set this.**
- `REFUTED` — killed by argument, citation, or experiment. Record why; refuted claims are not deleted.
- `SPECULATIVE` — interesting, unfalsifiable as stated. Must be re-specified or parked.
- `PARKED` — good, out of scope. Lives in the parking lot with a one-line reason.

## Special rounds
- **DEVIL'S ADVOCATE** — triggered by three consecutive agreeing turns. Red Team must argue that the
  entire project is not worth doing, in its strongest form. Proposer must answer it. This has caught
  more bad projects than any other single practice.
- **PRE-MORTEM** — run monthly. "It is 18 months later and this failed." Each agent gives one cause.
  Compare against §7.1; if a new cause appears, it goes in the risk register.
- **REVIEWER SIMULATION** — before any writeup. Reviewer agent produces three reviews (a security-PC
  reviewer, an ML-theory reviewer, an FCL-methods author whose work we audited) with scores and the
  single objection each would lead with. Address the lead objection or explain why you accept the
  rejection risk.
- **RED-LINE CHECK** — before any external release. Ethics obligations in §9, disclosure timeline,
  and "would we be comfortable if the audited authors read this today?"

## Rules of order
- The Moderator never argues the substance; it only frames, enforces, and records.
- The Proposer may not also adjudicate.
- Any agent may call `HALT: UNVERIFIED` on a factual claim; the claim is suspended until the
  Librarian checks it. This overrides everything else in the queue.
- Dissent is always recorded, never overruled into silence. A minority view that later proves right
  should be traceable to whoever held it.
- The human (Mohaimanul) is the only source of `CONFIRMED`, the only one who can unlock `[LOCKED]`,
  and the final arbiter on scope and ethics.
