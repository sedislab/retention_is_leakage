# P3 — Privacy Leakage and Lifelong DP Accounting in Federated Continual Learning

Everything needed to run this as a multi-agent research project on a local LLM.

```
RESEARCH_PLAN.md          The plan. ~12k words. The shared ground truth. Start here.
agents/                   Multi-agent scaffolding
  README.md               How to wire the roster into a local LLM
  00_SHARED_CONTEXT.md    Load into EVERY agent
  DEBATE_PROTOCOL.md      The Round structure and statuses
  01..09_*.md             Nine role prompts
  OPEN_QUESTIONS.md       The register — H1-H12, T1-T2, contested decisions. THE PROJECT'S STATE
code/                     Runnable skeleton (numpy-only core, 9 passing tests)
notes/                    Working notes, including preliminary observations
```

## The idea in three lines
FCL exists because of privacy, but what leaves the client is a curated summary of its history —
prototypes, prompts, LoRA factors, Gram matrices — engineered to *survive across tasks*, because
surviving across tasks is what defeats forgetting. The mechanism that prevents forgetting is the
mechanism that prevents privacy from decaying. We measure that, and we build the accountant for a
federation that trains forever.

## How to start
1. Read `RESEARCH_PLAN.md` end to end.
2. Read `agents/README.md`, wire up at minimum Moderator + Proposer + Red Team + DP Theorist,
   **in separate contexts** (a single model playing all roles will converge into agreement — that
   failure mode is what the whole design exists to prevent).
3. Run the first Round on **H2**, the flagship hypothesis, and specifically test whether the project
   survives H2 being false. It should — §7.2 is the fallback — but confirm it before committing.
4. In parallel and needing no cluster allocation: run H5's deciding experiment (`code/README.md`,
   first week, item 2). It is two days of linear algebra and it settles the sharpest claim.

## Three things to hold onto
- **Attack first, theory second.** `[LOCKED]`, for reasons in §8. The audit builds the codebase the
  theory needs, is de-risked (negative results publish at security venues), and gives Paper B real
  measured motivation instead of asserted plausibility.
- **Build the artifact ledger before any method.** The pre-mortem in §7.1 says the most likely way
  this project dies is months lost to reproducing seven codebases. The ledger plus a truncatable
  plug-in queue of methods is the countermeasure.
- **H2 is upside, not load-bearing.** Design so the paper lands even if the flagship hypothesis dies.

## Building the system

`build/` holds the implementation instructions for Claude Code — phases, cluster setup, datasets,
and the full figure/table specification. Start at [`build/README.md`](build/README.md).
Compute moved to **Baylor Kodiak** on 2026-09-14 (PBS, not Slurm); the `code/scripts/*.slurm` files
are stale and are replaced in Phase P0.
