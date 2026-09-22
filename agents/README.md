# How to run the multi-agent process

## Setup for a local LLM
Give **every** agent, in its system prompt, in this order:
1. `agents/00_SHARED_CONTEXT.md`
2. `agents/DEBATE_PROTOCOL.md`
3. that agent's own role file
4. `RESEARCH_PLAN.md` (full text — it is ~12k words; if the context window cannot hold it, give
   §§1–4 to the theory agents and §§3, 5–9 to the empirical agents, and always give §2 to everyone
   since it is the shared vocabulary)
5. the current `agents/OPEN_QUESTIONS.md`

## The roster
| File | Agent | Invoke when |
|---|---|---|
| `01_moderator.md` | Moderator / PI | Always. Runs every Round |
| `02_proposer.md` | Proposer | Always |
| `03_red_team.md` | Red Team | Always |
| `04_threat_modeler.md` | Threat Modeler | Any attack claim |
| `05_dp_theorist.md` | DP Theorist | Any privacy/accounting claim |
| `06_empiricist.md` | Empiricist | Any experiment design or result |
| `07_systems.md` | Systems | Any proposal with a compute or engineering cost |
| `08_librarian.md` | Librarian | Every Round, at SCOUT. Never skip |
| `09_reviewer_sim.md` | Reviewer Sim | Before any writeup; monthly otherwise |

## Minimum viable loop (if you can only run 3 agents)
Moderator + Proposer + Red Team, with the Librarian's SCOUT step done by hand via literature search.
Add the DP Theorist as the fourth — the privacy claims are where errors are most expensive and least
visible.

## Practical notes for local models
- Smaller models collapse into agreement fast. Rule 2 in `00_SHARED_CONTEXT.md` (no agreement without
  a reason) is the main defence; consider also raising temperature for Red Team relative to Proposer.
- Run agents in **separate contexts**, not as personas inside one conversation. A single model playing
  all roles in one context will converge — that is the failure mode this whole design exists to prevent.
- Persist `OPEN_QUESTIONS.md` between sessions. It is the only memory the process has.
- Do not let agents edit `RESEARCH_PLAN.md` directly. Debates update the register; a human folds
  register outcomes back into the plan periodically. Otherwise the ground truth drifts.

## Suggested first week
1. Round on **H2** (flagship; §7.2 exists because it may be false).
2. Round on **H5** (cheapest to settle; two days of real work would decide it).
3. **DEVIL'S ADVOCATE** on the project as a whole.
4. Round on **H6/T1** (determines whether Paper B has a theorem).
5. **REVIEWER SIMULATION** on the plan as if it were the paper's introduction.
