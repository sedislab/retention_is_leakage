# 04 — METHOD ZOO AND ATTACK SUITE

The audit is organised by **artifact family**, not by method name (`RESEARCH_PLAN.md §2.2`,
`[LOCKED]`). That is what makes the result generalise past the nine papers we happen to test, and it
is what lets you cut a method without cutting a claim — as long as each family keeps one working
instantiation.

---

## 1. The method zoo

Every method: subclass `methods.base.FCLMethod`, emit `ArtifactRecord`s with honest `touched`,
declare `families` and `cacheable`. Nothing else in the project may read a method's internals.

> **ID scheme.** These IDs are the ones in `RESEARCH_PLAN.md §5`, and `agents/OPEN_QUESTIONS.md`
> refers to them. Do not renumber. **M9 is the only new ID** (our mechanism, claim C4).

| ID | Method | Venue | Families | Cacheable | Retention knob (for FIG03) | Role |
|---|---|---|---|---|---|---|
| **M0** | FedAvg + sequential fine-tuning | — | F1 | yes (head only) | local epochs | retention lower bound; attack calibration point |
| **M1** | GLFC | CVPR'22 | F1, F7, F8 | partial (feature space) | exemplar budget / distillation weight | the classic FCIL reference |
| **M2** | TARGET | ICCV'23 | F1, F6 | partial (feature-space generator) | generator capacity / replay ratio | generative replay; expected worst case |
| **M3** | FOT | ICLR'24 | F1 + subspace | yes (head only) | projection strength / subspace rank | orthogonality-regularisation family |
| **M4** | PILoRA | ECCV'24 | F2, F4 | prototype part **yes**, LoRA part **no** | LoRA rank; prototype momentum | prototype + LoRA |
| **M5** | Hybrid Replay FCIL | ICLR'25 | F8 | partial (latent exemplars) | buffer size per class | raw-exemplar upper bound on leakage |
| **M6** | C²Prompt | NeurIPS'25 | F3, F7 | **no** | prompt pool size; top-k selection | current prompt SOTA; publishes class distributions |
| **M7** | TPGP | ICCV'25 | F3 + projection | **no** | projection strength | prompt + gradient projection |
| **M8** | Analytic FCL | preprint '26 | F5 | yes | ridge λ; class-conditional on/off | the inversion target, and M9's platform |
| **M9** | **DP-Contractive-FCL** *(ours, C4)* — `RESEARCH_PLAN.md §4.4` calls this **DP-Analytic-FCL**; same mechanism, renamed to match the claim | new | F5 | yes | sketch capacity; influence horizon | the constructive contribution |

`code/src/p3fcl/methods/proto.py` implements **M4's prototype half** (its docstring already says so);
`methods/analytic.py` is **M8**. Everything else is unbuilt.

**The GPU methods are M6, M7, and M4's LoRA half.** They modify the forward pass and cannot use the
feature cache, so they get the offline LiRA variant with 64–128 OUT shadows. M0/M1/M2/M3/M5 reduce
to head-level or feature-space work on a frozen backbone and stay on CPU — verify that assumption per
method when you implement it, and record any method that turns out not to be cacheable.

Missing on purpose: F1's "full model delta" in the literal sense (we freeze the backbone, so the
delta is the head). Say so in the paper — it makes our F1 numbers a *lower* bound on what a
full-finetuning federation would leak, which is the conservative direction and worth one sentence.

### 1.1 Rules

- **M3 and M4 are the only GPU methods.** They need backprop through the ViT and cannot use the
  feature cache. Give them the offline LiRA variant with 64–128 OUT shadows and say so in the paper.
  Everything else is CPU linear algebra on cached features.
- **Retention knobs are the experiment, not a detail.** Every method exposes exactly one scalar knob
  that monotonically trades plasticity for retention, with ≥6 usable levels, and it is named in the
  config. FIG03 and FIG04 are impossible without this, so build it in from the first commit.
- **Reproduction policy** (`CLAUDE.md` rule 8): reimplement from the paper inside our harness,
  compare against the published number, record the gap in **TAB07**, time-box to one session. Do not
  clone seven repos and fight their dependency trees — that is the failure mode the project's own
  pre-mortem names as most likely to kill it.
- **Classify each method's retention as `semantic` or `individual`** in a class attribute. FIG04
  depends on it, and it is the honest answer to the Red Team's strongest objection.

### 1.2 M9, the contribution — design constraints

M9 must satisfy, provably and checkably:

1. **Bounded influence horizon.** Any datum influences at most `h` releases, `h` fixed and
   independent of T. The contractive sketch has a fixed capacity and old mass decays out of it.
2. **Certifiable by our own checker.** `dp.accountant.check_disjointness(ledger)` must return
   `task_disjoint=True` (or the bounded-horizon equivalent you add) on M9's real ledger. If it will
   not certify, **fix the mechanism, not the checker.** This is the one place where it would be
   easiest and most damaging to cheat.
3. **Analytic sensitivity.** Keep the closed-form advantage: `dp.mechanisms.gram_sensitivity` is
   exact for a clipped second-moment statistic, unlike DP-SGD where effective sensitivity is a
   hyperparameter fighting the optimiser. That is the constructive argument.
4. **Honest correlated noise.** Either implement a real banded matrix-factorisation mechanism
   (Choquette-Choo et al., ICML 2023) or state plainly that binary-tree aggregation is used and pay
   the polylog(T) factor. `tree_aggregation_factor_PLACEHOLDER` must not survive into a reported number.
5. **Measure the non-private gap first.** If M8 non-private is already far behind M3 non-private,
   "best at equal ε" is a hollow claim. Put that gap in TAB08 and address it in the text rather than
   letting a reviewer find it.

---

## 2. The attack suite

Every attack declares a `ThreatModel` (`attacks/base.py`). An attack whose threat-model cell is
unspecified does not run — that guard exists for a reason, keep it.

> **ID scheme.** A1–A6 are `RESEARCH_PLAN.md §3`'s numbering and `agents/OPEN_QUESTIONS.md` cites
> them (H3→A1, H4→A4, H10→**A6**, H11→A1/A4). Do not renumber. A7 is our label for the one-run
> auditor, which is an *auditor*, not an attack.

| ID | Attack | Target | Aux | Passive | Survives secure agg | Status |
|---|---|---|---|---|---|---|
| **A1** | Cross-task LiRA over the transcript | membership | shadows | yes | yes | interface fixed, runner **to build** |
| **A2** | Analytic inversion of F5 (Gram) | reconstruction | public ref | yes | no — needs per-client R | re-specify H5 first |
| **A3** | Prototype-difference attribution | reconstruction / membership | counts (F7) | yes | yes | **to build** |
| **A4** | Onset / composition inference | event | none | yes | yes | needs a multi-round method |
| **A5** | Client attribution | *which* client holds a known sample | public ref | yes | no — needs per-client records | **to build** |
| **A6** | Property inference over time | client c's task-k class histogram / property | public ref | yes | yes | **to build — this is H10's deciding test** |
| **A7** | One-run canary audit | ε lower bound | canaries | n/a | n/a | needs wiring |

**A6 carries claim C1, not A1.** H10 — "property-inference accuracy about client c's task-k data
decays with (T − k) at a rate that differs by artifact family, with F5/F8 showing no decay" — is the
register's own cleanest operationalisation of the persistence idea, and its note says it "may end up
being a better headline metric than H2's correlation". So FIG01 plots **both** A1 membership and A6
property curves against the accuracy curve. If they disagree, that disagreement is a finding and goes
in the paper.

Passivity is the differentiator against PromptMIA, which needs an *active* malicious server. Keep
every attack passive unless there is a stated reason, and say so in the paper's threat model section.

### A1 — cross-task LiRA (the core)

Standard LiRA scores a target by the likelihood ratio of its statistic under shadow models trained
with and without it. Our extension: the statistic is a **trajectory over the transcript**, and the
score function is per-family.

- Score functions: `score_prototype`, `score_gram`, `score_lowrank` exist. Add F1 (logit margin /
  loss on the released head) and F3 (prompt-key similarity, plus selection frequency — which prompt
  a client's data pulls is itself a signal).
- `lira.trajectory` summarises the per-round series. **Ablate it** against last-round-only: that
  comparison is hypothesis H3 and produces FIG07. Keep the summariser simple and auditable; a learned
  summariser is a later upgrade that needs its own justification.
- **The cross-task axis is the novelty.** Target data from task k, attack from the round-T
  transcript, sweep k and T independently. That two-index sweep is what produces FIG01.
- Online variant (IN and OUT shadows) for cacheable methods; offline (OUT only) for M3/M4.
- Calibrate thresholds on the shadow split, **never** on the evaluation split.

### A3 — prototype-difference attribution (new, and sharp)

Two release conventions, and **the attack is stronger under the one our code already implements**:

- **Per-round mean** (what `methods/proto.py` does today: `X[m].mean(0)` over *this round's* shard):
  released mean × released count **is the exact sum of that shard's features**. No differencing
  needed. At a shard of one sample this is the sample.
- **Running mean** (what a server-side aggregate looks like): the adversary differences successive
  releases, `n_t·μ_t − n_{t−1}·μ_{t−1}`, to recover the sum of what was newly contributed.

Either way the quantity that matters is the **increment size**, and leakage as a function of that
increment is the result: it says precisely when "we only share class means" stops being a privacy
argument. The dangerous regime — small increments — is the late-stream, rare-class, low-data corner,
which is exactly where a hospital's unusual cases live.

Implement it, and report **leakage as a function of increment size**, because that function is the
result: it says precisely when "we only share class means" stops being a privacy argument.
Ablate releasing counts (F7) on and off; counts are the join key and turning them off should cost
the adversary real accuracy. That ablation is a concrete, cheap defensive recommendation for the paper.

### A5 — client attribution

Given a sample known to be somewhere in the federation, identify *which client* holds it. Needs
per-client records, so it does **not** survive secure aggregation — say so rather than overclaiming;
the honest split between attacks that do and do not survive it is what makes FIG11 credible. This
attack is also the machinery a later federated-continual-unlearning paper would reuse
(`RESEARCH_PLAN.md` parking lot, P7).

### A6 — property inference over time

From round-T artifacts, decide whether client c held class y at task k, and estimate c's task-k class
histogram. Report balanced accuracy against the marginal-frequency baseline — not raw accuracy,
which is trivially high under class imbalance. Cheap, and second only to A4 for harm legibility. **This is the deciding test for H10 and therefore for claim C1** — budget for it accordingly.

### Secure-aggregation mode

For A1, A3, A4, A6: a mode consuming only the aggregate broadcast `A^r`. Near-zero extra cost, and
it produces FIG11, which pre-empts the single most common dismissal of this kind of work. Prioritise
it — do not leave it to the end.

---

## 3. The shadow-federation runner

The engineering that decides whether this project is feasible on Kodiak.

```
scripts/run_shadows.py --dataset D --method M --start i --count k --workers 36 --out DIR
```

Design requirements:

1. **Write statistics, not ledgers.** One shadow emits only the per-target score vectors the attack
   consumes. Target ~10–100 KB per shadow, so 4,000 shadows per (dataset, method) is a few hundred MB
   and the whole shadow store stays manageable.
2. **Deterministic.** `shadow_id` seeds the membership vector and every draw, through
   `rng.seeded(name, seed)`. Re-running index 37 must reproduce its `.npz` byte-identically — make
   that a test, because silent non-determinism here poisons every attack number downstream.
3. **Resumable.** Skip if the output exists; write `.tmp` then `os.replace`. A requeued array task
   must cost nothing.
4. **Process-parallel, single-threaded BLAS.** `OMP_NUM_THREADS=1`, 36 worker processes. Measure the
   alternative once, record the number in `notes/`, then stop thinking about it.
5. **Chunked for the scheduler.** ~25 shadows per array task, 20–40 minutes each (`01_KODIAK.md §4.2`).
6. **Budget.** ~4,000 shadows per (dataset × cacheable method) for the headline; 64–128 offline
   shadows for M3/M4. Sanity-check the shadow count by plotting attack TPR against shadow count and
   showing it has plateaued — that plot belongs in the appendix and pre-empts "how do you know 4,000
   was enough?".

---

## 4. Order of implementation

Sharpest and cheapest first, so the project has a result early:

1. **A2 on real CUB and CIFAR features** (FIG13). Linear algebra on cached features, no shadows, no
   GPU, one to two days. It settles H5, the project's most scoopable claim, and it either produces a
   striking reconstruction figure or an honest curve. Either outcome is worth having in week one.
2. **The accountant on real ledgers** (FIG05). No new science, and it produces a required figure from
   work that is already written.
3. **A3 on M4's prototype half** (prototype-difference). Cheap, new, and it hits family F2, which the literature assumes is obviously safe.
4. **A6 then A1, on M4-proto/M8** — A6 first because it carries claim C1; A1 is the bigger infrastructure investment and unlocks FIG03 and FIG07.
5. **M9 and the Pareto** (FIG08, FIG09).
6. **M6/M7 and M4's LoRA half** (GPU) last: the most expensive and the least load-bearing. They add
   families F3 and F4 to the audit, but no claim in the paper dies if they arrive late.

## 5. What would make this paper fail review, and the countermeasure

| Failure | Countermeasure, already in the plan |
|---|---|
| "Just use secure aggregation." | FIG11 — the attacks run on the aggregate broadcast. |
| "Just add DP." | FIG05 — under a lifelong unit, ε diverges; FIG08/FIG09 is the constructive answer. |
| "Your clients are a Dirichlet artifact." | FIG18 — Camelyon17's real hospital partition. |
| "Attacks are weak / you only report AUC." | TPR@1%FPR and @0.1%FPR everywhere, log-log ROC (FIG16), and A7's audited ε lower bound as the floor. |
| "Retention is semantic, membership is individual." | FIG04 — the split is designed into the sweep, and the honest answer is reported either way. |
| "Only tested on toy class splits." | Camelyon17 required, FMoW as stretch. |
| "Is 4,000 shadows enough?" | The shadow-count plateau plot. |
| "Did you tune on test?" | Calibration on the shadow split, stated, plus the provenance check. |
| "The numbers in the text don't match the tables." | `make verify` — no number reaches the paper except through a CSV. |
