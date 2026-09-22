# P3 — Privacy Leakage and Lifelong DP Accounting in Federated Continual Learning

**A full research plan.**
Author of record: Mohaimanul Islam · Drafted 2026-08-25 · Compute: **Baylor Kodiak** (changed 2026-09-14 from NCSA Delta; PBS not Slurm, ~10 GPUs, see `build/01_KODIAK.md`)
Status: `PLANNING` — no experiment has been run. Every number in this document is a *target* or an *assumption*, never a result.

> **Read this if you are an LLM agent.** This document is the shared ground truth for a multi-agent
> research process. It is written to be argued with. Sections marked **[CONTESTED]** are places where
> the author is deliberately uncertain and wants adversarial pressure. Sections marked **[LOCKED]**
> are scope decisions that should not be relitigated without new evidence. Claims labelled `H1`…`H12`
> are falsifiable hypotheses tracked in `agents/OPEN_QUESTIONS.md`; do not add a new hypothesis to
> the prose without registering it there.

---

## 0. One-paragraph thesis

Federated Continual Learning exists because of privacy. Every survey opens by saying raw data never
leaves the client. Yet in the current generation of FCL methods the thing that actually leaves the
client is not a gradient — it is a *curated summary of the client's history*: class prototypes, prompt
pools, class-distribution vectors, LoRA factors, autocorrelation (Gram) matrices, generative replay
models. These artifacts are engineered to **survive across tasks**, because surviving across tasks is
exactly what defeats catastrophic forgetting. That is the central irony this project is built on:
**the mechanism that prevents forgetting is the same mechanism that prevents privacy from decaying.**
Nobody has measured this, and nobody has an accounting framework for a federation that trains forever.
We will do both: audit the leakage empirically with security-grade methodology, and build the first
differential-privacy accounting framework whose guarantee does not degrade to vacuity as the task
stream grows without bound.

---

## 1. Why this is the right problem

### 1.1 The gap, stated precisely

Three literatures nearly touch here and never meet.

1. **FCL methods** (GLFC CVPR'22 → C²Prompt NeurIPS'25 → GFedCL/C²-AFCL ICML'26). These papers assert
   privacy as motivation and then share prototypes, prompts, statistics, and synthetic data with zero
   quantitative privacy analysis. The typical treatment is one sentence: *"we do not share raw data,
   hence privacy is preserved."* That is not a privacy argument; it is a data-flow observation.

2. **FL privacy attacks** (gradient inversion, MIA surveys, FedInverse, PromptMIA). Mature, but
   **single-task and snapshot-based**. The adversary attacks one round, or the final model of one
   training run. There is no notion of a *task index*, so no notion of leakage about data that the
   client no longer even holds.

3. **DP theory for streams** (Dwork–Naor–Pitassi–Rothblum continual observation STOC'10; privacy
   filters and odometers, Rogers et al. NeurIPS'16; Rényi filters / individual accounting,
   Feldman & Zrnic NeurIPS'21; DP-FTRL, Kairouz et al. ICML'21; banded matrix factorization,
   Choquette-Choo et al.; BLT mechanisms 2024–25; fully-adaptive composition, Whitehouse et al.
   ICML'23). These give the right machinery for unbounded release, but they were built for
   *fixed-horizon* federated training of a *single* model on a *stationary* task. None of them models
   a task-structured, boundary-bearing, client-asynchronous stream.

**The unclaimed intersection:** temporal, cross-task privacy analysis of federated continual learning,
plus a privacy accountant that stays non-vacuous as T → ∞.

### 1.2 What already exists that we must not pretend doesn't

**[LOCKED]** These are the nearest neighbours. Any agent claiming "nobody has done X" must first
explain why it is not one of these.

| Work | Venue | What it does | Why P3 is still open |
|---|---|---|---|
| *Quantifying Privacy Risks of Prompts in Visual Prompt Learning*, Wu et al. | USENIX Sec '24 | MIA + property inference on visual prompts | Centralized, single-task, no federation, no temporal axis |
| *PromptMIA: Leveraging Soft Prompts for Privacy Attacks in Federated Prompt Tuning* | ICML (2026) | Malicious server injects adversarial prompts into the pool, watches selection | **Single-task**; requires an *active/malicious* server; attacks the selection mechanism, not retained knowledge |
| *Differentially Private Continual Learning using Pre-Trained Models*, Amit et al. | NeurIPS '24 workshop | DP cosine classifier + DP-SGD PEFT ensemble over a task stream | **Centralized**; relies on *parallel composition*, valid only under strong assumptions (S1: full label set known upfront; S2: strictly non-overlapping tasks, no replay). Both assumptions **fail in federation** — see §4.3. Workshop, not archival |
| *Differentially Private Federated Continual Learning with Heterogeneous Cohort Privacy* | IEEE (ICASSP-family, 2023) | Per-cohort ε in a federated continual setting | Pre-foundation-model era; fixed horizon; no leakage audit; no unbounded-stream accounting |
| DP-FTRL / banded MF / BLT | ICML '21–'25 | Correlated-noise mechanisms for FL under continual observation | Fixed T, no task structure, no forgetting objective, deployed for a single stationary task |
| Federated unlearning survey | IEEE TNNLS '24 | Removal of a client's contribution | Removal ≠ measurement of what leaked; complementary (see P7) |

**Honest summary of novelty:** the *tools* all exist. The *composition of tools onto this setting*,
and the empirical finding we expect from it (§3.1), do not. This is a "first rigorous treatment"
paper, not a "new primitive" paper. That framing is a strength at a security venue and a weakness at
a theory venue — which drives the venue strategy in §8.

### 1.3 Why now

- FCL has converged on a small set of *shared artifact types* (prompts, prototypes, LoRA, Gram
  matrices). That convergence makes a systematic audit tractable — you audit ~5 artifact families,
  not 40 bespoke methods.
- The NeurIPS'25 resource-constraint paper established that the community accepts adversarial-audit
  papers about FCL ("what does matter?"). There is a receptive audience for a second reality check.
- One-run privacy auditing (Steinke et al., NeurIPS'23) made rigorous empirical ε lower bounds cheap
  enough to run at this scale. Five years ago this project would have been compute-infeasible.

---

## 2. Formal framework

This section defines the objects the whole project manipulates. Agents should treat these definitions
as the project's type system: if a claim cannot be phrased in this vocabulary, the claim is not yet
well posed.

### 2.1 The FCL release channel

A federation of clients `c ∈ C` (|C| = N), a stream of tasks `k = 1, 2, 3, …` with no assumed upper
bound. Client `c` at task `k` holds a private dataset `D_c^k`. Task `k` spans communication rounds
`R_k`, and in round `r` a participating client uploads an **artifact** `a_c^r`. The server aggregates
and broadcasts `A^r`.

Define the **release channel**

```
M : ({D_c^k}) ↦ (A^1, A^2, A^3, …)          — the entire, unbounded broadcast transcript
```

Every privacy statement in this project is a statement about `M`, not about a single round and not
about the final model. **This is the single most important framing decision in the project.** Prior
FL privacy work almost always analyses `A^r` for one `r`, or the final `A^T`. Analysing the transcript
is what makes the temporal phenomena in §3 visible at all.

**Adversary's view.** We define three observation sets:
- `V_final = {A^T}` — the weakest; what a model-release adversary sees.
- `V_task = {A^{last round of each task k}}` — a checkpoint-publishing adversary; T observations.
- `V_full = {A^r : all r}` — an honest-but-curious server or a persistent participant; Σ_k R_k observations.

**[CONTESTED]** Is `V_full` realistic? A participating client sees every broadcast, so yes for
cross-silo (hospitals, banks) where clients participate in nearly every round. For cross-device it is
the *server's* view. Red Team should press on whether `V_full` results are interesting if secure
aggregation is deployed. Author's current answer: secure aggregation hides *individual* `a_c^r` but
not the *aggregate* `A^r`, and every attack in §3 that works on `V_full` therefore survives secure
aggregation. That is a selling point, not a weakness — say so explicitly in the paper.

### 2.2 The artifact taxonomy — what actually leaves the client

**[LOCKED]** The audit is organized by artifact family, not by method name. This is what makes the
result generalize beyond the specific papers we test.

| Family | Symbol | Methods that ship it | Sufficient-statistic content | A-priori leakage risk |
|---|---|---|---|---|
| **F1 Model deltas** | `Δθ` | FedAvg, GLFC, FOT | full gradient signal | High (known); our baseline |
| **F2 Class prototypes / feature moments** | `μ_c,y`, `Σ_c,y` | PILoRA, HGP, STSA, pFedMxF | mean (and covariance) of the client's features for class y | **High** — a mean over n samples is a linear query; with small n it is nearly a sample |
| **F3 Prompt pools + keys** | `P`, `k_P` | C²Prompt, Fed-CPrompt, TPGP | learned input-space perturbations optimized on local data | High (USENIX'24 showed it centrally) |
| **F4 Low-rank adapters** | `A`, `B` | PILoRA, DOLFIN, FCIT-style | rank-r update subspace | Medium–High; row space of updates |
| **F5 Analytic Gram / autocorrelation** | `R_c = XᵀX + λI`, `Q_c = XᵀY` | FedRAN and the analytic-FCL line | **an exact sufficient statistic of the entire local feature matrix** | **Critical** — see §3.4 |
| **F6 Generative replay / synthetic data** | `G` | TARGET, FedCIL, FedER | a generative model fitted to client data | Critical (the model *is* the leak) |
| **F7 Class-distribution / count vectors** | `n_c,y` | C²Prompt's compensation, GLFC | exact per-class counts per client per task | Medium alone; **high as a join key** — it deanonymizes and it reveals events (§3.3) |
| **F8 Exemplar buffers** | `E_c` | Hybrid Replay (ICLR'25) | literal raw samples | Trivially critical; included as an upper-bound reference point |

Note F5 and F7 are the families the FCL literature treats as *most* privacy-safe and *least* worth
discussing. That inversion — "the artifacts believed safest have the cleanest closed-form inversions" —
is a headline candidate.

### 2.3 Units of privacy in FCL — a taxonomy nobody has written down

**This subsection is, on its own, a contribution.** DP is meaningless without an adjacency relation,
and FCL admits at least five defensible ones. Papers that say "we add DP noise" without picking one
are unfalsifiable.

Let `D = {D_c^k}` and `D'` be a neighbouring dataset. Then:

- **U1 Example-level.** `D'` differs in one example of one client at one task. *Weakest.* What most
  DP-SGD-in-FL papers silently use. Says almost nothing to a hospital.
- **U2 Task-level (client-task).** `D'` differs in the entirety of `D_c^k` for one `(c,k)`. "What my
  clinic saw in Q3 is protected." Natural for FCL, **and it is the unit under which parallel
  composition can be made to work** (§4.3).
- **U3 Client-level, bounded horizon.** `D'` differs in all data of one client across tasks 1…T.
  Standard cross-silo target. Requires a fixed T.
- **U4 Client-level, unbounded (lifelong).** Same but for the infinite stream. **This is the unit the
  FCL premise implicitly promises and the one no existing accountant can deliver non-vacuously.**
- **U5 Individual-level (person-level) under renewal.** `D'` differs in the data of one *person*,
  who may appear in several tasks of one client and possibly across clients. The realistic unit for
  healthcare and for cross-device populations that churn. Enables the renewal argument in §4.5.

**Definition (Lifelong DP).** A release channel `M` is `(ε, δ)`-lifelong-DP under adjacency `~` if
for every `T`, the truncated transcript `M_{≤T}` is `(ε, δ)`-DP under `~`, with `ε` **independent of
T**. Any accountant whose ε grows like `√T` or `T` fails this definition, i.e. it makes no lifelong
promise at all — it just hasn't been asked to run long enough to embarrass itself.

`H1` (**the vacuity claim**): *For every FCL method published to date, applying the standard DP-FL
accountant under U4 yields ε that grows without bound in T; therefore no published FCL method makes a
non-trivial lifelong client-level privacy claim.* — This should be provable/demonstrable on paper and
is the theoretical hook of Paper B.

### 2.4 Metrics

Attack-side (report **all** of these; AUC alone is not acceptable at a security venue):
- `AUC` of the membership decision.
- `TPR @ FPR = 1%` and `TPR @ FPR = 0.1%` (Carlini et al.'s standard — low-FPR regime is what matters).
- `ε_emp` — empirical ε **lower bound** from one-run auditing, with its Clopper–Pearson confidence level.
- Reconstruction: LPIPS ↓, SSIM ↑, and a human-judgeable panel of recovered images.
- Onset inference: precision/recall of "task boundary at round r for client c", and MAE in rounds.

Utility-side (must be reported alongside every privacy number or the trade-off is unreadable):
- Final average accuracy `A_T`, average incremental accuracy.
- Backward transfer / forgetting `BWT`.
- Communication bytes/round, wall-clock, and GPU-hours (the resource-realism lesson from NeurIPS'25).

Joint:
- **Retention–Leakage coefficient** `ρ_RL` := Spearman correlation, across methods, between
  `−BWT` (retention) and `AUC_old` (leakage measured on task-1 data from the task-T transcript).
  `H2` predicts `ρ_RL > 0` and significantly so.
- **Privacy–Forgetting frontier**: the Pareto surface of (ε, A_T, BWT). The deliverable plot of Paper B.

---

## 3. Part A — The attack / audit (Paper A)

### 3.1 The three temporal leakage phenomena

These are new *phenomena*, not new *attack algorithms*. Naming them correctly is most of the paper's
value; the attack implementations are largely adapted from known techniques.

#### 3.1.1 Persistence leakage — "forgetting is a privacy feature, and FCL removes it"

In ordinary FL, an adversary must be observing during the rounds when the target data participated.
Model drift subsequently *destroys* that signal — catastrophic forgetting is, accidentally, a privacy
mechanism. FCL methods are explicitly engineered to defeat catastrophic forgetting. Therefore:

`H2` (**retention–leakage trade-off, the flagship hypothesis**):
*Across FCL methods, the strength of anti-forgetting is positively associated with the strength of
cross-task membership leakage. Concretely: a method's `−BWT` on task 1 measured at task T correlates
positively with `AUC` / `TPR@1%FPR` for membership inference on task-1 data using only round-T
artifacts. Effect present with Spearman ρ ≥ 0.5, p < 0.05, over ≥ 8 methods × ≥ 3 datasets.*

Why this is groundbreaking if true: it reframes the field's central dilemma. Continual learning has
lived under **stability vs plasticity** for thirty years. This makes it a **stability–plasticity–privacy
trilemma**, and shows that the FCL community's entire optimization direction (more retention) is also
a privacy-degradation direction. Every FCL paper's leaderboard becomes, in part, a leakage leaderboard.

Why it might be false — Red Team's strongest line, and we should say it in the paper: retention is
about *class-level / task-level* knowledge, membership is about *individual samples*. A method could
retain the concept "zebra" perfectly while retaining nothing about *which* zebra photos client 3 held.
If H2 fails, the fallback finding is that split itself — **"retention of semantics without retention of
individuals is achievable, and here is which artifact families achieve it"** — which is a *design
guideline* result and still publishable. Plan explicitly for both outcomes (§7.2).

#### 3.1.2 Accumulation leakage — the transcript is worth more than the checkpoint

`H3`: *`TPR@1%FPR` under `V_full` substantially exceeds that under `V_final`, and grows monotonically
with the number of observed rounds. Quantitatively: the attack advantage scales at least like
`Θ(√(#observations))` in the low-signal regime, matching the intuition that repeated noisy views of
the same client average out the noise.*

Implication for practice: publishing intermediate checkpoints — which every FCL paper does implicitly
by broadcasting each round — is a privacy decision nobody has priced. Practical attack: build a
per-example *trajectory* feature (the sequence of losses / prototype distances / prompt-key affinities
across rounds) and classify membership from the trajectory rather than from a single scalar. Prior FL
MIA work (CS-MIA) used confidence *series* within one task; we extend across tasks and artifact types.

#### 3.1.3 Onset & composition leakage — inferring events, not memberships

A new inference target. From the transcript alone, infer:
- **Task-onset inference:** at which round did client `c`'s distribution shift?
- **Class-arrival inference:** which new class did client `c` acquire, and when?
- **Composition inference:** the per-class counts `n_c,y` (directly published by some methods, F7;
  estimable from prototype-update magnitudes for others).

`H4`: *Task-onset can be localized to within ±2 rounds with precision > 0.8 for prototype- and
statistics-based methods, without any auxiliary data.*

**Why this matters more than MIA for real deployments.** Membership inference is abstract to a hospital
CISO. "The federation reveals that St. Mary's began seeing cases of disease X in week 32, and roughly
how many" is a concrete, actionable, reportable harm — it is business intelligence and, in some
jurisdictions, a notifiable disclosure. This is the framing that gets a security PC excited and the
framing that makes the paper matter outside ML. **[LOCKED: onset inference stays in Paper A even if
it costs space.]**

### 3.2 Threat models

Each attack must declare its cell in this grid. Underspecified threat models are the #1 reason
security papers get rejected; the Threat Modeler agent has veto power here.

| Axis | Options | Our primary choice |
|---|---|---|
| Adversary role | honest-but-curious server / participating client / external model-release observer | **honest-but-curious server** and **participating client** (both passive) |
| Activity | passive vs active (can inject prompts, poison, or choose participation) | **Passive.** Deliberate contrast with PromptMIA, which needs an active malicious server. A passive attack is a much stronger result |
| Observation set | `V_final` / `V_task` / `V_full` | all three, as an ablation axis |
| Auxiliary knowledge | none / public data from the same distribution / shadow-model capability | **shadow-model capability on public data** (LiRA standard), plus a *no-auxiliary* variant for onset inference |
| Target | membership / attribute-property / reconstruction / onset / client-attribution | all five, but membership + onset are the load-bearing ones |
| Defenses assumed present | none / secure aggregation / DP noise / clipping | report with and without secure aggregation; DP is Part B |

### 3.3 Attack suite

- **A1 — Cross-task LiRA.** Offline/online LiRA (Carlini et al.) adapted to the FCL transcript.
  Shadow federations trained with the target example IN/OUT; score = likelihood ratio of the observed
  *trajectory* statistic. Key novelty: the score function is per-artifact-family (loss for F1;
  distance-to-prototype for F2/F5; prompt-key affinity for F3; projection onto adapter row-space for F4).
- **A2 — Analytic inversion of F5.** See §3.4; closed-form, no shadow models, no training.
- **A3 — Prototype-difference attribution.** Track `μ_c,y^{r} − μ_c,y^{r-1}`; with small per-round
  sample counts this approximates individual feature vectors. Then feature-space → image-space via a
  feature inversion network trained on public data.
- **A4 — Onset/composition inference.** Change-point detection (CUSUM / Bayesian online change point)
  on artifact-sequence statistics. No auxiliary data required. Baseline: random guess and a
  "norm-of-update spike" heuristic.
- **A5 — Client attribution.** Given a sample known to be in the federation, identify *which client*
  held it. Bridges to fairness (P8): if minority clients are more attributable, that is a distinct harm.
- **A6 — Property inference over time.** Does client `c`'s task-`k` data over-represent property `P`
  (e.g. a scanner vendor, a demographic proxy)? Track how long that inference remains possible after
  task `k` ends — the **leakage half-life**, a metric we introduce.

### 3.4 The analytic-FCL inversion — the sharpest single result available

Analytic FCL (the FedRAN line) shares regularized autocorrelation statistics
`R_c = X_cᵀX_c + λI ∈ R^{d×d}` and cross-correlation `Q_c = X_cᵀ Y_c`, where `X_c ∈ R^{n×d}` is the
client's frozen-backbone feature matrix. Elementary but consequential facts:

1. `R_c − λI = X_cᵀX_c` is the **exact Gram-transpose**; it determines `X_c` up to a left orthogonal
   transform `X_c ↦ U X_c` with `UᵀU = I`. No information about the *span* is lost. Ever.
2. When `n < d` — the normal regime for a client at one task with a ViT-B/16 feature dim of 768 and a
   few hundred samples per class — `rank(XᵀX) = n`, and the eigenvectors of `R_c` span exactly the
   client's sample subspace. Recovering individual `x_i` reduces to resolving one `n×n` rotation.
3. With **class-conditional** statistics (which the methods publish, because they need them for the
   closed-form head), `n` per class can be single- or double-digit. At `n = 1` recovery is exact up to
   sign.
4. Truncated-SVD compression (FedRAN's communication trick) is a *rank* reduction, not a privacy
   mechanism, and the retained top-r subspace is precisely the part with the most sample energy.

`H5`: *For analytic FCL with class-conditional Gram statistics and realistic per-class sample counts
(n ≤ 64) on a frozen ViT-B/16, feature-space reconstruction achieves cosine similarity > 0.8 to the
true features for a majority of samples, and image-space reconstructions are visually identifiable.*

This is a strong, cheap, closed-form result requiring **no shadow models and almost no GPU time**. It
is the project's de-risking anchor: even if every learned attack underperforms, §3.4 alone is a
publishable finding, and it also sets up the constructive twist in §4.4 (analytic methods are the
*worst* offenders when naive, and the *best* platform for DP when done right — a satisfying arc).

**[CONTESTED]** Whether authors of the analytic line will have added noise by the time we publish.
Mitigation: pre-register and move fast on this one; it is the most scoopable component.

---

## 4. Part B — Lifelong DP accounting (Paper B)

### 4.1 The problem in one sentence

Every deployed DP-FL accountant calibrates noise to a **known, finite horizon T**. A continual
federation has no T. If you guess T and the federation outlives it, your guarantee is void; if you
budget for T = ∞ with sequential composition, per-round noise must go to zero utility. Nobody has
resolved this for FCL.

### 4.2 Four structural levers

The only way to get a finite ε over an infinite stream is to find structure that stops composition
from being sequential. There are exactly four such levers available, and Paper B is the systematic
study of all four in the FCL setting.

**L1 — Disjointness (parallel composition).** If each datum influences the transcript through exactly
one task, ε_total = max_k ε_k, not Σ_k ε_k. Finite for free. §4.3.

**L2 — Data renewal (individual accounting).** If each *person* participates in at most `m` task-epochs
and the population churns, then person-level ε is bounded by the cost of `m` epochs regardless of T. §4.5.

**L3 — Correlated noise (continual observation).** Tree-aggregation / matrix-factorization mechanisms
pay `polylog(T)` instead of `√T` for releasing a running sum. §4.6.

**L4 — Adaptive stopping (filters & odometers).** Do not fix T; run until a per-client privacy filter
is exhausted, then transition that client to a zero-cost mode. §4.7.

A key thesis of Paper B: **these four levers are not alternatives, they compose**, and the right FCL
design uses L1 for the head, L3 for the backbone-adjacent updates, L2 across the population, and L4 as
the outer safety envelope. The composite mechanism is the paper's artifact.

### 4.3 L1 — When is parallel composition legal in FCL?

The DP-CL workshop paper (NeurIPS'24 W) gets a finite budget from parallel composition under two
assumptions: (S1) the full label set is known upfront, (S2) tasks are non-overlapping with no replay.
**Both fail in federation.** Enumerate the failure modes precisely — this enumeration is a contribution:

- **V1 Client-side task overlap.** Different clients hit task `k` at different wall-clock rounds
  (asynchronous drift, cf. P2). The server's release at round r mixes data from several task indices,
  so the "partition" the parallel-composition argument needs does not exist globally.
- **V2 Replay.** Exemplar buffers (F8) and generative replay (F6) *deliberately* reuse task-`j` data
  during task `k`. Each replayed datum is touched again → sequential composition returns.
- **V3 Anti-forgetting regularizers.** FOT's subspace projection, TPGP's gradient projection, and
  distillation losses all read functions of old-task statistics. Even if old *data* is gone, old
  *statistics* re-enter the release, which is what the adjacency relation cares about.
- **V4 Persistent per-client state.** Personalized components (pFedMxF), prompt keys, and analytic
  Gram accumulators are running sums over all history — the definition of a non-disjoint release.
- **V5 Repeated participation within a task.** `R_k` rounds per task means each datum is touched `R_k`
  times even inside one task; parallel composition across tasks does not save you inside a task.

**Definition (task-disjoint mechanism).** An FCL mechanism is *task-disjoint* if there exists a
partition of the private data by task index such that the transcript restricted to task `k` is a
function of `D^k` alone, and no other transcript element depends on `D^k`.

`H6`: *Of the eight artifact families in §2.2, only F5 (analytic Gram) and F2-with-single-pass
prototypes admit a task-disjoint implementation without accuracy collapse; F3, F4, F6, F8 provably
cannot without abandoning their anti-forgetting mechanism.*

**Theorem shape to prove (T1).** *For a task-disjoint FCL mechanism where each task-epoch is released
with a `(ε_k, δ_k)`-DP mechanism under unit U2, the lifelong transcript is `(max_k ε_k, max_k δ_k)`-DP
under U2 — for every T, hence lifelong-DP. Conversely, under unit U4 (client-level, all tasks), no
task-disjoint mechanism with `ε_k ≥ ε₀ > 0` for infinitely many k can be lifelong-DP.*

That converse is the honest bad news and should be stated loudly: **lifelong client-level DP is
impossible for a client that keeps contributing new information forever.** The resolution is that the
promise must be re-scoped to U2 or U5. Establishing *which promise is even attainable* is itself the
theoretical contribution — it tells the field what it is allowed to claim.

### 4.4 The constructive proposal: DP-Analytic-FCL

The design that falls out of §4.3, and the reason Paper A's §3.4 sets it up:

- **Head:** closed-form ridge/analytic classifier over frozen features. Each sample enters the released
  statistics `(R_c, Q_c)` **exactly once** ⇒ single-pass ⇒ task-disjoint by construction ⇒ L1 applies.
- **Mechanism:** the Gaussian mechanism on the *statistics*, not on gradients. Sensitivity of
  `R_c = Σ x_i x_iᵀ` under U1 with `‖x‖₂ ≤ B` is `B²` in Frobenius norm — a *tight, analytic*
  sensitivity, unlike DP-SGD where clipping bias is an empirical nuisance. This is the key technical
  advantage and should be argued hard: **closed-form learning gives closed-form sensitivity.**
- **Composition inside a task:** none needed. One release per task per client.
- **Across the stream:** L1 gives `max_k ε_k`. Constant in T. Done.
- **For the population:** L2 refreshes as new people enter.
- **Safety envelope:** L4 filters cap any client that ends up over-contributing.

`H7`: *DP-Analytic-FCL attains, at client-task-level ε = 1, final average accuracy within 5 points of
the non-private analytic baseline and above every DP-SGD-based FCL baseline at the same ε, on
CIFAR-100/10-task and ImageNet-R/10-task with a frozen ViT-B/16.*

`H8`: *The advantage widens with T. At T = 50 tasks, DP-SGD-based FCL under U2-sequential accounting
is at chance while DP-Analytic-FCL is flat in T.* — The "flat line vs collapsing line" figure is the
paper's money plot.

**[CONTESTED]** Is DP-Analytic-FCL too simple to be a NeurIPS/ICML contribution on its own? Author's
position: the mechanism is simple; the *impossibility result* (T1 converse), the *taxonomy of units*
(§2.3), the *violation enumeration* (V1–V5) and the composite accountant (§4.8) are the contribution,
and the simple mechanism is the constructive proof that the taxonomy is actionable. Red Team should
push here; this is the most likely rejection axis.

### 4.5 L2 — Individual accounting and the renewal argument

Use Rényi filters / individual privacy accounting (Feldman & Zrnic, NeurIPS'21): each *person* carries
their own odometer, charged only in rounds where their data actually participates.

Model the population as a renewal process: person `p` is present in client `c` for a window of `m_p`
task-epochs, then departs (patients are discharged; devices are replaced; users churn).

**Theorem shape (T2).** *Under a renewal model with `E[m_p] = m` and a task-disjoint per-epoch
mechanism at `ε₀`, person-level (U5) privacy loss over the infinite stream is bounded by a function of
`m` and `ε₀` alone, independent of T, with high probability over the renewal process.*

This is the argument that makes an *actually infinite* deployment defensible, and it is the argument a
hospital consortium would need. It is also, as far as we can determine, unmade in the FL literature.

**[CONTESTED]** Realism of the renewal assumption. A chronic-care patient appears for years. Handle by
reporting the guarantee as a function of `m` and being explicit that long-tenure individuals get a
weaker guarantee — and *measure* that tail, don't hide it. The honest version of this ("privacy is not
equally distributed across people, and here is the distribution") is a better paper than a fake
uniform bound, and connects to fairness (P8).

### 4.6 L3 — Correlated noise for the parts that cannot be disjoint

Where the design does need a running sum over rounds (backbone-adjacent adaptation, momentum, running
prototypes), use matrix-factorization mechanisms — DP-FTRL's tree aggregation, banded MF, BLT. These
are literally continual-observation mechanisms and are the right import.

**Open theory question (the real theory contribution, `Q-T3`).** All existing MF mechanisms optimize
the factorization for a *known* T and a *homogeneous* participation pattern. FCL gives a stream with
**block structure of unknown, unequal block lengths** (tasks) and **block-correlated participation**.
What is the optimal (or a good, provably near-optimal) factorization for a block-structured stream of
unbounded length? Candidate: an infinite banded Toeplitz factorization whose band width tracks the
task length, extended online — the BLT line's buffered-linear-Toeplitz parameterization is the natural
starting point precisely because it is streaming-friendly and has O(1) state.

This is the piece most likely to yield a genuinely new theorem. It is also the piece most likely to
fail. Timebox it (§6) and be ready to ship Paper B without it.

### 4.7 L4 — Per-client privacy filters and the drift they induce

Give every client a privacy filter (Rogers et al. '16; Whitehouse et al. ICML'23 for fully-adaptive
composition). A client whose filter is exhausted stops contributing new private information.

`H9`: *Filter exhaustion is heterogeneous across clients (data-rich and high-drift clients exhaust
first), which silently converts a privacy mechanism into a participation-bias mechanism, measurably
degrading accuracy on those clients' distributions over time.*

Call this **privacy-induced participation drift**. It is a *new failure mode of DP in continual
federations* and it is exactly the kind of second-order finding that makes a paper memorable. It is
also the bridge to P8 (fairness over time) and, if it lands, a natural third paper.

### 4.8 The composite accountant (the software artifact)

A single library that, given an FCL method's *artifact schedule* (which family, which unit touched,
how many times, under what participation model), emits a lifelong ε or a certificate that none exists.
Concretely it must:
1. Take a declarative description of the method's releases (see `code/src/p3fcl/artifacts.py`).
2. Statically check task-disjointness (V1–V5 checker).
3. Route each release to the right accounting regime (parallel / RDP-sequential / MF / filter).
4. Return `ε(T)` as a *function*, and flag divergence.

Shipping this as an open-source tool is what makes the paper adopted rather than cited-and-forgotten.
It is also the deliverable that most directly serves the FCL community, which currently has no way at
all to check a privacy claim.

---

## 5. Experimental design

### 5.1 Methods under audit

**[LOCKED]** Nine systems, chosen to span the artifact families, not to be exhaustive.

| # | Method | Venue | Families | Role |
|---|---|---|---|---|
| M0 | FedAvg + sequential fine-tuning | — | F1 | Lower bound on retention, calibration point for attacks |
| M1 | GLFC | CVPR'22 | F1, F7, F8 | The classic FCIL reference |
| M2 | TARGET | ICCV'23 | F1, F6 | Generative replay; expected worst case |
| M3 | FOT | ICLR'24 | F1 + subspace | Orthogonality/regularization family |
| M4 | PILoRA | ECCV'24 | F2, F4 | Prototype + LoRA |
| M5 | Hybrid Replay FCIL | ICLR'25 | F8 | Raw-exemplar upper bound on leakage |
| M6 | C²Prompt | NeurIPS'25 | F3, F7 | Current prompt SOTA; publishes class distributions |
| M7 | TPGP | ICCV'25 | F3 + projection | Prompt + gradient projection |
| M8 | Analytic FCL (FedRAN-style reimpl.) | preprint '26 | F5 | The closed-form inversion target and Part B's platform |

For each: reproduce reported accuracy within ~2 points before running any attack. **A failed
reproduction invalidates every downstream privacy claim about that method** — this gate is
non-negotiable and is the Reproducibility agent's primary job.

### 5.2 Datasets and streams

| Dataset | Stream | Clients | Why |
|---|---|---|---|
| CIFAR-100 | 10 tasks × 10 classes, Dirichlet β ∈ {0.1, 0.5} | 10 / 50 | The field's default; needed for comparability |
| ImageNet-R | 10 and 20 tasks | 10 / 50 | Standard for PEFT-era FCL; harder |
| CUB-200 | 10 tasks | 10 | Fine-grained; small per-class n ⇒ **best case for the inversion attack** |
| DomainNet | 6 domains, domain-IL | 12 | Domain drift rather than class arrival |
| **Camelyon17-WILDS** (or NIH ChestX-ray14 split by site+time) | hospitals as clients, time as stream | 5 | **The sensitive-domain motivation.** A security venue needs a real-harm story, not CIFAR |
| **Long-horizon CIFAR-100** | 50 tasks × 2 classes | 10 | The T → ∞ stress test that produces the §4.4 money plot |

The long-horizon stream is deliberately unrealistic as a learning benchmark and deliberately essential
as an *accounting* benchmark. State that reasoning in the paper before a reviewer states it for you.

### 5.3 The compute plan on NCSA Delta

> **[SUPERSEDED 2026-09-14 — human decision, not an agent edit.]** Compute moved to **Baylor Kodiak**.
> This section is `[LOCKED]` so its text is left intact as the historical record, but **every number
> in it is void**: there are no A100s (the accessible pool is 10 GPUs on `gpu001`–`gpu005`), there is
> no Slurm (Kodiak is OpenPBS 23.06), and the ≈2,900 A100-hour budget does not transfer. The live
> cost model and the binding constraints — including a **512 concurrent-core per-user cap** that
> reshapes every array job — are in **`build/01_KODIAK.md §2 and §6`**. Read that instead.
> The *structure* of this section still holds: cacheable-vs-not is still the axis that decides
> feasibility, and it matters more on Kodiak than it did on Delta, not less.

**[LOCKED]** The dominant cost is shadow models for LiRA. Everything else is rounding error.

**The key efficiency trick: split the method zoo by whether features can be cached.**

- **Cacheable (M4-prototype-part, M8, and all F2/F5/F7 analysis):** the backbone is frozen and inputs
  are unmodified, so extract ViT-B/16 features **once per dataset** (CIFAR-100 60k images ≈ minutes on
  one A100) and run the entire FCL simulation as linear algebra on cached `d = 768` features.
  A full shadow federation costs **seconds of CPU**. Budget **thousands** of shadow models here —
  which means near-tight LiRA and genuinely tight one-run audits.
- **Non-cacheable (M6, M7, and M4's LoRA part):** prompts/adapters modify the forward pass, so every
  shadow needs real backprop through the frozen ViT. Budget ≈ 1–2 A100-hours per shadow federation.

Indicative budget:

| Line item | Unit cost | Count | GPU-hours |
|---|---|---|---|
| Feature extraction, all datasets | 0.5 h | 6 | 3 |
| Reproduction of M0–M8 (all datasets, 3 seeds) | ~4 h | 9×6×3 | ~650 |
| Shadow federations, cacheable methods | ~0 (CPU) | 2000+ | ~20 |
| Shadow federations, non-cacheable (M4/M6/M7) | 1.5 h | 3 methods × 3 datasets × 64 | ~865 |
| One-run auditing runs | 2 h | 9 methods × 3 datasets × 5 ε | ~270 |
| DP-Analytic-FCL sweeps (Paper B) | ~0.2 h | ~600 configs | ~120 |
| Long-horizon 50-task runs | 6 h | 9 × 3 seeds | ~160 |
| Slack / reruns / failed jobs (assume 40%) | — | — | ~830 |
| **Total** | | | **≈ 2,900 A100-hours** |

That is a realistic single-year Delta allocation request. Write the allocation proposal around the
reproduction + shadow-model lines, since those are the defensible, itemizable ones.

**Engineering requirements this implies:**
- Deterministic seeding and a manifest per run; SLURM job arrays for the shadow sweep.
- Feature caches on the parallel filesystem in a fixed layout, written once, read-only thereafter.
- Every run writes an **artifact ledger** (§`code/src/p3fcl/artifacts.py`) so that attacks operate on
  a uniform record format rather than on nine bespoke checkpoint formats. This single abstraction is
  what makes the audit tractable — build it first, before any method.

### 5.4 Statistical discipline

The Empiricist agent enforces all of this:
- ≥ 3 seeds for every reported number; report mean ± std, never a single run.
- Low-FPR TPR is primary; AUC is secondary. A method that looks safe at AUC 0.55 can be catastrophic
  at TPR@0.1%FPR.
- Clopper–Pearson intervals for all empirical ε lower bounds; state the confidence level.
- **Pre-register H1–H12 and the analysis plan** (OSF or a timestamped repo commit) before running the
  audit. This costs nothing and pre-empts "you fished for this" at review.
- Where H2 is a correlation over 9 methods, `n = 9` is small: report the correlation *and* a
  per-dataset replication, and do not overclaim. Consider adding intermediate method variants
  (e.g. the same method with retention strength dialed up/down) to get a **within-method** dose–response
  curve, which is far stronger evidence than a cross-method correlation. **Do this — it converts H2
  from correlational to quasi-experimental and is the single highest-value methodological upgrade in
  the plan.**

---

## 6. Timeline (18 months, two papers)

| Phase | Months | Output | Gate to pass |
|---|---|---|---|
| **P0 Infrastructure** | 1–2 | Artifact ledger, feature caches, FCL sim loop, M0/M8 running | M8 reproduces analytic-FCL accuracy |
| **P1 Cheap wins** | 2–4 | §3.4 inversion result + A4 onset inference | Inversion works on CUB-200 (H5) |
| **P2 Reproduction** | 3–6 | M1–M7 reproduced within 2 pts | Reproduction table complete and honest |
| **P3 The audit** | 5–9 | A1–A6 across all methods; H2/H3/H4 tested | H2 resolved either way |
| **P4 Paper A** | 9–11 | Submission to USENIX Sec / IEEE S&P (see §8) | Pre-registration honoured |
| **P5 Theory** | 8–13 | T1, T2; V1–V5 formalized; Q-T3 attempted (**timebox: 8 weeks**) | T1 + T2 proved; Q-T3 optional |
| **P6 Mechanism** | 12–16 | DP-Analytic-FCL, composite accountant, H7/H8/H9 | H7 holds or the negative is characterized |
| **P7 Paper B** | 16–18 | Submission to NeurIPS/ICML | Money plot exists |

Parallelism note: P1 needs almost no GPU and can start on day 1 while the allocation is pending. Do
not sequence the whole project behind the Delta allocation.

---

## 7. Risk register

| # | Risk | Likelihood | Impact | Mitigation / fallback |
|---|---|---|---|---|
| R1 | **H2 is false** — retention and leakage are uncorrelated | Medium | High (kills the flagship framing) | Fallback framing in §7.2. Also: the within-method dose–response design (§5.4) means even a null is a *clean* null, which is publishable as a rigorous negative audit |
| R2 | Attacks are weak across the board (all AUC ≈ 0.52) | Medium | High | One-run auditing turns weak attacks into **rigorous ε lower bounds**, which is a legitimate security result. And §3.4 (closed-form, no learning) is unlikely to fail with the rest |
| R3 | Reproduction of M1–M7 fails | **High** (FCL code quality is uneven) | Medium | Budget 40% slack (already in §5.3); drop a method rather than audit a broken reimplementation; report reproduction failures as a finding — the NeurIPS'25 paper made a career out of exactly this |
| R4 | Scooped on the analytic inversion | Medium | Medium | Move first (P1, months 2–4); it needs no allocation |
| R5 | Q-T3 (block-structured MF) does not yield a theorem | High | Low | Timeboxed; Paper B ships on T1+T2+mechanism |
| R6 | Reviewers say "this is just DP-SGD applied to FCL" | Medium | High | The T1 converse (impossibility of lifelong U4) and the U1–U5 taxonomy are the defence. Lead with them, not with the mechanism |
| R7 | Medical data access (Camelyon17 is public; ChestX-ray14 needs care) | Low | Medium | Camelyon17-WILDS is fully public — default to it; treat ChestX-ray14 as optional |
| R8 | Ethics/disclosure friction with audited authors | Low | Medium | §9 protocol; notify authors before submission, not after acceptance |

### 7.1 The pre-mortem

*It is month 18. The project failed. What happened?* Most likely story: months 3–8 disappeared into
reproducing seven codebases that do not reproduce, the attack suite was built late against a moving
target, and the theory was never started. **Countermeasure, and it is the most important operational
decision in this document: build the artifact ledger and the attack harness against M0 and M8 FIRST
(months 1–4, no allocation needed), and treat M1–M7 as a plug-in queue that can be truncated at any
time.** A paper auditing five methods lands; a paper auditing nine methods in month 24 does not.

### 7.2 The fallback narrative if H2 dies

Retitle from *"Anti-forgetting is anti-privacy"* to *"What FCL actually leaks: a systematic audit of
shared artifacts"*, and lead with (a) the artifact taxonomy, (b) the closed-form inversion of the
family the field believes is safest, (c) onset/composition inference as a new and more deployment-
relevant harm class, (d) rigorous empirical ε lower bounds where prior work offered none. That is a
solid USENIX/S&P paper with no dependence on H2 whatsoever. **Confirm this before starting: the project
must be designed so that H2 is upside, not load-bearing.**

---

## 8. Venue and publication strategy

**Paper A — the audit.** Primary: USENIX Security or IEEE S&P. Rationale: security venues reward
systematic measurement and rigorous negative results; ML venues punish papers without a new method.
Secondary: CCS. Tertiary: NDSS. If it must go to an ML venue, NeurIPS Datasets & Benchmarks (framed as
a leakage benchmark + tooling) rather than the main track.

**Paper B — the accounting.** Primary: NeurIPS or ICML main track, framed *theory-first*
(units taxonomy → impossibility → constructive mechanism), with the empirical section as
confirmation. Secondary: PETS (Privacy Enhancing Technologies Symposium) — a natural home and a
friendlier review process for exactly this work. Tertiary: SaTML.

**Sequencing rationale (the answer to "which half first"):** attack first. Three reasons: (1) the audit
builds the entire codebase the theory paper needs, so it is not sequential cost, it is shared cost;
(2) the audit is *de-risked* — negative results are publishable at security venues, which is not true
at ML venues; (3) the theory paper's motivation section is dramatically stronger when it can cite your
own measured leakage rather than assert that leakage is plausible. Do not invert this order.

**Artifact release.** Both papers ship code. Target USENIX Artifact Evaluation "Available + Functional +
Reproduced" badges — cheap credibility for a first-time security submission, and the composite
accountant (§4.8) is exactly the kind of tool AE committees like.

---

## 9. Ethics and responsible disclosure

**[LOCKED]** Non-negotiable.

1. **No real-person re-identification.** All membership/reconstruction experiments run on public
   research datasets. For Camelyon17, report aggregate leakage metrics only; publish no reconstructed
   patient-derived image that could be traced to a slide.
2. **Coordinated disclosure.** Contact the authors of every audited method ≥ 60 days before submission
   with the specific finding and a chance to respond. Include their responses in the paper. This is
   standard security practice, it makes reviewers trust you, and it turns potential enemies at review
   time into collaborators.
3. **Frame as a systemic finding, not as an accusation.** The point is that the *field's shared
   assumption* is wrong, not that any one group was careless. Write it that way; it is also true.
4. **IRB.** Public benchmark data → likely exempt, but file for a determination anyway before touching
   any medical dataset. Do this in month 1; it is free and slow.
5. **Dual-use.** Release attack code, but as an *auditing* library with a documented intended use, and
   consider withholding the highest-fidelity reconstruction weights pending disclosure timelines.

---

## 10. What "groundbreaking" would actually look like here

Three specific outcomes, ranked by how much they would change the field:

1. **The trilemma (H2).** If retention and leakage are genuinely coupled, every FCL leaderboard is
   partially a leakage leaderboard, and the field's optimization target is revealed to be in tension
   with its stated purpose. This is a *reframing*, and reframings are what get cited for a decade.
2. **The impossibility result (T1 converse).** A clean statement of what lifelong privacy promise is
   and is not attainable in a federation that trains forever. It tells 200 future papers what they are
   allowed to claim in their abstract. Small theorem, large blast radius.
3. **The safe-architecture claim (H6 + H7).** "Closed-form/analytic FCL is the DP-native architecture
   for continual federations, and here is the accountant that proves it" — a constructive, adoptable
   design principle that also connects P3 to P6 and P7 in the broader agenda.

Anything less than one of these three and the project is a competent audit paper. That is a fine
floor. Aim at all three; design so that the floor is guaranteed.

---

## 11. Connections to the rest of the agenda

- **P6 (analytic FCL)** — §4.4 makes analytic FCL the DP platform. Shared codebase. Do P6 and P3-B
  together; they are one project wearing two hats.
- **P7 (federated continual unlearning)** — the audit machinery (A1, A5) *is* the unlearning
  verification machinery. An unlearning claim without an MIA audit is unfalsifiable, and we will own
  the audit.
- **P8 (fairness over time)** — §4.7's privacy-induced participation drift and A5's client attribution
  both produce per-client harm distributions. Natural third paper.
- **P1 (natural-drift benchmark)** — if P1 exists, run the audit on it: leakage under *natural* drift
  vs synthetic splits is a strong secondary result and a reason to build P1 first.

---

## 12. Immediate next actions (first 30 days)

1. Build `code/src/p3fcl/artifacts.py` — the artifact ledger. Everything depends on it.
2. Implement M8 (analytic FCL) on cached ViT-B/16 CIFAR-100 features. Runs on a laptop.
3. Run the §3.4 closed-form inversion at n = 1, 8, 64. This is a two-day experiment that either
   validates the project's sharpest claim or forces a rethink in month 1 rather than month 12.
4. Implement the one-run auditing harness against M8 (cheap, cacheable, thousands of shadows).
5. Write the Delta allocation request around §5.3.
6. File the IRB determination request.
7. Pre-register H1–H12.
8. Set up the multi-agent debate loop (`agents/`) and run the first adversarial pass on this document.

---

*End of plan. Contest it.*
