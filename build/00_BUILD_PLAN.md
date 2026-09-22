# 00 — BUILD PLAN

The master instruction. Phases **P0 → P9**, in order. Each phase has a gate: do not start the next
phase until the gate passes. Where a phase can run in parallel with another it says so explicitly.

Read `../CLAUDE.md` first. Companion specs:
`01_KODIAK.md` (cluster) · `02_DATASETS.md` (data) · `03_RESULTS_SPEC.md` (every figure/table and its
CSV schema) · `04_METHODS_AND_ATTACKS.md` (the method zoo and attack suite).

**Working style.** Small commits, one concern each. After every phase, write a dated note in
`notes/` saying what happened, what surprised you, and what the next person needs to know. If a gate
fails twice for the same reason, stop and write the problem down rather than trying a third variant.

---

## P0 — The package, the harness, provenance

**Goal.** Build `code/` from nothing to the spec in `06_PACKAGE_SPEC.md`, with `make smoke`,
`make test` and `make figures` all working end to end on synthetic features, and every run
provenance-stamped. Nothing downloads, nothing touches a GPU, except the one probe job.

### Tasks

1. **Probe job first, then keep coding while it queues.** Submit the 5-minute GPU probe and have it
   answer the five open items in `01_KODIAK.md §2`: does `-q gpu -l select=1:ncpus=8:ngpus=1` grant a
   GPU; what does `nvidia-smi` report for model and driver; can a compute node reach the internet
   (`curl -sI https://pypi.org`); is there node-local scratch; which CUDA module matches the driver.
   Write the answers into §2. Do not wait on it — start task 2 immediately.

2. **Environment.** `module load python/3.10.4`, then `python -m venv /data/islamm/retention_leakage/envs/p3fcl`.
   **There is no conda module on Kodiak.** Pin exact versions in `env/requirements.txt`: numpy, scipy,
   pandas, pyyaml, matplotlib, tqdm; torch + torchvision + timm matching the CUDA the driver supports;
   wilds; pytest, ruff. Nothing large in `$HOME`.

3. **Build the package** in dependency order, testing as you go:
   `artifacts.py` (the ledger — first, everything depends on it) → `units.py` → `rng.py` →
   `provenance.py` → `config.py` → `metrics.py` → `streams.py` → `features.py` (cache API; leave
   `extract` for P1) → `methods/base.py` → `sim.py` → `dp/mechanisms.py` → `dp/accountant.py` →
   `audit/one_run.py` → `attacks/base.py` → `plotting.py` → `cli.py`. Spec: `06_PACKAGE_SPEC.md`.

4. **Two methods, so the loop is exercisable end to end:** **M8** (analytic closed-form head, family
   F5) and **M4's prototype half** (family F2 + F7). These two are cacheable, run on a laptop, and
   between them exercise the ledger, the simulator, the accountant and the inversion attack. The
   rest of the zoo is P2.

5. **The four claim tests** (`06_PACKAGE_SPEC.md §9`) plus the ordinary unit tests. The claim tests
   encode T1-fwd, H1, analytic task-disjointness and H5-exact-at-n=1. If one of them ever fails, a
   load-bearing argument has broken — say so rather than patching the test.

6. **Makefile** with `setup test smoke figures verify paper`. `make verify` is a real script
   (`analysis/verify_provenance.py`): parse `paper/**/*.tex`, extract numeric literals, fail on any
   that appears in no `results/*.csv` and no generated `tables/*.tex`. Allow a `\providedvalue{}`
   escape for genuinely non-experimental numbers and list them in its report.

7. **PBS scripts** under `code/scripts/pbs/`, OpenPBS syntax only (`select`/`ncpus`/`ngpus`, `-J`,
   `$PBS_ARRAY_INDEX`), literal log paths — `#PBS` directives are not shell-expanded. Plus
   `scripts/submit.sh` which submits, records the job id to `results/JOBS.jsonl`, and
   `scripts/wait_job.sh` which polls with a bounded timeout and exits non-zero on failure.

8. **Seed discipline.** No bare `np.random.*` anywhere in the package — everything goes through
   `rng.seeded(name, seed)`. Add a test that greps the source and fails if a bare call appears.

**Gate P0.** `make test` green including all four claim tests; `make smoke` under 60 s on synthetic
features; `make verify` runs clean on an empty paper; `01_KODIAK.md §2` has the probe answers; one
job successfully submitted and returned. Report the numbers and ask to proceed.

## P1 — Data and features

Runs on GPU nodes. One-time cost. **This is the only large GPU expenditure in the project.**

### Tasks

1. Implement `scripts/get_data.py` per `02_DATASETS.md`: download, verify checksums, extract,
   and write `data/MANIFEST.json` recording per-dataset URL, bytes, sha256, license, n_images,
   n_classes, and the natural client/domain field where one exists. Idempotent and resumable.
   Datasets: **CIFAR-100, ImageNet-R, CUB-200-2011, Camelyon17-WILDS** are required;
   **FMoW-WILDS and DomainNet** are stretch (see `02_DATASETS.md §6`).

2. Implement `features.extract()` (currently `NotImplementedError`, deliberately). timm,
   `pretrained=True, num_classes=0`, `torch.no_grad()`, AMP on, batch 256 on V100 / 128 on P100.
   Backbones: `vit_base_patch16_224.augreg_in21k` (primary) and `vit_base_patch14_dinov2.lvd142m`
   (robustness check). Save via the existing `features.save_cache`, which already keys on
   dataset|backbone|split. Store under `/data/$USER/retention_leakage/features`.

3. **Store both pooled features and the pre-norm CLS token.** The Gram-inversion attack (A2) needs
   the exact vector the method would consume; do not silently L2-normalise at extraction time.
   Normalisation is a *method* decision and belongs in the method.

4. Extract a **public reference split** per dataset — a disjoint slice the adversary is allowed to
   hold (`split="ref"`). A2 and A1's offline variant need it. Record the split indices in the
   manifest so it is reproducible and auditable.

5. Sanity check every cache: linear probe on the full cached training set, report top-1. If
   CIFAR-100 with `augreg_in21k` does not land in the high 80s / low 90s, the extraction is wrong —
   stop and debug rather than continuing. Write the numbers to `results/tab01_feature_sanity.csv`.

**Gate P1.** `data/MANIFEST.json` complete for the four required datasets; feature caches exist for
both backbones × {train, test, ref}; `tab01_feature_sanity.csv` written and plausible; total GPU hours
recorded in the run log.

---

## P2 — Method zoo

Depends on P1 for real features but can be written and unit-tested against synthetic features in
parallel with P1.

Implement M0–M9 to the spec in `04_METHODS_AND_ATTACKS.md §1`. Every method subclasses
`methods.base.FCLMethod`, emits `ArtifactRecord`s with honest `touched`, and declares `cacheable`.

**The cacheable/non-cacheable split is load-bearing** (`CLAUDE.md`, Compute reality). M0, M1, M2,
M3, M5, M8, M9 reduce to head-level or feature-space work on a frozen backbone and run on CPU.
**M6 (C²Prompt), M7 (TPGP) and M4's LoRA half** modify the forward pass, need backprop through the
ViT, and get a reduced shadow budget — plan for that from the start rather than discovering it later.
Verify the cacheability assumption per method as you implement it, and record any method that turns
out not to be cacheable after all.

Deliverables: the utility baseline table **TAB05** (final average accuracy, BWT, average incremental
accuracy, per method × dataset × {10, 20} tasks × 3 seeds), and the reproduction-gap table **TAB07**
comparing our reimplementation against each paper's published number with the tolerance we hit.

**Gate P2.** Every method runs on every required dataset; TAB05 and TAB07 written; each method's
disjointness report (`dp.accountant.check_disjointness`) is recorded in `results/disjointness_report.csv`
— this is the empirical half of hypothesis H6 and it must be produced *before* any DP claim.

---

## P3 — Attack suite and the shadow-federation runner

The compute-heavy phase, and the one that must be engineered rather than scripted.

### Tasks

1. **Shadow-federation runner** (`scripts/run_shadows.py`). Given a config and a shadow index, run
   one full federation on cached features with a known IN/OUT membership vector, and write only the
   *statistics the attack needs* — not the whole ledger — to a compact `.npz`. Shadow outputs must
   be small enough that 4,000 of them fit comfortably in `/data`.
   Parallelism: a PBS **job array** over shadow indices on CPU nodes (`01_KODIAK.md §4`). Chunk so
   one array task does ~25 shadows and runs 20–40 minutes; an array task that runs 30 seconds is
   scheduler abuse, and one that runs 12 hours cannot be resumed.

2. **A1 cross-task LiRA.** Finish `attacks/lira.py` (the interface is fixed, the runner is stubbed).
   Both **online** (IN and OUT shadows) and **offline** (OUT only, for the expensive non-cacheable
   methods) variants. Per-family score functions are already sketched; add F1 (loss/logit-margin on
   the released head), F3 (prompt-key similarity) and F4 (energy in the adapter row space, already
   there as `score_lowrank`). Trajectory features per `lira.trajectory` — and **ablate them**, since
   "is the transcript worth more than the checkpoint" is hypothesis H3 and needs its own number.

3. **A3 prototype-difference attack.** New. Successive released prototypes plus released counts
   (family F7 is the join key) let an adversary solve for the *sum of newly contributed features*
   between rounds: `n_t·μ_t − n_{t−1}·μ_{t−1}`. When the increment is small — which is exactly the
   late-stream, low-data regime — this approaches individual reconstruction. Implement, and
   report its dependence on the increment size.

4. **A6 property inference over time.** Infer, from round-T artifacts, whether client c held class
   y at task k, and estimate client c's task-k class histogram. Cheap, clean harm story — and
   **this is H10's deciding test, so it carries claim C1.** Build it before A1, not after.
   **A5 client attribution** (which client holds a known sample) is the smaller sibling; build it
   with A6 since they share machinery, and note that it does *not* survive secure aggregation.

5. **Secure-aggregation variants.** For A1, A3, A4, A6, implement a mode that consumes only the
   *aggregate* broadcast rather than per-client records. This is hypothesis H11 and it is the claim
   that pre-empts "just use secure aggregation" in review. Near-zero extra cost. Prioritise it.

6. **A2 and A4 are cheap — build them early.** Re-specify A2's hypothesis first: the toy-scale check in
   `notes/2026-08-25_preliminary.md` shows the "n ≤ 64" threshold is too aggressive. Replace it with
   an **n-threshold curve as a function of reference quality and feature anisotropy**, per the note.
   A4 needs a multi-round method to be exercised at all — M0/M1/M3/M6 provide that now (M8 releases once per task, which is why the smoke test detects nothing).

7. **A7 one-run audit.** Wire `audit/one_run.py` to M8/M9 with DP on, and to at least one
   non-private method as a control. The audited ε lower bound versus the analytical ε is result
   **FIG10** and it is the project's insurance policy if every learned attack comes back weak.

**Gate P3.** Every attack runs end to end on at least CIFAR-100 and produces a membership report
with TPR@1%FPR, TPR@0.1%FPR, AUC and exact CIs. Shadow runner validated: re-running one shadow index
with the same seed reproduces its `.npz` byte-identically.

---

## P4 — The decoupling experiment (claim C1)

The headline. Everything before this was infrastructure.

**Protocol.** For each dataset × method × seed: run the full stream; at every task index T, run A1
against every earlier task k ≤ T; record `TPR@1%FPR(k, T)` alongside `acc(k, T)` from the accuracy
matrix the simulator already produces. The two surfaces over `(k, T)` are the experiment.

Derived quantities:
- **Accuracy half-life** `h_acc(k)` — elapsed tasks until accuracy on task k falls to half its peak.
- **Leakage half-life** `h_leak(k)` — same for TPR@1%FPR above the chance floor.
- **Decoupling ratio** `h_leak / h_acc`, per artifact family, bootstrapped CI.

Fit an exponential decay per family; report R² and, if the fit is poor, report the raw curve and say
so rather than forcing a fit. Fitting a half-life to a flat curve is a reporting error — for
append-only families the honest answer may be "no decay detected over the observed horizon, lower
bound on half-life > H", and that is a stronger sentence anyway.

**Deliverables.** FIG01 (decoupling, headline), FIG02 (half-life forest plot), TAB03 (main leakage table),
TAB04 (half-life estimates). Long-horizon variant at T = 50 tasks on CIFAR-100 and ImageNet-R for FIG06.

**Gate P4.** FIG01 regenerates from its CSV; the decoupling ratio has a CI that excludes 1.0 for at
least one family, or a clearly reported null.

---

## P5 — Dose–response (claim C2)

Within-method causal version. Sweep retention strength over ≥6 levels on ≥3 methods × ≥2 datasets
× 3 seeds: distillation / exemplar weight (M1 GLFC), projection strength (M3 FOT), latent-exemplar
buffer size (M5 Hybrid Replay), prompt-pool size (M6 C²Prompt, GPU), ridge λ (M8/M9). Plot retention
(−BWT) against leakage, carrying **both** A6 property balanced-accuracy and A1 TPR@1%FPR, with final
accuracy as a third channel.

**The known strongest objection, from the Red Team, is unanswered and you must engage it head on:**
retention is *semantic* while membership is *individual* — a method can retain class means perfectly
and leak nothing example-specific. Design the sweep so it can distinguish those: include at least
one method whose retention is purely semantic (**M4's prototype half, counts off**) and one whose
retention is example-level (**M5 Hybrid Replay's latent exemplars**). If the dose–response holds for the second and not the first,
that is the finding, and it is a *better* paper than a uniform correlation. Report it that way.

**Deliverables.** FIG03, FIG04 (the semantic-vs-individual split), TAB06.

**Gate P5.** Monotone trend with CI, or a reported null with the semantic/individual split shown.

---

## P6 — Accounting and the theory artifacts (claim C3)

Mostly paper exercise plus machine-checked numerics. No GPU.

1. **Unit taxonomy table TAB02.** For every method in the zoo and for the published methods we cite,
   state which unit of privacy its privacy claim (if any) implicitly uses. Cite the sentence.
   This table is a contribution in itself; it is also the thing that makes the rest non-pedantic.

2. **ε(T) curves, FIG05.** Run `dp.accountant.account` over every method's real ledger at matched σ,
   for units U1–U5, T ∈ {1 … 1000}. The skeleton already reproduces the shape numerically
   (U2 flat; U4 at 15 → 262 → 3901 for T = 1, 50, 1000). Re-run it on *real* ledgers from P2 and
   report it as the measured divergence. This is H1.

3. **The characterisation.** Formalise contractive vs accumulating retention operators and prove
   T1-fwd and T1-conv, or reduce them to a clearly stated open problem. Obligations are tracked in
   `agents/OPEN_QUESTIONS.md § Theorem obligations`; `ASPIRATIONAL → SKETCHED` needs a written proof
   sketch, `SKETCHED → PROVED` needs a full proof and human sign-off. **Do not write a theorem into
   the paper at a status above SKETCHED without that sign-off.**

4. **Machine-check what is checkable.** Every ε in the paper comes from the accountant, not from
   hand algebra. Add property tests: ε monotone in T under sequential composition; ε constant in T
   under certified task-disjointness; the accountant refuses to certify when `touched` is empty.

**Gate P6.** FIG05 and TAB02 produced from real ledgers; theorem statuses updated; the claim tests in
`code/tests` still encode T1-fwd and H1 and still pass.

---

## P7 — The mechanism (claim C4)

The constructive half. Build **M9, DP-Contractive-FCL**: a fixed-capacity, contractive sketch of the
per-class second-moment statistics released through a continual-observation mechanism, so the
influence horizon of any datum is bounded and ε does not grow with T.

Requirements:
- Bounded influence horizon must be a *provable property of the operator*, checkable by
  `dp.accountant.check_disjointness` on the produced ledger. If the checker will not certify it,
  the mechanism is not what you claim it is — fix the mechanism, not the checker.
- Replace `tree_aggregation_factor_PLACEHOLDER` with a real banded matrix-factorisation mechanism
  (Choquette-Choo et al., ICML 2023) or state explicitly that we use binary-tree aggregation and pay
  the polylog factor.
- Sweep ε ∈ {0.5, 1, 2, 4, 8} × T ∈ {10, 50} × 3 seeds × all four datasets.
- Baselines at matched ε: DP-SGD federated linear probe, DP-FedAvg, and the non-private analytic
  head as the upper reference.
- **Measure the non-private gap first.** The Red Team's objection to H7 is that if the non-private
  analytic head is already far behind prompt-based SOTA, "best at equal ε" is a hollow win. Put the
  non-private gap in the table and address it in the text.

**Deliverables.** FIG08 (privacy–utility Pareto at T = 10 and T = 50), FIG09 (advantage widening with T
— the money plot), TAB08 (DP utility table), TAB09 (compute and communication cost).

**Gate P7.** Pareto frontier produced with CIs; the ledger from M9 is certified task-disjoint by the
checker; H7 and H8 status updated.

---

## P8 — Robustness, ablations, and the reviewer-facing results

Run these in parallel with P7 where the compute allows; they are cheap and they are what turns a
plausible paper into an accepted one.

- **FIG07 adversary-view ablation** (H3): TPR vs number of observed rounds for `view ∈ {final, task, full}`.
- **FIG11 secure-aggregation ablation** (H11): every attack with and without secure aggregation.
- **FIG12 onset inference** (H4): precision and MAE vs tolerance, plus one timeline visualisation.
- **FIG13 Gram-inversion n-curve** (H5): cosine similarity vs n per reference quality per dataset,
  with a decoded-image montage for the qualitative panel. CUB-200 is the best case here because
  per-class n is small; say so and show it.
- **FIG10 audit tightness**: analytical ε vs one-run empirical ε lower bound.
- **Backbone robustness**: repeat FIG01 on DINOv2 features. If the effect is backbone-dependent, that
  is a limitation to state, not to hide.
- **Non-IID sensitivity**: Dirichlet β ∈ {0.1, 0.5, 1.0, ∞} on the headline experiment.
- **Natural-federation check**: repeat the headline on Camelyon17 with its *real* hospital partition
  rather than a Dirichlet partition. A reviewer will ask whether the effect is an artifact of
  synthetic client splits. Having the answer is worth more than another dataset.
- **Log-log ROC curves** for every headline attack, appendix.
- **Seed-variance panel**: the per-seed spread of the headline number, so nobody has to wonder.

**Gate P8.** Every figure in `03_RESULTS_SPEC.md` marked *required* exists and regenerates.

---

## P9 — Paper assembly

1. `make figures` regenerates every figure and table from `results/` from a clean results/ directory, no manual
   steps, no stale artifacts. Delete `figs/` and `tables/` and rebuild to prove it.
2. All tables emitted as booktabs `.tex` from the CSVs; `\input{}` them, never paste.
3. `make verify` passes with an empty exception list, or a short list of `\providedvalue{}` entries
   each justified in a comment.
4. Reproducibility appendix: dataset manifest with checksums, exact package versions, the cluster
   discovery block, total compute in node-hours split GPU/CPU, and the command to regenerate each
   figure.
5. Ethics statement: this is offensive security work on published methods. State the responsible
   disclosure position, that no real patient data was re-identified, that Camelyon17 and FMoW are
   public research datasets used under their licenses, and that the attack code is released with
   the defensive mechanism.
6. `agents/OPEN_QUESTIONS.md` fully reconciled: every hypothesis is `CONFIRMED`, `REFUTED`, or
   explicitly carried as an open question in the paper's limitations section. No hypothesis is left
   `OPEN` without a sentence in the paper saying so.

**Gate P9.** `make setup && make figures && make paper` from an empty `figs/` and `tables/` produces the submitted PDF.

---

## Ordering summary

```
P0 ──► P1 ──► P2 ──► P3 ──► P4 ──► P5 ──┐
       │       │              │         ├──► P9
       └───────┴──► P6 ───────┴──► P7 ──┤
                                  P8 ───┘
```

P2 can be written against synthetic features while P1's extraction jobs run. P6 needs only ledgers,
so it starts as soon as P2 lands. P8 runs alongside P7.

## If you are short on time

Cut in this order, and say in the paper what was cut: FMoW and DomainNet (P1 stretch) → the 20-task
variants → the DINOv2 backbone repeat → M7 TPGP and M4's LoRA half (keep M6 C²Prompt — it is the current SOTA and the more-cited family) →
the T = 50 long-horizon run on ImageNet-R (keep it on CIFAR-100, FIG09 depends on it).

**Never cut:** the Camelyon17 natural-federation check, the secure-aggregation ablation, the
log-log ROCs, the seed variance, or `make verify`. Those four are what reviewers check.
