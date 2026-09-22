# 06 — PACKAGE SPEC

**Build `code/` from scratch to this spec.** There is no prior code to reuse. This file is the
source of truth for the abstractions; `00_BUILD_PLAN.md` says when to build them,
`04_METHODS_AND_ATTACKS.md` says what goes in `methods/` and `attacks/`.

```
code/
  pyproject.toml            package `p3fcl`, src layout, py>=3.10
  Makefile                  setup test smoke figures verify paper
  configs/                  base.yaml, attack_lira.yaml, dp.yaml, per-dataset overrides
  scripts/                  CLI entry points + PBS job scripts (scripts/pbs/*.pbs)
  tests/                    unit tests + the four CLAIM tests (§9)
  src/p3fcl/
    artifacts.py    ledger      ← build first, everything depends on it
    units.py        Unit U1-U5 + the unit-of-privacy predicate helpers
    streams.py      task streams + client partitions
    features.py     frozen-backbone feature cache
    sim.py          the FCL simulation loop
    metrics.py      accuracy, BWT, AUC, TPR@FPR, Clopper-Pearson
    provenance.py   run manifests, RUN_LOG.jsonl, .meta.json
    config.py       YAML load + deep-merge + schema validation + hashing
    rng.py          call-site-seeded generators
    cli.py          one entry point: `python -m p3fcl.cli <subcommand>`
    methods/        base.py + m0_fedavg.py … m9_contractive.py
    attacks/        base.py + a1_lira.py … a6_property.py
    dp/             accountant.py, mechanisms.py
    audit/          one_run.py
    plotting.py     shared style: family colours, CI rendering, fonts
```

Core dependency is **numpy only**. torch/timm are needed for `features.extract` and the two GPU
methods and nowhere else — keep that boundary sharp so the whole audit runs on a CPU node.

---

## 1. `artifacts.py` — the ledger. Build this first.

Every method declares what it **releases**. Attacks and the accountant consume the ledger and
nothing else. Nine methods with nine checkpoint formats is how this project dies; one ledger is how
it ships.

```python
class Family(str, Enum):        # artifact families, RESEARCH_PLAN.md §2.2
    MODEL_DELTA = "F1"   # gradients / weight deltas
    PROTOTYPE   = "F2"   # class means, covariances
    PROMPT      = "F3"   # prompt pools and keys
    LOW_RANK    = "F4"   # LoRA A/B factors
    GRAM        = "F5"   # X^T X + lambda I, X^T Y — exact sufficient statistic
    GENERATIVE  = "F6"   # replay generator parameters
    COUNTS      = "F7"   # per-class sample counts
    EXEMPLAR    = "F8"   # literal raw samples

@dataclass
class ArtifactRecord:
    round: int
    task: int
    client: int
    family: Family
    payload: np.ndarray | dict[str, np.ndarray]
    touched: frozenset[int]        # datum ids whose VALUES influenced this release
    n_touched: int = 0
    passes_over_data: int = 1      # >1 ⇒ sequential composition inside the task
    meta: dict = field(default_factory=dict)
    def nbytes(self) -> int: ...
```

`touched` is the privacy-critical field and the easiest one to get quietly wrong. **A datum touched
only through a replay buffer, a distillation term or a regulariser still counts.** Over-report rather
than under-report. If `touched` is empty the disjointness checker must refuse to certify anything
rather than certify vacuously.

```python
class Ledger:                      # an append-only transcript = the release channel M
    def add(rec) / extend(recs) / __len__ / __iter__
    def view(kind: str) -> list[ArtifactRecord]
        # "full"  — every record (honest-but-curious server / persistent participant)
        # "task"  — last round of each task (checkpoint-publishing adversary)
        # "final" — last round only (model-release adversary)
    def by_family(fam) / by_client(c) / tasks() / summary()
    def aggregate_view() -> list[ArtifactRecord]
        # per-round SUM over clients — the secure-aggregation view. Attacks that still work
        # on this are the ones that survive secure agg (H11, FIG11). Required, not optional.
```

Serialisation: `save_npz(path)` / `load_npz(path)` that round-trips exactly, so a ledger produced on
a compute node can be analysed later. Test the round-trip.

## 2. `units.py` — units of privacy

`Unit` enum `U1 EXAMPLE, U2 TASK (client-task), U3 CLIENT_BOUNDED, U4 CLIENT_LIFELONG, U5 INDIVIDUAL`.
Every DP claim in this project must name its unit; a function that computes an ε without a `Unit`
argument is a bug. Provide `neighbouring(unit)` returning a short prose description used in captions
and in TAB02, so the paper's definitions and the code's definitions cannot drift apart.

## 3. `streams.py`

- `class_incremental_tasks(labels, n_tasks, seed)` — shuffle the class list with `seed`, split into
  `n_tasks` disjoint groups.
- `dirichlet_partition(labels, idx, n_clients, beta, seed)` — standard non-IID partition.
- `natural_partition(domain_field, idx)` — client = real domain (Camelyon17 hospitals). **Required**,
  it produces FIG18.
- `build_stream(...) -> list[list[Shard]]` where `Shard(client, task, idx, ids)` and `ids` are
  **stable global datum ids** — the ledger's `touched` refers to these, so they must survive
  subsetting and shuffling unchanged.
- `domain_incremental_tasks(...)` for Camelyon17/DomainNet.

## 4. `features.py`

- `cache_key(dataset, backbone, split)` — sha1 of the triple, stable across runs.
- `save_cache / load_cache` — `.npz` + a sidecar `.json` with n, d, dataset, backbone, split.
- `extract(dataset, backbone, split, batch_size, device)` — timm,
  `pretrained=True, num_classes=0`, `torch.no_grad()`, AMP. **Store the pooled feature and the
  pre-norm CLS token, unnormalised.** L2 normalisation is a *method* decision (it sets the DP
  sensitivity) and belongs in the method, not the extractor. A2 needs the exact vector the method
  consumes.
- Splits `train / test / ref / canary`. **`ref` must be disjoint from `train`** — if the adversary's
  reference contains target records, every attack number in the paper is inflated. Assert it, in a
  claim test.

## 5. `sim.py` — the FCL loop

```python
def run(method, X, y, stream_cfg, seed) -> dict:
    # returns acc_matrix[T,T] (acc on task k after finishing task t), final_avg_acc,
    # bwt, avg_incremental_acc, ledger
```
Honours `participation` (partial client participation per round) and `method.rounds_per_task()`.
Evaluates on every task seen so far after each task. The accuracy matrix and the ledger are the two
outputs everything downstream consumes — nothing else reads the method's internals.

## 6. `metrics.py`

`accuracy`, `backward_transfer` (mean over k<T of `A[T-1,k] - A[k,k]`; negative = forgetting),
`average_incremental_accuracy`, `roc_auc` (rank-based, tie-corrected), `tpr_at_fpr(scores, labels,
target_fpr)`, `membership_report` (returns AUC **and** TPR@1%FPR **and** TPR@0.1%FPR with n_pos/n_neg),
`clopper_pearson(k, n, alpha)` exact binomial CI, `bootstrap_ci(fn, data, n=2000)`.

**`membership_report` must never return AUC alone** — reviewers at this venue treat a bare AUC as
hiding the low-FPR regime, which is the regime that matters.

## 7. `dp/`

`mechanisms.py`
- `gram_sensitivity(clip_norm)` → `clip_norm**2`, exact for a clipped second-moment statistic.
- `analytic_gaussian_sigma(eps, delta, sensitivity)` — **Balle & Wang (ICML 2018)**, valid for all ε.
  Implement this one, not the classical `sqrt(2 ln(1.25/δ))/ε` form, which is only valid for ε ≤ 1
  and will silently understate noise above it.
- `banded_mf_factor(T, bands)` — correlated-noise factor for the continual-observation mechanism.
  If you ship binary-tree aggregation instead, name the function `tree_aggregation_factor` and state
  the polylog(T) cost in the paper. **Do not ship a placeholder that a reported number depends on.**
- `sym_gaussian_noise(d, sigma, rng)` — symmetric noise for a symmetric statistic, so sensitivity is
  not double-counted.

`accountant.py`
- `rdp_gaussian(alpha, sigma)`, `rdp_to_dp(rdp, alpha, delta)`, `eps_gaussian_composed(sigma, k, delta)`
  minimised over a grid of α.
- `check_disjointness(ledger) -> DisjointnessReport` implementing violations **V1–V5**:
  V0 no `touched` ids recorded ⇒ unverifiable, refuse to certify;
  V1 a round mixes more than one task index (clients out of phase);
  V2/V3 a datum influences more than one task-epoch (replay, regulariser, persistent state);
  V5 a datum is touched in more than one round within a task; V5b `passes_over_data > 1`.
- `account(ledger, sigma, unit, delta) -> LifelongResult` with `eps_of_T: Callable[[int], float]`,
  a `regime` string, and `lifelong: bool`. Routing: task-disjoint + U2 ⇒ parallel composition,
  ε constant in T (**T1-fwd**); U4 ⇒ sequential, ε grows without bound (**H1**); otherwise sequential
  RDP at `max_task_touches × rounds_per_task` per T.
- `filter_exhaustion(eps_budget, sigma, delta)` — how many task-epochs until a per-client privacy
  filter is spent (feeds H9 / FIG21).

## 8. `attacks/base.py`

```python
@dataclass(frozen=True)
class ThreatModel:
    who: str              # hbc_server | participating_client | external
    active: bool          # False = passive. Passive is our differentiator vs PromptMIA — keep it.
    view: str             # full | task | final
    auxiliary: str        # none | public_data | shadow_training
    target: str           # membership|property|reconstruction|onset|attribution
    survives_secure_agg: bool
```
An attack whose threat-model cell is unspecified **must refuse to run**. Keep that guard.

## 9. The four CLAIM tests — `tests/test_claims.py`

These are not unit tests. Each encodes a load-bearing argument, and a failure means an argument in
the paper has broken, not that a helper regressed. Label them as such in the docstrings.

1. **T1-fwd** — a synthetic task-disjoint ledger accounts to an ε that is *constant* in T under U2,
   for T ∈ {1, 10, 100, 1000}.
2. **H1** — the same ledger under U4 gives ε that is strictly increasing and unbounded in T.
3. **Task-disjointness of the analytic head** — M8's single-pass ledger is certified
   `task_disjoint=True`; a variant with a replay buffer is certified `False` with a V2/V3 violation.
4. **H5 exactness at n = 1** — Gram inversion recovers a single feature vector to |cos| = 1 up to
   sign, exactly, as the linear algebra requires.

Plus: `ref ∩ train = ∅` for every dataset; ledger `.npz` round-trip; shadow determinism (the same
shadow index reproduces its output byte-identically).

## 10. `provenance.py`, `config.py`, `rng.py`

- `source_hash()` → SHA-256 over the sorted contents of `code/src/**/*.py` and `code/configs/**`.
  **This project does not use git**, so this is the identity of the code that produced a result.
  Cache it per process; it is cheap but not free.
- `run_manifest(config)` → source hash, config SHA-256, seed, hostname, `$PBS_JOBID`,
  `$PBS_ARRAY_INDEX`, package versions, UTC start.
- `finalize(manifest, outputs)` → writes `<output>.meta.json` beside each output, appends one line to
  `results/RUN_LOG.jsonl`. Two results with different source hashes came from different code —
  that is the invariant `make verify` leans on.
- `@logged_run` decorator on every CLI subcommand.
- `config.load(path, overrides)` → frozen dict + hash; every config that produced a result is copied
  to `results/configs/<hash>.yaml`.
- `rng.seeded(name, seed)` hashes the call-site name into the seed, so adding a new random draw
  somewhere never shifts every downstream stream. **No bare `np.random.*` anywhere in the package** —
  add a test that greps for it.

## 11. `plotting.py`

One module owning the visual system, imported by every `analysis/figNN_*.py`:
- a fixed **artifact-family → colour + marker** mapping, identical in every figure in the paper;
- colour-blind-safe, and legible in greyscale because marker shape also varies;
- `ci_band(ax, x, lo, hi)` and `ci_point(ax, ...)` so intervals are drawn the same way everywhere;
- `save(fig, stem)` writing `figs/<stem>.pdf` and `figs/<stem>.png` at 200 dpi;
- a `column_width()` helper so fonts are checked at final printed width, not zoomed.

Plot scripts **read a CSV and nothing else**. No computation at plot time — that is what makes
`make figures` a regeneration rather than a re-run.

## 12. `cli.py`

One entry point, subcommands: `data --fetch|--prepare|--audit`, `extract`, `run`, `shadows`,
`attack`, `account`, `audit`, `sweep`, `collect`. Every subcommand is `@logged_run`, takes
`--config` plus `--set key.path=value` overrides, and writes tidy CSVs to `results/`.
