# Retention Is Leakage — code, results and figures

This repository contains the experimental system behind the submission
*"Retention Is Leakage: A Stability–Plasticity–Privacy Trilemma in Federated Continual Learning."*
It holds the full implementation (`code/`), every committed result table (`results/`), and the
scripts that turn those tables into every figure and table of the paper (`analysis/`, `figs/`,
`tables/`).

**Thesis.** In federated continual learning (FCL) the mechanism that prevents forgetting is the
mechanism that prevents privacy from decaying. Centralised work shows forgetting *passively*
amplifies privacy; once an adversary observes the whole broadcast transcript instead of a final
model, that result inverts.

| Claim | Statement | Main artifacts |
|---|---|---|
| **C1** | Leakage half-life differs by released-artifact family and decouples from accuracy decay | FIG01, FIG02, FIG16, FIG17, TAB03, TAB04 |
| **C2** | Within a method, retention strength causally drives leakage (dose–response) | FIG03, FIG04 |
| **C3** | Contractive retention admits a T-independent lifelong ε; accumulating retention does not | FIG05, TAB02, TAB10 |
| **C4** | A bounded-influence (contractive, DP) retention operator keeps ε flat in T | FIG08, FIG10, TAB08 |

Appendix-level artifacts: FIG11 (secure aggregation), FIG13 (Gram inversion), FIG17 (seed variance),
FIG18 (natural federation), FIG19 (retention vs. release), FIG A6 (property inference).

---

## 1. Quick start (CPU only, no data download, no GPU)

Everything a reviewer needs to **check the reported numbers** runs from the committed `results/`
tables. Requirements: Linux or macOS, Python ≥ 3.10, `make`, ~1 GB disk.

```bash
make setup      # creates envs/p3fcl, installs pinned requirements + the p3fcl package (editable)
make test       # unit + property + "claim" tests (318 tests; well under a minute)
make smoke      # fast end-to-end check on synthetic features (no downloads)
make figures    # regenerates every figure (figs/) and table (tables/) from results/*.csv
make verify     # provenance + consistency checks (see §6)
```

* `make figures` only *reads* `results/*.csv`; no plotting script recomputes an experiment.
* `make verify` fails loudly if a figure is older than the CSV it reads, if a headline number no
  longer traces to a results file, or if any result is missing provenance.
* On a machine with an environment-modules system the Makefiles run `module load python/3.10.4`
  and silently ignore its absence elsewhere. To use a different interpreter, create
  `envs/p3fcl` yourself (`python3.X -m venv envs/p3fcl && envs/p3fcl/bin/pip install -r env/requirements.txt && envs/p3fcl/bin/pip install -e code`).

## 2. Repository layout

```
code/
  src/p3fcl/            the package
    artifacts.py          ArtifactRecord / Ledger — the ONLY interface between methods and attacks/accounting
    streams.py sim.py     class-incremental stream construction, Dirichlet client partition, simulation loop
    methods/              M0 FedAvg, M1 GLFC, M2 Gaussian feature replay, M3 FOT, M4 prototypes,
                          M5 exemplar replay, M8 analytic FCL, M9 contractive DP-analytic (ours), M6 (prompt, GPU)
    attacks/              LiRA membership inference (lira.py), onset inference, property inference,
                          prototype diff, Gram inversion
    dp/ audit/            record-composition accountant; empirical ε lower bounds (Clopper–Pearson)
    halflife.py           non-parametric leakage/accuracy half-life, hierarchical bootstrap
    shadow_runner.py      shadow-federation generation and scoring (the LiRA workhorse)
    provenance.py         run manifests, source/config hashes, .meta.json sidecars, RUN_LOG.jsonl
  scripts/              CLI entry points: run_*.py (experiments), build_*/merge_*.py (summaries), check_*.py
  scripts/pbs/          PBS/Torque job templates (see §4)
  configs/              YAML configs (base, LiRA attack, DP)
  tests/                pytest suite
analysis/               one script per figure/table; reads results/*.csv, writes figs/ and tables/
results/                every result CSV, with a sibling *.meta.json (provenance) and RUN_LOG.jsonl
figs/  tables/          generated PDF/PNG and CSV/TeX (never edited by hand)
env/                    pinned requirements and bootstrap script
data/MANIFEST.json      dataset provenance/licences
```

## 3. What is (and is not) included

Included: all code, all result CSVs (`results/`, ~500 files), all generated figures and tables,
provenance sidecars, and the run log.

**Not** included (large and regenerable): raw datasets, cached ViT features (~3 GB), and the
per-shadow score files (`shadows*/…/*.npz`, several hundred GB). Consequently the *scoring* stages of
§4 need those files regenerated; the *reporting* path of §1 does not.

## 4. Reproducing the experiments from scratch (cluster)

The full campaign was run on a PBS/Torque cluster: a handful of GPU jobs (feature extraction only)
and large CPU job arrays (shadow federations). The templates in `code/scripts/pbs/` show the resource
requests used. They assume: submission from the repository root (they use `$PBS_O_WORKDIR`), a
`logs/` directory (`mkdir -p logs`), and site module names (`module load python/3.10.4 …`) that you
should adapt to your cluster. GPU is used **only** for frozen-backbone feature extraction (and the
optional prompt method M6); every shadow federation is linear algebra on cached features.

Stages, in order (all commands run from the repository root unless noted):

1. **Data.** `python -m p3fcl.cli data --fetch|--prepare|--audit --dataset {cifar100,cub200,imagenet_r,camelyon17}`
   (run from `code/`; templates `pbs/fetch_data.pbs`, `pbs/prepare_data.pbs`). Licences and sources: `data/MANIFEST.json`.
2. **Features (GPU).** `pbs/extract_features.pbs` — frozen ViT-B/16 (`vit_base_patch16_224.augreg_in21k`),
   splits `train,test,ref`, written to `features/`. Pinned GPU stack: `env/requirements-gpu.txt`.
3. **Real federations / utility.** `code/scripts/run_accuracy_matrix.py` (accuracy matrices),
   `run_utility_baseline.py` + `collect_tab05.py` (TAB05), `run_fx9_gate.py` (replay-weight gate).
4. **Shadow federations (CPU arrays).** `pbs/shadow_array.pbs`, `pbs/shadow_v3.pbs`
   (manifest-driven; manifests are produced by `build_wave_v2_manifest.py` / `build_wave_v3_manifest.py`
   into `build/waves/` — create it with `mkdir -p build/waves`). Underlying command:
   `python -m p3fcl.cli shadows --dataset D --method M --start S --count N`.
5. **Scoring and summaries.** Per (dataset, method, view): `run_lira_pertask.py D M SEED VIEW` (×3 seeds) →
   `build_fx2_summary.py` → `build_fx9_joint.py` (paired seed/task/target bootstrap); orchestrated per
   combination by `run_fx9_score_combo.py` (one PBS array index per combination). Shared tables are then
   rebuilt **only from per-combination files**: `merge_fx2_summary.py`, `build_retention_curves.py`,
   `merge_fx9_ratios.py`, `build_fx3_views_summary.py`, `build_fx9_roc.py`, `build_fig17.py`.
6. **Dose–response (C2).** `build_fx8_manifest.py` (writes `build/waves/fx9_dose_response.tsv`) →
   `pbs/fx9_dose_response.pbs` (shadow array) → `run_dose_response_accuracy.py`,
   `run_dose_response_lira.py` → `merge_fx9_dose.py`.
7. **DP method M9 (C3/C4).** `run_m9_hparam_selection.py` (on the public `ref` split) → `run_m9_sweep.py` →
   `merge_m9_sweep.py`; empirical audit `run_m9_audit.py`; accountant/certified ε: `run_fig05.py`,
   `complete_fx9_m9_certified.py`, `build_tab08.py`, `build_fig08_pareto.py`.
8. **Appendix experiments.** `run_gram_inversion.py`, `run_onset_inference.py`, `run_property_inference.py`,
   `run_fig18_report.py` (+ `pbs/camelyon17_fig18_v2_matched5.pbs`), `check_secure_agg_a1.py`,
   `run_buffer_coverage.py` (→ `results/buffer_coverage.csv`).
9. **Numbers and checks.** `build_paper_numbers.py` transcribes cells into `results/paper_numbers.csv`
   (no arithmetic; every row records its source CSV, row filter and column), then `make figures && make verify`.

Every experiment script writes its outputs together with a sidecar recording the source hash,
configuration hash, seed, PBS job id and wall time, and appends one line to `results/RUN_LOG.jsonl`.

## 5. Experimental protocol (summary)

* Backbone: frozen ViT-B/16 (IN-21k), **raw** cached features (no standardisation).
* Streams: 10 class-incremental tasks × 10 clients, Dirichlet β = 0.5; one FedAvg round per task,
  30 local epochs, lr 0.5 (replay-weight overrides come from the training-side gate `results/fx9_gate.csv`,
  never from test data). Accuracy is on the test split with predictions restricted to seen classes.
* Datasets: CIFAR-100, CUB-200, ImageNet-R (main); Camelyon17 (natural-federation appendix pilot).
* Membership inference: LiRA with **1,024 shadow federations per (dataset, method, seed)**, 3 seeds
  (5 for M0), attack hyper-parameters chosen on held-out shadows (calibration fraction 0.8), fixed-K task
  population K = {0,1,2,3}, elapsed horizon E = 6. Reported as **TPR@1%FPR and TPR@0.1%FPR** (AUC is never
  the only metric); log–log ROC in FIG16.
* Intervals: 95 %. Rates use Clopper–Pearson; leakage/accuracy curves use a 2,000-replicate hierarchical
  bootstrap (seeds → tasks → targets).
* Half-life: first elapsed value at which the normalised quantity drops to ≤ ½ of its e = 0 value,
  **right-censored at E = 6 and never extrapolated**; a ratio is reported only where both sides are defined
  (M0), with lower-bound ratios labelled as such.
* Accounting: record-composition accountant with σ = 2, δ = 1e-5, window W = 3; M9 uses its own
  analytic-Gaussian noise multiplier. Empirical ε lower bounds use one-sided Clopper–Pearson bounds
  (pointwise, not simultaneous).

## 6. Provenance and integrity checks

* Each `results/*.csv`, `figs/*`, `tables/*` has a `*.meta.json` sidecar (source hash over `code/src` +
  `code/configs`, config hash, seed, job id, wall time). Account names and absolute paths in sidecars were
  replaced by placeholders (`anonymous`, `<repo>`) for double-blind review.
* `make verify` runs `code/scripts/check_fx9_consistency.py` — accuracy diagonals vs. curve files; every
  shared summary equals the union of its per-combination files; every `paper_numbers.csv` row equals its
  source cell; no figure is older than its CSV; width/font limits — and `analysis/verify_provenance.py`
  (no post-phase result without a job id; numeric literals in any `.tex` trace to a results file).
* `results/RUN_LOG.jsonl` also contains a few entries produced by *running the test suite* on the
  cluster (their output paths are under `/tmp/pytest-…`, `pbs_jobid: null`); none of them corresponds to a
  file used in a figure or table.

## 7. Scope and known limitations

* Methods are **feature-space reimplementations** on a frozen backbone, not reproductions of the original
  trainable-backbone systems; M2 is a diagonal-Gaussian feature-replay approximation; M4 is the prototype half
  only; M6/M7 (prompt/LoRA) are not part of the reported evidence. Gaps to published numbers are reported in
  `results/tab07_reproduction_gap.csv`.
* The balanced-replay gate is **not** met for 7 of the 9 (dataset × {M1, M2, M5}) configurations
  (`gate_pass` in `results/fx9_gate_summary.csv`); those runs are kept and labelled, not tuned away.
* Not delivered: DP-FedAvg / DP linear-probe baselines (no equal-ε superiority claim), the 50-task
  utility run, a hospital-as-client Camelyon17 design, and a full-trajectory empirical M9 ε certificate
  (the audit is a one-release pointwise bound).
* Retention ratios are reported only for M0; other methods' accuracy half-lives are censored within E = 6.
* Comments inside some source files refer to internal planning notes and rule numbers that are not part of
  this release.

## 8. Results index

| Artifact | Script | Data |
|---|---|---|
| FIG01 retention/leakage curves | `analysis/fig01_decoupling.py` | `results/retention_curves.csv` |
| FIG02 retention at horizon | `analysis/fig02_halflife.py` | `results/retention_curves.csv`, `results/fig02_retention_at_horizon.csv` |
| FIG03 / FIG04 dose–response | `analysis/fig03_*.py`, `fig04_*.py` | `results/fx9_dose_summary.csv` |
| FIG05 ε(T) | `analysis/fig05_eps_of_T.py` | `results/fig05_eps_of_T.csv` |
| FIG08 privacy–utility | `analysis/fig08_pareto.py` | `results/fig08_pareto.csv` |
| FIG10 empirical audit | `analysis/fig10_m9_audit.py` | `results/fig10_m9_audit.csv` |
| FIG11 secure aggregation | `analysis/fig11_secure_agg.py` | `results/fx3_views_summary.csv` |
| FIG13 Gram inversion | `analysis/fig13_gram_inversion.py` | `results/fig13_gram_inversion_summary.csv` |
| FIG16 ROC, FIG17 seeds | `analysis/fig16_roc.py`, `fig17_seed_variance.py` | `results/fig16_roc.csv`, `fig17_seed_variance.csv` |
| FIG18 natural federation | `analysis/fig18_natural_federation.py` | `results/fig18_natural_federation.csv` |
| FIG19 retention vs. release | `analysis/fig19_retention_vs_release.py` | `results/retention_curves.csv` |
| FIG A6 property inference | `analysis/fig_a6_property_inference.py` | `results/a6_property_inference.csv` |
| TAB02 / TAB10 | `analysis/tab02_units.py`, `tab10_taxonomy.py` | definitional |
| TAB03 / TAB04 | `analysis/tab03_leakage.py`, `tab04_halflife.py` | `results/fig01_decoupling.csv`, `results/fig02_halflife.csv` |
| TAB05 utility | `code/scripts/collect_tab05.py` | `results/tab05_utility_baselines.csv`, `fx9_utility_summary.csv` |
| TAB07 reproduction gap | `code/scripts/build_tab07.py` | `results/tab07_reproduction_gap.csv` |
| TAB08 DP utility + ε accounting | `code/scripts/build_tab08.py` | `results/m9_sweep.csv`, `m9_certified.csv` |
| Buffer coverage (M1/M5) | `code/scripts/run_buffer_coverage.py` | `results/buffer_coverage.csv` |
| Every quoted number | `code/scripts/build_paper_numbers.py` | `results/paper_numbers.csv` |
