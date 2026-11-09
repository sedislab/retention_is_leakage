# Code-reading answers for the authors (not rebuttal prose)

## Q. How the five targets are chosen (mjT5 Q2)

In every (client, task) shard of the base stream (10 tasks × 10 clients, Dirichlet β = 0.5, built
with `stream.base_seed` = the run's seed), 5 targets are drawn uniformly at random without
replacement by a permutation seeded by (client, task, base_seed), which gives a pool of 500 targets
that is fixed across all 1,024 shadows of that seed. In every shadow, the membership of every
training record, targets included, is redrawn independently as IN with probability 0.5 from a single
generator seeded by the shadow index, so a target is treated like every other record and differs
only in that its score is recorded.

Sources:
- Target draw: `code/src/p3fcl/shadow_runner.build_targets`, lines 132–151 (per-shard
  `rng.seeded("shadow_runner.build_targets::{client}::{task}", base_seed).permutation(...)[:5]`).
- Pool fixed per seed: `run_shadow_range` computes it once from the base stream and config, lines
  591–599. It is a pure function of the config. A copy is written to `targets.json`.
- Membership: `_population_mask`, lines 154–160 (`r.random(n_total) < p_in`, `p_in = 0.5`), applied
  to every id in `_apply_population_mask`, lines 163–172, per shadow in `_run_one_shadow`, line 456.
- Config: `code/configs/attack_lira.yaml` (`targets.per_shard: 5`, `targets.p_in: 0.5`,
  `stream: n_tasks 10, n_clients 10, beta 0.5`).
- Checked on disk: every `shadows_v3/<ds>/m2_target/seed{0,1,2}/targets.json` and
  `shadows_v2/<ds>/m3_fot/seed{0,1,2}/targets.json` has 500 targets in 100 shards of exactly 5.

Detail the authors should know before writing the answer (found while reading, 2026-11-06): the
mask's seed string is `"shadow_runner.population_mask::{dataset}::{method}"` with `shadow_id`. It
does **not** include the run seed, and every seed's shadow store uses shadow ids 0–1023
(`build/waves/v3_main.tsv`). So shadow i has the **same IN/OUT vector over record ids** in seeds 0,
1 and 2. The streams, client partitions and target pools differ across seeds, so the seeds are not
duplicates. But the membership draws are shared, so the three seeds are not fully independent
replicates of the membership randomness. "Resampled independently" is true across records within a
shadow and across shadows. It is not true across seeds for the same shadow index.

## R3 preamble. The paper's n_clients arm reused a clip norm calibrated for other client sizes

The paper's n_clients arm (`code/scripts/run_m9_sweep.py`, lines 148–157) runs U2 at n ∈ {50, 100},
but takes `clip_C` from `hparams[("U2", eps)]`. That C was set once per p as the 95th percentile of
pseudo-client pair norms, using the fixed 10-task × 10-client split of `ref-train`
(`code/scripts/run_m9_hparam_selection.py`, lines 75–88 and 120–121; `N_CLIENTS = 10` at line 47).
Each client's pair norm grows with its number of examples, while the noise scales with C
(`code/src/p3fcl/methods/m9_contractive.py`: clip at line 130, sensitivity = C at line 116). So at
n = 50 or 100 the clients were much smaller than the size C was calibrated for. The noise did not
shrink, and the arm could not show the expected gain.

A related point found while reading, to be quantified in R3: the pseudo-clients are built from
`ref-train`, which is half of `ref` (`_ref_train_val_split`, `frac = 0.5`, lines 53–63). So even at
n = 10 they are much smaller than the real clients. The mean pseudo-client sizes are about 50 / 4 / 15
examples (CIFAR-100 / CUB-200 / ImageNet-R: 5,000 / 400 / 1,544 ref-train examples over 10 × 10; corrected 2026-11-09,
CUB ref-train is 400, not 500).
The mean real client sizes are about 400 / 50 / 239 (40,000 / 4,994 / 23,918 train examples over
10 × 10). That is a factor of about 8–16. So the paper's U2 runs at n = 10 probably clip most real
clients well below their norm. R3's size-matched C(n) also covers n = 10, so its n = 10 row tests
this directly.

## R5. Deviation from the plan: DP-FedAvg aggregation (decided by the user, 2026-11-06)

The plan (§9) specified DP-FedAvg as "divide by the number of participating clients". With no noise
and no clipping that uniform mean missed the M0 correctness gate (0.01) in 3 of 9 cells:
CIFAR-100 seed 1 (gap 0.0265), CUB-200 seed 1 (0.0505) and CUB-200 seed 2 (0.0109). The reason is
that M0 weights clients by size, and Dirichlet(0.5) client sizes vary a lot. DP-SGD and an exact
size-weighted mode both matched M0 exactly. Details are in
`notes/2026-11-06_rb_r5_uniform_mean_gate.md`. The uniform mean is not reported as a baseline.
R5 instead uses McMahan et al. (ICLR 2018)'s fixed-denominator weighted mean. The cap ŵ, the gate
rerun and the procedure are in `rebuttal/R5_PROTOCOL.md`, amendment 1.

**Privacy assumption to state in the paper.** Each client's update is clipped to norm S and given
weight w_c = min(n_c / ŵ, 1) ≤ 1, where the cap ŵ is fixed from public data. The server divides the
noisy weighted sum by the fixed denominator W = Σ_c w_c of the base stream. Like McMahan et al., we
treat W as a public constant, not a quantity that changes between neighbouring datasets. Under that
assumption, adding or removing one client's task data changes the weighted sum by at most S (each
w_c · clip_S(Δ_c) has norm ≤ S). So the Gaussian noise is calibrated to sensitivity S with the
analytic Gaussian mechanism (σ = σ_AG(ε0, 10⁻⁵, S)), and each round is (ε0, 10⁻⁵)-DP at the
client-task level (U2). Each shard is read in one round, so the lifelong U2 guarantee is the same
(ε0, 10⁻⁵).
