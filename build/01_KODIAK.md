# 01 — KODIAK (Baylor HPC)

Experiments run on **Kodiak**, Baylor's cluster (`kodiak.baylor.edu`). This file is both an
instruction and a **living record**: §2 is blank until you fill it in with real discovered values.

## 0. Account and authoritative documentation

| | |
|---|---|
| Host | `kodiak.baylor.edu` |
| Username | `islamm` (Mohaimanul Islam, ECE; advisor Dr. Ernest Bonnah) |
| Login | `ssh islamm@kodiak.baylor.edu` — **change the initial password with `passwd` on first login** |
| Support | Research_Technology@baylor.edu · helpdesk@baylor.edu · HPRCS@baylor.edu |

**Credentials are never stored in this repository.** Not in a config, not in a job script, not in a
comment, not in `.env`. Use SSH keys after the first login (`ssh-keygen -t ed25519` locally,
`ssh-copy-id islamm@kodiak.baylor.edu`). If a credential ever lands in a commit, rotate it and
rewrite the history — do not just delete it in a later commit.

### Read these before the discovery protocol

Both are Baylor-authenticated and could not be fetched off-campus; open them from the user's machine
and fold what they say into §2, overriding anything below that they contradict.

- **Kodiak usage information** — <https://rtservices.baylor.edu/hpc/support.html>
- **Full HPC documentation (PDF)** — <https://baylor.app.box.com/v/kodiak-documentation>

These are authoritative. The table in §0.1 is assembled from public marketing pages and is a
starting point only.

### Data classification — a hard site rule

> Data classified **"public" or "non-public"** may be stored on Kodiak.
> Data classified **"Protected" or higher is not permitted.**
> <https://its.web.baylor.edu/security/data-classification-standards>

This is satisfied by design — CIFAR-100, ImageNet-R, CUB-200-2011, Camelyon17-WILDS and FMoW-WILDS
are all public research datasets released for open use, and the project never handles real patient
records. Two consequences that are not optional:

1. **Never stage anything Protected onto Kodiak** — no clinical data, no PHI, no identifiable
   records, at any point, including "just to test the pipeline". If a collaborator later offers real
   hospital data for the FCL story, it does not go on Kodiak; ask Research_Technology@baylor.edu first.
2. **Say this in the paper's ethics statement.** "Experiments were run on a cluster whose data policy
   forbids protected data; all datasets are public research releases used under their licenses, and
   no re-identification of any individual was attempted" is a true, specific sentence that costs
   nothing and answers the reviewer question before it is asked.

---

## 0.1 What is known from public documentation

| | |
|---|---|
| Login | `kodiak.baylor.edu` (login001), SSH, Baylor credentials |
| OS | Rocky Linux |
| Scheduler | **PBS / Torque** — `qsub`, `qstat`, `qdel`, `#PBS` directives. **Not Slurm.** |
| Default queue | `batch` |
| CPU nodes | dual-socket 18-core (36 cores/node) |
| GPU nodes | 2 × dual NVIDIA **P100 16 GB**; 3 × dual NVIDIA **V100 32 GB** — roughly 10 GPUs total |
| Interconnect | Mellanox HDR 200 Gb/s (100 Gb/s to compute) |
| Storage | PixStor, 2.5 PB multi-tier (200 TB NVMe tier-0, 2.3 PB HDD tier-1) |
| Paths | `/home/$USER` (small — code only) · `/data/$USER` (large — everything else). **Site rule: store files under `/data`, not `/home`.** |
| Modules | `module avail | load | list | purge`; `module load use.own` for `~/privatemodules` |
| Support | HPRCS@baylor.edu |

Public docs are thin on GPU request syntax and queue limits, so §1 is mandatory before any real job.

> **The existing `code/scripts/delta_extract.slurm` and `delta_shadow_array.slurm` were written for
> NCSA Delta and are wrong for this cluster.** Delete them; replace with the PBS templates below.
> Keep the file names descriptive: `scripts/pbs/extract_features.pbs`, `scripts/pbs/shadow_array.pbs`.

---

## 1. Discovery protocol — run this first, record the answers in §2

**Step 0: read the two authoritative documents linked in §0.** They may answer most of this
directly, in which case quote them into §2 and use the commands below only to confirm.

Do not guess any of this. Run it on the login node and paste real output.

```bash
# scheduler and version
which qsub sbatch 2>&1; qstat --version 2>&1 | head -3

# queues, their limits, and who can use them
qstat -Q                        # queue list
qmgr -c "print server" 2>&1 | head -60
qmgr -c "print queue @default" 2>&1 | head -40

# nodes: how many, how many cores, which have GPUs and how they are advertised
pbsnodes -a 2>&1 | head -120
pbsnodes -a 2>&1 | grep -iE "gpu|ngpus|properties|np =" | sort | uniq -c

# what GPU request syntax does this PBS accept?  (Torque vs PBS Pro differ)
#   Torque:  -l nodes=1:ppn=8:gpus=1
#   PBS Pro: -l select=1:ncpus=8:ngpus=1
# Submit a 2-minute probe job with each form; whichever is accepted is the answer.

# software
module avail 2>&1 | tr ' ' '\n' | grep -iE "cuda|python|anaconda|gcc|cudnn" | sort -u
module load cuda 2>&1; nvcc --version 2>&1 | tail -2

# storage: quotas and what is actually writable
df -h /home/$USER /data/$USER 2>&1
quota -s 2>&1 || lfs quota -u $USER /data 2>&1 || echo "no quota tool"
ls -ld /scratch /tmp /dev/shm 2>&1        # is there node-local scratch?
```

Then submit one probe job and read its output:

```bash
cat > probe.pbs <<'EOS'
#!/bin/bash
#PBS -N p3probe
#PBS -l nodes=1:ppn=4:gpus=1
#PBS -l walltime=00:05:00
#PBS -j oe
cd $PBS_O_WORKDIR
hostname; echo "---"; nproc; free -g | head -2
echo "---"; nvidia-smi || echo "NO GPU VISIBLE"
echo "---"; echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "---"; cat $PBS_NODEFILE
EOS
qsub probe.pbs && qstat -u $USER
```

**If a GPU is not visible with `-l nodes=1:ppn=4:gpus=1`,** try the PBS Pro `select` form, then a
GPU queue name discovered from `qstat -Q`, then email HPRCS@baylor.edu. Record which worked.

---

## 2. DISCOVERED — verified on Kodiak, 2026-09-14, probe job confirmed 2026-09-15

**This section is now authoritative. Where §0.1 (public marketing pages) disagrees with this, this wins.**

```
scheduler flavour     : OpenPBS 23.06.06  (PBS Pro lineage — NOT Torque, NOT Slurm)
resource syntax       : -l select=<N>:ncpus=<c>[:ngpus=<g>][:mem=<m>gb]
                        nodes=/ppn= are NOT defined on this server. Never use them.
qsub GPU syntax       : -q gpu -l select=1:ncpus=8:ngpus=1  — CONFIRMED. Probe job 133583.bcm11
                        landed on gpu001 with CUDA_VISIBLE_DEVICES set to exactly one GPU UUID.
queues                : batch (default, walltime max 372:00:00) | gpu (168:00:00)
                        interactive (240:00:00)
                        condo queues — DO NOT TARGET: swint rose cray hep poderoso gallagher
                        kevlishvili zeke maggard craven capybara instructional ig
per-user cap          : batch max_run_res.ncpus = 512 concurrent cores  ← THE binding constraint
                        batch queued_jobs_threshold = 3000
job arrays            : -J 0-159%20  — the %N throttle DOES work (attr max_run_subjobs).
                        server max_array_size = 10000.  Index var: $PBS_ARRAY_INDEX
GPU pool (usable)     : gpu001–gpu005, 2 GPUs each = 10 GPUs, 36 cores/node, queue `gpu`
GPU pool (physical)   : 14 nodes / 39 GPUs — the other 29 sit behind condo queues
GPU model             : CONFIRMED MIXED POOL, matching the §0.1 public-docs claim exactly: gpu001
                        observed as Tesla P100-PCIE-16GB (Pascal, sm_60); a later job on the same
                        `gpu` queue landed on a Tesla V100-PCIE-32GB (Volta, sm_70). Do not assume
                        uniform hardware across gpu001-gpu005 — jobs can land on either generation.
GPU driver            : 580.178.04 (P100 host), reporting max supported CUDA runtime 13.0
                        (nvidia-smi header). cuda/13.1 and cuda/13.2 modules compile (nvcc) but
                        likely EXCEED what this driver supports at run time — do not use them for
                        real GPU work. cuda/12.9 (already what extract_features.pbs loads) and
                        cuda/13.0 are both <= the driver's ceiling and are the safe choices.
torch compatibility    : CONFIRMED — `torch==2.5.1+cu121` (env/requirements-gpu.txt) has
                        `torch.cuda.get_arch_list() == ['sm_50','sm_60','sm_70','sm_75','sm_80',
                        'sm_86','sm_90']`, i.e. it was built with BOTH Pascal (sm_60, the P100) and
                        Volta (sm_70, the V100) kernels included. A real matmul executed successfully
                        on a V100 in `code/scripts/pbs/torch_gpu_probe.pbs`'s run. P100 execution was
                        not separately observed but is covered by the same compiled arch list.
CPU nodes             : 121 vnodes, 1/2/32/36/48/64/96/128/192 cores. batch nodes are mostly 48–64c.
node-local scratch    : CONFIRMED on GPU nodes — /local (empty, world-writable+sticky) and /tmp,
                        both on local disk /dev/sda3 (208G total, 159G free at probe time on
                        gpu001). Not confirmed yet on plain `batch`-queue CPU nodes; assume similar
                        until checked, it costs nothing to fall back to /data if a job finds none.
/data                 : GPFS. 2.1 PB total, 88% USED, ~270 TB free. Per-user quota STILL UNKNOWN —
                        quota/lfs/mmlsquota all unavailable on login. ASK HPRCS before budgeting
                        the shadow store. Current usage /data/islamm: ~6 MB. Not blocking until P3.
/home                 : 70 TB total, 23% used.
CUDA modules          : cuda/{11, 11.8, 12, 12.9, 13, 13.0, 13.1, 13.2}; nvcc 12.9.86 (May 2025).
                        Use 12.9 or 13.0 — see "GPU driver" above.
Python modules        : python/{3.7.2, 3.8.18, 3.10.4, 3.12.8, 3.13.0, 3.14.6}. Use 3.10.4 — CONFIRMED
                        present and working on compute nodes too (probe job: `Python 3.10.4`).
conda                 : NO conda/anaconda/miniconda module exists. Use `python -m venv` (done, see
                        env/bootstrap_kodiak.sh — venv already built at envs/p3fcl/).
gcc                   : 11.2 / 11.3 / 11.5 / 14.2 / 15.2
internet from compute : CONFIRMED YES. Probe job's `curl -sI https://pypi.org` and
                        `curl -sI https://data.caltech.edu` both returned HTTP/2 200 from gpu001.
                        Dataset fetching for P1 can run as a normal batch job — no login-node
                        trickle needed. (CUB-200's mirror, data.caltech.edu, was checked
                        specifically since 02_DATASETS.md flags its old URL as dead.)
```

### Formerly open, now resolved by probe job 133583.bcm11 (ran 2026-09-15 on gpu001)

1. ~~Does `-q gpu -l select=1:ncpus=8:ngpus=1` actually grant a GPU?~~ **Yes.**
2. ~~What GPU is it?~~ **Tesla P100-PCIE-16GB**, driver 580.178.04, driver's CUDA ceiling 13.0.
3. ~~Can a compute node reach the internet?~~ **Yes** — P1 dataset fetching is a plain batch job.
4. ~~Is there compute-node local scratch?~~ **Yes**, `/local` and `/tmp` on GPU nodes (159G free
   observed). CPU (`batch`-queue) nodes not yet individually confirmed.
5. **What is my `/data` quota?** Still open — needs HPRCS, not a probe job. Not blocking until P3.

## 3. Storage and network layout

```
/data/islamm/retention_leakage/         git checkout. NOT /home — the site rule is "store files under
                                 /data", and $HOME quota is small. Claude Code runs from here.
/data/islamm/retention_leakage/
    raw/                         downloaded archives (tar/zip), kept for checksum re-verification
    datasets/                    extracted images
    features/                    .npz feature caches  ← the project's real working set
    shadows/                     shadow-federation statistics (thousands of small .npz)
    runs/                        per-run outputs before promotion to results/
    envs/p3fcl/                  venv (NOT in $HOME). There is NO conda module on Kodiak —
                                 `module load python/3.10.4 && python -m venv`.
    logs/                        PBS .o/.e files
```

**Downloads, and the login-node rule.** Kodiak's stated policy is that the login node is for
*software development, editing files, and submitting jobs* — **not** production work. A multi-hour
parallel download is production work. So:
- first test whether compute nodes have outbound internet (§1) and record the answer. If they do,
  fetch from a short batch job and the question is closed;
- if they do not, download from the login node but keep it small and polite: one stream,
  `wget -c --limit-rate=10m`, under `tmux`, into `/data/$USER/retention_leakage/raw/`, and never several at once.
  The largest required archive is Camelyon17 at ~10 GB, which is one overnight trickle, not a raid.
  If the total grows (FMoW is ~100 GB), **email Research_Technology@baylor.edu and ask for the right
  way to stage it** rather than hammering the login node — they may have a data-transfer node;
- make `scripts/get_data.py` strictly two-phase: `--fetch` (login node, network) and
  `--prepare` (compute node, no network), so the compute path never touches a URL;
- `wilds` and `torchvision` both want to download at import; pass `download=False` on the compute
  path and pre-stage instead. A job that dies at hour six because torchvision tried to phone home is
  an avoidable, and very annoying, failure.

The checkout already lives on `/data`, so `results/`, `figs/` and `tables/` need no symlinks — but
keep the **CSVs themselves in git** — they are the paper's evidence
and they are small. Binary artifacts (`.npz`, images, PDFs over ~5 MB) are gitignored.

---

## 4. Job templates

These are written for **OpenPBS 23.06** as verified in §2. Three things that were open questions
in the first draft of this file and are now settled:

1. **`#PBS` directives are not shell-expanded.** `-o /data/$USER/logs/` writes to a directory
   literally named `$USER`. Literal paths only, below.
2. **Resource syntax is `select`/`ncpus`/`ngpus`.** `nodes=`/`ppn=` are not defined on this server
   and will be rejected. The Torque form is gone from this file — do not reintroduce it.
3. **Job arrays are `-J 0-N%K`, index `$PBS_ARRAY_INDEX`,** and the `%K` throttle *does* work here
   (`max_run_subjobs`), contrary to the usual PBS Pro caveat.

And one new hard constraint that shapes every array job:

> **`batch` caps a generic user at 512 concurrently running cores** (`max_run_res.ncpus`). GPU jobs
> consume ncpus from the same budget. So array geometry must satisfy
> `ncpus_per_task × max_concurrent ≤ ~480`, leaving headroom for a GPU job. Asking for
> `ncpus=36` at `%20` is 720 cores and will simply not schedule as intended.

### 4.1 Feature extraction (GPU, one-time, P1)

```bash
#!/bin/bash
#PBS -N p3_extract
#PBS -q gpu
#PBS -l select=1:ncpus=8:ngpus=1:mem=64gb
#PBS -l walltime=08:00:00
#PBS -j oe
#PBS -o /data/islamm/retention_leakage/logs/
set -euo pipefail
cd "$PBS_O_WORKDIR"
module purge; module load python/3.10.4 cuda/12.9   # match CUDA to the driver — see §2 open item 2
source /data/$USER/retention_leakage/envs/p3fcl/bin/activate
export HF_HOME=/data/$USER/retention_leakage/hf
export TORCH_HOME=/data/$USER/retention_leakage/torch
export OMP_NUM_THREADS=8

python -m p3fcl.cli extract \
  --dataset "${DATASET}" --backbone "${BACKBONE}" \
  --splits train,test,ref \
  --out /data/$USER/retention_leakage/features \
  --batch-size 256 --amp
```

Submit per dataset × backbone: `qsub -v DATASET=cifar100,BACKBONE=vit_base_patch16_224.augreg_in21k extract_features.pbs`.
Serialise these — with three V100 nodes, queueing eight extraction jobs at once helps nobody.

### 4.2 Shadow federations (CPU array, the bulk of the compute, P3)

This is where the project's compute actually goes, and it wants **CPU**, not GPU.

```bash
#!/bin/bash
#PBS -N p3_shadow
#PBS -q batch
#PBS -l select=1:ncpus=8:mem=32gb
#PBS -l walltime=12:00:00
#PBS -J 0-249%60                      # 60 x 8 = 480 cores < the 512 per-user cap, with headroom
#PBS -j oe
#PBS -o /data/islamm/retention_leakage/logs/
set -euo pipefail
cd "$PBS_O_WORKDIR"
source /data/$USER/retention_leakage/envs/p3fcl/bin/activate
export OMP_NUM_THREADS=1            # we parallelise over shadows, not inside BLAS
export MKL_NUM_THREADS=1

CHUNK=16                              # 16 shadows / task, 8 workers => 2 waves, ~20-40 min
IDX="${PBS_ARRAY_INDEX:-0}"
START=$(( IDX * CHUNK ))
python -m p3fcl.cli shadows \
  --config configs/attack_lira.yaml \
  --dataset "${DATASET}" --method "${METHOD}" \
  --start "$START" --count "$CHUNK" \
  --workers 8 \
  --out /data/$USER/retention_leakage/shadows/"${DATASET}"/"${METHOD}"
```

Notes that matter:
- **The `%60` is load-bearing, not decoration.** 60 tasks x 8 cores = 480 < the 512-core per-user
  cap, leaving 32 cores for a concurrent GPU job. Raise `ncpus` and you must lower `%K` to match.
  4,000 shadows at CHUNK=16 is 250 array tasks — well under `max_array_size=10000`.
- `OMP_NUM_THREADS=1` plus 8 worker processes beats 8 BLAS threads on one shadow, by a lot, for
  this workload. Measure it once and record the number.
- **Every array task must be resumable**: check for the output `.npz` and skip if present, so a
  requeue is free. Never write a partial `.npz` — write to `.tmp` and `os.replace`.
- Arrays are confirmed working here, so the loop-and-stagger fallback should not be needed. If you
  hit the per-user core cap anyway, lower `%K` before lowering the shadow count.

### 4.3 Prompt/LoRA training (GPU, the only other GPU consumer, P2/P3)

```bash
#PBS -q gpu
#PBS -l select=1:ncpus=16:ngpus=2:mem=96gb
#PBS -l walltime=24:00:00
```
`gpu001`–`gpu005` are the only generally accessible GPU nodes (36 cores, 2 GPUs each). Confirm the
actual GPU model with `nvidia-smi` before sizing batches — §2 open item 2. — ViT-B/16 prompt tuning with a reasonable batch will not fit comfortably on
a P100 16 GB alongside a second job. Request the V100 node explicitly if §2 exposes a property or
queue for it. These methods get the **offline LiRA** variant (OUT shadows only) with N ≈ 64–128;
budget for that from the start.

---

## 5. Cluster etiquette, because this cluster is small

- Ten GPUs is a shared resource among the whole university. Never hold more than one GPU node for
  routine work, and never idle a GPU allocation while debugging — debug on CPU with synthetic
  features (`make smoke` exists for this).
- Long CPU array jobs at `%20` concurrency are fine and are the right shape for this project.
- Put a realistic `walltime` on everything. Short accurate walltimes schedule sooner.
- Check `qstat -u $USER` before submitting another wave.
- **No production work on the login node** — it is a stated site rule, not a convention. Debug on
  CPU with `make smoke`, then submit.
- Everything lives under `/data/$USER`, not `/home/$USER` — also a stated site rule.
- **Never submit to a condo queue** (`hep`, `kevlishvili`, `zeke`, `swint`, `poderoso`, `gallagher`,
  `instructional`, `cray`, `rose`, …). Those 29 GPUs belong to specific PIs. Use `-q gpu`.
- `/data` is **88% full cluster-wide**. The shadow store is the thing that will grow. Check
  `du -sh /data/islamm/retention_leakage/*` between waves, and get your real quota from HPRCS before the big run.
- If you need more than a few node-days in a burst, email HPRCS@baylor.edu first and say what you
  are doing. Ask about a project allocation or a reservation — this is a normal request and they
  would rather know in advance.

## 6. Cost model to plan against (revise once §2 is real)

| Work | Where | Rough scale |
|---|---|---|
| Feature extraction, 4 datasets × 2 backbones × 3 splits | GPU | ~15–30 GPU-hours total, once |
| Cacheable method runs (M0, M1, M2, M3, M5, M8, M9), all sweeps | CPU `batch` | minutes each; thousands of them, capped at 480 concurrent cores |
| Shadow federations, cacheable methods | CPU | the bulk — plan ~4,000 shadows/dataset/method |
| Prompt/LoRA training (M6, M7, M4-LoRA), full stream | GPU | hours per run; keep the run count small |
| Shadow federations, M6/M7/M4-LoRA (offline LiRA) | GPU | 64–128 per config, the real GPU constraint |
| Accounting, theory numerics, all plotting | laptop | seconds |

The single most important budgeting sentence: **if a design needs a GPU per shadow model, it is the
wrong design on this cluster.** Push it onto cached features, or use the offline variant, or cut it.
