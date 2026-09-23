#!/bin/bash
set -euo pipefail
cd /data/islamm/retention_leakage/code
module load python/3.10.4
source /data/islamm/retention_leakage/envs/p3fcl/bin/activate
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2

# FX4g (08_FIX_PLAN.md): "results/tab05_utility_baselines.csv for 10 tasks. Add 20 tasks only if
# time allows, and mark which." -- this regenerates the mandatory 10-task cifar100/cub200/imagenet_r
# configs (7 methods x 3 datasets x 3 seeds = 63 runs) with the post-fix method code, overwriting the
# stale pre-fix runs/utility_baseline/*.json cache in place (same filenames, same collect_tab05.py
# downstream). Camelyon17 (t5) is deliberately excluded -- its stream needs Wave V3's matched-5-client
# redesign first, not yet done; regenerating it now would just need redoing again.
for METHOD in M0 M1 M2 M3 M4 M5 M8; do
  for DATASET in cifar100 cub200 imagenet_r; do
    for SEED in 0 1 2; do
      python scripts/run_utility_baseline.py --method "$METHOD" --dataset "$DATASET" --n_tasks 10 --seed "$SEED"
    done
  done
done
