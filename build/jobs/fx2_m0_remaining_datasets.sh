#!/bin/bash
set -euo pipefail
cd /data/islamm/retention_leakage/code
module load python/3.10.4
source /data/islamm/retention_leakage/envs/p3fcl/bin/activate
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2

DATASETS=(cub200 imagenet_r)
DATASET="${DATASETS[$PBS_ARRAY_INDEX]}"

for SEED in 0 1 2 3 4; do
  OUT_NPZ="/data/islamm/retention_leakage/results/a1_lira_pertask_${DATASET}_m0_fedavg_seed${SEED}_full.npz"
  if [ -f "$OUT_NPZ" ]; then
    echo "skipping $DATASET/m0_fedavg/seed$SEED/full: already exists"
    continue
  fi
  python scripts/run_lira_pertask.py "$DATASET" m0_fedavg "$SEED" full
done

python scripts/build_fx2_summary.py "$DATASET" m0_fedavg F1 full 0 1 2 3 4
