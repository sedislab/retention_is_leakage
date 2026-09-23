#!/bin/bash
set -euo pipefail
cd /data/islamm/retention_leakage/code
module load python/3.10.4
source /data/islamm/retention_leakage/envs/p3fcl/bin/activate
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2

for SEED in 0 1 2; do
  python scripts/run_lira_pertask.py cifar100 m0_fedavg "$SEED" full 1024
done
python scripts/build_a1_m0_budget_check.py
