#!/bin/bash
set -euo pipefail
cd /data/islamm/retention_leakage/code
module load python/3.10.4
source /data/islamm/retention_leakage/envs/p3fcl/bin/activate
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
python scripts/run_m9_sweep.py cifar100
