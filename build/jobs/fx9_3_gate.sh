#!/bin/bash
set -euo pipefail
cd /data/islamm/retention_leakage
module load python/3.10.4
source envs/p3fcl/bin/activate
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
DATASETS=(cifar100 cub200 imagenet_r)
python -u code/scripts/run_fx9_gate.py "${DATASETS[$PBS_ARRAY_INDEX]}"
