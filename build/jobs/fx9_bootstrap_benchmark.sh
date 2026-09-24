#!/bin/bash
set -euo pipefail
cd /data/islamm/retention_leakage
module load python/3.10.4
source envs/p3fcl/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python -u code/scripts/benchmark_fx9_bootstrap.py
