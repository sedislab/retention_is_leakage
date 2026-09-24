#!/bin/bash
set -euo pipefail
cd /data/islamm/retention_leakage
exec > >(tee -a "build/logs/${PBS_JOBID}.live.log") 2>&1
module load python/3.10.4
source envs/p3fcl/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python code/scripts/build_wave_v3_manifest.py
python code/scripts/build_fx8_manifest.py
