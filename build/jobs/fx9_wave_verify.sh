#!/bin/bash
set -euo pipefail
cd /data/islamm/retention_leakage
exec > >(tee -a "build/logs/${PBS_JOBID}.live.log") 2>&1
module load python/3.10.4
source envs/p3fcl/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python -u code/scripts/verify_wave_v2.py --root "$(python -c 'from p3fcl.paths import REPO_ROOT,V3_SHADOW_ROOT; print(REPO_ROOT/V3_SHADOW_ROOT)')" --methods m1_glfc m2_target m4_proto m5_hybrid_replay
