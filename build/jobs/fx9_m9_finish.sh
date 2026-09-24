#!/bin/bash
set -euo pipefail
cd /data/islamm/retention_leakage
exec > >(tee -a "build/logs/${PBS_JOBID}.live.log") 2>&1
module load python/3.10.4
source envs/p3fcl/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python code/scripts/check_fx9_preserved.py
python code/scripts/build_tab08.py
python code/scripts/build_fig08_pareto.py
python analysis/fig08_pareto.py
