#!/bin/bash
set -euo pipefail
cd /data/islamm/retention_leakage
exec > >(tee -a "build/logs/${PBS_JOBID}.live.log") 2>&1
module load python/3.10.4
source envs/p3fcl/bin/activate
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
python code/scripts/build_method_descriptions.py
python -u code/scripts/run_gram_inversion.py
python code/scripts/build_fig13.py
python analysis/fig13_gram_inversion.py
