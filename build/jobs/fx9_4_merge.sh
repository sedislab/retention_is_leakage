#!/bin/bash
set -euo pipefail
cd /data/islamm/retention_leakage
exec > >(tee -a "build/logs/${PBS_JOBID}.live.log") 2>&1
module load python/3.10.4
source envs/p3fcl/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python -u code/scripts/merge_fx9_accuracy.py
python code/scripts/collect_tab05.py
python code/scripts/build_tab07.py
python - <<'PY'
import subprocess, sys
from p3fcl.experiment import DATASETS, METHODS
from p3fcl.shadow_runner import METHOD_REGISTRY
for ds in DATASETS:
    for method in METHODS:
        family=METHOD_REGISTRY[method][1].value
        for view in (['full', 'aggregate', 'global'] if method in ['m4_proto', 'm8_analytic'] else ['full']):
            subprocess.run([sys.executable, 'code/scripts/build_fx2_accuracy_summary.py', ds, method, family, view], check=True)
PY
