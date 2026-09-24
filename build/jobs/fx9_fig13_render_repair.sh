#!/bin/bash
set -euo pipefail
cd /data/islamm/retention_leakage
exec > >(tee -a "build/logs/${PBS_JOBID}.live.log") 2>&1
module load python/3.10.4
source envs/p3fcl/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python - <<'PY'
import csv
from pathlib import Path
with Path('results/fig13_gram_inversion_summary.csv').open() as f:
    rows=list(csv.DictReader(f))
assert all(int(r['n_trials'])==25 for r in rows)
print(f'FX9-8 FIG13 ACCEPT cells={len(rows)} min_trials=25 statistic=median bootstrap_replicates=2000',flush=True)
PY
python analysis/fig13_gram_inversion.py
