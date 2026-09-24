#!/bin/bash
set -euo pipefail
cd /data/islamm/retention_leakage
module load python/3.10.4
source envs/p3fcl/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
for dataset in cifar100 cub200 imagenet_r; do
  python code/scripts/build_fx2_accuracy_summary.py "$dataset" m0_fedavg F1 full
 done
python - <<'PY'
import csv
import numpy as np
from pathlib import Path
from p3fcl.halflife import accuracy_curve
for ds in ['cifar100', 'cub200', 'imagenet_r']:
    with Path(f'results/accuracy_matrix_{ds}.csv').open() as f:
        rows = [r for r in csv.DictReader(f) if r['method'] == 'm0_fedavg']
    curve = accuracy_curve(rows)
    diagonal = np.mean([float(r['acc']) for r in rows if int(r['task_k']) in range(4) and int(r['elapsed']) == 0])
    assert abs(curve.raw[0]-diagonal) < 1e-12
    assert curve.norm[0] == 1
    print(f'FX9-1 ACCEPT {ds} raw_e0={curve.raw[0]:.12f} raw_e6={curve.raw[6]:.12f} diagonal={diagonal:.12f} diagonal_error={abs(curve.raw[0]-diagonal):.3g} norm_e0={curve.norm[0]}')
PY
