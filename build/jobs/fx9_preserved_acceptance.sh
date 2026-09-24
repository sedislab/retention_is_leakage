#!/bin/bash
set -euo pipefail
cd /data/islamm/retention_leakage
exec > >(tee -a "build/logs/${PBS_JOBID}.live.log") 2>&1
module load python/3.10.4
source envs/p3fcl/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python code/scripts/check_fx9_preserved.py
python analysis/fig05_eps_of_T.py
python - <<'PYDOC'
import sys
sys.path.insert(0,'code/scripts')
from build_fx9_documents import ROOT, read, value, update_hypothesis
rows=[r for r in read(ROOT/'results/fx9_m9_gap_summary.csv') if r['unit']=='U2' and float(r['eps'])==1]
assert all(float(r['gap_m9_minus_m8_ci_hi'])<-.05 for r in rows)
lines=[f"{r['dataset']} U2 epsilon=1: paired M9−M8 accuracy gap {value(r,'gap_m9_minus_m8')}." for r in rows]
lines += ['Source: results/fx9_m9_gap_summary.csv, preserved M9 sweep. The within-five-points utility criterion is refuted; the separate DP-baseline superiority comparison remains untested.']
p=ROOT/'agents/OPEN_QUESTIONS.md'
p.write_text(update_hypothesis(p.read_text(),'H7','REFUTED',lines))
print('FX9-10 H7 status=REFUTED paired_datasets=3')
PYDOC
