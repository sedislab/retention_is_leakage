#!/bin/bash
set -euo pipefail
cd /data/islamm/retention_leakage/code
module load python/3.10.4
source /data/islamm/retention_leakage/envs/p3fcl/bin/activate
for s in 0 1 2 3 4; do
  python scripts/run_lira_pertask.py cifar100 m0_fedavg "$s" full
done
