#!/bin/bash
set -euo pipefail
cd /data/islamm/retention_leakage/code
module load python/3.10.4
source /data/islamm/retention_leakage/envs/p3fcl/bin/activate
python scripts/build_fx2_summary.py cifar100 m0_fedavg F1 full 0 1 2 3 4
