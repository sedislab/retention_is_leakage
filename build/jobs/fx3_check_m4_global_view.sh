#!/bin/bash
set -euo pipefail
cd /data/islamm/retention_leakage/code
module load python/3.10.4
source /data/islamm/retention_leakage/envs/p3fcl/bin/activate
python scripts/run_lira_pertask.py cifar100 m4_proto 0 full
python scripts/run_lira_pertask.py cifar100 m4_proto 0 aggregate
python scripts/run_lira_pertask.py cifar100 m4_proto 0 global
