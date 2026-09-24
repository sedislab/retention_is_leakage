#!/bin/bash
set -euo pipefail
cd /data/islamm/retention_leakage
exec > >(tee -a "build/logs/${PBS_JOBID}.live.log") 2>&1
module load python/3.10.4
source envs/p3fcl/bin/activate
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
DATASETS=(cifar100 cub200 imagenet_r)
METHODS=(m0_fedavg m1_glfc m2_target m3_fot m4_proto m5_hybrid_replay m8_analytic)
DS=${DATASETS[$((PBS_ARRAY_INDEX / 7))]}
METHOD=${METHODS[$((PBS_ARRAY_INDEX % 7))]}
python -u code/scripts/run_accuracy_matrix.py "$DS" "$METHOD"
SHORT=${METHOD%%_*}
for SEED in 0 1 2 3 4; do
  python -u code/scripts/run_utility_baseline.py --method "${SHORT^^}" --dataset "$DS" --n_tasks 10 --seed "$SEED"
done
