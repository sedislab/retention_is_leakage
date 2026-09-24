#!/bin/bash
set -euo pipefail
cd /data/islamm/retention_leakage
exec > >(tee -a "build/logs/${PBS_JOBID}.live.log") 2>&1
module load python/3.10.4
source envs/p3fcl/bin/activate
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
pytest -q code/tests
python code/scripts/check_fx9_preserved.py
python -u code/scripts/merge_fx2_summary.py
python -u code/scripts/build_retention_curves.py
python code/scripts/build_fig01_decoupling.py
python code/scripts/merge_fx9_ratios.py
python code/scripts/build_fx3_views_summary.py
python code/scripts/build_fx9_roc.py
python code/scripts/build_fig17.py
python code/scripts/merge_fx9_dose.py
python code/scripts/build_fig03_dose_response.py
python code/scripts/build_fig04_semantic_vs_individual.py
python code/scripts/build_fx9_reporting.py
python code/scripts/build_fx9_m9_gap.py
python code/scripts/merge_fx9_fig05.py
python code/scripts/build_fig08_pareto.py
python code/scripts/build_tab08.py
python code/scripts/build_method_descriptions.py
make figures
python code/scripts/build_paper_numbers.py
python code/scripts/build_fx9_documents.py
make verify
