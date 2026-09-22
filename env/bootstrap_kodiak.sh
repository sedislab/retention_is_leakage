#!/bin/bash
# One-time environment bootstrap, run from the login node. Builds the venv under
# /data/islamm/retention_leakage/envs/p3fcl — never in $HOME (Kodiak site rule).
set -euo pipefail

REPO=/data/islamm/retention_leakage
ENV_DIR="$REPO/envs/p3fcl"

module purge
module load python/3.10.4

if [ ! -d "$ENV_DIR" ]; then
  python3 -m venv "$ENV_DIR"
fi
# shellcheck disable=SC1091
source "$ENV_DIR/bin/activate"

pip install --upgrade pip setuptools wheel
pip install -r "$REPO/env/requirements.txt"
pip install -e "$REPO/code"

python -c "import p3fcl; print('p3fcl', p3fcl.__version__, 'importable OK')"
echo "Bootstrap complete. Activate with: source $ENV_DIR/bin/activate"
