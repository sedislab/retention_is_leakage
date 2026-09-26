#!/bin/bash
# One-time environment bootstrap. Builds a virtualenv under <repo>/envs/p3fcl, installs the pinned
# CPU requirements and the p3fcl package in editable mode.
#
# On a cluster that provides environment modules, python 3.10 is loaded through them; elsewhere any
# python >= 3.10 on PATH works. Run from anywhere:  bash env/bootstrap.sh
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_DIR="$REPO/envs/p3fcl"

if command -v module >/dev/null 2>&1; then
  module purge
  module load python/3.10.4 || echo "note: 'module load python/3.10.4' failed; using python3 from PATH" >&2
fi

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
