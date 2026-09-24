#!/bin/bash
set -euo pipefail
cd /data/islamm/retention_leakage
test -n "${PBS_JOBID:-}"
module load python/3.10.4
source envs/p3fcl/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
dest=archive/2026-09-23b_pre_fx9
test ! -e "$dest"
mkdir -p "$dest"/{paper,agents,build}
cp -a results figs tables "$dest/"
cp -a paper/PAPER_BRIEFING.md "$dest/paper/"
cp -a agents/OPEN_QUESTIONS.md "$dest/agents/"
cp -a build/STATE.md "$dest/build/"
cat > "$dest/README.md" <<'EOF'
# Pre-FX9 archive

This is the unmodified pre-FX9 snapshot of results, figures, tables, the paper briefing, hypothesis register, and state file, copied with `cp -a` before the second fix round. See `build/09_FIX_PLAN.md` §1 for B1–B5 and I1–I4, the review findings that require replacements. Historical completion claims and numbers here are superseded by the FX9 acceptance checks. No data were deleted and neither existing shadow store was changed. SHA256SUMS covers the copied files and this README; verify from this directory with `sha256sum -c SHA256SUMS`.
EOF
(
  cd "$dest"
  find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS
  sha256sum --quiet -c SHA256SUMS
)
python - <<'PY'
from pathlib import Path
from p3fcl import provenance
dest = Path('archive/2026-09-23b_pre_fx9')
files = sum(1 for _ in (dest / 'SHA256SUMS').open())
print(f'FX9-0 ACCEPT copied_and_verified_files={files} checksum_failures=0 source_sets=6 shadows_modified=0', flush=True)
manifest = provenance.run_manifest({'phase': 'FX9-0', 'hypotheses': ['H2', 'H7', 'H11'], 'archive': str(dest)})
provenance.finalize(manifest, [])
PY
