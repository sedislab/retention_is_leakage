#!/bin/bash
# Submit a PBS script and record the job id to results/JOBS.jsonl. Usage:
#   scripts/submit.sh path/to/job.pbs [qsub-args...]
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT="$1"
shift || true

JOBID=$(qsub "$@" "$SCRIPT")
TS=$(date -u +%Y-%m-%dT%H:%M:%SZ)

mkdir -p "$REPO/results"
python3 - "$REPO" "$JOBID" "$SCRIPT" "$TS" "$*" <<'EOF'
import json, sys
repo, jobid, script, ts, qsub_args = sys.argv[1:6]
line = {"job_id": jobid, "script": script, "submitted_utc": ts, "qsub_args": qsub_args}
with open(repo + "/results/JOBS.jsonl", "a") as f:
    f.write(json.dumps(line) + "\n")
EOF

echo "$JOBID"
