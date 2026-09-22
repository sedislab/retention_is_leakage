#!/bin/bash
# Submit a PBS script and record the job id to results/JOBS.jsonl. Usage:
#   scripts/submit.sh path/to/job.pbs [qsub-args...]
set -euo pipefail

REPO=/data/islamm/retention_leakage
SCRIPT="$1"
shift || true

JOBID=$(qsub "$@" "$SCRIPT")
TS=$(date -u +%Y-%m-%dT%H:%M:%SZ)

mkdir -p "$REPO/results"
python3 - "$JOBID" "$SCRIPT" "$TS" "$*" <<'EOF'
import json, sys
jobid, script, ts, qsub_args = sys.argv[1:5]
line = {"job_id": jobid, "script": script, "submitted_utc": ts, "qsub_args": qsub_args}
with open("/data/islamm/retention_leakage/results/JOBS.jsonl", "a") as f:
    f.write(json.dumps(line) + "\n")
EOF

echo "$JOBID"
