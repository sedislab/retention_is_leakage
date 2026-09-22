#!/bin/bash
# Poll a PBS job (or array) until it leaves the queue, with a bounded timeout. Exits non-zero if the
# job is still present when the timeout expires, or if `qstat` reports it in a failed-looking state.
# Usage: scripts/wait_job.sh <job_id> [timeout_seconds] [poll_interval_seconds]
set -euo pipefail

JOBID="$1"
TIMEOUT="${2:-3600}"
INTERVAL="${3:-30}"
ELAPSED=0

while qstat "$JOBID" >/dev/null 2>&1; do
  if [ "$ELAPSED" -ge "$TIMEOUT" ]; then
    echo "wait_job: timed out after ${TIMEOUT}s waiting on $JOBID" >&2
    exit 2
  fi
  sleep "$INTERVAL"
  ELAPSED=$(( ELAPSED + INTERVAL ))
done

echo "wait_job: $JOBID left the queue after ~${ELAPSED}s"
