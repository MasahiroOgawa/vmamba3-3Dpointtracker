#!/usr/bin/env bash
# Standalone deadline for the already-running v90c trainer.
#
# chain_v90c.sh's own cut_at polled for the trainer the instant after launching it, before `uv run`
# had exec'd the venv python, so it saw no process, concluded the job had finished, and returned --
# leaving v90c with no deadline. The chain is now parked on `wait` and will still run the evaluation
# once the trainer exits, so only the stop is missing. Takes the pid directly: it is known.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
PID=$1
CUT=$(date -d "$2" +%s)
S=result/OVERNIGHT_STATUS.md
while [ -d "/proc/$PID" ]; do
  if [ "$(date +%s)" -ge "$CUT" ]; then
    printf '  [v90c] deadline %s reached; stopping pid %s\n' "$2" "$PID" | tee -a "$S"
    kill -INT "$PID" 2>/dev/null
    for _ in $(seq 1 60); do sleep 5; [ -d "/proc/$PID" ] || break; done
    [ -d "/proc/$PID" ] && { printf '  [v90c] SIGINT ignored; SIGTERM\n' | tee -a "$S"; kill -TERM "$PID"; }
    exit 0
  fi
  sleep 60
done
printf '  [v90c] trainer %s exited on its own at %s\n' "$PID" "$(date '+%H:%M')" | tee -a "$S"
