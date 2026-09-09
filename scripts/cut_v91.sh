#!/usr/bin/env bash
# Hard deadline for v91 so its final evaluation lands before 13:00. run_v91.sh scores whatever
# checkpoint exists once the trainer exits, so stopping the trainer is all that is needed.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
PID=$1; CUT=$(date -d "$2" +%s)
S=result/OVERNIGHT_STATUS.md
while [ -d "/proc/$PID" ]; do
  if [ "$(date +%s)" -ge "$CUT" ]; then
    printf '  [cut] v91 deadline %s reached; stopping %s\n' "$2" "$PID" | tee -a "$S"
    kill -INT "$PID" 2>/dev/null
    for _ in $(seq 1 40); do sleep 5; [ -d "/proc/$PID" ] || break; done
    [ -d "/proc/$PID" ] && kill -TERM "$PID" 2>/dev/null
    exit 0
  fi
  sleep 60
done
printf '  [cut] v91 trainer exited on its own at %s\n' "$(date '+%H:%M')" | tee -a "$S"
