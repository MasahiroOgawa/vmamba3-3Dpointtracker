#!/usr/bin/env bash
# Keep queue_waft_arms.sh alive for the ~2 days it needs, and make any death loud.
#
# Why this exists: the sweep died at 15:36 on 2026-08-16 from a FileNotFoundError on one clip of
# 4419 and sat dead for two hours before anyone looked. A job of this length with no liveness check
# can lose a whole day silently, and the only signal was a progress line that simply stopped moving.
#
# Two failure modes are covered, because they look different:
#   dead    -- no queue process at all. Restart; step 0 resumes from the tracks already on disk.
#   stalled -- process alive but the log has not been written for STALL_MIN minutes. Kill, restart.
#              A live-but-silent run is the worse case: pgrep alone reports it healthy.
#
# Stops on its own when the last arm has been evaluated, so it does not outlive the work.
set -uo pipefail
cd "$(dirname "$0")/.."

LOG=result/waft_arms.log
WD_LOG=result/watchdog.log
POLL_SEC=${POLL_SEC:-300}
STALL_MIN=${STALL_MIN:-45}      # generous: a 300-frame adt clip can take minutes
MAX_RESTARTS=${MAX_RESTARTS:-20}

say () { echo "[watchdog $(date -Is)] $*" | tee -a "$WD_LOG"; }

restarts=0
say "started; polling every ${POLL_SEC}s, stall threshold ${STALL_MIN}min"

while :; do
  sleep "$POLL_SEC"

  # Finished? The last arm is v60; its eval metrics file is the completion marker.
  if [ -f result/v60_waft/eval/metrics.json ]; then
    say "v60 evaluated; sweep complete, watchdog exiting"
    exit 0
  fi

  if [ "$restarts" -ge "$MAX_RESTARTS" ]; then
    say "FAILED: $restarts restarts reached without completing; not restarting again"
    exit 1
  fi

  running=$(pgrep -fc "bash scripts/queue_waft_arms.sh" || true)
  : "${running:=0}"

  if [ "$running" -eq 0 ]; then
    say "queue not running -- restarting (restart #$((restarts+1)))"
    tail -c 600 "$LOG" 2>/dev/null | tr '\r' '\n' | grep -viE "^\s*(drivetrack|pstudio|adt):" \
      | tail -3 | sed 's/^/    last: /' | tee -a "$WD_LOG"
    nohup bash scripts/queue_waft_arms.sh >> "$LOG" 2>&1 &
    restarts=$((restarts+1))
    continue
  fi

  # Alive but silent: compare log mtime against the stall threshold.
  if [ -f "$LOG" ]; then
    age_min=$(( ( $(date +%s) - $(stat -c %Y "$LOG") ) / 60 ))
    if [ "$age_min" -ge "$STALL_MIN" ]; then
      say "log silent for ${age_min}min while the process is alive -- killing and restarting"
      pkill -f "bash scripts/queue_waft_arms.sh"
      pkill -f "scripts/eval_waft.py"
      pkill -f "scripts/train_depth_refined_tracker.py"
      sleep 10
      pkill -9 -f "scripts/eval_waft.py" 2>/dev/null
      pkill -9 -f "scripts/train_depth_refined_tracker.py" 2>/dev/null
      sleep 5
      nohup bash scripts/queue_waft_arms.sh >> "$LOG" 2>&1 &
      restarts=$((restarts+1))
    fi
  fi
done
