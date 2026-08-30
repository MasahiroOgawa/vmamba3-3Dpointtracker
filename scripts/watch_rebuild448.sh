#!/usr/bin/env bash
# Watcher for queue_448_rebuild.sh. Exists because that queue halted at its verification gate and
# the GPU then sat idle for 15 hours with nothing reporting it.
#
# Reports every stage transition, and any halt, to result/rebuild448.watch. A queue stopped at stage
# 1 is indistinguishable from a queue working on stage 3 unless something says which.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
LOG=result/rebuild448.log
W=result/rebuild448.watch
POLL=${POLL:-300}
CACHE_DIR=${CACHE_DIR:-$HOME/data/tapvid3d_baseline_preds/waft_minival_is896_s-1_i4}
say () { echo "[watch $(date -Is)] $*" | tee -a "$W"; }

say "started; polling every ${POLL}s"
last=""
while :; do
  sleep "$POLL"
  # halted?
  if ! pgrep -f "queue_448_rebuild.sh" >/dev/null; then
    if grep -aqE "\[448\] done at" "$LOG" 2>/dev/null; then
      say "QUEUE COMPLETE"
      grep -aE "^\[v6[789]\] " "$LOG" | tee -a "$W"
    else
      say "QUEUE HALTED without completing -- last lines follow"
      grep -aE "\[448\]|\[v6[789]\]|FAILED|STOP|Error" "$LOG" | tail -5 | tee -a "$W"
    fi
    exit 0
  fi
  # stage transitions and finished arms
  now=$(grep -aE "\[448\] cache complete|\[448\] verification|\[448\] cache and live agree|^\[v6[789]\] (training|drivetrack)" "$LOG" 2>/dev/null | tail -1)
  if [ "$now" != "$last" ] && [ -n "$now" ]; then
    say "stage: ${now:0:110}"
    last="$now"
  fi
  # stall detection: the log must move
  # The cache-wait stage polls silently for hours by design, so a silent log only means something
  # once the queue has passed it. Before that, report progress instead of warning.
  if [ -f "$LOG" ]; then
    age=$(( ( $(date +%s) - $(stat -c %Y "$LOG") ) / 60 ))
    if grep -aq "\[448\] cache complete" "$LOG" 2>/dev/null; then
      [ "$age" -ge 60 ] && say "WARNING: log silent for ${age} min while the queue process is alive"
    elif [ $(( age % 30 )) -eq 0 ] && [ "$age" -gt 0 ]; then
      say "waiting on cache: $(find "$CACHE_DIR" -name '*.npz' 2>/dev/null | wc -l)/150 clips"
    fi
  fi
done
