#!/usr/bin/env bash
# Start v90b as soon as v90a has been trained AND evaluated, without a manual go-ahead.
#
# Waits for run_v73_plan.sh to exit rather than editing it: that script is mid-run, and bash reads a
# script incrementally, so editing a running one can corrupt its execution. The plan ends right
# after v90a's evaluation, so its exit is exactly the signal we want.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
S=result/OVERNIGHT_STATUS.md
note () { printf '%s\n' "$*" | tee -a "$S"; }

for _ in $(seq 1 900); do          # up to 7.5h
  running=""
  for p in $(ls /proc | grep -E '^[0-9]+$'); do
    c=$(tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null) || continue
    case "$c" in *run_v73_plan.sh*) case "$c" in *chain_v90b*) ;; *) running=1;; esac;; esac
  done
  [ -z "$running" ] && break
  sleep 30
done

if [ ! -f result/v90a/eval/metrics.json ]; then
  note "  [$(date +%H:%M)] v90a has no evaluation; starting v90b anyway from its best checkpoint"
else
  note "  [$(date +%H:%M)] v90a evaluated; starting v90b automatically (no confirmation needed)"
fi
exec bash scripts/run_v90b.sh
