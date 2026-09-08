#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
HARD_STOP=$(date -d "2026-09-08 11:00" +%s)
STATUS=result/OVERNIGHT_STATUS.md
for attempt in 1 2 3 4 5; do
  [ "$(date +%s)" -ge "$HARD_STOP" ] && { printf '%s\n' "## supervisor2: hard stop $(date -Is)" >> "$STATUS"; break; }
  printf '%s\n' "## supervisor2: launching run_overnight2.sh (attempt $attempt) $(date -Is)" >> "$STATUS"
  bash scripts/run_overnight2.sh >> result/overnight2.log 2>&1
  printf '%s\n' "## supervisor2: exited $? $(date -Is)" >> "$STATUS"
  grep -q "PART 2 ended" "$STATUS" && break
  sleep 120
done
printf '%s\n' "## supervisor2 finished $(date -Is)" >> "$STATUS"
