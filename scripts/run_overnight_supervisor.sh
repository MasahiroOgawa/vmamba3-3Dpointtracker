#!/usr/bin/env bash
# Keeps run_overnight.sh alive. That script already survives step-level failures; this covers the
# process itself dying (OOM-killer, driver reset, transient cuDNN death at import).
# run_overnight.sh is idempotent, so a relaunch resumes rather than repeating work.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
HARD_STOP=$(date -d "2026-09-08 11:00" +%s)
STATUS=result/OVERNIGHT_STATUS.md
for attempt in 1 2 3 4 5; do
  now=$(date +%s)
  [ "$now" -ge "$HARD_STOP" ] && { printf '%s\n' "## supervisor: hard stop reached $(date -Is)" >> "$STATUS"; break; }
  printf '%s\n' "## supervisor: launching run_overnight.sh (attempt $attempt) $(date -Is)" >> "$STATUS"
  bash scripts/run_overnight.sh >> result/overnight.log 2>&1
  rc=$?
  printf '%s\n' "## supervisor: run_overnight.sh exited $rc $(date -Is)" >> "$STATUS"
  grep -q "overnight run ended" "$STATUS" && break
  sleep 120
done
printf '%s\n' "## supervisor finished $(date -Is)" >> "$STATUS"
