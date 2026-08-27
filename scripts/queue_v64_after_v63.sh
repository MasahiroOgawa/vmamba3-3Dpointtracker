#!/usr/bin/env bash
# Run v64 once v63 has finished. v63 is the DA3-g WAFT arm at iters=5, deliberately left to complete
# rather than restarted; v64 is the DA3-l WAFT arm at iters=4, matched to v50's SEA-RAFT settings.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
until [ -f result/v63_waft_live_2pool/eval/metrics.json ]; do
  pgrep -f "run_v63_waft_live" >/dev/null || { echo "[queue] v63 is no longer running and produced no eval; stopping"; exit 1; }
  sleep 300
done
echo "[queue] v63 complete at $(date -Is); starting v64"
exec bash scripts/run_v64_waft_live_da3l.sh
