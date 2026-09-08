#!/usr/bin/env bash
# Does admitting the depth-map scale stage help v63 at all? Answered by SETTING the gate and
# scoring, with v63 frozen inside every variant -- no training, so no optimiser noise.
#
# Motivation: v88a trained the gate for 500 steps and it reached -0.005 (a 1.003x effect) with an
# inconsistent gradient sign, i.e. the optimiser declined to open it. A sweep measures the answer
# directly in ~15 min per point instead of hours of watching a scalar random-walk.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
# shellcheck source=/dev/null
source "$(dirname "$0")/cudnn_env.sh"
CACHE=$HOME/data/tapvid3d_baseline_preds/waft_minival_is896_s-1_i4
DA3G=$HOME/data/tapvid3d_da3nested
STATUS=result/OVERNIGHT_STATUS.md
HARD_STOP=$(date -d "2026-09-08 11:00" +%s)
note () { printf '%s\n' "$*" | tee -a "$STATUS"; }

note ""
note "### GATE SWEEP $(date -Is): how much of stage 1 helps, with v63 frozen?"
note "  gate 0.00 = v63 exactly = 0.2238 (measured). TARGET DELTA = 0.2274."
for g in 0.10 0.25 0.50 1.00; do
  [ "$(date +%s)" -ge "$HARD_STOP" ] && { note "  hard stop reached; sweep truncated"; break; }
  d=result/v88_gate$g
  if [ ! -f "$d/eval/metrics.json" ]; then
    uv run python scripts/eval_metric3d.py --method v88 --ckpt "$d/ckpt_0.pt" \
        --da3-depth-root "$DA3G" --split minival --waft-pred-dir "$CACHE" --out-dir "$d/eval" \
      || { note "  gate $g EVAL FAILED"; continue; }
  fi
  note "  [$(date '+%H:%M')] gate $g -> $(uv run python -c "
import json;print(f\"{json.load(open('$d/eval/metrics.json'))['overall']['metric_average_jaccard']:.4f}\")")"
  uv run python scripts/report_metric_aj.py "$d/eval" >> "$STATUS"
done
note "### GATE SWEEP done $(date -Is)"
