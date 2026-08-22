#!/usr/bin/env bash
# Short paired runs to test one hypothesis: does WAFT's more accurate visibility hurt by zeroing
# more input tokens? v59 (gated) is the control, v61 (ungated) the treatment; everything else,
# including the WAFT front-end for both training and validation, is identical.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
# shellcheck source=/dev/null
source "$(dirname "$0")/cudnn_env.sh"
STEPS=${STEPS:-6000}
WAFT_TRAIN=$HOME/data/tapvid3d_baseline_preds/waft_full_eval
for tag in v59 v61; do
  out=result/probe_${tag}_${STEPS}
  [ -f "$out/loss_history.json" ] && { echo "[probe] $tag done"; continue; }
  echo "=== [probe $tag] $STEPS steps ($(date -Is)) ==="
  uv run python scripts/train_depth_refined_tracker.py --config "configs/${tag}.yaml" \
      --out-dir "$out" --steps "$STEPS" --waft-pred-dir "$WAFT_TRAIN" \
    || { echo "[probe] $tag FAILED"; continue; }
done
echo "[probe] done at $(date -Is)"
