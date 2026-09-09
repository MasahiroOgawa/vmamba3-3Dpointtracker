#!/usr/bin/env bash
# v91: the v90 swap with a corrected learning rate. Same architecture, same warm start as v90a.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
# shellcheck source=/dev/null
source "$(dirname "$0")/cudnn_env.sh"
CACHE=$HOME/data/tapvid3d_baseline_preds/waft_minival_is896_s-1_i4
S=result/OVERNIGHT_STATUS.md
note () { printf '%s\n' "$*" | tee -a "$S"; }
note ""; note "### v91 (v90 swap, lr 1e-4 -> 5e-5, real stable phase) $(date -Is)"
uv run python scripts/train_depth_refined_tracker.py --config configs/v91.yaml \
  || note "  v91 exited nonzero; scoring what exists"
ck=$(ls -1t result/v91/best/ckpt_*.pt result/v91/ckpt_*.pt 2>/dev/null | head -1)
if [ -n "$ck" ] && [ ! -f result/v91/eval_final/metrics.json ]; then
  note "  final scoring on $ck"
  uv run python scripts/eval_metric3d.py --method v73 --ckpt "$ck" --depth da3g \
      --split minival --waft-pred-dir "$CACHE" --out-dir result/v91/eval_final || note "  v91 EVAL FAILED"
fi
[ -f result/v91/eval_final/metrics.json ] && uv run python scripts/report_metric_aj.py result/v91/eval_final | tee -a "$S"
note "### v91 done $(date '+%H:%M')   bars: warm start 0.2191 | v63 0.2238 | DELTA 0.2274"
