#!/usr/bin/env bash
# v71: DA3-g + de-flicker + VSSD-2pool with positional noise on the training track only.
#
# Everything experimental is in configs/v71.yaml; the config path is the only argument passed.
# Evaluation reads the same 448/iters-4 track set every other rebuilt arm was scored against, and
# eval_metric3d checks that set's manifest against this run before using it.
#
# Comparisons:
#   v45  SEA-RAFT-trained, WAFT-evaluated  0.2320   <- the target, the paper's row
#   v63  same arm, noise off               0.2238   (trained at iters 5; iters made no difference
#                                                    at eval, 0.2238 either way)
#   v68  same iters, no de-flicker         0.2218
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
# shellcheck source=/dev/null
source "$(dirname "$0")/cudnn_env.sh"

CACHE=$HOME/data/tapvid3d_baseline_preds/waft_minival_is896_s-1_i4
OUT=result/v71_waft448_da3g_tracknoise
mkdir -p "$OUT"

for _ in $(seq 1 240); do
  [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c .)" -eq 0 ] && break
  sleep 60
done
echo "[v71] GPU idle at $(date -Is)"

if [ ! -f "$OUT/ckpt_20000.pt" ]; then
  uv run python scripts/train_depth_refined_tracker.py --config configs/v71.yaml \
    || { echo "[v71] TRAIN FAILED"; exit 1; }
fi
if [ ! -f "$OUT/eval/metrics.json" ]; then
  uv run python scripts/eval_metric3d.py --method v45 --ckpt "$OUT/ckpt_20000.pt" \
      --da3-depth-root "$HOME/data/tapvid3d_da3nested" --split minival \
      --waft-pred-dir "$CACHE" --out-dir "$OUT/eval" \
    || { echo "[v71] EVAL FAILED"; exit 1; }
fi
uv run python -c "
import json; d=json.load(open('$OUT/eval/metrics.json')); ps=d['per_subset']
print('[v71] ' + ' '.join(f\"{s}={ps[s]['metric_average_jaccard']:.4f}\" for s in ('drivetrack','pstudio','adt'))
      + f\" mean={d['overall']['metric_average_jaccard']:.4f} frames={d['frames']} fail={d.get('failures')}\")
print('[v71] target v45 0.2320 | no-noise v63 0.2238 | no-deflicker v68 0.2218')"
echo "[v71] done at $(date -Is)"
