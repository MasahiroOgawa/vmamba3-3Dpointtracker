#!/usr/bin/env bash
# v70 (DA3-g + vmamba3-2pool + per-frame scale head) once the current rebuild queue finishes.
#
# Deferred earlier to stay inside the time budget, and requested afterwards. It completes the DA3-g
# table: every row then has a self-consistent arm whose training and evaluation use the same flow
# front-end at the same resolution and iteration count.
#
# Reads the same 448/iters-4 track set the other arms were scored against; eval_metric3d checks that
# set's manifest against this run and refuses to score if they disagree.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
# shellcheck source=/dev/null
source "$(dirname "$0")/cudnn_env.sh"

CACHE=$HOME/data/tapvid3d_baseline_preds/waft_minival_is896_s-1_i4
OUT=result/v70_waft448_da3g_scalehead
DEPTH=$HOME/data/tapvid3d_da3nested

# Wait for the rebuild queue to release the GPU, but do not wait forever on a queue that died.
for _ in $(seq 1 2880); do
  pgrep -f "queue_448_rebuild.sh" >/dev/null || break
  sleep 60
done
for _ in $(seq 1 120); do
  [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c .)" -eq 0 ] && break
  sleep 60
done
echo "[v70] GPU idle at $(date -Is)"

mkdir -p "$OUT"
if [ ! -f "$OUT/ckpt_20000.pt" ]; then
  uv run python scripts/train_depth_refined_tracker.py --config configs/v70.yaml \
    || { echo "[v70] TRAIN FAILED"; exit 1; }
fi
if [ ! -f "$OUT/eval/metrics.json" ]; then
  uv run python scripts/eval_metric3d.py --method v35 --ckpt "$OUT/ckpt_20000.pt" \
      --da3-depth-root "$DEPTH" --split minival \
      --waft-pred-dir "$CACHE" --out-dir "$OUT/eval" \
    || { echo "[v70] EVAL FAILED"; exit 1; }
fi
uv run python -c "
import json; d=json.load(open('$OUT/eval/metrics.json')); ps=d['per_subset']
print('[v70] ' + ' '.join(f\"{s}={ps[s]['metric_average_jaccard']:.4f}\" for s in ('drivetrack','pstudio','adt'))
      + f\" mean={d['overall']['metric_average_jaccard']:.4f} frames={d['frames']} fail={d.get('failures')}\")
print('[v70] against: paper scale-head 0.229 | v68 no-scale-head 0.2218 | v63 deflicker 0.2238')"
echo "[v70] done at $(date -Is)"
