#!/usr/bin/env bash
# v72: Mamba3DepthScaleRefiner on DA3-g -- per-frame depth scale read from the depth map, with its
# own loss (L_dsr). Everything experimental is in configs/v72.yaml.
#
# Replaces the de-flicker stage, whose four defects were each measured:
#   no supervision of its own (+0.0020 delivered against a +0.0707 flicker ceiling); every input
#   arriving through the optical flow; an output bound that clipped 17.4% of frames; and a name
#   describing a tenth of the job, the correction being dominated by a per-clip constant.
#
# Targets, all DA3-g: v68 0.2218 (no stage), v63 0.2238 (old stage), v45 0.2320 (mismatched row).
# Oracle ceiling for a per-frame scale is 0.5033 against 0.2484 as-is.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
# shellcheck source=/dev/null
source "$(dirname "$0")/cudnn_env.sh"

CACHE=$HOME/data/tapvid3d_baseline_preds/waft_minival_is896_s-1_i4
OUT=result/v72_depth_scale_refiner
mkdir -p "$OUT"

for _ in $(seq 1 240); do
  [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c .)" -eq 0 ] && break
  sleep 60
done
echo "[v72] GPU idle at $(date -Is)"

if [ ! -f "$OUT/ckpt_20000.pt" ]; then
  uv run python scripts/train_depth_refined_tracker.py --config configs/v72.yaml \
    || { echo "[v72] TRAIN FAILED"; exit 1; }
fi
if [ ! -f "$OUT/eval/metrics.json" ]; then
  uv run python scripts/eval_metric3d.py --method v72 --ckpt "$OUT/ckpt_20000.pt" \
      --da3-depth-root "$HOME/data/tapvid3d_da3nested" --split minival \
      --waft-pred-dir "$CACHE" --out-dir "$OUT/eval" \
    || { echo "[v72] EVAL FAILED"; exit 1; }
fi
uv run python -c "
import json; d=json.load(open('$OUT/eval/metrics.json')); ps=d['per_subset']
print('[v72] ' + ' '.join(f\"{s}={ps[s]['metric_average_jaccard']:.4f}\" for s in ('drivetrack','pstudio','adt'))
      + f\" mean={d['overall']['metric_average_jaccard']:.4f} frames={d['frames']} fail={d.get('failures')}\")
print('[v72] v68 0.2218 (none) | v63 0.2238 (old stage) | v45 0.2320 (mismatched) | oracle 0.5033')"
echo "[v72] done at $(date -Is)"
