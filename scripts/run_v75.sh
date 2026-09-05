#!/usr/bin/env bash
# v75: WAFT + (DA3-g + frozen Mamba-3 depth-scale refiner) + vmamba3-2pool.
#
# The scale refiner is a fixed depth pre-processor -- (DA3-g + scale refiner) is just a better depth
# source -- and only the vmamba3 refiner trains, warm-started from v63.
#
# Targets, metric-AJ on DA3-g:
#   scale refiner alone on raw tracks  0.2385   <- already beats DELTA
#   DELTA + DA3-g                      0.2270
#   v63  de-flicker + vmamba3-2pool    0.2238
#   oracle per-frame scale             0.3298
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
# shellcheck source=/dev/null
source "$(dirname "$0")/cudnn_env.sh"
CACHE=$HOME/data/tapvid3d_baseline_preds/waft_minival_is896_s-1_i4
OUT=$(uv run python -c "import yaml;print(yaml.safe_load(open('configs/v75.yaml'))['train']['out_dir'])")
STEPS=$(uv run python -c "import yaml;print(yaml.safe_load(open('configs/v75.yaml'))['train']['steps'])")
mkdir -p "$OUT"

for _ in $(seq 1 240); do
  [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c .)" -eq 0 ] && break
  sleep 60
done
echo "[v75] GPU idle at $(date -Is)"

CK=$(ls -t "$OUT"/best/ckpt_*.pt 2>/dev/null | head -1)
if [ -z "$CK" ] && [ ! -f "$OUT/ckpt_${STEPS}.pt" ]; then
  uv run python scripts/train_depth_refined_tracker.py --config configs/v75.yaml \
    || { echo "[v75] TRAIN FAILED"; exit 1; }
fi
CK=$(ls -t "$OUT"/best/ckpt_*.pt 2>/dev/null | head -1)
[ -n "$CK" ] || CK=$(ls -t "$OUT"/ckpt_*.pt 2>/dev/null | head -1)
[ -n "$CK" ] || { echo "[v75] no checkpoint"; exit 1; }
echo "[v75] scoring $CK"
if [ ! -f "$OUT/eval/metrics.json" ]; then
  uv run python scripts/eval_metric3d.py --method v73 --ckpt "$CK" \
      --da3-depth-root "$HOME/data/tapvid3d_da3nested" --split minival \
      --waft-pred-dir "$CACHE" --out-dir "$OUT/eval" || { echo "[v75] EVAL FAILED"; exit 1; }
fi
uv run python -c "
import json; d=json.load(open('$OUT/eval/metrics.json')); ps=d['per_subset']
print('[v75] ' + ' '.join(f\"{s}={ps[s]['metric_average_jaccard']:.4f}\" for s in ('drivetrack','pstudio','adt'))
      + f\" mean={d['overall']['metric_average_jaccard']:.4f} frames={d['frames']} fail={d.get('failures')}\")
print('[v75] vs: scale alone 0.2385 | DELTA 0.2270 | v63 0.2238 | oracle 0.3298')"
echo "[v75] done at $(date -Is)"
