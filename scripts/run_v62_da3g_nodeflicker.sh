#!/usr/bin/env bash
# v62: WAFT + DA3-g + VSSD-2pool, WITHOUT the de-flicker stage.
#
# Trained AND evaluated on WAFT, so it is directly comparable to v60 (same thing with de-flicker).
# v62 vs v60 isolates the de-flicker stage; nothing else differs.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
# shellcheck source=/dev/null
source "$(dirname "$0")/cudnn_env.sh"
STEPS=${STEPS:-20000}
WAFT_TRAIN=$HOME/data/tapvid3d_baseline_preds/waft_full_eval
WAFT_EVAL=$HOME/data/tapvid3d_baseline_preds/waft_minival_cudnn925
OUT=result/v62_waft

for _ in $(seq 1 1440); do
  [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c .)" -eq 0 ] && break
  sleep 60
done
echo "[v62] GPU idle at $(date -Is)"

if [ ! -f "$OUT/ckpt_${STEPS}.pt" ]; then
  uv run python scripts/train_depth_refined_tracker.py --config configs/v62.yaml \
      --out-dir "$OUT" --steps "$STEPS" --waft-pred-dir "$WAFT_TRAIN" \
    || { echo "[v62] TRAIN FAILED"; exit 1; }
fi
CK=$(ls -t "$OUT"/ckpt_*.pt 2>/dev/null | head -1)
[ -n "$CK" ] || { echo "[v62] no checkpoint"; exit 1; }
echo "[v62] trained $(basename "$CK") at $(date -Is)"

uv run python scripts/eval_metric3d.py --method v35 --ckpt "$CK" \
    --da3-depth-root "$HOME/data/tapvid3d_da3nested" --split minival \
    --waft-pred-dir "$WAFT_EVAL" --out-dir "$OUT/eval" \
  || { echo "[v62] EVAL FAILED"; exit 1; }
python3 -c "
import json; d=json.load(open('$OUT/eval/metrics.json')); ps=d['per_subset']; ov=d['overall']
print('[v62] ' + ' '.join(f\"{s}={ps[s]['metric_average_jaccard']:.4f}\" for s in ('drivetrack','pstudio','adt'))
      + f\" mean={ov['metric_average_jaccard']:.4f} frames={d['frames']} fail={d.get('failures')}\")
print('[v62] compare: v60 (with de-flicker) 0.2211 | v59 (1pool, de-flicker) 0.2194 | v45 (SEA-RAFT-trained) 0.2320')"
echo "[v62] done at $(date -Is)"
