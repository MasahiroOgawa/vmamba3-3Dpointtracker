#!/usr/bin/env bash
# v63: WAFT + DA3-g + de-flicker + VSSD-2pool, with WAFT run LIVE on the augmented images.
#
# This is the arm of most interest, and the one configuration in which the flow network is the only
# variable: the SEA-RAFT-trained runs re-track on the augmented images every step, so a cached WAFT
# track -- computed once from clean images -- is not the same protocol. Both front-ends are consumed
# by the same track_clip, so with the cache gone nothing else differs.
#
# Comparisons it lands against, all WAFT-evaluated on the same tracks:
#   v45  SEA-RAFT live,  1 pool, de-flicker    0.2320   <- the number to beat
#   v59  WAFT cached,    1 pool, de-flicker    0.2194
#   v60  WAFT cached,    2 pool, de-flicker    0.2211   <- same config as this, cached not live
#   v62  WAFT cached,    2 pool, no de-flicker 0.2228
#
# v63 against v60 isolates the augmentation asymmetry exactly: identical config, cached versus live.
#
# Cost: WAFT inference is 9.1x heavier per step than SEA-RAFT's (4 vs 40 steps/min, measured), so
# 20000 steps is about 76 h against v45's 8.3 h.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
# shellcheck source=/dev/null
source "$(dirname "$0")/cudnn_env.sh"
STEPS=${STEPS:-20000}
OUT=result/v63_waft_live_2pool
mkdir -p "$OUT"   # train script writes cfg.json into it before creating it
WAFT_EVAL=$HOME/data/tapvid3d_baseline_preds/waft_minival_cudnn925

for _ in $(seq 1 2880); do
  [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c .)" -eq 0 ] && break
  sleep 60
done
echo "[v63] GPU idle at $(date -Is); ~76 h expected"

if [ ! -f "$OUT/ckpt_${STEPS}.pt" ]; then
  uv run python scripts/train_depth_refined_tracker.py --config configs/v60.yaml \
      --out-dir "$OUT" --steps "$STEPS" --waft-live \
    || { echo "[v63] TRAIN FAILED"; exit 1; }
fi
CK=$(ls -t "$OUT"/ckpt_*.pt 2>/dev/null | head -1)
[ -n "$CK" ] || { echo "[v63] no checkpoint"; exit 1; }
echo "[v63] trained $(basename "$CK") at $(date -Is)"

uv run python scripts/eval_metric3d.py --method v45 --ckpt "$CK" \
    --da3-depth-root "$HOME/data/tapvid3d_da3nested" --split minival \
    --waft-pred-dir "$WAFT_EVAL" --out-dir "$OUT/eval" \
  || { echo "[v63] EVAL FAILED"; exit 1; }
python3 -c "
import json; d=json.load(open('$OUT/eval/metrics.json')); ps=d['per_subset']; ov=d['overall']
print('[v63] ' + ' '.join(f\"{s}={ps[s]['metric_average_jaccard']:.4f}\" for s in ('drivetrack','pstudio','adt'))
      + f\" mean={ov['metric_average_jaccard']:.4f} frames={d['frames']} fail={d.get('failures')}\")
print('[v63] matched comparison: v45 (SEA-RAFT live) 0.2320 | v59 (WAFT cached) 0.2194')"
echo "[v63] done at $(date -Is)"
