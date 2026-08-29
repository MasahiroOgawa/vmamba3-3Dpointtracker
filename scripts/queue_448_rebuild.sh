#!/usr/bin/env bash
# Rebuild every mismatched paper row as a self-consistent arm at half resolution.
#
# The rows being replaced were trained on SEA-RAFT tracks and evaluated on WAFT tracks. Half
# resolution is SEA-RAFT's own operating point and where every other row in the tables already sits,
# so moving the WAFT rows there makes each table internally consistent. Full resolution was dropped:
# 8.7x the training cost for at most +0.0013, which is the whole v51-v64 gap.
#
# Evaluation reads a precomputed 448/iters-4 track set. That is not the training cache mistake: at
# evaluation there is no augmentation, so a cache and a live run at the same scale and iteration
# count are the same computation. Step 0 proves it by re-scoring v64, whose published 0.2547 came
# from a live evaluation -- if the cache does not reproduce it, everything downstream is suspect and
# this script stops.
#
# Arms already in hand and NOT rerun: v64 (DA3-l+vmamba3-2pool, 0.2547), v50 (SEA-RAFT counterpart,
# 0.2350), v63 (DA3-g+deflicker+vmamba3-2pool, 0.2238).
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
# shellcheck source=/dev/null
source "$(dirname "$0")/cudnn_env.sh"

CACHE=$HOME/data/tapvid3d_baseline_preds/waft_minival_is896_s-1_i4
until [ "$(find "$CACHE" -name '*.npz' 2>/dev/null | wc -l)" -ge 150 ]; do
  pgrep -f "eval_waft.py --split minival" >/dev/null || { echo "[448] cache generation died"; exit 1; }
  sleep 120
done
echo "[448] cache complete ($(find "$CACHE" -name '*.npz' | wc -l) clips) at $(date -Is)"

# --- step 0: prove cache == live at evaluation, using an arm whose live score is known -----------
V64=result/v64_waft_live_da3l_2pool
if [ ! -f "$V64/eval_from_cache/metrics.json" ]; then
  uv run python scripts/eval_metric3d.py --method v35 --ckpt "$V64/ckpt_20000.pt" \
      --da3-depth-root "$HOME/data/tapvid3d_da3" --split minival \
      --waft-pred-dir "$CACHE" --out-dir "$V64/eval_from_cache" \
    || { echo "[448] verification eval FAILED"; exit 1; }
fi
uv run python - <<'PY' || exit 1
import json, sys
live = json.load(open("result/v64_waft_live_da3l_2pool/eval/metrics.json"))["overall"]["metric_average_jaccard"]
cach = json.load(open("result/v64_waft_live_da3l_2pool/eval_from_cache/metrics.json"))["overall"]["metric_average_jaccard"]
print(f"[448] verification: v64 live {live:.4f} vs from cache {cach:.4f}  delta {cach-live:+.4f}")
if abs(cach - live) > 5e-4:
    print("[448] STOP: the cache does not reproduce the live evaluation; do not trust anything built on it")
    sys.exit(1)
print("[448] cache and live agree -- every eval below can use the cache")
PY

run_arm () {
  local cfg=$1 method=$2 depthkey=$3 label=$4
  local out depth
  out=$(uv run python -c "import yaml;print(yaml.safe_load(open('$cfg'))['train']['out_dir'])")
  depth=$HOME/data/$depthkey
  mkdir -p "$out"
  if [ ! -f "$out/ckpt_20000.pt" ]; then
    echo "[$label] training $(date -Is)"
    uv run python scripts/train_depth_refined_tracker.py --config "$cfg" \
      || { echo "[$label] TRAIN FAILED"; return 1; }
  fi
  if [ ! -f "$out/eval/metrics.json" ]; then
    uv run python scripts/eval_metric3d.py --method "$method" --ckpt "$out/ckpt_20000.pt" \
        --da3-depth-root "$depth" --split minival \
        --waft-pred-dir "$CACHE" --out-dir "$out/eval" \
      || { echo "[$label] EVAL FAILED"; return 1; }
  fi
  uv run python -c "
import json; d=json.load(open('$out/eval/metrics.json')); ps=d['per_subset']
print('[$label] ' + ' '.join(f\"{s}={ps[s]['metric_average_jaccard']:.4f}\" for s in ('drivetrack','pstudio','adt'))
      + f\" mean={d['overall']['metric_average_jaccard']:.4f} frames={d['frames']} fail={d.get('failures')}\")"
}

run_arm configs/v67.yaml v35 tapvid3d_da3       v67   # DA3-l + vmamba3 1-pool     ~8.3 h
run_arm configs/v68.yaml v35 tapvid3d_da3nested v68   # DA3-g + vmamba3-2pool      ~27 h
run_arm configs/v69.yaml v44 tapvid3d_da3nested v69   # DA3-g + de-flicker alone   ~27 h
echo "[448] done at $(date -Is)"
