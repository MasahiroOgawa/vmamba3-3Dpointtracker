#!/usr/bin/env bash
# Two arms, in this order:
#   v72  Mamba3DepthScaleRefiner ALONE          -- isolates the new stage's own ability
#   v73  that stage + vmamba3-2pool             -- the best-DA3-g candidate
#
# v72 is the analogue of v69 (de-flicker standalone, 0.1998) and v73 of v63 (stage + refiner,
# 0.2238), so each has a like-for-like predecessor. Running the standalone first says whether the
# stage itself works before its result is entangled with the appearance refiner: if v72 does not
# clear v69 by a wide margin, the composite's number would be hard to attribute.
#
# Both read the same 448/iters-4 track set as every other rebuilt arm; the manifest is checked.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
# shellcheck source=/dev/null
source "$(dirname "$0")/cudnn_env.sh"
CACHE=$HOME/data/tapvid3d_baseline_preds/waft_minival_is896_s-1_i4

run_arm () {
  local cfg=$1 method=$2 label=$3 out steps ckpt
  out=$(uv run python -c "import yaml;print(yaml.safe_load(open('$cfg'))['train']['out_dir'])")
  # Read the step count from the config rather than assuming 20000: the arms were shortened to meet
  # a deadline, and a hardcoded ckpt_20000.pt made v72's evaluation fail after it had trained.
  steps=$(uv run python -c "import yaml;print(yaml.safe_load(open('$cfg'))['train']['steps'])")
  ckpt="$out/ckpt_${steps}.pt"
  mkdir -p "$out"
  if [ ! -f "$ckpt" ]; then
    echo "[$label] training $(date -Is)"
    uv run python scripts/train_depth_refined_tracker.py --config "$cfg" \
      || { echo "[$label] TRAIN FAILED"; return 1; }
  fi
  if [ ! -f "$out/eval/metrics.json" ]; then
    uv run python scripts/eval_metric3d.py --method "$method" --ckpt "$ckpt" \
        --da3-depth-root "$HOME/data/tapvid3d_da3nested" --split minival \
        --waft-pred-dir "$CACHE" --out-dir "$out/eval" \
      || { echo "[$label] EVAL FAILED"; return 1; }
  fi
  uv run python -c "
import json; d=json.load(open('$out/eval/metrics.json')); ps=d['per_subset']
print('[$label] ' + ' '.join(f\"{s}={ps[s]['metric_average_jaccard']:.4f}\" for s in ('drivetrack','pstudio','adt'))
      + f\" mean={d['overall']['metric_average_jaccard']:.4f} frames={d['frames']} fail={d.get('failures')}\")"
}

for _ in $(seq 1 240); do
  [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c .)" -eq 0 ] && break
  sleep 60
done
echo "[queue] GPU idle at $(date -Is)"

run_arm configs/v72.yaml v72 v72   # against v69 0.1998 (old stage, standalone)
run_arm configs/v73.yaml v73 v73   # against v63 0.2238, v45 0.2320, v68 0.2218
echo "[queue] both done at $(date -Is)"
