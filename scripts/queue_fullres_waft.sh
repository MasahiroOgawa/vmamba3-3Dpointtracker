#!/usr/bin/env bash
# WAFT at FULL resolution (scale 0), live for both training and evaluation, on DA3-l then DA3-g.
#
# The half-resolution arms do not answer the question. The standing best is v51 -- SEA-RAFT-trained,
# evaluated with WAFT at full resolution -- at 0.2560, and v64 (WAFT live, half-res both sides)
# reached 0.2547, below it. Since v64's evaluation front-end is weaker than v51's, that comparison
# cannot say whether WAFT is the better front-end. These arms run WAFT at the resolution the best
# result uses, on both sides, so the evaluation matches v51's and train/eval stay consistent.
#
# Targets to beat, both evaluated with full-resolution WAFT:
#   DA3-l 2 pools             v51 0.2560   (SEA-RAFT-trained, WAFT full-res eval)
#   DA3-g de-flicker 1 pool   v45 0.2320   (SEA-RAFT-trained, WAFT full-res eval)
#
# One argument per run, the config path; everything experimental lives in the config.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
# shellcheck source=/dev/null
source "$(dirname "$0")/cudnn_env.sh"

run_arm () {
  local cfg=$1 method=$2 label=$3
  local out depth
  out=$(uv run python -c "import yaml;print(yaml.safe_load(open('$cfg'))['train']['out_dir'])")
  depth=$(uv run python -c "import os,yaml;print(os.path.expanduser(yaml.safe_load(open('$cfg'))['data']['da3_depth_root']))")
  mkdir -p "$out"

  if [ ! -f "$out/ckpt_20000.pt" ]; then
    echo "[$label] training from $cfg at $(date -Is)  (resumes if a checkpoint exists)"
    uv run python scripts/train_depth_refined_tracker.py --config "$cfg" \
      || { echo "[$label] TRAIN FAILED"; return 1; }
  fi

  if [ ! -f "$out/eval/metrics.json" ]; then
    echo "[$label] evaluating at $(date -Is)"
    uv run python scripts/eval_metric3d.py --method "$method" --ckpt "$out/ckpt_20000.pt" \
        --da3-depth-root "$depth" --split minival \
        --run-cfg "$out/cfg.json" --out-dir "$out/eval" \
      || { echo "[$label] EVAL FAILED"; return 1; }
  fi

  uv run python -c "
import json; d=json.load(open('$out/eval/metrics.json')); ps=d['per_subset']
print('[$label] ' + ' '.join(f\"{s}={ps[s]['metric_average_jaccard']:.4f}\" for s in ('drivetrack','pstudio','adt'))
      + f\" mean={d['overall']['metric_average_jaccard']:.4f} frames={d['frames']} fail={d.get('failures')}\")"
}

for _ in $(seq 1 2880); do
  [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c .)" -eq 0 ] && break
  sleep 60
done
echo "[fullres] GPU idle at $(date -Is)"

run_arm configs/v65.yaml v35 v65   # DA3-l + 2pool           -> against v51 0.2560
run_arm configs/v66.yaml v45 v66   # DA3-g + deflicker+2pool -> against v45 0.2320
echo "[fullres] done at $(date -Is)"
