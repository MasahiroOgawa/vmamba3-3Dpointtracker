#!/usr/bin/env bash
# Evaluate v72 and v73, deriving each checkpoint name from its config.
#
# Exists because queue_v72_v73.sh had ckpt_20000.pt hardcoded while the arms were shortened to 3000
# and 14000 steps to meet a deadline, so v72 trained successfully and then failed to score. The
# queue process had already parsed that function into memory, so fixing the file does not help the
# run in flight; this picks up both arms instead.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
# shellcheck source=/dev/null
source "$(dirname "$0")/cudnn_env.sh"
CACHE=$HOME/data/tapvid3d_baseline_preds/waft_minival_is896_s-1_i4

eval_arm () {
  local cfg=$1 method=$2 label=$3 out steps ckpt
  out=$(uv run python -c "import yaml;print(yaml.safe_load(open('$cfg'))['train']['out_dir'])")
  steps=$(uv run python -c "import yaml;print(yaml.safe_load(open('$cfg'))['train']['steps'])")
  ckpt="$out/ckpt_${steps}.pt"
  # wait for training to produce it, but not forever
  for _ in $(seq 1 1440); do
    [ -f "$ckpt" ] && break
    sleep 60
  done
  [ -f "$ckpt" ] || { echo "[$label] checkpoint never appeared: $ckpt"; return 1; }
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

eval_arm configs/v72.yaml v72 v72   # against v69 0.1998 (old stage, standalone)
eval_arm configs/v73.yaml v73 v73   # against v63 0.2238, v45 0.2320, v68 0.2218
echo "[evalq] done at $(date -Is)"
