#!/usr/bin/env bash
# v64: WAFT + DA3-l + VSSD-2pool, WAFT run LIVE for both training and evaluation.
#
# The comparison this exists for, the front-end being the only difference:
#   v50  SEA-RAFT train + SEA-RAFT eval, scale -1, iters 4   0.2350
#   v64  WAFT     train + WAFT     eval, scale -1, iters 4   <- this
#
# Everything experimental is in configs/v64.yaml; the config path is the only argument passed.
# Nothing is precomputed: the evaluator reads flow.source and flow.scale from this run's cfg.json
# and runs WAFT live on the evaluation frames, so no cache can disagree with what training saw.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
# shellcheck source=/dev/null
source "$(dirname "$0")/cudnn_env.sh"

OUT=$(uv run python -c "import yaml;print(yaml.safe_load(open('configs/v64.yaml'))['train']['out_dir'])")
# Depth cache read from the same config the training used, so eval cannot silently score a
# DA3-l arm against the DA3-g cache.
DEPTH=$(uv run python -c "import os,yaml;print(os.path.expanduser(yaml.safe_load(open('configs/v64.yaml'))['data']['da3_depth_root']))")
mkdir -p "$OUT"

for _ in $(seq 1 2880); do
  [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c .)" -eq 0 ] && break
  sleep 60
done
echo "[v64] GPU idle at $(date -Is); ~76 h expected; out=$OUT"

uv run python scripts/train_depth_refined_tracker.py --config configs/v64.yaml \
  || { echo "[v64] TRAIN FAILED"; exit 1; }

CK=$(ls -t "$OUT"/ckpt_*.pt 2>/dev/null | head -1)
[ -n "$CK" ] || { echo "[v64] no checkpoint"; exit 1; }
echo "[v64] trained $(basename "$CK") at $(date -Is)"

uv run python scripts/eval_metric3d.py --method v35 --ckpt "$CK" \
    --da3-depth-root "$DEPTH" --split minival \
    --run-cfg "$OUT/cfg.json" --out-dir "$OUT/eval" \
  || { echo "[v64] EVAL FAILED"; exit 1; }
python3 -c "
import json; d=json.load(open('$OUT/eval/metrics.json')); ps=d['per_subset']; ov=d['overall']
print('[v64] ' + ' '.join(f\"{s}={ps[s]['metric_average_jaccard']:.4f}\" for s in ('drivetrack','pstudio','adt'))
      + f\" mean={ov['metric_average_jaccard']:.4f} frames={d['frames']} fail={d.get('failures')}\")
print('[v64] matched comparison: v45 0.2320 | v60 (same config, cached) 0.2211')"
echo "[v64] done at $(date -Is)"
