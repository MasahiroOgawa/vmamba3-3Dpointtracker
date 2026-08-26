#!/usr/bin/env bash
# v63: WAFT + DA3-g + de-flicker + VSSD-2pool, WAFT run LIVE on the augmented images.
#
# Everything experimental is in configs/v63.yaml -- front-end, output directory, step count -- and
# the config path is the only argument this script passes. cfg.json is then a complete record of
# what produced the numbers, which a command-line flag would not be.
#
# v63 differs from v60 by one line, flow.source: waft_live vs waft_cached, so the pair isolates the
# augmentation asymmetry: the SEA-RAFT path re-tracks on the augmented images every step, while a
# cached track is computed once from clean images and never moves.
#
# Comparisons, all WAFT-evaluated on the same tracks:
#   v45  SEA-RAFT live,  1 pool, de-flicker    0.2320   <- the number to beat
#   v59  WAFT cached,    1 pool, de-flicker    0.2194
#   v60  WAFT cached,    2 pool, de-flicker    0.2211   <- identical config, cached not live
#   v62  WAFT cached,    2 pool, no de-flicker 0.2228
#
# The earlier version of this run had WAFT at its own default scale=0, i.e. 896x896, against
# SEA-RAFT's spring-M scale=-1, i.e. 448x448 -- 4x the pixels and 4x the ViT tokens. That made
# WAFT look 9.1x heavier per step and, worse, meant the arms differed in more than the front-end.
# flow.scale is now -1 for both, so this compares flow networks at one operating point.
#
# Nothing is precomputed. Training runs WAFT on each augmented window as SEA-RAFT is run, and the
# evaluation runs it live too, reading flow.source and flow.scale from this run's own cfg.json --
# so a cache cannot silently disagree with what training saw, at any resolution.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
# shellcheck source=/dev/null
source "$(dirname "$0")/cudnn_env.sh"

OUT=$(uv run python -c "import yaml;print(yaml.safe_load(open('configs/v63.yaml'))['train']['out_dir'])")
mkdir -p "$OUT"

for _ in $(seq 1 2880); do
  [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c .)" -eq 0 ] && break
  sleep 60
done
echo "[v63] GPU idle at $(date -Is); ~76 h expected; out=$OUT"

uv run python scripts/train_depth_refined_tracker.py --config configs/v63.yaml \
  || { echo "[v63] TRAIN FAILED"; exit 1; }

CK=$(ls -t "$OUT"/ckpt_*.pt 2>/dev/null | head -1)
[ -n "$CK" ] || { echo "[v63] no checkpoint"; exit 1; }
echo "[v63] trained $(basename "$CK") at $(date -Is)"

uv run python scripts/eval_metric3d.py --method v45 --ckpt "$CK" \
    --da3-depth-root "$HOME/data/tapvid3d_da3nested" --split minival \
    --run-cfg "$OUT/cfg.json" --out-dir "$OUT/eval" \
  || { echo "[v63] EVAL FAILED"; exit 1; }
python3 -c "
import json; d=json.load(open('$OUT/eval/metrics.json')); ps=d['per_subset']; ov=d['overall']
print('[v63] ' + ' '.join(f\"{s}={ps[s]['metric_average_jaccard']:.4f}\" for s in ('drivetrack','pstudio','adt'))
      + f\" mean={ov['metric_average_jaccard']:.4f} frames={d['frames']} fail={d.get('failures')}\")
print('[v63] matched comparison: v45 0.2320 | v60 (same config, cached) 0.2211')"
echo "[v63] done at $(date -Is)"
