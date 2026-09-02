#!/usr/bin/env bash
# The two no-refiner WAFT rows, rescored on the same 448/iters-4 track set as every rebuilt arm.
#
# They need no training -- they are WAFT's 2-D track unprojected with DA3 depth -- but they were
# scored against the OLD track set (image_size 512, scale 0). Leaving them there would put the
# tables back to mixing front-end settings across rows, which is the defect being removed.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
# shellcheck source=/dev/null
source "$(dirname "$0")/cudnn_env.sh"
CACHE=$HOME/data/tapvid3d_baseline_preds/waft_minival_is896_s-1_i4

for spec in "da3l:$HOME/data/tapvid3d_da3" "da3g:$HOME/data/tapvid3d_da3nested"; do
  IFS=: read -r tag depth <<<"$spec"
  out=result/waft_baseline_448_$tag
  [ -f "$out/metrics.json" ] && { echo "[base-$tag] done; skipping"; continue; }
  uv run python scripts/eval_metric3d.py --method searaft --waft-pred-dir "$CACHE" \
      --da3-depth-root "$depth" --split minival --image-size 896 --out-dir "$out" \
    || { echo "[base-$tag] FAILED"; continue; }
  uv run python -c "
import json; d=json.load(open('$out/metrics.json')); ps=d['per_subset']
print('[base-$tag] metricAJ ' + ' '.join(f\"{s}={ps[s]['metric_average_jaccard']:.4f}\" for s in ('drivetrack','pstudio','adt'))
      + f\" mean={d['overall']['metric_average_jaccard']:.4f}\")
print('[base-$tag] AJ       ' + ' '.join(f\"{s}={ps[s]['average_jaccard']:.4f}\" for s in ('drivetrack','pstudio','adt'))
      + f\" mean={d['overall']['average_jaccard']:.4f}\")"
done
echo "[base] done at $(date -Is)"
