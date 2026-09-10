#!/usr/bin/env bash
# After v93b: score the checkpoint its own runner will miss, then sweep the visibility head's
# class weight. Ordered, because both want the GPU.
#
# run_v93b.sh globs result/v93b/ckpt_*.pt, which does not match result/v93b/best/ckpt_*.pt, so
# the best-validation checkpoint would go unscored. On this arm validation has mispredicted
# metric-AJ 5 times out of 5, so "best" is not a reason to skip it -- nor to trust it.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
# shellcheck source=/dev/null
source "$(dirname "$0")/cudnn_env.sh"
CACHE=$HOME/data/tapvid3d_baseline_preds/waft_minival_is896_s-1_i4
S=result/OVERNIGHT_STATUS.md
note () { printf '%s\n' "$*" | tee -a "$S"; }
idle () { local p a; for p in $(pgrep -f 'train_depth_refined_tracker|eval_metric3d' 2>/dev/null); do
  a=$(tr '\0' '\n' < "/proc/$p/cmdline" 2>/dev/null | head -1)
  case "$a" in */.venv/bin/python*) return 1;; esac; done; return 0; }

sleep 120                      # let run_v93b.sh claim the GPU for its own scoring first
while ! idle; do sleep 60; done

for ck in result/v93b/best/ckpt_*.pt; do
  [ -f "$ck" ] || continue
  tag=$(basename "$ck" .pt)
  [ -f "result/v93b/eval_best_$tag/metrics.json" ] && continue
  note "  scoring $ck  $(date '+%H:%M')"
  uv run python scripts/eval_metric3d.py --method v73 --ckpt "$ck" --depth da3g --split minival \
      --waft-pred-dir "$CACHE" --out-dir "result/v93b/eval_best_$tag" >> result/v93b_eval.log 2>&1 \
    && uv run python scripts/report_metric_aj.py "result/v93b/eval_best_$tag" | tee -a "$S"
  while ! idle; do sleep 30; done
done

note ""; note "### visibility head, class-weight sweep $(date '+%H:%M')"
note "    bar to beat, flow mask: adt 0.7666 | drivetrack 0.7790 | pstudio 0.7551"
for pw in 0.3 0.5 2.0; do
  log=result/fit_vis_pw$pw.log
  uv run python scripts/fit_vis_head.py --cache result/vis_feature_cache \
    --out "result/vis_head_pw$pw" --steps 4000 --pos-weight "$pw" > "$log" 2>&1 \
    || { note "  pos_weight=$pw FAILED"; continue; }
  note "  pos_weight=$pw"
  grep -aE "^  (adt|drivetrack|pstudio) +0\." "$log" | tee -a "$S"
done
note "### morning chain done $(date '+%H:%M')"
