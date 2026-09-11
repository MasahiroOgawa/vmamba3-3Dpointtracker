#!/usr/bin/env bash
# Stage 2 of v94: wait for the minival flow cache, then score v93b's ckpt_8297 with the
# learned visibility in place of the forward-backward mask.
#
# The refiner is frozen and --vis-source changes only the visibility that is *reported*:
# the mask is still what feeds the trunk's 4th channel and gates its pooling, so the 3-D
# positions are identical to the 0.2274 baseline and the comparison isolates visibility.
set -u
cd "$(dirname "$0")/.."
source scripts/cudnn_env.sh
CACHE=$HOME/data/tapvid3d_baseline_preds/waft_minival_is896_s-1_i4
FLOW=$HOME/data/tapvid3d_flowvis/minival
CK=result/v93b/ckpt_8297.pt
S=result/v94_status.log
note() { echo "$*" | tee -a "$S"; }

note ""; note "### v94 stage 2 started $(date '+%H:%M')"
while pgrep -f '[c]ache_flow_vis.py' >/dev/null; do sleep 60; done
note "  minival flow cache finished $(date '+%H:%M'): $(for ss in adt drivetrack pstudio; do
  printf '%s=%s ' "$ss" "$(ls $FLOW/$ss 2>/dev/null | wc -l)"; done)"
note "  mask mismatches over all minival clips: $(grep -c 'MASK MISMATCH' result/v94_cache.log || echo 0)"
note "  baseline (--vis-source flow): drivetrack=0.1631 pstudio=0.2092 adt=0.3100 mean=0.2274"

for tag in causal noncausal; do
  [ -f "result/v94_$tag/best.pt" ] || { note "  no head for $tag"; continue; }
  out="result/v93b/eval_v94_$tag"
  note "  scoring v94_$tag $(date '+%H:%M')"
  uv run python scripts/eval_metric3d.py --method v73 --ckpt "$CK" --depth da3g \
      --split minival --waft-pred-dir "$CACHE" --vis-source v94 \
      --vis-head-ckpt "result/v94_$tag/best.pt" --flowvis-dir "$FLOW" \
      --out-dir "$out" >> "result/v94_eval_$tag.log" 2>&1 \
    || { note "  EVAL FAILED: $tag (see result/v94_eval_$tag.log)"; continue; }
  note "    $(uv run python scripts/report_metric_aj.py "$out")"
done
note "### v94 stage 2 done $(date '+%H:%M')"
