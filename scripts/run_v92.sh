#!/usr/bin/env bash
# v92: train only the visibility head, then score adt with it.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
# shellcheck source=/dev/null
source "$(dirname "$0")/cudnn_env.sh"
CACHE=$HOME/data/tapvid3d_baseline_preds/waft_minival_is896_s-1_i4
S=result/OVERNIGHT_STATUS.md
note () { printf '%s\n' "$*" | tee -a "$S"; }

note ""; note "### v92 (visibility head only) $(date -Is)"
uv run python scripts/train_depth_refined_tracker.py --config configs/v92.yaml \
  >> result/v92.log 2>&1 || note "  v92 exited nonzero; scoring what exists"

ck=$(ls -1t result/v92/ckpt_*.pt result/v92/best/ckpt_*.pt 2>/dev/null | head -1)
[ -n "$ck" ] || { note "  v92 produced no checkpoint"; exit 1; }
note "  scoring $ck"

# adt with the learned head -- the number the whole experiment is about.
uv run python scripts/eval_metric3d.py --method v73 --ckpt "$ck" --depth da3g --split minival \
    --subsets adt --waft-pred-dir "$CACHE" --vis-source model \
    --out-dir result/v92/eval_adt_model >> result/v92_eval.log 2>&1 || note "  adt/model EVAL FAILED"
[ -f result/v92/eval_adt_model/metrics.json ] && uv run python -c "
import json
m=json.load(open('result/v92/eval_adt_model/metrics.json'))['per_subset']['adt']
print(f\"  adt, learned visibility : AJ {m['metric_average_jaccard']:.4f}  pts {m['metric_average_pts_within_thresh']:.4f}  occ {m['occlusion_accuracy']:.4f}\")" | tee -a "$S"

# The same checkpoint on the flow mask, to confirm the depth path really is untouched.
uv run python scripts/eval_metric3d.py --method v73 --ckpt "$ck" --depth da3g --split minival \
    --subsets adt --waft-pred-dir "$CACHE" \
    --out-dir result/v92/eval_adt_flow >> result/v92_eval.log 2>&1 || note "  adt/flow EVAL FAILED"
[ -f result/v92/eval_adt_flow/metrics.json ] && uv run python -c "
import json
m=json.load(open('result/v92/eval_adt_flow/metrics.json'))['per_subset']['adt']
print(f\"  adt, flow mask (control): AJ {m['metric_average_jaccard']:.4f}  pts {m['metric_average_pts_within_thresh']:.4f}  occ {m['occlusion_accuracy']:.4f}\")" | tee -a "$S"

note "### v92 done $(date '+%H:%M')   adt bars: DELTA 0.3244 | v91 flow 0.3062 | oracle ceiling 0.4194"
