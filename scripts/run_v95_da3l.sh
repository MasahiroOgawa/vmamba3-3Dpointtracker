#!/usr/bin/env bash
# v94 on the DA3-l line. The DA3-l winner (v64 = WAFT+DA3-l+vmamba3-2pool) carries no scale
# refiner -- that stage exists to correct DA3-g's per-frame drift -- so this is the counterpart
# of the DA3-g arm, not a literal depth swap on the same checkpoint.
#
# The control run comes first and must reproduce v64's recorded 0.2547 abs / 0.1207 norm. If it
# does not, the invocation differs from how v64 was scored and the v95 numbers cannot be read.
set -u
cd "$(dirname "$0")/.." || exit 1
source scripts/cudnn_env.sh
CACHE=$HOME/data/tapvid3d_baseline_preds/waft_minival_is896_s-1_i4
FLOW=$HOME/data/tapvid3d_flowvis/minival
CK=result/v64_waft_live_da3l_2pool/ckpt_20000.pt
S=result/v95_status.log
note() { echo "$*" | tee -a "$S"; }
aj() { python3 -c "
import json;m=json.load(open('$1/metrics.json'))['overall']
print(f\"abs={m['metric_average_jaccard']:.4f} norm={m['average_jaccard']:.4f}\")" 2>/dev/null; }

note ""; note "### v95 on DA3-l (v64) started $(date '+%H:%M')"
note "  recorded v64: abs=0.2547 norm=0.1207"

out=result/v64_waft_live_da3l_2pool/eval_control_flow
note "  control (--vis-source flow) $(date '+%H:%M')"
uv run python scripts/eval_metric3d.py --method v35 --ckpt "$CK" --depth da3l \
    --split minival --waft-pred-dir "$CACHE" --vis-source flow \
    --out-dir "$out" >> result/v95_da3l_control.log 2>&1 \
  && note "    control: $(aj "$out")" || { note "    CONTROL FAILED"; exit 1; }

out=result/v64_waft_live_da3l_2pool/eval_v95
note "  v95 head $(date '+%H:%M')"
uv run python scripts/eval_metric3d.py --method v35 --ckpt "$CK" --depth da3l \
    --split minival --waft-pred-dir "$CACHE" --vis-source v94 \
    --vis-head-ckpt result/v95/best.pt --flowvis-dir "$FLOW" \
    --out-dir "$out" >> result/v95_da3l_eval.log 2>&1 \
  && note "    v95 head: $(aj "$out")" || note "    V95 EVAL FAILED"

note "### v95 DA3-l done $(date '+%H:%M')"
