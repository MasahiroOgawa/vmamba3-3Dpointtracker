#!/usr/bin/env bash
# After v91's own evaluation of its final checkpoint, spend what is left of the window scoring
# earlier checkpoints. Checkpoint choice cannot be left to the validation loss here: on this line
# it picked the worse checkpoint three times out of three, so the best is found by measuring.
# Each scoring is ~26 min, so one is only started if it can finish before the hard stop.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
# shellcheck source=/dev/null
source "$(dirname "$0")/cudnn_env.sh"
CACHE=$HOME/data/tapvid3d_baseline_preds/waft_minival_is896_s-1_i4
STOP=$(date -d "$1" +%s)
S=result/OVERNIGHT_STATUS.md
note () { printf '%s\n' "$*" | tee -a "$S"; }

for _ in $(seq 1 240); do [ -f result/v91/eval_final/metrics.json ] && break; sleep 30; done
[ -f result/v91/eval_final/metrics.json ] || { note "  [extra] v91 final eval never landed"; exit 1; }

for st in 1000 800; do
  ck=result/v91/ckpt_$st.pt
  [ -f "$ck" ] || continue
  [ -f "result/v91/eval_$st/metrics.json" ] && continue
  if [ "$(( $(date +%s) + 1800 ))" -ge "$STOP" ]; then
    note "  [extra] not enough window left for ckpt_$st; stopping here"; break
  fi
  note "  [extra] scoring ckpt_$st at $(date '+%H:%M')"
  uv run python scripts/eval_metric3d.py --method v73 --ckpt "$ck" --depth da3g \
      --split minival --waft-pred-dir "$CACHE" --out-dir "result/v91/eval_$st" \
    || { note "  [extra] ckpt_$st eval failed"; continue; }
  uv run python scripts/report_metric_aj.py "result/v91/eval_$st" | tee -a "$S"
done
note "### v91 all scored checkpoints $(date '+%H:%M')"
for d in result/v91/eval_400 result/v91/eval_800 result/v91/eval_1000 result/v91/eval_final; do
  [ -f "$d/metrics.json" ] && uv run python scripts/report_metric_aj.py "$d" | tee -a "$S"
done
