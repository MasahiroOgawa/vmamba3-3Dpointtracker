#!/usr/bin/env bash
# v93b: continue v93, the only change that has moved the metric.
#
# v93 gained +0.0064 over its own step-0 (0.2189 -> 0.2253) with the repaired sampler, and is
# 0.0021 from DELTA. Its loss curve was flat the whole way, so this run is judged by scoring
# checkpoints, not by watching the loss -- validation has mispredicted metric-AJ 5 times out of 5
# on this arm.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
# shellcheck source=/dev/null
source "$(dirname "$0")/cudnn_env.sh"
CACHE=$HOME/data/tapvid3d_baseline_preds/waft_minival_is896_s-1_i4
STOP=$(date -d 'tomorrow 06:15' +%s)
S=result/OVERNIGHT_STATUS.md
note () { printf '%s\n' "$*" | tee -a "$S"; }

mkdir -p result/v93b
cp -n result/v93/ckpt_3191.pt result/v93b/     # resume with optimiser and scheduler state intact
rem=$(( (STOP - $(date +%s)) / 5 ))
tot=$(( 3191 + rem ))
uv run python - "$tot" <<'PYEOF'
import sys, yaml, pathlib
tot = int(sys.argv[1])
c = yaml.safe_load(pathlib.Path("configs/v93.yaml").read_text())
c.pop("version", None)
c["train"].update(out_dir="result/v93b", init_ckpt="result/v93/ckpt_3191.pt",
                  steps=tot, decay=max(1, tot // 5), ckpt_every=500, val_every=500)
pathlib.Path("configs/v93b.yaml").write_text(
    "# v93b: v93 continued. Same sampler, same rate; the run resumes from v93/ckpt_3191 with its\n"
    "# optimiser and scheduler state, so the two form one trajectory rather than two fine-tunes.\n"
    + yaml.safe_dump(c, sort_keys=False) + "version: v73\n")
print(f"  v93b -> {tot} total steps")
PYEOF
note ""; note "### v93b start $(date -Is)  continuing v93 (0.2253) toward DELTA 0.2274"
uv run python scripts/train_depth_refined_tracker.py --config configs/v93b.yaml >> result/v93b.log 2>&1 &
tp=$!
while kill -0 $tp 2>/dev/null; do
  [ "$(date +%s)" -ge "$STOP" ] && { note "  [cut] $(date '+%H:%M')"; kill -INT $tp; sleep 90; break; }
  sleep 60
done
wait $tp 2>/dev/null

# Score the last two checkpoints: the loss cannot tell us which is best, only the metric can.
for ck in $(ls -1t result/v93b/ckpt_*.pt 2>/dev/null | head -2); do
  tag=$(basename "$ck" .pt)
  note "  scoring $ck  $(date '+%H:%M')"
  uv run python scripts/eval_metric3d.py --method v73 --ckpt "$ck" --depth da3g --split minival \
      --waft-pred-dir "$CACHE" --out-dir "result/v93b/eval_$tag" >> result/v93b_eval.log 2>&1 \
    || { note "  eval failed: $tag"; continue; }
  uv run python scripts/report_metric_aj.py "result/v93b/eval_$tag" | tee -a "$S"
done
note "### MORNING $(date '+%H:%M')  DELTA 0.2274 | v93 0.2253 | v91 0.2251 | v63 0.2238"
