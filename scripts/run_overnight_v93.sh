#!/usr/bin/env bash
# Overnight: two independent shots at DELTA's 0.2274 on DA3-g.
#
#  A. The visibility head, fitted on cached trunk features. adt's whole deficit is that channel
#     (+0.1146 with an oracle, against a 0.0196 gap), and the head is only useful if it agrees
#     with ground truth MORE OFTEN than the flow mask, which is wrong on 20-23% of observations.
#     Costs minutes, so it runs first and is discarded on the spot if it loses to the mask.
#
#  B. v93: the depth path retrained with data.reanchor_window ON. Every prior arm was supervised
#     by a median of 8 tracks per step because the sampler kept only queries anchored inside the
#     8-frame window; it is now ~120. That is a 15x change in supervision and it has never been
#     run for the depth path. Highest-value untested change, so it gets the night.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
# shellcheck source=/dev/null
source "$(dirname "$0")/cudnn_env.sh"
CACHE=$HOME/data/tapvid3d_baseline_preds/waft_minival_is896_s-1_i4
STOP=$(date -d 'tomorrow 05:40' +%s)
S=result/OVERNIGHT_STATUS.md
note () { printf '%s\n' "$*" | tee -a "$S"; }
idle () { local p a; for p in $(pgrep -f 'train_depth_refined_tracker|cache_trunk_features' 2>/dev/null); do
  a=$(tr '\0' '\n' < "/proc/$p/cmdline" 2>/dev/null | head -1)
  case "$a" in */.venv/bin/python*) return 1;; esac; done; return 0; }

while ! idle; do sleep 30; done          # wait for the feature cache to finish
note ""; note "## night $(date -Is)   stop $(date -d @"$STOP" '+%H:%M')"

# ---- A. visibility head ----------------------------------------------------
note "### A: fit the visibility head on cached features $(date '+%H:%M')"
uv run python scripts/fit_vis_head.py --cache result/vis_feature_cache \
  --out result/vis_head --steps 4000 >> result/fit_vis_head.log 2>&1 || note "  fit FAILED"
grep -aE "^  (adt|drivetrack|pstudio) " result/fit_vis_head.log | tail -3 | tee -a "$S"

# ---- B. v93: depth path with the repaired sampler --------------------------
rem=$(( (STOP - $(date +%s)) / 12 ))
[ "$rem" -lt 300 ] && rem=300
uv run python - "$rem" <<'PYEOF'
import sys, yaml, pathlib
steps = int(sys.argv[1])
c = yaml.safe_load(pathlib.Path("configs/v91.yaml").read_text())
c.pop("version", None)
c["data"]["reanchor_window"] = True
c["train"].update(out_dir="result/v93", init_ckpt="result/v73fix_warmstart/ckpt_0.pt",
                  lr=5.0e-5, warmup=0, steps=steps, decay=max(1, steps // 5),
                  log_every=25, val_every=250, ckpt_every=250, val_at_step0=True,
                  early_stop_patience=10**6)
pathlib.Path("configs/v93.yaml").write_text(
    "# v93: v91's arm with data.reanchor_window ON -- the first depth run to see the full\n"
    "# supervision. Every earlier arm was fed a median of 8 tracks per step; it is now ~120.\n"
    "# Same warm start, rate and schedule shape as v91 so the sampler is the only difference.\n"
    + yaml.safe_dump(c, sort_keys=False) + "version: v73\n")
print(f"  v93: {steps} steps")
PYEOF
note "### B: v93 start $(date -Is)  reanchor_window ON"
uv run python scripts/train_depth_refined_tracker.py --config configs/v93.yaml \
  >> result/v93.log 2>&1 &
tp=$!
while kill -0 $tp 2>/dev/null; do
  [ "$(date +%s)" -ge "$STOP" ] && { note "  [cut] $(date '+%H:%M')"; kill -INT $tp; sleep 60; break; }
  sleep 60
done
wait $tp 2>/dev/null

# ---- score every checkpoint v93 left, best-first is decided by measurement --
for ck in $(ls -1t result/v93/ckpt_*.pt result/v93/best/ckpt_*.pt 2>/dev/null | head -2); do
  tag=$(basename "$ck" .pt)
  [ -f "result/v93/eval_$tag/metrics.json" ] && continue
  note "  scoring $ck  $(date '+%H:%M')"
  uv run python scripts/eval_metric3d.py --method v73 --ckpt "$ck" --depth da3g \
      --split minival --waft-pred-dir "$CACHE" --out-dir "result/v93/eval_$tag" \
    >> result/v93_eval.log 2>&1 || { note "  eval failed for $tag"; continue; }
  uv run python scripts/report_metric_aj.py "result/v93/eval_$tag" | tee -a "$S"
done
note "### MORNING $(date '+%H:%M')  target DELTA 0.2274 | v63 0.2238 | v91 best 0.2251 | warm start 0.2191"
