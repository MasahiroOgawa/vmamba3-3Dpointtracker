#!/usr/bin/env bash
# Post-reboot sequence.
#
# 1. v63 re-evaluation. Its training finished (ckpt_20000), but the eval that ran at the end of the
#    job used iters=4 while training had used 5: the job launched before flow.iters was plumbed
#    through to WAFT, so WAFT fell back to tar-c-t.json's default, and the eval -- starting hours
#    later -- picked up the new code and read the config's 4. cfg.json now records 5, the value
#    actually used, so this re-eval is train/eval consistent. The mismatched 0.2238 is kept in
#    eval_iters4_MISMATCHED/ rather than deleted.
#
# 2. v64: WAFT + DA3-l + VSSD-2pool, live on both sides, iters 4 throughout now that the plumbing
#    exists. This is the arm that compares against v50 (SEA-RAFT train + SEA-RAFT eval, 0.2350),
#    the two differing only in the flow network.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
# shellcheck source=/dev/null
source "$(dirname "$0")/cudnn_env.sh"

V63=result/v63_waft_live_2pool
if [ ! -f "$V63/eval/metrics.json" ]; then
  echo "[post] v63 re-eval at iters=5 (matching its training) $(date -Is)"
  uv run python scripts/eval_metric3d.py --method v45 --ckpt "$V63/ckpt_20000.pt" \
      --da3-depth-root "$HOME/data/tapvid3d_da3nested" --split minival \
      --run-cfg "$V63/cfg.json" --out-dir "$V63/eval" \
    || { echo "[post] v63 RE-EVAL FAILED"; exit 1; }
  uv run python -c "
import json; d=json.load(open('$V63/eval/metrics.json')); ps=d['per_subset']
print('[post] v63 (train5/eval5, consistent): ' + ' '.join(f\"{s}={ps[s]['metric_average_jaccard']:.4f}\" for s in ('drivetrack','pstudio','adt'))
      + f\" mean={d['overall']['metric_average_jaccard']:.4f} frames={d['frames']} fail={d.get('failures')}\")
print('[post] the mismatched train5/eval4 run gave 0.2238; v60 (cached, full-res) gave 0.2211')"
fi

echo "[post] starting v64 $(date -Is)"
exec bash scripts/run_v64_waft_live_da3l.sh
