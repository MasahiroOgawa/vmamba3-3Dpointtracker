#!/usr/bin/env bash
# One more timing, so Tables 5/6 can quote all rows on ONE accounting. v68 is the same pipeline
# as v93b minus the scale refiner, so v68 -> v93b -> v94 isolates each added stage's cost at a
# single operating point (896 px, 4 iters, flow live and counted, depth cached).
set -u
cd "$(dirname "$0")/.."
source scripts/cudnn_env.sh
S=result/v94_timing.log
out=result/timing_v68
uv run python scripts/eval_metric3d.py --method v35 --ckpt result/v68_waft448_da3g_2pool/ckpt_20000.pt \
    --depth da3g --split minival --max-clips-per-subset 6 --vis-source flow --out-dir "$out" \
    >> result/v94_timing_run.log 2>&1 \
  && echo "  v68 (vmamba3-2pool, no scale refiner):       $(python3 -c "
import json;m=json.load(open('$out/metrics.json'))
print(f\"fps={m['fps']:.2f} frames={m['frames']} elapsed={m['elapsed_s']:.0f}s\")")" | tee -a "$S" \
  || echo "  v68 TIMING FAILED" | tee -a "$S"
echo "### baseline timing done $(date '+%H:%M')" | tee -a "$S"
