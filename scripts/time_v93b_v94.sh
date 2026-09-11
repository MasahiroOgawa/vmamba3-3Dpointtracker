#!/usr/bin/env bash
# Throughput for the paper's Tables 5/6 under the paper's own accounting: optical flow IS
# counted (run live from the run's cfg.json), metric depth is NOT (read from the offline cache),
# matching the footnotes on the existing rows. fps is a rate, so a clip subset is representative
# and avoids two full 2.8 h live-flow passes.
set -u
cd "$(dirname "$0")/.." || exit 1
source scripts/cudnn_env.sh
N=6                       # clips per subset
CK=result/v93b/ckpt_8297.pt
FLOW=$HOME/data/tapvid3d_flowvis/minival
S=result/v94_timing.log
note() { echo "$*" | tee -a "$S"; }
fps() { python3 -c "
import json;m=json.load(open('$1/metrics.json'))
print(f\"fps={m['fps']:.2f} frames={m['frames']} elapsed={m['elapsed_s']:.0f}s\")" 2>/dev/null; }

note "### timing started $(date '+%H:%M')  ($N clips/subset, flow live, depth cached)"

out=result/timing_v93b
uv run python scripts/eval_metric3d.py --method v73 --ckpt "$CK" --depth da3g --split minival \
    --max-clips-per-subset $N --vis-source flow --out-dir "$out" \
    >> result/v94_timing_run.log 2>&1 \
  && note "  v93b (scale refiner + vmamba3-2pool):        $(fps $out)" \
  || note "  v93b TIMING FAILED"

out=result/timing_v94
uv run python scripts/eval_metric3d.py --method v73 --ckpt "$CK" --depth da3g --split minival \
    --max-clips-per-subset $N --vis-source v94 \
    --vis-head-ckpt result/v94_noncausal/best.pt --flowvis-dir "$FLOW" --out-dir "$out" \
    >> result/v94_timing_run.log 2>&1 \
  && note "  v93b + v94 visibility head:                  $(fps $out)" \
  || note "  v94 TIMING FAILED"

note "### timing done $(date '+%H:%M')   paper's comparable rows: WAFT+DA3-g+vmamba3 ~6.3, DELTA+DA3-g ~7.7"
