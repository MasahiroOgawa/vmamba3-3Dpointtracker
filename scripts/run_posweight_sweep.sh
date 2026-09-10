#!/usr/bin/env bash
# Re-fit the visibility head with class weighting, once the GPU is free.
#
# The head collapsed toward "always visible": it predicted visible 81% of the time on adt where
# truth is 57%, and lost to the flow mask 0.605 vs 0.767. pos_weight was 1.0 against a 57-82%
# positive target, which rewards exactly that. This is the one untried knob and, against the
# cached features, each setting costs a couple of minutes rather than a GPU run.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
# shellcheck source=/dev/null
source "$(dirname "$0")/cudnn_env.sh"
S=result/OVERNIGHT_STATUS.md
note () { printf '%s\n' "$*" | tee -a "$S"; }
idle () { local p a; for p in $(pgrep -f 'train_depth_refined_tracker|eval_metric3d' 2>/dev/null); do
  a=$(tr '\0' '\n' < "/proc/$p/cmdline" 2>/dev/null | head -1)
  case "$a" in */.venv/bin/python*) return 1;; esac; done; return 0; }

while ! idle; do sleep 60; done
note ""; note "### visibility head, class-weight sweep $(date '+%H:%M')  (bar: flow mask 0.767 adt / 0.779 drive / 0.755 pstudio)"
for pw in 0.3 0.5 2.0; do
  log=result/fit_vis_pw$pw.log
  uv run python scripts/fit_vis_head.py --cache result/vis_feature_cache \
    --out "result/vis_head_pw$pw" --steps 4000 --pos-weight "$pw" > "$log" 2>&1 || {
      note "  pos_weight=$pw FAILED"; continue; }
  note "  pos_weight=$pw"
  grep -aE "^  (adt|drivetrack|pstudio) +0\." "$log" | tee -a "$S"
done
note "### sweep done $(date '+%H:%M')  -- a head is only worth evaluating if it beats the flow mask"
