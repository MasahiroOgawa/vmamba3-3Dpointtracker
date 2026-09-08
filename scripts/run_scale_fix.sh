#!/usr/bin/env bash
# Retrain the DA3-g scale refiner against the CORRECTED target, then verify and score it.
# Gated: if it still does not reduce the minival scale error, do not spend an eval on it.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
# shellcheck source=/dev/null
source "$(dirname "$0")/cudnn_env.sh"
export DA3_ROOT="$HOME/data/tapvid3d_da3nested"
S=result/OVERNIGHT_STATUS.md
note () { printf '%s\n' "$*" | tee -a "$S"; }

note ""
note "### RETRAIN with the corrected target $(date -Is)"
uv run python scripts/train_scale_standalone.py --out result/scale_standalone_da3g_fixed \
  2>&1 | grep -E "depth root|cache|cached|train /|step|best val" | tee -a "$S"

note "  [$(date '+%H:%M')] verifying on MINIVAL: does it reduce the scale error now?"
uv run python scripts/check_scale_on_minival.py \
    --ckpt result/scale_standalone_da3g_fixed/best.pt 2>&1 | grep -vE "xFormers|Loading|it/s" | tee -a "$S"
