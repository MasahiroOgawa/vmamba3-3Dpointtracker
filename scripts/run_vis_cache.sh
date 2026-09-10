#!/usr/bin/env bash
# Build the trunk-feature cache, balanced at 3M observations per subset.
# 3M and not more because pstudio has only 106 training clips at ~28.6k
# observations each: it caps at ~3.0M, and matching the others to it keeps the
# mix even rather than letting drivetrack and adt dominate.
# Build the trunk-feature cache. Sources cudnn_env.sh: without it every convolution fails with
# CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
# shellcheck source=/dev/null
source "$(dirname "$0")/cudnn_env.sh"
S=result/OVERNIGHT_STATUS.md
printf '\n### visibility feature cache %s\n' "$(date -Is)" | tee -a "$S"
uv run python scripts/cache_trunk_features.py --config configs/v92.yaml \
  --obs-per-subset 3000000 --out result/vis_feature_cache >> result/vis_cache.log 2>&1
printf '### cache done %s  %s clips, %s\n' "$(date '+%H:%M')" \
  "$(ls -1 result/vis_feature_cache 2>/dev/null | wc -l)" \
  "$(du -sh result/vis_feature_cache 2>/dev/null | cut -f1)" | tee -a "$S"
