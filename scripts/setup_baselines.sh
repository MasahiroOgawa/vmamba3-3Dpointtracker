#!/usr/bin/env bash
# Check out the baseline repositories the comparisons are run against. They are optional
# submodules (`update = none` in .gitmodules), so a recursive clone skips them.
#
#   bash scripts/setup_baselines.sh                   # all baselines
#   bash scripts/setup_baselines.sh SpaTrackerV2      # only the named ones
#
# Each baseline then needs its own venv -- they pin torch/CUDA versions this repository's
# environment cannot share. See "Baselines" in README.md for how each one was built.
#
# LFS smudging is skipped: DenseTrack3Dv2's example media is stored in LFS and one of those
# objects is missing on the upstream server, which otherwise aborts the checkout. No baseline
# needs its example media to run.
set -euo pipefail
cd "$(dirname "$0")/.."

BASELINES=(SEA-RAFT SpaTrackerV2 TrackCraft3R DELTA_densetrack3d DenseTrack3Dv2)
names=("${@:-${BASELINES[@]}}")
for n in "${names[@]}"; do
  [[ " ${BASELINES[*]} " == *" $n "* ]] || { echo "unknown baseline: $n (one of: ${BASELINES[*]})" >&2; exit 1; }
  GIT_LFS_SKIP_SMUDGE=1 git submodule update --init --checkout "third_party/$n"
done
git submodule status "${names[@]/#/third_party/}"
