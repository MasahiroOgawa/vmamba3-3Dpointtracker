#!/usr/bin/env bash
# Cache the v95 flow pool, then train the visibility head on it. One command, because the
# cache is ~4 days and nobody should have to be present when it ends.
#
# The cache is resumable (it skips any clip already written), so this script is safe to
# re-run after an interruption: it picks up where it stopped and only then trains.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python

echo "[chain] $(date -Is) caching -> configs/v95_cache.yaml"
"$PY" scripts/cache_flow_vis.py configs/v95_cache.yaml

echo "[chain] $(date -Is) training -> configs/v95.yaml"
"$PY" scripts/train_flow_vis_head.py configs/v95.yaml

echo "[chain] $(date -Is) done"
