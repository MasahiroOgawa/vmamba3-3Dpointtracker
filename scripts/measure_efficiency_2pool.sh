#!/usr/bin/env bash
# Throughput for the VSSD-2pool arms, on the protocol the efficiency tables already use: run
# eval_metric3d.py over all of minival and read `fps` out of metrics.json. Every existing row in
# those tables came from exactly that field (SEA-RAFT baseline 12.45, +mamba3 12.51, +vmamba3 7.74
# on the 3314-frame drivetrack subset), so a comparable 2-pool number has to come from there too.
#
# The v50 eval already in result/ cannot supply it: it ran while ETH3D training held the GPU and
# covered 15914 of 25814 frames with 33 failures, giving 4.12 fps against sibling runs of the same
# method at 10-62. v51's 30.51 is a different basis again -- it reads WAFT flow from cache, so the
# front-end cost is excluded. Neither is comparable to the published rows.
#
# Contention is the whole point of this script, so it waits for an idle GPU and refuses to start
# while anything else is resident.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

STEPS_DIR=result/efficiency_2pool
mkdir -p "$STEPS_DIR"

gpu_busy () {
  local n
  n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c . || echo 0)
  [ "$n" -gt 0 ]
}

for _ in $(seq 1 1440); do
  gpu_busy || break
  sleep 60
done
gpu_busy && { echo "[eff] GPU still busy after 24 h; aborting rather than reporting a contended fps"; exit 1; }
echo "[eff] GPU idle at $(date -Is)"

# Both arms share one checkpoint; they differ only in the flow front-end, which is what the fps
# column is mostly measuring.
CK=result/v50_twopool/ckpt_20000.pt
[ -f "$CK" ] || { echo "[eff] missing $CK"; exit 1; }

run () {
  local tag=$1; shift
  local out="$STEPS_DIR/$tag"
  [ -f "$out/metrics.json" ] && { echo "[eff] $tag done; skipping"; return 0; }
  echo "=== [$tag] $(date -Is) ==="
  uv run python scripts/eval_metric3d.py --method v35 --ckpt "$CK" \
      --da3-depth-root "$HOME/data/tapvid3d_da3" --split minival --out-dir "$out" "$@" \
    || { echo "[eff] $tag FAILED"; return 1; }
  python3 -c "
import json; d=json.load(open('$out/metrics.json'))
print(f\"[eff] $tag fps={d['fps']:.2f} frames={d['frames']} failures={len(d.get('failures') or [])}\")"
}

run searaft_2pool
run waft_2pool --waft-pred-dir "$HOME/data/tapvid3d_baseline_preds/waft"

echo "[eff] done at $(date -Is)"
echo "  a row is only usable if frames=25814; anything short means clips failed and the fps is"
echo "  an average over a different workload than the published rows."
