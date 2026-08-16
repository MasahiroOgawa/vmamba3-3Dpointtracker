#!/usr/bin/env bash
# v56 = DA3-g + de-flicker + VSSD-2pool: the DA3-g counterpart of v50/v51.
#
# This arm was documented as "arm B" in queue_twopool_arms.sh but that script only ever invoked
# arm A, so it never ran. configs/v56.yaml is configs/v45.yaml plus two_pool: true, exactly as
# v50.yaml is v35.yaml plus two_pool.
#
# PROTOCOL, and why it is not the one queue_waft_arms.sh describes: the comparison target is v45's
# WAFT-evaluated 0.2320, which was TRAINED on SEA-RAFT and only EVALUATED on WAFT. v50/v51 were
# produced the same way. Training this arm on WAFT instead would change two variables at once and
# the resulting number could not be placed beside 0.2320. So: train without --waft-pred-dir, then
# evaluate twice -- once without it (the SEA-RAFT row, comparable to v50) and once with it (the
# WAFT row, comparable to v51 and to v45's 0.2320).
#
# Cold start, full 20000 steps, identical to v45 in every respect except the second pool, so any
# difference is attributable to the operator rather than to extra training.
#
# The refiner now runs the collapse path, which is linear in track length, so the adt
# out-of-memory failures that truncated v50/v51's first evaluation cannot recur here.
set -uo pipefail
cd "$(dirname "$0")/.."

STEPS=${STEPS:-20000}
DA3G="$HOME/data/tapvid3d_da3nested"
WAFT="$HOME/data/tapvid3d_baseline_preds/waft"
OUT=result/v56_da3g_twopool

[ -f configs/v56.yaml ] || { echo "[v56] configs/v56.yaml missing"; exit 1; }
[ -d "$DA3G" ] || { echo "[v56] DA3-g depth cache missing: $DA3G"; exit 1; }

for _ in $(seq 1 1440); do
  [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c .)" -eq 0 ] && break
  sleep 60
done
echo "[v56] GPU idle at $(date -Is); training $STEPS steps"

if [ ! -f "$OUT/ckpt_${STEPS}.pt" ]; then
  uv run python scripts/train_depth_refined_tracker.py \
      --config configs/v56.yaml --out-dir "$OUT" --steps "$STEPS" \
    || { echo "[v56] TRAIN FAILED"; exit 1; }
fi
CK=$(ls -t "$OUT"/ckpt_*.pt 2>/dev/null | head -1)
[ -n "$CK" ] || { echo "[v56] no checkpoint produced"; exit 1; }
echo "[v56] trained: $(basename "$CK") at $(date -Is)"

evaluate () {
  local tag=$1; shift
  local out="$OUT/eval_$tag"
  [ -f "$out/metrics.json" ] && { echo "[v56] $tag already evaluated"; return 0; }
  uv run python scripts/eval_metric3d.py --method v45 --ckpt "$CK" \
      --da3-depth-root "$DA3G" --split minival --out-dir "$out" "$@" \
    || { echo "[v56] eval $tag FAILED"; return 1; }
  python3 - "$out/metrics.json" "$tag" <<'PY'
import json, sys
d = json.load(open(sys.argv[1])); f = d.get("failures")
n = len(f) if isinstance(f, list) else f
ps, ov = d["per_subset"], d["overall"]
print(f"[v56] {sys.argv[2]}: frames={d['frames']} failures={n} fps={d['fps']:.2f}")
print("[v56]   " + " ".join(f"{s}={ps[s]['metric_average_jaccard']:.4f}"
                            for s in ("drivetrack","pstudio","adt"))
      + f" mean={ov['metric_average_jaccard']:.4f}")
PY
}

evaluate searaft
evaluate waft --waft-pred-dir "$WAFT"

echo "[v56] done at $(date -Is)"
echo "  compare the WAFT row against v45 = 0.2320 (same protocol, one pool)."
echo "  frames must read 25814; anything short means clips failed and the mean is not comparable."
