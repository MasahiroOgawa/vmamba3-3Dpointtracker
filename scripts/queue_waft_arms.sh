#!/usr/bin/env bash
# The 2x2 that the paper's WAFT rows should have been all along: depth backbone x mixer, with the
# flow front-end held fixed at WAFT for BOTH training and evaluation.
#
#   v53  DA3-l                  VSSD-gamma      (one pool)
#   v54  DA3-l                  VSSD-beta,gamma (two pool)
#   v55  DA3-g + de-flicker     VSSD-gamma      (one pool)
#   v56  DA3-g + de-flicker     VSSD-beta,gamma (two pool)
#
# Why this exists: every WAFT row reported so far was TRAINED on SEA-RAFT and only EVALUATED on
# WAFT. The refiner sees flow during training (track_clip runs in the training loop), so it learns
# the front-end's error characteristics; scoring it with a different front-end is a train/test
# mismatch. It is a mismatch the baselines share, which is why the v51 comparison against 0.2458
# is still one-variable and valid -- but it means none of those numbers answers "how good is this
# pipeline on WAFT", only "how does a SEA-RAFT-trained refiner behave when fed WAFT".
#
# Step 0 is unavoidable: the published WAFT predictions cover minival (150 clips), which is the
# EVALUATION split. Training clips (full_eval, 4419) have no WAFT tracks, and training on the
# minival ones would be training on the test set.
set -uo pipefail
cd "$(dirname "$0")/.."

WAFT_TRAIN=${WAFT_TRAIN:-$HOME/data/tapvid3d_baseline_preds/waft_full_eval}
WAFT_EVAL=${WAFT_EVAL:-$HOME/data/tapvid3d_baseline_preds/waft}
STEPS=${STEPS:-20000}

# Wait for any GPU job, in this repo or the sibling one.
for _ in $(seq 1 1440); do
  pgrep -f "\.venv/bin/python3? -m (depth\.run|eval\.run_cifar)" >/dev/null && { sleep 60; continue; }
  pgrep -f "\.venv/bin/python3? .*(train_depth_refined_tracker|eval_metric3d|eval_waft)" >/dev/null && { sleep 60; continue; }
  break
done
echo "[waft-arms] GPU free at $(date -Is)"

# --- Step 0: generate WAFT tracks for the 4419 training clips. Resumable: eval_waft skips any
# clip whose output already exists, so an interrupted run continues where it stopped.
echo "=== [step 0] WAFT tracks for full_eval ($(date -Is)) ==="
uv run python scripts/eval_waft.py --split full_eval --out-dir "$WAFT_TRAIN" \
  || { echo "[waft-arms] track generation FAILED; arms cannot run"; exit 1; }
n=$(find "$WAFT_TRAIN" -name '*.npz' | wc -l)
echo "[waft-arms] $n/4419 training clips have tracks"
[ "$n" -lt 4000 ] && { echo "[waft-arms] too few tracks; stopping rather than training on a partial set"; exit 1; }

run_arm () {
  local tag=$1 method=$2 depth=$3
  local out="result/${tag}_waft"
  [ -f "$out/eval/summary.md" ] && { echo "[waft-arms] $tag already evaluated; skipping"; return 0; }
  echo "=== [$tag] train $STEPS steps on WAFT ($(date -Is)) ==="
  uv run python scripts/train_depth_refined_tracker.py \
      --config "configs/${tag}.yaml" --out-dir "$out" --steps "$STEPS" \
      --waft-pred-dir "$WAFT_TRAIN" \
    || { echo "[waft-arms] $tag TRAIN FAILED"; return 1; }
  local ck; ck=$(ls -t "$out"/ckpt_*.pt 2>/dev/null | head -1)
  [ -n "$ck" ] || { echo "[waft-arms] $tag produced no checkpoint"; return 1; }
  echo "=== [$tag] eval $(basename "$ck") ($(date -Is)) ==="
  # --waft-pred-dir at eval too: the whole point is that train and test share the front-end.
  uv run python scripts/eval_metric3d.py \
      --method "$method" --ckpt "$ck" --da3-depth-root "$depth" \
      --waft-pred-dir "$WAFT_EVAL" --split minival --out-dir "$out/eval" \
    || { echo "[waft-arms] $tag EVAL FAILED"; return 1; }
  sed -n '/| subset/,/mean/p' "$out/eval/summary.md"
}

# DA3-l pair first: it is the headline backbone, and the one-pool arm is the control that makes
# the two-pool number interpretable, so it runs before its own treatment.
run_arm v53 v35 "$HOME/data/tapvid3d_da3"
run_arm v54 v35 "$HOME/data/tapvid3d_da3"
run_arm v55 v45 "$HOME/data/tapvid3d_da3nested"
run_arm v56 v45 "$HOME/data/tapvid3d_da3nested"

echo "[waft-arms] finished at $(date -Is)"
echo "  reference rows, SEA-RAFT-trained/WAFT-evaluated: DA3-l 0.2458 | DA3-g+de-flicker 0.232"
echo "  the v53..v56 numbers are WAFT-trained AND WAFT-evaluated and are not directly comparable"
echo "  to those; v53 is the control for v54, and v55 for v56."
