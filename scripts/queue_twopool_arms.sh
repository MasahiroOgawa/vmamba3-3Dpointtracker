#!/usr/bin/env bash
# Overnight: swap the refiner's mixer to VSSD-beta,gamma (two-pool) in the two best pipelines,
# train them from scratch, and evaluate on TAPVid-3D minival.
#
#   arm A  v50 = v35 + two_pool   WAFT + DA3-l + vmamba3                published mean metric-AJ 0.246
#   arm B  v51 = v45 + two_pool   WAFT + DA3-g + de-flicker + vmamba3    published mean metric-AJ 0.232
#
# COLD START, full 20000 steps, deliberately. Warm-starting from the published checkpoint and
# fine-tuning 8000 steps was the cheap option and it is confounded: the two-pool model would have
# seen 20000 one-pool steps PLUS 8000 two-pool steps = 28000, against the baseline's 20000, so any
# gain could be the extra training rather than the second pool. Identical config, identical steps,
# identical data, two_pool the only difference, is the only comparison that attributes the result
# to the operator. The cost is ~9.5 h per arm, so only the first finishes overnight; the second is
# queued behind it and completes the next day. One attributable number beats two ambiguous ones.
#
# pool2_gate still starts at zero. At cold start that is an init choice, not a confound: the second
# pool begins inert and has to earn its contribution. Gradient does reach the gate, because m2's
# zero-initialised bias gives softplus(0)=0.693, a nonzero pool output.
#
# The submodule third_party/visionMamba3 is deliberately NOT bumped to the parent repo's HEAD.
# Between its pinned commit and HEAD, projections.py and self_attention.py change substantially,
# which could shift the operator numerically and invalidate the very baselines being compared
# against. Only the two_pool addition is applied, on a branch off the pinned commit.
set -uo pipefail
cd "$(dirname "$0")/.."

STEPS=${STEPS:-20000}
LSSM=/home/mas/proj/study/largescale3Dreconstruction_using_SSM
# Baselines to compare against, NOT used for initialisation -- recorded so the comparison target
# is unambiguous and so a later warm-start experiment knows where they are.
A_BASE=$LSSM/result/20260701_v35/ckpt_20000.pt        # mean metric-AJ 0.246
B_BASE=result/v45_scale/ckpt_20000.pt                 # mean metric-AJ 0.232

# Wait for the GPU. Match the venv interpreter, not a bare command substring, so this does not
# see its own shell wrapper and exit immediately.
for _ in $(seq 1 900); do
  pgrep -f "\.venv/bin/python3? -m (depth\.run|eval\.run_cifar)" >/dev/null || break
  pgrep -f "\.venv/bin/python3? .*train_depth_refined_tracker" >/dev/null || sleep 5
  sleep 60
done
echo "[twopool] GPU free at $(date -Is); steps=$STEPS per arm"

run_arm () {
  local tag=$1 cfg=$2 method=$3 depth=$4
  local out="result/${tag}_twopool"
  if [ -f "$out/eval/summary.md" ]; then
    echo "[twopool] $tag already evaluated; skipping"; return 0
  fi
  echo "=== [$tag] train $STEPS steps from scratch ($(date -Is)) ==="
  uv run python scripts/train_depth_refined_tracker.py \
      --config "$cfg" --out-dir "$out" --steps "$STEPS" \
    || { echo "[twopool] $tag TRAIN FAILED"; return 1; }

  # newest checkpoint, whatever step it landed on
  local ck; ck=$(ls -t "$out"/ckpt_*.pt 2>/dev/null | head -1)
  [ -n "$ck" ] || { echo "[twopool] $tag no checkpoint produced"; return 1; }

  echo "=== [$tag] eval $(basename "$ck") on minival ($(date -Is)) ==="
  uv run python scripts/eval_metric3d.py \
      --method "$method" --ckpt "$ck" --da3-depth-root "$depth" \
      --split minival --out-dir "$out/eval" \
    || { echo "[twopool] $tag EVAL FAILED"; return 1; }
  echo "=== [$tag] done ($(date -Is)) ==="
  sed -n '/| subset/,/mean/p' "$out/eval/summary.md" 2>/dev/null
}

# Arm A first: it is the headline configuration, so if only one finishes it should be this one.
run_arm v50 configs/v50.yaml v35 "$HOME/data/tapvid3d_da3"
run_arm v51 configs/v51.yaml v45 "$HOME/data/tapvid3d_da3nested"

echo "[twopool] all arms finished at $(date -Is)"
echo "  compare against: v35/DA3-l = 0.246 | v45/DA3-g+de-flicker = 0.232 (mean metric-AJ)"
