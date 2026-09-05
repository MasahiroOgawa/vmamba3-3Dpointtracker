#!/usr/bin/env bash
# Overnight queue, gated so a dead branch does not consume the night.
#
# A: score the standalone scale model in official metric-AJ          ~15 min
#    Its regression result is a 66.6% reduction in scale error against the linear fit's ~30%, but
#    that is not the paper's metric. This substitutes its prediction into the tracks exactly as the
#    oracle did, so the three are comparable: oracle 0.3298, v63 0.2238, DELTA 0.2270.
#
# GATE: if A does not beat v63's 0.2238, the scale route is finished as a standalone correction and
#       B is skipped -- there is no point composing a stage that does not help on its own.
#
# B: v74 = v63 with the de-flicker stage replaced by the new scale refiner,       ~6 h
#    warm-started from v63 (its v35 refiner weights carry over unchanged, and the
#    scale stage from the standalone run), then evaluated.
#
# C: a second seed of v63, whatever happened above                                ~5 h
#    The paper's single-seed limitation is exactly what makes the -0.0032 gap to DELTA hard to
#    read. One repeat does not give a variance estimate, but it bounds whether a 0.003 difference
#    is meaningful at all. Runs regardless: it is useful under every outcome.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
# shellcheck source=/dev/null
source "$(dirname "$0")/cudnn_env.sh"
CACHE=$HOME/data/tapvid3d_baseline_preds/waft_minival_is896_s-1_i4
say () { echo "[overnight $(date -Is)] $*"; }

# ---- A ------------------------------------------------------------------------------------------
if [ ! -f result/learned_scale_eval/metrics.json ]; then
  say "A: scoring the learned scale in metric-AJ"
  uv run python scripts/eval_metric3d.py --method external \
      --pred-dir "$HOME/data/tapvid3d_baseline_preds/learned_scale_minival" \
      --da3-depth-root "$HOME/data/tapvid3d_da3nested" --split minival \
      --out-dir result/learned_scale_eval || say "A FAILED"
fi
A=$(uv run python -c "
import json
try: print(json.load(open('result/learned_scale_eval/metrics.json'))['overall']['metric_average_jaccard'])
except Exception: print(0.0)")
say "A result: learned scale = $A metric-AJ  (v63 0.2238, DELTA 0.2270, oracle 0.3298)"

GATE=$(uv run python -c "print(1 if $A > 0.2238 else 0)")
if [ "$GATE" = "1" ]; then
  say "B: gate passed, training v74"
  uv run python scripts/train_depth_refined_tracker.py --config configs/v74.yaml || say "B TRAIN FAILED"
  OUT=$(uv run python -c "import yaml;print(yaml.safe_load(open('configs/v74.yaml'))['train']['out_dir'])")
  CK=$(ls -t "$OUT"/best/ckpt_*.pt 2>/dev/null | head -1); [ -n "$CK" ] || CK=$(ls -t "$OUT"/ckpt_*.pt 2>/dev/null | head -1)
  if [ -n "$CK" ]; then
    uv run python scripts/eval_metric3d.py --method v73 --ckpt "$CK" \
        --da3-depth-root "$HOME/data/tapvid3d_da3nested" --split minival \
        --waft-pred-dir "$CACHE" --out-dir "$OUT/eval" || say "B EVAL FAILED"
    uv run python -c "
import json; d=json.load(open('$OUT/eval/metrics.json')); ps=d['per_subset']
print('[v74] ' + ' '.join(f\"{s}={ps[s]['metric_average_jaccard']:.4f}\" for s in ('drivetrack','pstudio','adt'))
      + f\" mean={d['overall']['metric_average_jaccard']:.4f}\")"
  fi
else
  say "B: gate NOT passed ($A <= 0.2238); skipping v74 and going straight to C"
fi

# ---- C ------------------------------------------------------------------------------------------
say "C: second seed of v63"
uv run python scripts/train_depth_refined_tracker.py --config configs/v63_seed2.yaml || say "C TRAIN FAILED"
OUT2=result/v63_seed2
CK2=$(ls -t "$OUT2"/ckpt_*.pt 2>/dev/null | head -1)
if [ -n "$CK2" ]; then
  uv run python scripts/eval_metric3d.py --method v45 --ckpt "$CK2" \
      --da3-depth-root "$HOME/data/tapvid3d_da3nested" --split minival \
      --waft-pred-dir "$CACHE" --out-dir "$OUT2/eval" || say "C EVAL FAILED"
  uv run python -c "
import json; d=json.load(open('$OUT2/eval/metrics.json'))
print(f\"[v63-seed2] mean={d['overall']['metric_average_jaccard']:.4f}  (seed 1 was 0.2238)\")"
fi
say "all done"
