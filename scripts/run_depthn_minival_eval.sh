#!/usr/bin/env bash
# Overnight evaluation: compare depth_preds vs DA3 for SEA-RAFT and TAPIP3D on minival.
#
# Runs four evaluations:
#   1. SEA-RAFT + DA3      (baseline, already done — here for log completeness)
#   2. SEA-RAFT + depthn   (TAPVid-3D built-in depth_preds, no model needed)
#   3. TAPIP3D + DA3       (already done; skipped if predictions cached)
#   4. TAPIP3D + depthn    (new: TAPIP3D with better depth)
#
# Results land in result/. Compare median-AJ and absolute metric-AJ.
# Expected total wall time: ~5-6 h (steps 2+4 are the heavy ones).
set -euo pipefail

REPO=/home/mas/proj/study/largescale3Dreconstruction_using_SSM
TAPIP3D=/home/mas/proj/study/TAPIP3D
VENV=$TAPIP3D/.venv
CU13=$VENV/lib/python3.11/site-packages/nvidia/cu13
export LD_LIBRARY_PATH=$VENV/lib/python3.11/site-packages/torch/lib:$CU13/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=$TAPIP3D:${PYTHONPATH:-}

TS=$(date +%Y%m%d-%H%M)
cd "$REPO"

echo "================================================================"
echo "STEP 0: Extract depth_preds annotations (fast, ~5 min)"
echo "================================================================"
uv run python scripts/make_depthn_annotations.py

echo ""
echo "================================================================"
echo "STEP 1: SEA-RAFT + depthn on minival (ADT only — only ADT has depth_preds)"
echo "================================================================"
uv run python scripts/eval_metric3d.py \
    --method searaft \
    --da3-depth-root result/tapvid3d_depthn \
    --split minival \
    --subsets adt \
    --out-dir "result/eval/${TS}_metric3d_searaft_depthn_minival" \
    2>&1 | tee "result/eval/${TS}_metric3d_searaft_depthn_minival.log"

echo ""
echo "================================================================"
echo "STEP 2: TAPIP3D + depthn on minival"
echo "================================================================"
TAPIP3D_CKPT=$TAPIP3D/checkpoints/tapip3d_final.pth

cd "$TAPIP3D"
$VENV/bin/python train_eval.py \
    --config-name tapip3d_depthn_minival_eval \
    eval_only=true \
    +train.eval_only=true \
    wandb.enable=false \
    "train.checkpoint=$TAPIP3D_CKPT" \
    '~test_datasets.kubric_24frames_384trajs_200samples' \
    2>&1 | tee "$REPO/result/${TS}_tapip3d_depthn_eval.log"
cd "$REPO"

echo ""
echo "================================================================"
echo "ALL DONE — summary"
echo "================================================================"
echo "SEA-RAFT + depthn:  result/eval/${TS}_metric3d_searaft_depthn_minival/"
echo "TAPIP3D + depthn:   $TAPIP3D/result/${TS}_tapip3d_depthn_minival/"
echo ""
echo "Compare against baselines:"
echo "  SEA-RAFT + DA3:   result/20260618-1718_metric3d_searaft_fix/"
echo "  TAPIP3D + DA3:    $TAPIP3D/result/auto_generated/*/"
