#!/usr/bin/env bash
# Overnight pipeline: MegaSAM annotations + full method comparison on minival.
#
# Stages:
#   1. MegaSAM annotation generation — drivetrack (50 clips, ~1.7 h)
#   2. MegaSAM annotation generation — pstudio   (50 clips, ~7 h)
#   3. Extract H5 depths → NPZ for SEA-RAFT eval
#   4. SEA-RAFT + MegaSAM eval   (drivetrack + pstudio)
#   5. TAPIP3D + MegaSAM eval    (drivetrack + pstudio)
#
# Total wall time: ~9-10 h.  Results land in result/.
set -euo pipefail

REPO=/home/mas/proj/study/largescale3Dreconstruction_using_SSM
TAPIP3D=/home/mas/proj/study/TAPIP3D
VENV=$TAPIP3D/.venv
CU13=$VENV/lib/python3.11/site-packages/nvidia/cu13
TAPIP3D_CKPT=$TAPIP3D/checkpoints/tapip3d_final.pth

export CUDA_HOME=$CU13
export LD_LIBRARY_PATH=$VENV/lib/python3.11/site-packages/torch/lib:$CU13/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=$TAPIP3D:${REPO}/scripts:${PYTHONPATH:-}

TS=$(date +%Y%m%d-%H%M)
LOG_DIR="$REPO/result/eval/${TS}_megasam_eval"
mkdir -p "$LOG_DIR"

echo "============================================================"
echo "Run started: $(date)"
echo "Logs: $LOG_DIR"
echo "============================================================"

# ---------------------------------------------------------------
# Stage 1: MegaSAM annotations — drivetrack
# ---------------------------------------------------------------
echo ""
echo "STAGE 1: MegaSAM annotations — drivetrack (50 clips, ~1.7h)"
echo "============================================================"
$VENV/bin/python "$REPO/scripts/make_megasam_annotations.py" \
    --subsets drivetrack \
    2>&1 | tee "$LOG_DIR/megasam_drivetrack.log"
echo "Stage 1 done: $(date)"

# ---------------------------------------------------------------
# Stage 2: MegaSAM annotations — pstudio
# ---------------------------------------------------------------
echo ""
echo "STAGE 2: MegaSAM annotations — pstudio (50 clips, ~7h)"
echo "============================================================"
$VENV/bin/python "$REPO/scripts/make_megasam_annotations.py" \
    --subsets pstudio \
    2>&1 | tee "$LOG_DIR/megasam_pstudio.log"
echo "Stage 2 done: $(date)"

# ---------------------------------------------------------------
# Stage 2b: Write combined TAPIP3D config (drivetrack + pstudio)
# ---------------------------------------------------------------
# Running subsets separately overwrites the config each time.
# Re-run with both subsets specified (skips cached clips) to write merged config.
$VENV/bin/python "$REPO/scripts/make_megasam_annotations.py" \
    --subsets drivetrack pstudio \
    --start 0 --end 0 \
    2>&1 | tee -a "$LOG_DIR/megasam_pstudio.log"

# ---------------------------------------------------------------
# Stage 3: Extract depths from H5 → NPZ for SEA-RAFT eval
# ---------------------------------------------------------------
echo ""
echo "STAGE 3: Extract depths H5→NPZ for SEA-RAFT eval"
echo "============================================================"
$VENV/bin/python - << 'PYEOF'
import sys, h5py, numpy as np
from pathlib import Path

sys.path.insert(0, '/home/mas/proj/study/TAPIP3D')
from evaluation.tapvid3d_splits import MINIVAL_FILES

REPO = Path('/home/mas/proj/study/largescale3Dreconstruction_using_SSM')
ANNO_ROOT = REPO / 'result' / 'tapip3d_annotations'
NPZ_ROOT = REPO / 'result' / 'tapvid3d_megasam'

for subset in ['drivetrack', 'pstudio']:
    h5_dir = ANNO_ROOT / f'{subset}_megasam_minival' / 'megasam'
    npz_dir = NPZ_ROOT / subset
    npz_dir.mkdir(parents=True, exist_ok=True)
    clips = sorted(MINIVAL_FILES[subset])
    for seq_id, fname in enumerate(clips):
        clip_id = fname.removesuffix('.npz')
        h5_path = h5_dir / f'{seq_id}.h5'
        if not h5_path.exists():
            print(f'  [{subset}] seq {seq_id}: H5 missing — skipping')
            continue
        out_npz = npz_dir / (clip_id + '.npz')
        if out_npz.exists():
            continue
        with h5py.File(h5_path, 'r') as f:
            depth = f['depths'][:]  # (T, H, W) float32
        np.savez_compressed(out_npz, depth=depth)
    print(f'[{subset}] NPZ files extracted → {npz_dir}')
PYEOF
echo "Stage 3 done: $(date)"

# ---------------------------------------------------------------
# Stage 4: SEA-RAFT + MegaSAM eval — drivetrack + pstudio
# ---------------------------------------------------------------
echo ""
echo "STAGE 4: SEA-RAFT + MegaSAM eval (drivetrack + pstudio)"
echo "============================================================"
cd "$REPO"
uv run python scripts/eval_metric3d.py \
    --method searaft \
    --da3-depth-root result/tapvid3d_megasam \
    --split minival \
    --subsets drivetrack pstudio \
    --out-dir "result/eval/${TS}_metric3d_searaft_megasam_minival" \
    2>&1 | tee "$LOG_DIR/searaft_megasam_eval.log"
echo "Stage 4 done: $(date)"

# ---------------------------------------------------------------
# Stage 5: TAPIP3D + MegaSAM eval — drivetrack + pstudio
# ---------------------------------------------------------------
echo ""
echo "STAGE 5: TAPIP3D + MegaSAM eval (drivetrack + pstudio)"
echo "============================================================"
cd "$TAPIP3D"
$VENV/bin/python train_eval.py \
    --config-name tapip3d_megasam_minival_eval \
    eval_only=true \
    +train.eval_only=true \
    wandb.enable=false \
    "train.checkpoint=$TAPIP3D_CKPT" \
    '~test_datasets.kubric_24frames_384trajs_200samples' \
    2>&1 | tee "$REPO/$LOG_DIR/tapip3d_megasam_eval.log"
cd "$REPO"
echo "Stage 5 done: $(date)"

# ---------------------------------------------------------------
# Summary
# ---------------------------------------------------------------
echo ""
echo "============================================================"
echo "ALL DONE: $(date)"
echo "============================================================"
echo "Results:"
echo "  SEA-RAFT + MegaSAM:  result/eval/${TS}_metric3d_searaft_megasam_minival/"
echo "  TAPIP3D + MegaSAM:   $TAPIP3D/result/  (look for megasam run)"
echo ""
echo "Existing baselines (all 3 subsets):"
echo "  SEA-RAFT + DA3:      result/20260618-1139_metric3d_searaft/"
echo "  v33:                 result/20260618-1139_metric3d_v33/"
echo "  SEA-RAFT + depthn (ADT): result/20260625-2015_metric3d_searaft_depthn_minival/"
