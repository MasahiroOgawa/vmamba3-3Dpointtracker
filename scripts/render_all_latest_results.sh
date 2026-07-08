#!/usr/bin/env bash
# Produce ALL results for a tracker run in one shot — eval + every visualisation:
#   1. TAPVid-3D metric eval (3D-AJ / APD / OA) on the chosen split
#   2. baseline comparison table + bar chart vs configs/tapvid3d_baselines.yaml
#   3. per-clip tracking MP4s
#   4. per-clip 3D xyz-space trajectory plots (PNG + interactive HTML)
#   5. per-clip space-time xyt/yzt/zxt plots (PNG + interactive HTML)
#   6. training/validation loss curves + motion-ratio plot
#
# Layout (see memory/feedback_output_dir_naming.md — datetime only at run-dir level):
#   <run_dir>/eval/metric_results/<subset>.json   per-clip metrics
#   <run_dir>/eval/summary.md                      per-subset roll-up
#   <run_dir>/eval/comparison.{png,md}             vs published baselines
#   <run_dir>/viz_step<N>/*.mp4                     tracking videos
#   <run_dir>/viz_step<N>/*_3d.{png,html}           xyz spatial plots
#   <run_dir>/viz_step<N>/*_st.{png,html}           space-time plots
#   <run_dir>/plots/{training_curve,motion_ratio}.png
#
# Usage:
#   scripts/render_all_latest_results.sh                                 # latest run dir, latest ckpt
#   scripts/render_all_latest_results.sh result/<dt>_track_v19          # specific run dir
#   scripts/render_all_latest_results.sh result/<dt>_track_v19/ckpt_5000.pt   # specific ckpt
#
# Flags (each mirrors an env var; either works):
#   --quick                fast preview (~1-2 min): skip eval+comparison, skip
#                          3D+space-time plots, render 1 tracker MP4 per subset,
#                          always do loss-curve plot. Sets QUICK=1.
#   --use-cpu, --use_cpu   run on CPU (set when GPU busy with training). Sets USE_CPU=1.
#   --no-eval              skip the metric eval + comparison, viz only. Sets RUN_EVAL=0.
#   --split <name>         which TAPVid-3D split (minival / full_eval / all).
#   --subsets "a b c"      subsets to process.
#   --clips-per-subset N   clips per subset for per-clip visualisations.
#   --max-frames N         frame cap per clip for the visualisations.
#
# Env vars (all optional; equivalent to flags above):
#   SPLIT=minival          (default minival; use held-out test for v19+ runs)
#   USE_CPU=1              (default 0 = GPU)
#   RUN_EVAL=0             (default 1)
#   SUBSETS="pstudio drivetrack adt"   (default all three)
#   CLIPS_PER_SUBSET=3     (default 3)
#   MAX_FRAMES=48          (default 48)
#   QUICK=1                (default 0)
#
# Argument resolution:
#   no arg                 -> ls -td result/track_v*_*/ | head -1 ; latest ckpt
#   path is a directory    -> that dir's latest ckpt
#   path is a .pt file     -> that exact ckpt

set -euo pipefail

# Parse flags. Each flag mirrors the corresponding env var; the env var still
# works when set in the environment. Anything not a flag is treated as the
# positional path argument (run dir or .pt ckpt).
ARG=""
while [ $# -gt 0 ]; do
    case "$1" in
        --quick)             QUICK=1; shift ;;
        --use-cpu|--use_cpu) USE_CPU=1; shift ;;
        --no-eval)           RUN_EVAL=0; shift ;;
        --split)             SPLIT="$2"; shift 2 ;;
        --subsets)           SUBSETS="$2"; shift 2 ;;
        --clips-per-subset)  CLIPS_PER_SUBSET="$2"; shift 2 ;;
        --max-frames)        MAX_FRAMES="$2"; shift 2 ;;
        -h|--help)
            sed -n '1,40p' "$0"; exit 0 ;;
        --) shift; ARG="${1:-}"; break ;;
        -*) echo "error: unknown flag: $1" >&2; exit 1 ;;
        *)  ARG="$1"; shift ;;
    esac
done

if [ -z "$ARG" ]; then
    RUN_DIR="$(ls -td result/track_v*_*/ 2>/dev/null | head -1)"
    RUN_DIR="${RUN_DIR%/}"
    if [ -z "$RUN_DIR" ] || [ ! -d "$RUN_DIR" ]; then
        echo "error: no run dir found under result/track_v*_*/" >&2
        exit 1
    fi
    CKPT=$(ls -t "$RUN_DIR"/ckpt_*.pt 2>/dev/null | head -1)
elif [ -d "$ARG" ]; then
    RUN_DIR="${ARG%/}"
    CKPT=$(ls -t "$RUN_DIR"/ckpt_*.pt 2>/dev/null | head -1)
elif [ -f "$ARG" ]; then
    CKPT="$ARG"
    RUN_DIR="$(dirname "$CKPT")"
else
    echo "error: $ARG is neither a directory nor a file" >&2
    exit 1
fi

if [ -z "$CKPT" ] || [ ! -f "$CKPT" ]; then
    echo "error: no checkpoint found (RUN_DIR=$RUN_DIR)" >&2
    exit 1
fi
STEP=$(basename "$CKPT" .pt | sed 's/ckpt_//')
OUT="${RUN_DIR}/viz_step${STEP}"
mkdir -p "$OUT"
LATEST_CKPT="$CKPT"

USE_CPU="${USE_CPU:-0}"
RUN_EVAL="${RUN_EVAL:-1}"
CLIPS_PER_SUBSET="${CLIPS_PER_SUBSET:-3}"
MAX_FRAMES="${MAX_FRAMES:-48}"
QUICK="${QUICK:-0}"
# Quick mode: skip the expensive eval and the secondary plot suites, keep only
# the tracker-MP4 viz at 1 clip per subset plus the loss-curve plot.
if [ "$QUICK" = "1" ]; then
    RUN_EVAL=0
    CLIPS_PER_SUBSET=1
fi
SUBSETS="${SUBSETS:-pstudio drivetrack adt}"
SPLIT="${SPLIT:-minival}"      # all | minival | full_eval — minival = held-out test set for v19+
AMP=$([ "$USE_CPU" = "1" ] && echo fp32 || echo bf16)
CUDA_PREFIX=$([ "$USE_CPU" = "1" ] && echo "CUDA_VISIBLE_DEVICES=" || echo "")
EVAL_DIR="${RUN_DIR}/eval"

echo "run    : $RUN_DIR"
echo "ckpt   : $LATEST_CKPT (step $STEP)"
echo "output : $OUT"
echo "device : $([ "$USE_CPU" = "1" ] && echo CPU || echo GPU)"
echo "subsets: $SUBSETS  | clips/subset: $CLIPS_PER_SUBSET  | max-frames: $MAX_FRAMES  | split: $SPLIT  | run_eval: $RUN_EVAL"
echo

# ---- Step 1+2: metric eval + baseline comparison ----------------------------
if [ "$RUN_EVAL" = "1" ]; then
    echo "[eval] TAPVid-3D metrics on split=$SPLIT → $EVAL_DIR"
    mkdir -p "$EVAL_DIR"
    env ${CUDA_PREFIX} PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
        uv run python scripts/eval_mamba3_tracker.py \
        --ckpt "$LATEST_CKPT" --out-dir "$EVAL_DIR" \
        --subsets $SUBSETS --split "$SPLIT" --amp "$AMP" \
        2>&1 | tee "$EVAL_DIR/eval.log" || echo "[eval] WARNING: eval exited non-zero (often just labels-only clips)"
    echo
    echo "[compare] vs published baselines → $EVAL_DIR/comparison.{png,md}"
    uv run python scripts/compare_tracker_baselines.py \
        --eval-dir "$EVAL_DIR" --label "$(basename "$RUN_DIR") (step $STEP)" \
        --out-dir "$EVAL_DIR" 2>&1 | tee -a "$EVAL_DIR/eval.log" || echo "[compare] WARNING: comparison failed"
    echo
    # ALWAYS produce the global "where are we" plot — every minival-comparable
    # run on disk, paper baselines, and SoTA reference (full-eval).
    GLOBAL_CMP="result/eval_tracker/comparison_all_methods"
    echo "[compare-all] every v* + baselines + SoTA reference → $GLOBAL_CMP/comparison.{png,md}"
    uv run python scripts/compare_tracker_baselines.py \
        --out-dir "$GLOBAL_CMP" 2>&1 | tee -a "$EVAL_DIR/eval.log" || echo "[compare-all] WARNING: failed"
    echo
fi

# ---- Step 3: per-clip tracking MP4s -----------------------------------------
echo "[viz] tracking MP4s → $OUT"
env ${CUDA_PREFIX} uv run python scripts/render_tracker_video.py \
    --ckpt "$LATEST_CKPT" --out-dir "$OUT" --subsets $SUBSETS \
    --clips-per-subset "$CLIPS_PER_SUBSET" --max-frames "$MAX_FRAMES" \
    --split "$SPLIT" --fps 15 --amp "$AMP" 2>&1 | tee "$OUT/render.log"

# ---- Steps 4+5: per-clip 3D + space-time plots — skipped in QUICK mode -----
if [ "$QUICK" != "1" ]; then
    echo
    echo "[3d] 3D-space track plots (xyz)"
    env ${CUDA_PREFIX} uv run python scripts/render_3d_tracks.py \
        --ckpt "$LATEST_CKPT" --out-dir "$OUT" --subsets $SUBSETS \
        --clips-per-subset "$CLIPS_PER_SUBSET" --max-frames "$MAX_FRAMES" \
        --split "$SPLIT" --amp "$AMP" 2>&1 | tee -a "$OUT/render.log"

    echo
    echo "[st] space-time track plots (xy / yz / zx × time)"
    env ${CUDA_PREFIX} uv run python scripts/render_space_time_tracks.py \
        --ckpt "$LATEST_CKPT" --out-dir "$OUT" --subsets $SUBSETS \
        --clips-per-subset "$CLIPS_PER_SUBSET" --max-frames "$MAX_FRAMES" \
        --split "$SPLIT" --amp "$AMP" 2>&1 | tee -a "$OUT/render.log"
else
    echo
    echo "[quick] skipping 3D + space-time plots (QUICK=1)"
fi

# ---- Step 6: training/val loss curves + motion-ratio ------------------------
echo
echo "[plot] training-curve + motion-ratio plots"
uv run python scripts/plot_training_curves.py --run-dir "$RUN_DIR" 2>&1 | tee -a "$OUT/render.log"

# Final summary — absolute paths so users can copy-paste and find them
# even after the surrounding logs scroll out of view.
RUN_ABS="$(realpath "$RUN_DIR")"
OUT_ABS="$(realpath "$OUT")"
PLOTS_ABS="$(realpath "$RUN_DIR/plots")"

echo
echo "================================================================"
echo "DONE — outputs for step $STEP"
echo "================================================================"
echo "run dir : $RUN_ABS"
if [ "$RUN_EVAL" = "1" ] && [ -f "$EVAL_DIR/comparison.md" ]; then
    echo
    echo "eval + comparison ($(realpath "$EVAL_DIR")):"
    echo "  $(realpath "$EVAL_DIR")/comparison.md      (table vs baselines)"
    echo "  $(realpath "$EVAL_DIR")/comparison.png     (bar chart)"
    echo "  $(realpath "$EVAL_DIR")/metric_results/*.json"
    echo
    echo "  --- comparison table ---"
    sed 's/^/  /' "$EVAL_DIR/comparison.md"
fi
echo
echo "MP4s    ($OUT_ABS):"
for f in "$OUT"/*.mp4; do
    [ -f "$f" ] && echo "  $(realpath "$f")"
done
echo
echo "3D PNGs ($OUT_ABS):"
for f in "$OUT"/*_3d.png; do
    [ -f "$f" ] && echo "  $(realpath "$f")"
done
echo
echo "3D HTML ($OUT_ABS) — open in a browser for interactive rotate/zoom:"
for f in "$OUT"/*_3d.html; do
    [ -f "$f" ] && echo "  $(realpath "$f")"
done
echo
echo "Space-time PNGs ($OUT_ABS) — xy / yz / zx vs frame index:"
for f in "$OUT"/*_st.png; do
    [ -f "$f" ] && echo "  $(realpath "$f")"
done
echo
echo "Space-time HTML ($OUT_ABS) — interactive 1×3 subplots:"
for f in "$OUT"/*_st.html; do
    [ -f "$f" ] && echo "  $(realpath "$f")"
done
echo
echo "plots   ($PLOTS_ABS):"
for f in "$PLOTS_ABS"/*.png; do
    [ -f "$f" ] && echo "  $f"
done
echo
echo "log     : $OUT_ABS/render.log"
echo "================================================================"
