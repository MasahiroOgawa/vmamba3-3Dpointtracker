---
name: project_delta_eval
description: DELTA (DenseTrack3D) added as the fast SOTA external baseline that actually works on this PC — 14.1% mean minival 3D-AJ
metadata:
  type: project
---

DELTA / DenseTrack3D (snap-research, ICLR 2025) is the fast, reliable external 3D
tracker added to the TAPVid-3D minival comparison (2026-07-08). It is the **only**
external SOTA method that produces reasonable, subset-balanced scores on the
RTX 4080 Laptop (12 GB) — unlike [[project_spatrackerv2_bidir]], [[project_tapip3d_comparison]],
[[project_trackcraft3r_eval]] which all collapse to ~1-2% (see [[project_external_methods_not_lying]]).

**Results (minival, our `eval_metric3d.py`, DELTA+DA3):** mean norm 3D-AJ **0.141**
(drivetrack 0.130, pstudio 0.141, adt 0.152), OA 0.835; absolute metric-AJ 0.183
(adt 0.344 = best of ANY method). Beats our SEA-RAFT+DA3 baseline (0.117) and v36
(0.097) on the leaderboard metric; v35 still wins absolute-metric mean (0.234).
Runtime 7.7 fps, full 150-clip minival in ~1 h (0 OOM, 0 failures).

**Setup:** cloned to `~/proj/study/DELTA_densetrack3d`, symlinked `third_party/DELTA_densetrack3d`,
own uv venv (py3.10, torch 2.2.2+cu121, numpy 1.26.4 — lean: only torch/numpy/einops/
jaxtyping/opencv/tqdm; UniDepth/DepthCrafter/pytorch3d/xformers NOT needed since we
feed our own depth). Checkpoint `checkpoints/densetrack3d.pth` via `gdown <id>` (old
gdown has no `--fuzzy`; use the raw file id).

**Wrapper:** `scripts/eval_delta.py` (run from DELTA venv). Uses `Predictor3D` sparse
path; feeds DA3 depth as videodepth + true `fx_fy_cx_cy` as `predefined_intrs` (NOT
None — else UniDepth invents intrinsics). Output `trajs_3d_dict["coords"]` is already
per-frame camera XYZ (matches GT [[project_metric3d_evaluation]]), no conversion.
Queries reordered (x,y,t)→(t,x,y); `backward_tracking=True` when any query t>0.

**Key VRAM trick:** feature maps are computed for ALL frames at once (memory ~ F·h·w),
so adt's 300-frame clips OOM at (384,512). The model's pos_emb is a fixed sincos buffer
it crops when input reso < build reso, so the wrapper builds the model once at (384,512)
and lowers the fed `interp_shape` per clip (adaptive ladder keyed on frame count, with
OOM-retry). `--max-side 512` bounds the full-res video tensor (drivetrack is 1280×1920).
Metric XYZ is invariant to both rescalings (ray direction preserved), so accuracy holds.
Preds at `/home/mas/data/tapvid3d_baseline_preds/delta/<subset>/`.
