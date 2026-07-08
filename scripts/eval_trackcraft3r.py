#!/usr/bin/env python3
"""
Run TrackCraft3R inference on TAPVid-3D minival clips and save predictions
in eval_metric3d.py --method external format.

Must be run from TrackCraft3R's uv environment:
  cd ~/proj/study/TrackCraft3R
  uv run python /home/mas/proj/study/largescale3Dreconstruction_using_SSM/scripts/eval_trackcraft3r.py \\
    --subsets pstudio adt drivetrack \\
    --out-dir /home/mas/data/tapvid3d_baseline_preds/trackcraft3r

Output per clip: <out-dir>/<subset>/<clip_name>.npz
  tracks_XYZ  (F, N, 3)  float32  per-frame camera-space XYZ
  visibility  (F, N)     float32  0/1 model-predicted visibility
"""

import argparse
import gc
import io
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# Limit CPU parallelism BEFORE any torch import side-effects fire.
# Loading large safetensors/pth files with 32-48 threads pegs all cores and
# exhausts RAM via swap.  8 threads is plenty for sequential tensor copies.
_N_THREADS = int(os.environ.get("TCR_NUM_THREADS", "8"))
os.environ.setdefault("OMP_NUM_THREADS", str(_N_THREADS))
os.environ.setdefault("MKL_NUM_THREADS", str(_N_THREADS))
os.environ.setdefault("OPENBLAS_NUM_THREADS", str(_N_THREADS))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# Expandable segments let the CUDA allocator grow existing memory blocks instead
# of requiring a new contiguous allocation — prevents OOM from fragmentation when
# model weights fill most of the 12 GB RTX 4080 VRAM.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# ── TrackCraft3R root (must run with its uv env) ─────────────────────────────
TCR_ROOT = Path(__file__).resolve().parent.parent.parent / "TrackCraft3R"
assert TCR_ROOT.exists(), f"TrackCraft3R not found at {TCR_ROOT}"
sys.path.insert(0, str(TCR_ROOT))

# Import canonical minival split from main project
_PROJ_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJ_ROOT / "src"))

from evaluation.wan_scene_flow_predictor import WanSceneFlowPredictor  # noqa: E402
from mamba3_tracker.data.tapvid3d_splits import MINIVAL_FILES  # noqa: E402

# ── paths ────────────────────────────────────────────────────────────────────
TAPVID3D_ROOT = Path("/home/mas/data/tapvid3d")
DA3_ROOT = Path("/home/mas/data/tapvid3d_da3")
TCR_CKPT = TCR_ROOT / "checkpoints" / "trackcraft3r" / "model.safetensors"
WAN_MODEL_ID = str(TCR_ROOT / "checkpoints" / "wan_models" / "Wan-AI" / "Wan2.1-T2V-1.3B")
NULL_CTX_CACHE = TCR_ROOT / "checkpoints" / "null_context.pt"

WIN_SIZE = 12  # TrackCraft3R window length


def ensure_null_context() -> None:
    """Pre-compute and cache the T5 null context if not already done.

    Loads ONLY the T5 encoder (11 GB) on CPU, encodes the empty string, saves
    the tiny result (~64 KB), then frees the encoder before the main model loads.
    CPU encoding is intentional: T5 is 11 GB and the GPU may be occupied by
    other services (e.g. Ollama). CPU is slower (~30 s) but safe.
    """
    if NULL_CTX_CACHE.exists():
        print(f"[tcr] null context cache found: {NULL_CTX_CACHE}", flush=True)
        return

    print("[tcr] Pre-computing null context from T5 encoder (one-time, ~11 GB on CPU) ...",
          flush=True)
    from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig  # noqa: E402

    pipe_t5 = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cpu",
        model_configs=[
            ModelConfig(model_id=WAN_MODEL_ID,
                        origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth"),
        ],
    )
    # Encode on CPU — no GPU move, avoids competing with other GPU services.
    with torch.no_grad():
        null_ctx = pipe_t5.prompter.encode_prompt(
            "", positive=True, device="cpu",
        ).to(dtype=torch.bfloat16, device="cpu")

    NULL_CTX_CACHE.parent.mkdir(parents=True, exist_ok=True)
    torch.save(null_ctx, NULL_CTX_CACHE)
    print(f"[tcr] null context cached → {NULL_CTX_CACHE} (shape: {null_ctx.shape})", flush=True)

    del pipe_t5, null_ctx
    gc.collect()
    torch.cuda.empty_cache()


def load_da3_depth(subset: str, clip_name: str, F: int) -> np.ndarray:
    """Load DA3 depth → float32 (F, H_da3, W_da3) in metres."""
    p = DA3_ROOT / subset / clip_name
    with np.load(p) as d:
        q = d["depth_q"].astype(np.float32)[:F]
        d_min = float(d["d_min"])
        d_max = float(d["d_max"])
    return d_min + q * ((d_max - d_min) / 65535.0)


def decode_images(jpeg_bytes_arr) -> list[Image.Image]:
    """Decode JPEG byte array → list of PIL RGB images."""
    return [Image.open(io.BytesIO(bytes(b))).convert("RGB") for b in jpeg_bytes_arr]


def world_to_camspace(xyz_w: np.ndarray, w2c: np.ndarray) -> np.ndarray:
    """Transform 3D points from world/frame-0 space to per-frame camera space.

    Args:
        xyz_w: (T, N, 3) 3D tracks in frame-0 camera (world) space
        w2c:   (T, 4, 4) world-to-camera transforms, w2c[0] should be identity

    Returns:
        (T, N, 3) tracks in per-frame camera space
    """
    R = w2c[:, :3, :3]   # (T, 3, 3)
    t = w2c[:, :3, 3:]   # (T, 3, 1)
    # (T, 3, 3) @ (T, 3, N) + (T, 3, 1) → (T, 3, N) → (T, N, 3)
    return (R @ xyz_w.transpose(0, 2, 1) + t).transpose(0, 2, 1)


def build_query_windows(queries_xyt: np.ndarray, F: int) -> dict:
    """Group query indices into 12-frame windows centred on their anchor frame.

    Each query at anchor t_q maps to window [t0, t1) = [t_q-5, t_q+7) clamped
    to [0, F).  Queries with the same (t0, t1) share one inference call.

    Returns: {(t0, t1): [query_indices]}
    """
    windows: dict[tuple, list] = {}
    for i, (_, _, t) in enumerate(queries_xyt):
        t_q = int(t)
        t0 = max(0, t_q - 5)
        t1 = min(F, t0 + WIN_SIZE)
        t0 = max(0, t1 - WIN_SIZE)  # adjust if near end
        windows.setdefault((t0, t1), []).append(i)
    return windows


def infer_clip(
    predictor: WanSceneFlowPredictor,
    data: dict,
    subset: str,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Run TrackCraft3R on one clip via query-centric windows.

    Returns:
        tracks_XYZ: (F, N, 3) per-frame camera-space XYZ
        visibility: (F, N) float32 0/1 model-predicted visibility
    """
    jpeg_bytes = data["images_jpeg_bytes"]
    queries_xyt = data["queries_xyt"].astype(np.float32)   # (N, 3): x_px, y_px, t
    fx_fy_cx_cy = data["fx_fy_cx_cy"].astype(np.float32)

    frames_pil = decode_images(jpeg_bytes)
    F = len(frames_pil)
    N = len(queries_xyt)
    orig_w, orig_h = frames_pil[0].width, frames_pil[0].height

    # Load DA3 depth
    clip_name = data["_clip_name"]
    depth_full = load_da3_depth(subset, clip_name, F)   # (F, H_da3, W_da3)

    # Extrinsics: identity for pstudio/adt; GT w2c for drivetrack
    if subset == "drivetrack" and "extrinsics_w2c" in data:
        w2c_full = data["extrinsics_w2c"].astype(np.float32)[:F]   # (F, 4, 4)
    else:
        w2c_full = np.tile(np.eye(4, dtype=np.float32)[None], (F, 1, 1))

    tracks_XYZ = np.zeros((F, N, 3), np.float32)
    visibility = np.zeros((F, N), np.float32)
    covered = np.zeros(N, bool)  # which queries have received window predictions

    windows = build_query_windows(queries_xyt, F)

    for (t0, t1), qids in windows.items():
        qids = list(qids)
        q_uv = queries_xyt[qids, :2].astype(np.float64)  # (M, 2) x,y in original pixels

        frames_w = frames_pil[t0:t1]
        depth_w = depth_full[t0:t1]                       # (T_w, H_da3, W_da3)
        w2c_w = w2c_full[t0:t1].copy()                    # (T_w, 4, 4)

        # Normalise so window-frame-0 = identity (divide all by w2c[t0])
        w2c_w_norm = w2c_w @ np.linalg.inv(w2c_w[:1])    # (T_w, 4, 4), [0] = I

        try:
            pred = predictor.predict(
                frames_w,
                q_uv,
                np.ones(len(qids), np.float32),    # visibility unused by model
                fx_fy_cx_cy,
                depth_w,
                w2c_w_norm,
            )
            # pred: (T_w, M', 3) in window-frame-0 camera space; M' <= len(qids)
        except Exception as e:
            print(f"  [warn] window ({t0},{t1}) failed: {e}")
            continue

        oob_mask = predictor._last_oob_mask   # (M,) bool, which queries survived OOB filter
        surviving_qids = [qids[k] for k in range(len(qids)) if oob_mask[k]]
        T_w = pred.shape[0]

        # Convert frame-0 space → per-frame camera space using normalised w2c
        pred_cam = world_to_camspace(pred, w2c_w_norm[:T_w])   # (T_w, M', 3)

        # Sample per-track visibility from dense vis field
        vis_dense = predictor._last_vis_dense   # (T_w, H_out, W_out) or None
        uv_model = predictor._last_query_uv_model[oob_mask]  # (M', 2) in model coords
        u_q = uv_model[:, 0].astype(int)
        v_q = uv_model[:, 1].astype(int)
        if vis_dense is not None:
            H_out, W_out = vis_dense.shape[1], vis_dense.shape[2]
            u_q = np.clip(u_q, 0, W_out - 1)
            v_q = np.clip(v_q, 0, H_out - 1)
            vis_at_queries = vis_dense[:T_w, v_q, u_q]   # (T_w, M')
        else:
            vis_at_queries = np.ones((T_w, len(surviving_qids)), np.float32)

        # Store window predictions; propagate by nearest copy outside window
        for local_t, abs_t in enumerate(range(t0, t1)):
            for k, qid in enumerate(surviving_qids):
                tracks_XYZ[abs_t, qid] = pred_cam[local_t, k]
                visibility[abs_t, qid] = vis_at_queries[local_t, k]

        # Backward fill (frames before window): last known position, vis=0
        for qid in surviving_qids:
            for abs_t in range(t0 - 1, -1, -1):
                tracks_XYZ[abs_t, qid] = tracks_XYZ[t0, qid]
                visibility[abs_t, qid] = 0.0

        # Forward fill (frames after window): last known position, vis=0
        for qid in surviving_qids:
            for abs_t in range(t1, F):
                tracks_XYZ[abs_t, qid] = tracks_XYZ[t1 - 1, qid]
                visibility[abs_t, qid] = 0.0

        for qid in surviving_qids:
            covered[qid] = True

        torch.cuda.empty_cache()

    n_uncovered = int((~covered).sum())
    if n_uncovered:
        print(f"  [warn] {n_uncovered}/{N} queries had all OOB in their window")

    return tracks_XYZ, visibility


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subsets", nargs="+", default=["pstudio", "adt", "drivetrack"])
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/home/mas/data/tapvid3d_baseline_preds/trackcraft3r"),
    )
    ap.add_argument("--device", default="cuda")
    ap.add_argument(
        "--max-clips",
        type=int,
        default=0,
        help="If >0, stop after this many clips (for smoke tests)",
    )
    ap.add_argument(
        "--height", type=int, default=480,
        help="Model resolution height (default: 480, same as paper)",
    )
    ap.add_argument(
        "--width", type=int, default=832,
        help="Model resolution width (default: 832, same as paper)",
    )
    args = ap.parse_args()

    # Limit CPU thread count before ANY model loading (including Phase 1 / T5).
    # torch.set_num_threads controls intra-op parallelism; setting it here covers
    # ensure_null_context() which runs before WanSceneFlowPredictor.__init__.
    torch.set_num_threads(_N_THREADS)
    torch.set_num_interop_threads(min(_N_THREADS, 4))
    print(f"[tcr] CPU threads limited to {_N_THREADS} (intra-op) / {min(_N_THREADS, 4)} (inter-op)")

    # ── phase 1: pre-compute null context (T5 only, freed before main load) ──
    ensure_null_context()

    # ── phase 2: load main model (DiT + VAE, no T5) ──────────────────────────
    print(f"Loading WanSceneFlowPredictor from {TCR_CKPT} ...", flush=True)
    predictor = WanSceneFlowPredictor(
        checkpoint_path=str(TCR_CKPT),
        model_id=WAN_MODEL_ID,
        height=args.height,
        width=args.width,
        device=args.device,
        null_context_cache=str(NULL_CTX_CACHE),
        num_threads=_N_THREADS,
    )
    print("Model loaded.", flush=True)

    # ── run inference ─────────────────────────────────────────────────────────
    for subset in args.subsets:
        if subset not in MINIVAL_FILES:
            print(f"[warn] unknown subset {subset!r}, skip")
            continue
        out_sub = args.out_dir / subset
        out_sub.mkdir(parents=True, exist_ok=True)

        clips = MINIVAL_FILES[subset]
        n_done = 0
        for clip_name in tqdm(clips, desc=subset):
            out_path = out_sub / clip_name
            if out_path.exists():
                n_done += 1
                continue
            if args.max_clips > 0 and n_done >= args.max_clips:
                break

            npz_path = TAPVID3D_ROOT / subset / clip_name
            # allow_pickle=True is safe here: these are our own local dataset files
            # under /home/mas/data/tapvid3d/ (TAPVid-3D benchmark data, not user input).
            data = dict(np.load(npz_path, allow_pickle=True))
            data["_clip_name"] = clip_name

            try:
                tracks_XYZ, vis = infer_clip(predictor, data, subset, args.device)
                np.savez_compressed(out_path, tracks_XYZ=tracks_XYZ, visibility=vis)
                n_done += 1
                torch.cuda.empty_cache()
            except torch.cuda.OutOfMemoryError as e:
                torch.cuda.empty_cache()
                print(f"[OOM]   {subset}/{clip_name}: {e}")
            except Exception as e:
                torch.cuda.empty_cache()
                print(f"[error] {subset}/{clip_name}: {e}")

    print(f"[trackcraft3r] done → {args.out_dir}")


if __name__ == "__main__":
    main()
