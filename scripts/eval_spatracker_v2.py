#!/usr/bin/env python3
"""
Run SpatialTrackerV2 inference on TAPVid-3D minival clips
and save predictions in eval_metric3d.py --method external format.

Must be run with SpaTrackerV2's own venv:
  cd ~/proj/study/SpaTrackerV2
  .venv/bin/python /home/mas/proj/study/largescale3Dreconstruction_using_SSM/scripts/eval_spatracker_v2.py \\
    --subsets pstudio adt drivetrack \\
    --out-dir /home/mas/data/tapvid3d_baseline_preds/spatrackerv2

Output format per clip: <out-dir>/<subset>/<clip_name>.npz
  tracks_XYZ  (F, N, 3)  float32  per-frame camera-space XYZ
  visibility  (F, N)     float32  0/1 visibility
"""

import argparse
import io
import sys
import types
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as Fn
from PIL import Image
from tqdm import tqdm

# ── SpaTrackerV2 root (must run with its venv) ───────────────────────────────
SPA_ROOT = Path(__file__).resolve().parent.parent.parent / "SpaTrackerV2"
assert SPA_ROOT.exists(), f"SpaTrackerV2 not found at {SPA_ROOT}"
sys.path.insert(0, str(SPA_ROOT))

# Import canonical minival split from the main project (no external deps in that file)
_PROJ_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJ_ROOT / "src"))

# utils3d 0.1.0 (PyPI) lacks the `torch` submodule used by SpaTrackerV2.
# Stub it before importing the model.  depth_edge returns all-False (no
# depth-discontinuity masking — slightly more noise near edges but otherwise
# correct), and points_to_normals returns zero normals / all-valid mask.
import utils3d as _u3d  # noqa: E402

if not hasattr(_u3d, "torch"):
    import torch as _torch

    def _depth_edge(depth, rtol=0.03, mask=None):
        return _torch.zeros(depth.shape, dtype=_torch.bool, device=depth.device)

    def _pts_to_normals(points, mask=None):
        n = _torch.zeros_like(points[..., :3])
        m = _torch.ones(points.shape[:-1], dtype=_torch.bool, device=points.device)
        return n, m

    def _image_pixel_center(width, height, dtype=None, device=None):
        xs = _torch.arange(width, dtype=dtype, device=device)
        ys = _torch.arange(height, dtype=dtype, device=device)
        grid_y, grid_x = _torch.meshgrid(ys, xs, indexing="ij")
        return _torch.stack([grid_x, grid_y], dim=-1)

    def _sliding_window_2d(x, window_size, stride=1, dim=(-2, -1)):
        return x.unfold(
            dim[0],
            window_size if isinstance(window_size, int) else window_size[0],
            stride,
        ).unfold(
            dim[1],
            window_size if isinstance(window_size, int) else window_size[1],
            stride,
        )

    def _image_uv(width, height, dtype=None, device=None):
        u = _torch.linspace(0, 1, width, dtype=dtype, device=device)
        v = _torch.linspace(0, 1, height, dtype=dtype, device=device)
        grid_v, grid_u = _torch.meshgrid(v, u, indexing="ij")
        return _torch.stack([grid_u, grid_v], dim=-1)

    _u3d.torch = types.SimpleNamespace(
        depth_edge=_depth_edge,
        points_to_normals=_pts_to_normals,
        image_pixel_center=_image_pixel_center,
        sliding_window_2d=_sliding_window_2d,
        image_uv=_image_uv,
    )
    del _depth_edge, _pts_to_normals, _image_pixel_center, _sliding_window_2d, _image_uv

from models.SpaTrackV2.models.predictor import Predictor  # noqa: E402
from mamba3_tracker.data.tapvid3d_splits import MINIVAL_FILES  # noqa: E402


def reproject_2d3d(uvd: torch.Tensor, K: np.ndarray) -> np.ndarray:
    """Reproject 2D pixel+depth to camera-space XYZ.

    Args:
        uvd: (T, N, 3) float32 tensor — (u, v) in pixels, d in metres
        K:   (T, 3, 3) float32 array — per-frame intrinsics

    Returns:
        (T, N, 3) float32 numpy array — camera-space XYZ
    """
    u, v, d = uvd[:, :, 0], uvd[:, :, 1], uvd[:, :, 2]
    Kt = torch.from_numpy(K).to(uvd.device)
    fx = Kt[:, 0, 0].unsqueeze(-1)
    fy = Kt[:, 1, 1].unsqueeze(-1)
    cx = Kt[:, 0, 2].unsqueeze(-1)
    cy = Kt[:, 1, 2].unsqueeze(-1)
    x = (u - cx) / fx * d
    y = (v - cy) / fy * d
    return torch.stack([x, y, d], dim=-1).cpu().numpy().astype(np.float32)


# ── data roots ───────────────────────────────────────────────────────────────
TAPVID3D_ROOT = Path("/home/mas/data/tapvid3d")
DA3_ROOT = Path("/home/mas/data/tapvid3d_da3")

# ── TAPVid-3D minival clip lists — imported from canonical source above ───────


def load_da3_depth(subset: str, clip_name: str, F: int) -> np.ndarray:
    """Load and decode DA3 depth → float32 (F, H_da3, W_da3) in metres."""
    p = DA3_ROOT / subset / clip_name
    with np.load(p) as d:
        q = d["depth_q"].astype(np.float32)[:F]
        d_min = float(d["d_min"])
        d_max = float(d["d_max"])
    return d_min + q * ((d_max - d_min) / 65535.0)


def decode_images(jpeg_bytes_arr) -> np.ndarray:
    """Decode JPEG byte array → uint8 (F, H, W, 3)."""
    frames = []
    for b in jpeg_bytes_arr:
        img = Image.open(io.BytesIO(bytes(b)))
        frames.append(np.array(img.convert("RGB")))
    return np.stack(frames)


def build_intrinsics(fx_fy_cx_cy, T: int) -> np.ndarray:
    """Build (T, 3, 3) intrinsics from [fx, fy, cx, cy]."""
    fx, fy, cx, cy = fx_fy_cx_cy.tolist()
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
    return np.tile(K[None], (T, 1, 1))


def resize_video_if_needed(
    video: np.ndarray, max_side: int = 960
) -> tuple[np.ndarray, float, float]:
    """
    Resize (T, H, W, 3) video if max side > max_side.
    Returns (resized_video, scale_h, scale_w).
    """
    T, H, W, C = video.shape
    scale = min(max_side / max(H, W), 1.0)
    if scale == 1.0:
        return video, 1.0, 1.0
    new_H, new_W = int(H * scale), int(W * scale)
    # round to even
    new_H = new_H - new_H % 2
    new_W = new_W - new_W % 2
    resized = np.stack([cv2.resize(video[t], (new_W, new_H)) for t in range(T)])
    return resized, new_H / H, new_W / W


def _run_forward(
    model: Predictor, video_t, depth_np, K, extrs_np, queries_txy, F: int
) -> tuple[np.ndarray, np.ndarray]:
    """Single model.forward call for one query batch; returns (F,N,3) and (F,N) numpy arrays.

    Uses fixed_cam=True (paper protocol): no VO estimation, each frame's 3D position
    is independently computed from 2D tracked position + per-frame depth.
    When extrs_np is provided (GT camera poses), those are passed to the model.
    """
    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        result = model.forward(
            video_t,
            depth=depth_np,
            intrs=K,
            extrs=extrs_np,
            queries=queries_txy,
            fps=1,
            full_point=False,
            iters_track=4,
            query_no_BA=True,
            fixed_cam=True,  # paper protocol: no VO, per-frame depth unproject
            stage=1,
            support_frame=F - 1,
            replace_ratio=1.0,  # paper protocol
        )
    # result = (c2w_traj, intrs_out, point_map, unc_metric,
    #           track3d_pred (T,N,6), track2d_pred (T,N,3), vis_pred (T,N,1), conf_pred, video)
    # Use track2d_pred (result[5]): globally-consistent 2D pixel coords + metric depth,
    # then reproject to camera-space XYZ with per-frame intrinsics.
    # track3d_pred (result[4]) is in window-local camera coordinates — with multiple
    # windows (s_wind < clip length), each window uses a different reference frame,
    # making the assembled 3D predictions inconsistent across windows.
    track2d = result[5]  # (T, N, 3): (u_px, v_px, depth_m) in video pixel space
    vis = result[6]  # (T, N, 1)
    xyz = reproject_2d3d(track2d[:F].float(), K[:F])
    vis_np = vis[:F, :, 0].float().cpu().numpy().astype(np.float32)
    return xyz, vis_np


def _run_batched(
    model: Predictor,
    video_t,
    depth_np,
    K,
    extrs_np,
    queries_txy,
    F: int,
    max_queries: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Run model on all queries, batching to avoid OOM if max_queries > 0."""
    N = len(queries_txy)
    if max_queries > 0 and N > max_queries:
        all_xyz, all_vis = [], []
        for i in range(0, N, max_queries):
            xyz_b, vis_b = _run_forward(
                model,
                video_t,
                depth_np,
                K,
                extrs_np,
                queries_txy[i : i + max_queries],
                F,
            )
            torch.cuda.empty_cache()
            all_xyz.append(xyz_b)
            all_vis.append(vis_b)
        return np.concatenate(all_xyz, axis=1), np.concatenate(all_vis, axis=1)
    return _run_forward(model, video_t, depth_np, K, extrs_np, queries_txy, F)


def infer_clip(
    model: Predictor, data: dict, subset: str, max_queries: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run SpaTrackerV2 on one TAPVid-3D clip using bidirectional tracking.
    Returns (tracks_XYZ (F,N,3), visibility (F,N)).

    Uses the paper's evaluation protocol:
      1. Forward pass: track from query frame forward (t >= t_query)
      2. Backward pass on time-reversed video: track from query frame backward (t < t_query)
      3. Combine: use forward predictions for t >= t_query, backward for t < t_query
    This ensures all visible frames (not just post-query) get valid predictions.
    """
    jpeg_bytes = data["images_jpeg_bytes"]
    queries_xyt = data["queries_xyt"].astype(np.float32)  # (N, 3) in (x,y,t)
    fx_fy_cx_cy = data["fx_fy_cx_cy"].astype(np.float32)

    # Decode images
    video_hwc = decode_images(jpeg_bytes)  # (T, H, W, 3)
    T_clip, H_orig, W_orig, _ = video_hwc.shape
    F = T_clip

    # Resize to 336px max
    video_hwc, scale_h, scale_w = resize_video_if_needed(video_hwc, max_side=336)
    _, H, W, _ = video_hwc.shape

    video_t = torch.from_numpy(video_hwc.transpose(0, 3, 1, 2)).float()

    # Load depth
    if subset == "adt" and "depth_preds" in data:
        depth_np = data["depth_preds"].astype(np.float32)[:F]
    elif subset == "drivetrack" and "depth_preds" in data:
        depth_np = data["depth_preds"].astype(np.float32)[:F]
    else:
        depth_np = load_da3_depth(subset, data["_clip_name"], F)

    # Resize depth to match video spatial dims if needed
    if depth_np.shape[1] != H or depth_np.shape[2] != W:
        depth_t = torch.from_numpy(depth_np).unsqueeze(1)
        depth_t = Fn.interpolate(
            depth_t, size=(H, W), mode="bilinear", align_corners=False
        )
        depth_np = depth_t.squeeze(1).numpy()

    # Intrinsics (T, 3, 3) scaled to match resized video
    K = build_intrinsics(fx_fy_cx_cy, F)
    if scale_h != 1.0 or scale_w != 1.0:
        K[:, 0, :] *= scale_w  # fx, cx
        K[:, 1, :] *= scale_h  # fy, cy

    # Extrinsics (drivetrack has GT w2c)
    extrs_np = None
    if subset == "drivetrack" and "extrinsics_w2c" in data:
        w2c = data["extrinsics_w2c"].astype(np.float32)[:F]
        extrs_np = np.linalg.inv(w2c)  # c2w

    # Queries: TAPVid-3D is (x, y, t) → V2 wants (t, x, y)
    queries_txy = queries_xyt[:, [2, 0, 1]].astype(np.float32)  # (N, 3)
    if scale_h != 1.0 or scale_w != 1.0:
        queries_txy[:, 1] *= scale_w  # x
        queries_txy[:, 2] *= scale_h  # y

    # ── Forward pass (t >= t_query) ──────────────────────────────────────────
    fwd_xyz, fwd_vis = _run_batched(
        model, video_t, depth_np, K, extrs_np, queries_txy, F, max_queries
    )

    # ── Backward pass on time-reversed video (t < t_query) ───────────────────
    # Reverse video, depth, and extrinsics in time; adjust query times accordingly.
    video_rev = video_t.flip(0)
    depth_rev = depth_np[::-1].copy()
    K_rev = K[::-1].copy()
    extrs_rev = None if extrs_np is None else extrs_np[::-1].copy()

    inv_queries_txy = queries_txy.copy()
    inv_queries_txy[:, 0] = (F - 1) - queries_txy[:, 0]  # flip time index

    bwd_xyz_rev, bwd_vis_rev = _run_batched(
        model, video_rev, depth_rev, K_rev, extrs_rev, inv_queries_txy, F, max_queries
    )
    # Reverse time axis back to original order
    bwd_xyz = bwd_xyz_rev[::-1].copy()  # (F, N, 3)
    bwd_vis = bwd_vis_rev[::-1].copy()  # (F, N)

    # ── Combine: forward for t >= t_query, backward for t < t_query ──────────
    t_queries = queries_xyt[:, 2].astype(int)  # (N,) original (unscaled) query times
    t_arr = np.arange(F)[:, None]  # (F, 1)
    fwd_mask = t_arr >= t_queries[None, :]  # (F, N) True → use forward

    tracks_XYZ = np.where(fwd_mask[:, :, None], fwd_xyz, bwd_xyz)
    visibility = np.where(fwd_mask, fwd_vis, bwd_vis)

    return tracks_XYZ, visibility


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subsets", nargs="+", default=["pstudio", "adt", "drivetrack"])
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/home/mas/data/tapvid3d_baseline_preds/spatrackerv2"),
    )
    ap.add_argument("--device", default="cuda")
    ap.add_argument(
        "--max-queries",
        type=int,
        default=0,
        help="If >0, split queries into batches of this size to save GPU memory",
    )
    args = ap.parse_args()

    # ── load model ───────────────────────────────────────────────────────────
    ckpt_path = str(SPA_ROOT / "checkpoints" / "SpaTrack3_offline.pth")
    model_args = {
        "Track_cfg": {
            "base": {"corr_radius": 3, "stride": 4, "window_len": 60},
            "base_ckpt": ckpt_path,
            "mode": "online",
            "overlap": 4,
            # s_wind=60: VRAM constraint on RTX 4080 Laptop (12 GB).
            # The paper's official magic_infer_offline.yaml uses s_wind=500,
            # which fits a 300-frame ADT clip in a single pass but exceeds 12 GB.
            # s_wind=300 already OOMs on this GPU; s_wind=60 is the maximum feasible.
            # This is the dominant remaining gap to the paper's 24.7% normalised AJ.
            "s_wind": 60,
            "stablizer": True,
        },
        "backbone_cfg": {"ckpt_dir": "doesnotexist"},  # moge_as_base=False → skipped
        "chunk_size": 24,
        "ckpt_fwd": True,
        "ft_cfg": {"mode": "fix", "paras_name": []},
        "max_len": 512,
        "resolution": 336,
        # 64 VO support points → minimal correlation volume; accuracy tradeoff is small
        # for short clips where camera motion is modest.
        "track_num": 64,
    }
    model = Predictor(args=model_args)
    model.eval()
    model.to(args.device)
    print(f"[spatracker_v2] model loaded from {ckpt_path}")

    # ── run inference ─────────────────────────────────────────────────────────
    for subset in args.subsets:
        if subset not in MINIVAL_FILES:
            print(f"[warn] unknown subset {subset!r}, skip")
            continue
        out_sub = args.out_dir / subset
        out_sub.mkdir(parents=True, exist_ok=True)

        clips = MINIVAL_FILES[subset]
        for clip_name in tqdm(clips, desc=subset):
            out_path = out_sub / clip_name
            if out_path.exists():
                continue

            npz_path = TAPVID3D_ROOT / subset / clip_name
            # allow_pickle=True is safe here: these are our own local dataset files
            # under /home/mas/data/tapvid3d/ (TAPVid-3D benchmark data, not user input).
            data = dict(np.load(npz_path, allow_pickle=True))
            data["_clip_name"] = clip_name

            try:
                tracks_XYZ, visibility = infer_clip(
                    model, data, subset, max_queries=args.max_queries
                )
                np.savez_compressed(
                    out_path, tracks_XYZ=tracks_XYZ, visibility=visibility
                )
                torch.cuda.empty_cache()
            except torch.cuda.OutOfMemoryError as e:
                torch.cuda.empty_cache()
                print(f"[OOM]   {subset}/{clip_name}: {e}")
            except Exception as e:
                torch.cuda.empty_cache()
                print(f"[error] {subset}/{clip_name}: {e}")

    print(f"[spatracker_v2] done → {args.out_dir}")


if __name__ == "__main__":
    main()
