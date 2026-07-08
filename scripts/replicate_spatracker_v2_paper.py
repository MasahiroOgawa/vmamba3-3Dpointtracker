#!/usr/bin/env python3
"""
Replicate SpatialTrackerV2 paper results on TAPVid-3D using the official protocol:
  - s_wind=300 (covers full ADT/pstudio clip in one window; official is s_wind=500)
  - track2d_pred + reproject_2d3d  (official evaluator, NOT track3d_pred)
  - bidirectional tracking

Compares three conditions on 5 ADT clips:
  A) Our buggy code:  s_wind=60  + track3d_pred  → expected ~2.64%
  B) Partial fix:     s_wind=60  + track2d_pred  → expected ~15-20%
  C) Full fix:        s_wind=300 + track2d_pred  → expected ~20-25% (paper-equivalent)

Run from SpaTrackerV2 venv:
  cd ~/proj/study/SpaTrackerV2
  .venv/bin/python /home/mas/proj/study/largescale3Dreconstruction_using_SSM/scripts/replicate_spatracker_v2_paper.py
"""

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

# ── SpaTrackerV2 root ────────────────────────────────────────────────────────
SPA_ROOT = Path(__file__).resolve().parent.parent.parent / "SpaTrackerV2"
assert SPA_ROOT.exists(), f"SpaTrackerV2 not found at {SPA_ROOT}"
sys.path.insert(0, str(SPA_ROOT))

_PROJ_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJ_ROOT / "src"))

# Stub utils3d.torch (not in public repo's installed version)
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
        return x.unfold(dim[0], window_size if isinstance(window_size, int) else window_size[0], stride).unfold(
            dim[1], window_size if isinstance(window_size, int) else window_size[1], stride
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

from models.SpaTrackV2.models.predictor import Predictor  # noqa: E402
from models.SpaTrackV2.evaluation.core.tapvid3d_metrics import compute_tapvid3d_metrics  # noqa: E402
from mamba3_tracker.data.tapvid3d_splits import MINIVAL_FILES  # noqa: E402

TAPVID3D_ROOT = Path("/home/mas/data/tapvid3d")
CKPT = str(SPA_ROOT / "checkpoints" / "SpaTrack3_offline.pth")
DEVICE = "cuda"


# ── utils ────────────────────────────────────────────────────────────────────

def decode_images(jpeg_bytes_arr) -> np.ndarray:
    return np.stack([np.array(Image.open(io.BytesIO(bytes(b))).convert("RGB")) for b in jpeg_bytes_arr])


def resize_video(video: np.ndarray, max_side: int = 336):
    T, H, W, C = video.shape
    scale = min(max_side / max(H, W), 1.0)
    if scale == 1.0:
        return video, 1.0, 1.0
    nH = int(H * scale) & ~1
    nW = int(W * scale) & ~1
    return np.stack([cv2.resize(video[t], (nW, nH)) for t in range(T)]), nH / H, nW / W


def build_K(fx_fy_cx_cy, T, scale_h=1.0, scale_w=1.0):
    fx, fy, cx, cy = fx_fy_cx_cy.tolist()
    K = np.array([[fx * scale_w, 0, cx * scale_w],
                  [0, fy * scale_h, cy * scale_h],
                  [0, 0, 1]], dtype=np.float32)
    return np.tile(K[None], (T, 1, 1))


def reproject_2d3d(uvd: torch.Tensor, K: np.ndarray) -> np.ndarray:
    """uvd: (T, N, 3), K: (T, 3, 3) → (T, N, 3) camera-space XYZ."""
    u = uvd[:, :, 0]
    v = uvd[:, :, 1]
    d = uvd[:, :, 2]
    Kt = torch.from_numpy(K).to(uvd.device)
    fx = Kt[:, 0, 0].unsqueeze(-1)   # (T, 1)
    fy = Kt[:, 1, 1].unsqueeze(-1)
    cx = Kt[:, 0, 2].unsqueeze(-1)
    cy = Kt[:, 1, 2].unsqueeze(-1)
    x = (u - cx) / fx * d
    y = (v - cy) / fy * d
    return torch.stack([x, y, d], dim=-1).cpu().numpy().astype(np.float32)


def run_model(model, video_t, depth_np, K, extrs_np, queries_txy, depth_init_np, s_wind, use_track2d):
    """Run one forward pass (forward OR backward); return (xyz, vis) numpy."""
    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        result = model.forward(
            video_t,
            depth=depth_np,
            intrs=K,
            extrs=extrs_np,
            queries=queries_txy,
            fps=1,
            full_point=False,
            iters_track=4,       # match eval_spatracker_v2.py to avoid OOM
            query_no_BA=True,
            fixed_cam=True,
            stage=1,
            support_frame=len(video_t) - 1,
            replace_ratio=1.0,
        )

    F = len(depth_np)  # clip length

    # result = (c2w_traj, intrs_out, point_map, unc_metric,
    #           track3d_pred (T,N,6), track2d_pred (T,N,3), vis_pred (T,N,1), conf_pred, video)
    vis = result[6][:F, :, 0].float().cpu().numpy().astype(np.float32)  # (F, N)

    if use_track2d:
        track2d = result[5][:F].float()  # (F, N, 3)
        uvd = track2d  # [..., :2] = px, [..., 2] = depth
        xyz = reproject_2d3d(uvd, K[:F])
    else:
        track3d = result[4][:F].float().cpu().numpy().astype(np.float32)
        xyz = track3d[:, :, :3]  # (F, N, 3) window-local camera-space XYZ (BUGGY)

    return xyz, vis


def run_bidir(model, video_np, depth_np, K, extrs_np, queries_txy, s_wind, use_track2d,
              max_queries: int = 700):
    """Bidirectional tracking with optional query batching to avoid OOM."""
    F, H, W, _ = video_np.shape
    N = len(queries_txy)
    video_t = torch.from_numpy(video_np.transpose(0, 3, 1, 2)).float()
    video_rev_t = video_t.flip(0)
    depth_rev = depth_np[::-1].copy()
    K_rev = K[::-1].copy()
    extrs_rev = None if extrs_np is None else extrs_np[::-1].copy()

    if max_queries > 0 and N > max_queries:
        # Batch queries to avoid OOM
        fwd_xyz_all, fwd_vis_all = [], []
        bwd_xyz_all, bwd_vis_all = [], []
        for i in range(0, N, max_queries):
            q_batch = queries_txy[i : i + max_queries]
            inv_q = q_batch.copy(); inv_q[:, 0] = (F - 1) - q_batch[:, 0]
            fxyz, fvis = run_model(model, video_t, depth_np, K, extrs_np, q_batch, depth_np, s_wind, use_track2d)
            bxyz_r, bvis_r = run_model(model, video_rev_t, depth_rev, K_rev, extrs_rev, inv_q, depth_rev, s_wind, use_track2d)
            torch.cuda.empty_cache()
            fwd_xyz_all.append(fxyz); fwd_vis_all.append(fvis)
            bwd_xyz_all.append(bxyz_r); bwd_vis_all.append(bvis_r)
        fwd_xyz = np.concatenate(fwd_xyz_all, axis=1)
        fwd_vis = np.concatenate(fwd_vis_all, axis=1)
        bwd_xyz = np.concatenate(bwd_xyz_all, axis=1)[::-1].copy()
        bwd_vis = np.concatenate(bwd_vis_all, axis=1)[::-1].copy()
    else:
        inv_queries = queries_txy.copy()
        inv_queries[:, 0] = (F - 1) - queries_txy[:, 0]
        fwd_xyz, fwd_vis = run_model(model, video_t, depth_np, K, extrs_np,
                                      queries_txy, depth_np, s_wind, use_track2d)
        bwd_xyz_rev, bwd_vis_rev = run_model(model, video_rev_t, depth_rev, K_rev, extrs_rev,
                                              inv_queries, depth_rev, s_wind, use_track2d)
        bwd_xyz = bwd_xyz_rev[::-1].copy()
        bwd_vis = bwd_vis_rev[::-1].copy()

    # Merge: forward for t >= t_query, backward for t < t_query
    t_q = queries_txy[:, 0].astype(int)
    t_arr = np.arange(F)[:, None]
    fwd_mask = t_arr >= t_q[None, :]
    xyz = np.where(fwd_mask[:, :, None], fwd_xyz, bwd_xyz)
    vis = np.where(fwd_mask, fwd_vis, bwd_vis)
    return xyz, vis


def eval_clip(xyz, vis, data):
    """Compute normalized AJ for one clip."""
    gt_xyz = data["tracks_XYZ"]          # (T, N, 3)
    gt_vis = data["visibility"]           # (T, N) bool
    fx_fy_cx_cy = data["fx_fy_cx_cy"].astype(np.float32)

    N, T = gt_xyz.shape[1], gt_xyz.shape[0]

    gt_occluded = (~gt_vis).T.astype(bool)  # (N, T)
    gt_tracks = gt_xyz.transpose(1, 0, 2)   # (N, T, 3)
    pred_occluded = (vis < 0.5).T           # (N, T)
    pred_tracks = xyz.transpose(1, 0, 2)    # (N, T, 3)

    metrics = compute_tapvid3d_metrics(
        gt_occluded=gt_occluded,
        gt_tracks=gt_tracks,
        pred_occluded=pred_occluded,
        pred_tracks=pred_tracks,
        intrinsics_params=fx_fy_cx_cy,
        scaling="median",
        order="n t",
        use_fixed_metric_threshold=False,
    )
    return float(np.asarray(metrics["average_jaccard"]).flat[0])


def load_model(s_wind: int):
    model_args = {
        "Track_cfg": {
            "base": {"corr_radius": 3, "stride": 4, "window_len": 60},
            "base_ckpt": CKPT,
            "mode": "online",
            "overlap": 4,
            "s_wind": s_wind,
            "stablizer": True,
        },
        "backbone_cfg": {"ckpt_dir": "doesnotexist"},
        "chunk_size": 24,
        "ckpt_fwd": True,
        "ft_cfg": {"mode": "fix", "paras_name": []},
        "max_len": 512,
        "resolution": 336,
        "track_num": 64,
    }
    model = Predictor(args=model_args)
    model.eval()
    model.to(DEVICE)
    return model


def run_condition(label, s_wind, use_track2d, clips, n_clips=5):
    print(f"\n{'='*60}")
    print(f"  Condition {label}: s_wind={s_wind}, use_track2d={use_track2d}")
    print(f"{'='*60}")

    model = load_model(s_wind)
    scores = []

    for clip_name in clips[:n_clips]:
        # allow_pickle=True: local TAPVid-3D benchmark files under /home/mas/data/ (not user input)
        data = dict(np.load(TAPVID3D_ROOT / "adt" / clip_name, allow_pickle=True))
        jpeg_bytes = data["images_jpeg_bytes"]
        queries_xyt = data["queries_xyt"].astype(np.float32)  # (N, 3) = (x, y, t)
        fx_fy_cx_cy = data["fx_fy_cx_cy"].astype(np.float32)

        video_np = decode_images(jpeg_bytes)  # (T, H, W, 3)
        T = len(video_np)
        video_np, sh, sw = resize_video(video_np, max_side=336)
        _, H, W, _ = video_np.shape

        depth_np = data["depth_preds"].astype(np.float32)[:T]
        if depth_np.shape[1] != H or depth_np.shape[2] != W:
            d_t = torch.from_numpy(depth_np).unsqueeze(1)
            depth_np = Fn.interpolate(d_t, (H, W), mode="bilinear", align_corners=False).squeeze(1).numpy()

        K = build_K(fx_fy_cx_cy, T, sh, sw)

        queries_txy = queries_xyt[:, [2, 0, 1]].copy()  # (N, 3) = (t, x, y)
        queries_txy[:, 1] *= sw
        queries_txy[:, 2] *= sh

        try:
            xyz, vis = run_bidir(model, video_np, depth_np, K, None, queries_txy, s_wind, use_track2d,
                                 max_queries=700)
            aj = eval_clip(xyz, vis, data)
            scores.append(aj)
            print(f"  {clip_name[:40]:40s}  AJ={aj*100:.2f}%")
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"  {clip_name[:40]:40s}  OOM — skipping")

    if scores:
        mean_aj = np.mean(scores)
        print(f"\n  → Mean normalized AJ: {mean_aj*100:.2f}% (n={len(scores)} clips)")
    else:
        mean_aj = 0.0
        print("  → All clips OOM")

    del model
    torch.cuda.empty_cache()
    return scores


def main():
    # Use low-N clips to avoid OOM on 12GB GPU
    # (GreenDecorationTall N=844 and Apartment N=695+ all OOM)
    low_n_clips = [
        "Apartment_release_meal_seq139_3.npz",        # N=258
        "Apartment_release_multiuser_clean_seq114_0.npz",  # N=294
        "Apartment_release_multiuser_clean_seq120_0.npz",  # N=350
    ]
    clips = low_n_clips

    print("\nReplicating SpatialTrackerV2 paper results on TAPVid-3D ADT")
    print(f"Clips: {[c.split('.')[0][-25:] for c in clips]}")
    print(f"GPU: {torch.cuda.get_device_name(0)}, {torch.cuda.get_device_properties(0).total_memory // 1024**2} MiB")

    results = {}

    # Run on all 3 low-N clips (each ~8 min with bidir, no batching needed)
    n = len(clips)

    # Condition A: our buggy code (s_wind=60, track3d_pred)
    results["A: s_wind=60,  track3d_pred (BUGGY)"] = run_condition(
        "A", s_wind=60, use_track2d=False, clips=clips, n_clips=n
    )

    # Condition B: fix Bug 2 only (s_wind=60, track2d_pred)
    results["B: s_wind=60,  track2d+reproject (partial fix)"] = run_condition(
        "B", s_wind=60, use_track2d=True, clips=clips, n_clips=n
    )

    # Condition C: fix both bugs (s_wind=300, track2d_pred)
    # s_wind=300 fits all 300 ADT frames in one window → paper-equivalent (they use 500)
    results["C: s_wind=300, track2d+reproject (paper-equiv)"] = run_condition(
        "C", s_wind=300, use_track2d=True, clips=clips, n_clips=n
    )

    print("\n" + "="*60)
    print("  Summary")
    print("="*60)
    for label, scores in results.items():
        if scores:
            print(f"  {label:55s} {np.mean(scores)*100:.2f}%")
        else:
            print(f"  {label:55s} OOM")
    print(f"  {'Paper result (full minival, s_wind=500)':55s} ~24.7%")
    print("="*60)


if __name__ == "__main__":
    main()
