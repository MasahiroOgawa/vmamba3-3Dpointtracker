"""Render qualitative SEA-RAFT+DA3 tracking results (training-free).

The repo's three render scripts (render_tracker_video / render_3d_tracks /
render_space_time_tracks) are all keyed to a trained Mamba3Tracker `--ckpt` and
reconstruct tracks from an xyz-delta cumsum. SEA-RAFT+DA3 has no ckpt, so this
thin driver reuses the SAME render functions but feeds them the training-free
SEA-RAFT-chaining + DA3-unproject predictions (identical inference path to
scripts/eval_searaft_tracker.py).

Per clip it writes the repo-standard artifact set:
    <sub>_<clip>.mp4       2D tracks overlaid on the video frames
    <sub>_<clip>_3d.png    static 3D trajectory (pred solid / GT dashed)
    <sub>_<clip>_3d.html   interactive 3D trajectory (plotly; rotate/zoom)
    <sub>_<clip>_st.png    space-time X-t / Y-t / Z-t (3D position over time)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from mamba3_tracker.data.dataset import filter_to_split
from mamba3_tracker.data.tapvid3d import SUBSETS, has_images, list_clips, load_clip
from mamba3_tracker.train.loss import _unproject_with_depth
from mamba3_tracker.viz.track_video import render_tracking_video
from searaft_flow import FlowModel, track_clip

# Reuse the existing 3D / space-time plot functions (pure numpy renderers).
sys.path.insert(0, str(Path(__file__).resolve().parent))
import render_3d_tracks as r3d          # noqa: E402
import render_space_time_tracks as rst  # noqa: E402


def _load_depth(da3_depth_root: Path, subset: str, clip_id: str, F_: int) -> torch.Tensor:
    depth_path = Path(da3_depth_root).expanduser() / subset / (clip_id + ".npz")
    with np.load(depth_path) as dd:
        if "depth_q" in dd:
            q = np.asarray(dd["depth_q"][:F_]).astype(np.float32)
            d_min, d_max = float(dd["d_min"]), float(dd["d_max"])
            depth_full = d_min + q * (max(d_max - d_min, 1e-6) / 65535.0)
        else:
            depth_full = np.asarray(dd["depth"][:F_], dtype=np.float32)
    return torch.from_numpy(depth_full).unsqueeze(0)   # (1, F, Hd, Wd)


@torch.no_grad()
def _infer_clip(flow_model, clip, image_size, fb_alpha, fb_beta, da3_depth_root, max_frames):
    """Return (pred_tracks (N,F,3) camera-frame XYZ, pred_vis (N,F))."""
    F_ = int(clip.images.shape[0]) if not max_frames else min(int(clip.images.shape[0]), max_frames)
    images = clip.images[:F_].clone()
    H_orig, W_orig = images.shape[-2], images.shape[-1]
    if (H_orig, W_orig) != (image_size, image_size):
        images = F.interpolate(images, size=(image_size, image_size), mode="bilinear", align_corners=False)
    images = images * 255.0

    sx, sy = image_size / float(W_orig), image_size / float(H_orig)
    q = clip.queries_xyt.clone()
    queries_xy = torch.stack([q[:, 0] * sx, q[:, 1] * sy], dim=-1)
    anchor_t = q[:, 2].long().clamp(0, F_ - 1)

    uv, vis = track_clip(flow_model, images, queries_xy, anchor_t, image_size, fb_alpha, fb_beta)

    depth_t = _load_depth(da3_depth_root, clip.subset, clip.clip_id, F_)
    K = clip.K.clone()
    K[0] *= sx   # intrinsics in the image_size² working space
    K[1] *= sy
    xyz = _unproject_with_depth(uv.unsqueeze(0), depth_t, K.unsqueeze(0), float(image_size))[0]  # (F,N,3)
    return xyz.transpose(0, 1).numpy(), vis.transpose(0, 1).numpy()


def _frames_to_uint8(images_01: torch.Tensor) -> np.ndarray:
    return (images_01.clamp(0, 1) * 255).byte().permute(0, 2, 3, 1).numpy()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, default=Path("~/data"))
    ap.add_argument("--da3-depth-root", type=Path, default=Path("~/data/tapvid3d_da3"))
    ap.add_argument("--subsets", nargs="+", default=list(SUBSETS))
    ap.add_argument("--split", choices=["all", "minival", "full_eval"], default="minival")
    ap.add_argument("--clips-per-subset", type=int, default=2)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--max-tracks", type=int, default=32)
    ap.add_argument("--image-size", type=int, default=896)
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--url", type=str, default="MemorySlices/Tartan-C-T-TSKH-spring540x960-M")
    ap.add_argument("--iters", type=int, default=None)
    ap.add_argument("--scale", type=int, default=None)
    ap.add_argument("--fb-alpha", type=float, default=0.05)
    ap.add_argument("--fb-beta", type=float, default=1.0)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.data_root = args.data_root.expanduser()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    flow_model = FlowModel(device, url=args.url, iters=args.iters, scale=args.scale)
    print(f"[viz] SEA-RAFT loaded ({args.url}) iters={flow_model.args.iters} scale={flow_model.args.scale}")

    for sub in args.subsets:
        clips = [p for p in filter_to_split(list_clips(args.data_root, [sub]), args.split)
                 if has_images(p)][: args.clips_per_subset]
        print(f"[viz] {sub}: rendering {len(clips)} clips")
        for path in clips:
            try:
                clip = load_clip(path)
                pred_NF3, vis_NF = _infer_clip(
                    flow_model, clip, args.image_size, args.fb_alpha, args.fb_beta,
                    args.da3_depth_root, args.max_frames,
                )
                F_ = pred_NF3.shape[1]
                gt_NF3 = np.transpose(clip.tracks_XYZ[:F_].numpy(), (1, 0, 2))   # (N,F,3)
                gt_vis_NF = np.transpose(clip.visibility[:F_].float().numpy(), (1, 0))  # (N,F)
                anchor_n = clip.queries_xyt[:, 2].long().clamp(0, F_ - 1).numpy()
                times_s = np.arange(F_, dtype=np.float32) / float(args.fps)
                K = clip.K.numpy()
                stem = f"{sub}_{clip.clip_id}"
                title = f"SEA-RAFT+DA3  —  {sub}/{clip.clip_id}"

                # 2D overlay video (camera-frame xyz projected with original K).
                frames = _frames_to_uint8(clip.images[:F_])
                render_tracking_video(frames, pred_NF3, vis_NF, K,
                                      args.out_dir / f"{stem}.mp4", fps=args.fps)
                # 3D trajectory (static + interactive). GT visibility for track picking.
                r3d._plot_clip_3d_png(pred_NF3, gt_NF3, gt_vis_NF, anchor_n,
                                      args.out_dir / f"{stem}_3d.png", title, args.max_tracks)
                r3d._plot_clip_3d_html(pred_NF3, gt_NF3, gt_vis_NF, anchor_n,
                                       args.out_dir / f"{stem}_3d.html", title, args.max_tracks)
                # Space-time (3D position over time).
                rst._plot_clip_st_png(pred_NF3, gt_NF3, gt_vis_NF, anchor_n, times_s,
                                      args.out_dir / f"{stem}_st.png", title, args.max_tracks)
                print(f"[viz] wrote {stem}.mp4 / _3d.png / _3d.html / _st.png", flush=True)
            except Exception as e:
                print(f"[viz] {sub}/{path.stem}: FAIL {type(e).__name__}: {e}", flush=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print(f"[viz] DONE — artifacts in {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
