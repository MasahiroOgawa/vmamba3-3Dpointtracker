"""Unified qualitative renderer for the SEA-RAFT+DA3 tracking family.

ONE script that outputs every artifact for either method:
  * SEA-RAFT + DA3 (training-free)            --method searaft
  * v33 Mamba-3 depth refiner (ckpt)          --method v33 --ckpt <path>

The repo's render_all_latest_results.sh / render_*_tracks.py only drive the
trained Mamba3Tracker (xyz-delta) path, so neither the training-free SEA-RAFT
result nor the v33 depth-refiner can be visualised with them. This driver
reuses the SAME render functions (render_tracking_video, render_3d_tracks and
render_space_time_tracks plot helpers) but feeds them whichever method's
predictions, plus a real-metric error report so absolute-depth quality (which
the median-scaled 3D-AJ metric hides) can be compared between methods.

Per clip it writes:
    <sub>_<clip>.mp4       2D tracks overlaid on the video frames
    <sub>_<clip>_3d.png    static 3D trajectory (pred solid / GT dashed)
    <sub>_<clip>_3d.html   interactive 3D trajectory (plotly)
    <sub>_<clip>_st.png    space-time X-t / Y-t / Z-t (3D position over time)
and a run-level `metrics.json` + printed table of mean real-metric 3D error
(metres), raw and after global median scaling.

`--scaling` controls what is plotted/measured:
    none    raw predicted depth (true output; starts will NOT overlap GT
            because DA3 metric depth is scale-biased)
    median  one global scale = median(GT_Z / pred_Z) over visible points
            (what TAPVid-3D 3D-AJ applies before scoring)
    anchor  per-track scale so each track's anchor-frame depth matches GT
            (injects the GT starting depth → starts overlap, shows depth drift)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

# Must be set before torch initialises the CUDA allocator.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
import torch.nn.functional as F

from mamba3_tracker.data.dataset import filter_to_split
from mamba3_tracker.data.tapvid3d import SUBSETS, has_images, list_clips, load_clip
from mamba3_tracker.train.loss import _unproject_with_depth
from mamba3_tracker.viz.track_video import (
    render_tracking_d4rt_html,
    render_tracking_video,
)
from searaft_flow import FlowModel, track_clip

sys.path.insert(0, str(Path(__file__).resolve().parent))
import render_3d_tracks as r3d  # noqa: E402
import render_space_time_tracks as rst  # noqa: E402


def _load_depth(
    da3_depth_root: Path, subset: str, clip_id: str, F_: int
) -> torch.Tensor:
    depth_path = Path(da3_depth_root).expanduser() / subset / (clip_id + ".npz")
    with np.load(depth_path) as dd:
        if "depth_q" in dd:
            q = np.asarray(dd["depth_q"][:F_]).astype(np.float32)
            d_min, d_max = float(dd["d_min"]), float(dd["d_max"])
            depth_full = d_min + q * (max(d_max - d_min, 1e-6) / 65535.0)
        else:
            depth_full = np.asarray(dd["depth"][:F_], dtype=np.float32)
    return torch.from_numpy(depth_full).unsqueeze(0)  # (1, F, Hd, Wd)


def _ray_from_uv(uv: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    fx, fy = K[:, 0, 0].view(-1, 1, 1), K[:, 1, 1].view(-1, 1, 1)
    cx, cy = K[:, 0, 2].view(-1, 1, 1), K[:, 1, 2].view(-1, 1, 1)
    return torch.stack([(uv[..., 0] - cx) / fx, (uv[..., 1] - cy) / fy], dim=-1)


@torch.no_grad()
def _infer(
    method,
    flow_model,
    model,
    clip,
    image_size,
    fb_alpha,
    fb_beta,
    da3_depth_root,
    max_frames,
    device,
    amp_dtype=torch.bfloat16,
):
    """Return (pred_tracks (N,F,3) camera-frame XYZ, pred_vis (N,F))."""
    F_ = (
        int(clip.images.shape[0])
        if not max_frames
        else min(int(clip.images.shape[0]), max_frames)
    )
    images = clip.images[:F_].clone()
    H_orig, W_orig = images.shape[-2], images.shape[-1]
    if (H_orig, W_orig) != (image_size, image_size):
        images = F.interpolate(
            images, size=(image_size, image_size), mode="bilinear", align_corners=False
        )
    images_255 = images * 255.0

    sx, sy = image_size / float(W_orig), image_size / float(H_orig)
    q = clip.queries_xyt.clone()
    queries_xy = torch.stack([q[:, 0] * sx, q[:, 1] * sy], dim=-1)
    anchor_t = q[:, 2].long().clamp(0, F_ - 1)

    uv, vis = track_clip(
        flow_model,
        images_255.to(device),
        queries_xy,
        anchor_t,
        image_size,
        fb_alpha,
        fb_beta,
    )
    depth_t = _load_depth(da3_depth_root, clip.subset, clip.clip_id, F_).to(device)
    K = clip.K.clone()
    K[0] *= sx
    K[1] *= sy
    K_t = K.unsqueeze(0).to(device)
    uv_d = uv.unsqueeze(0).to(device)

    if method == "searaft":
        xyz = _unproject_with_depth(uv_d, depth_t, K_t, float(image_size))[0]  # (F,N,3)
    elif method == "v35":
        ray = _ray_from_uv(uv_d, K_t)
        grid = (2.0 * uv_d / image_size - 1.0).view(F_, 1, -1, 2)
        z_raw = F.grid_sample(
            depth_t.squeeze(0).unsqueeze(1),
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        ).view(1, F_, -1)
        # Cast to amp_dtype to halve the ~1.35 GB fp32 image tensor
        images_b = images.unsqueeze(0).to(device, dtype=amp_dtype)
        with torch.autocast(device_type=device.type, dtype=amp_dtype):
            xyz = model(
                ray, z_raw, vis.unsqueeze(0).to(device), uv_d, depth_t, images_b, K_t
            ).xyz[0]
    else:  # v33
        ray = _ray_from_uv(uv_d, K_t)
        grid = (2.0 * uv_d / image_size - 1.0).view(F_, 1, -1, 2)
        z_raw = F.grid_sample(
            depth_t.squeeze(0).unsqueeze(1),
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        ).view(1, F_, -1)
        with torch.autocast(device_type=device.type, dtype=amp_dtype):
            xyz = model(ray, z_raw, vis.unsqueeze(0).to(device)).xyz[0]  # (F,N,3)
    return xyz.transpose(0, 1).cpu().numpy(), vis.transpose(0, 1).numpy()


def _apply_scaling(pred_NF3, gt_NF3, vis_NF, anchor_n, mode):
    """Return a scaled copy of pred_NF3 per the chosen mode (for plot + metric)."""
    if mode == "none":
        return pred_NF3
    pred = pred_NF3.copy()
    pZ, gZ = pred[..., 2], gt_NF3[..., 2]
    if mode == "median":
        m = (
            (vis_NF > 0.5)
            & np.isfinite(pZ)
            & np.isfinite(gZ)
            & (pZ > 1e-3)
            & (gZ > 1e-3)
        )
        s = np.median(gZ[m] / pZ[m]) if m.any() else 1.0
        return pred * s
    if mode == "anchor":  # per-track scale to match GT depth at the anchor frame
        N = pred.shape[0]
        for n in range(N):
            a = int(anchor_n[n])
            pa, ga = pZ[n, a], gZ[n, a]
            if pa > 1e-3 and ga > 1e-3:
                pred[n] *= ga / pa
        return pred
    raise ValueError(mode)


def _metric_err(pred_NF3, gt_NF3, vis_NF):
    """Mean real-metric 3D error (metres) over visible points."""
    m = (vis_NF > 0.5) & np.isfinite(pred_NF3).all(-1) & np.isfinite(gt_NF3).all(-1)
    if not m.any():
        return float("nan")
    return float(np.linalg.norm((pred_NF3 - gt_NF3)[m], axis=-1).mean())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["searaft", "v33", "v35"], required=True)
    ap.add_argument(
        "--ckpt", type=Path, default=None, help="required for --method v33/v35"
    )
    ap.add_argument(
        "--style",
        choices=["tapvid", "d4rt"],
        default="tapvid",
        help="tapvid: fading lines + circles; d4rt: vivid rainbow dot trails",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="output dir; defaults to <ckpt_parent>/viz_ckpt<step>_<style>/",
    )
    ap.add_argument("--data-root", type=Path, default=Path("~/data"))
    ap.add_argument("--da3-depth-root", type=Path, default=Path("~/data/tapvid3d_da3"))
    ap.add_argument("--subsets", nargs="+", default=list(SUBSETS))
    ap.add_argument(
        "--split", choices=["all", "minival", "full_eval"], default="minival"
    )
    ap.add_argument("--clips-per-subset", type=int, default=2)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--max-tracks", type=int, default=32)
    ap.add_argument("--image-size", type=int, default=896)
    ap.add_argument("--scaling", choices=["none", "median", "anchor"], default="none")
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument(
        "--tail-seconds",
        type=float,
        default=3.0,
        help="D4RT: seconds of track history shown as fading trail (default 1.0)",
    )
    ap.add_argument(
        "--pcloud-stride",
        type=int,
        default=16,
        help="D4RT point cloud: sample every N pixels (higher = sparser, smaller HTML)",
    )
    ap.add_argument(
        "--url", type=str, default="MemorySlices/Tartan-C-T-TSKH-spring540x960-M"
    )
    ap.add_argument("--iters", type=int, default=None)
    ap.add_argument("--scale", type=int, default=None)
    ap.add_argument("--fb-alpha", type=float, default=0.05)
    ap.add_argument("--fb-beta", type=float, default=1.0)
    ap.add_argument("--amp", choices=["bf16", "fp32"], default="bf16")
    args = ap.parse_args()

    if args.out_dir is None:
        if args.method == "searaft":
            ap.error("--out-dir is required for --method searaft (no checkpoint)")
        step = (
            args.ckpt.stem.split("_")[-1] if "_" in args.ckpt.stem else args.ckpt.stem
        )
        args.out_dir = args.ckpt.parent / f"viz_ckpt{step}_{args.style}"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.data_root = args.data_root.expanduser()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    flow_model = FlowModel(device, url=args.url, iters=args.iters, scale=args.scale)
    model = None
    if args.method == "v33":
        if args.ckpt is None:
            ap.error("--method v33 requires --ckpt")
        from mamba3_tracker.model.depth_refined_tracker import Mamba3DepthRefiner

        state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        mc = state.get("cfg", {}).get("model", {})
        model = Mamba3DepthRefiner(
            dim=int(mc.get("dim", 128)),
            state_dim=int(mc.get("state_dim", 64)),
            num_heads=int(mc.get("num_heads", 4)),
            num_layers=int(mc.get("num_layers", 2)),
            max_log_correction=float(mc.get("max_log_correction", 2.0)),
        ).to(device)
        model.load_state_dict(state["model"])
        model.eval()
        print(f"[viz] v33 ckpt {args.ckpt} (step={state.get('step', '?')})")
    elif args.method == "v35":
        if args.ckpt is None:
            ap.error("--method v35 requires --ckpt")
        from mamba3_tracker.model.depth_refined_tracker import Mamba3V35Refiner

        state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        mc = state.get("cfg", {}).get("model", {})
        model = Mamba3V35Refiner(
            dim=int(mc.get("dim", 128)),
            state_dim=int(mc.get("state_dim", 64)),
            num_heads=int(mc.get("num_heads", 4)),
            num_layers=int(mc.get("num_layers", 2)),
            max_log_correction=float(mc.get("max_log_correction", 2.0)),
            max_delta_uv=float(mc.get("max_delta_uv", 2.0)),
            patch_size=int(mc.get("patch_size", 5)),
            d_proj=int(mc.get("d_proj", 64)),
            dino_model=str(
                mc.get("dino_model", "facebook/dinov3-vits16-pretrain-lvd1689m")
            ),
            dino_image_size=int(mc.get("dino_image_size", 448)),
            image_size=int(mc.get("image_size", 896)),
        ).to(device)
        model.load_state_dict(state["model"])
        model.eval()
        print(f"[viz] v35 ckpt {args.ckpt} (step={state.get('step', '?')})")
    amp_dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}[args.amp]
    print(
        f"[viz] method={args.method}  style={args.style}  scaling={args.scaling}  "
        f"amp={args.amp}  SEA-RAFT iters={flow_model.args.iters} scale={flow_model.args.scale}"
    )

    raw_err: dict[str, list[float]] = defaultdict(list)
    med_err: dict[str, list[float]] = defaultdict(list)
    per_clip_metrics: list[dict] = []

    for sub in args.subsets:
        clips = [
            p
            for p in filter_to_split(list_clips(args.data_root, [sub]), args.split)
            if has_images(p)
        ][: args.clips_per_subset]
        print(f"[viz] {sub}: {len(clips)} clips")
        for path in clips:
            try:
                clip = load_clip(path)
                pred_NF3, vis_NF = _infer(
                    args.method,
                    flow_model,
                    model,
                    clip,
                    args.image_size,
                    args.fb_alpha,
                    args.fb_beta,
                    args.da3_depth_root,
                    args.max_frames,
                    device,
                    amp_dtype,
                )
                F_ = pred_NF3.shape[1]
                gt_NF3 = np.transpose(clip.tracks_XYZ[:F_].numpy(), (1, 0, 2))
                gt_vis_NF = np.transpose(clip.visibility[:F_].float().numpy(), (1, 0))
                anchor_n = clip.queries_xyt[:, 2].long().clamp(0, F_ - 1).numpy()
                times_s = np.arange(F_, dtype=np.float32) / float(args.fps)
                K = clip.K.numpy()
                stem = f"{sub}_{clip.clip_id}"

                e_raw = _metric_err(pred_NF3, gt_NF3, gt_vis_NF)
                e_med = _metric_err(
                    _apply_scaling(pred_NF3, gt_NF3, gt_vis_NF, anchor_n, "median"),
                    gt_NF3,
                    gt_vis_NF,
                )
                raw_err[sub].append(e_raw)
                med_err[sub].append(e_med)
                per_clip_metrics.append(
                    {
                        "subset": sub,
                        "clip_id": clip.clip_id,
                        "metric_err_raw_m": e_raw,
                        "metric_err_median_m": e_med,
                    }
                )

                pred_plot = _apply_scaling(
                    pred_NF3, gt_NF3, gt_vis_NF, anchor_n, args.scaling
                )
                title = (
                    f"{args.method.upper()}  —  {sub}/{clip.clip_id}\n"
                    f"real-metric 3D err: raw={e_raw:.2f} m, median-scaled={e_med:.2f} m "
                    f"(plot scaling={args.scaling})"
                )

                if args.style == "d4rt":
                    H_orig = clip.images.shape[-2]
                    W_orig = clip.images.shape[-1]
                    sx = args.image_size / W_orig
                    sy = args.image_size / H_orig
                    # images at image_size (same transform as in _infer)
                    imgs = clip.images[:F_].clamp(0, 1)
                    if (H_orig, W_orig) != (args.image_size, args.image_size):
                        imgs = F.interpolate(
                            imgs,
                            size=(args.image_size, args.image_size),
                            mode="bilinear",
                            align_corners=False,
                        )
                    frames_rgb = (imgs * 255).byte().permute(0, 2, 3, 1).numpy()
                    # depth resized to image_size so it's consistent with K_scaled
                    depth_raw = _load_depth(
                        args.da3_depth_root, clip.subset, clip.clip_id, F_
                    )
                    depth_np = F.interpolate(
                        depth_raw.squeeze(0).unsqueeze(1),  # (F_, 1, Hd, Wd)
                        size=(args.image_size, args.image_size),
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze(1).numpy()  # (F_, image_size, image_size)
                    K_scaled = K.copy()
                    K_scaled[0] *= sx
                    K_scaled[1] *= sy
                    render_tracking_d4rt_html(
                        pred_NF3,
                        vis_NF,
                        args.out_dir / f"{stem}_d4rt.html",
                        fps=args.fps,
                        tail_seconds=args.tail_seconds,
                        max_tracks=args.max_tracks,
                        images_rgb=frames_rgb,
                        depth_hw=depth_np,
                        K_mat=K_scaled,
                        pcloud_stride=args.pcloud_stride,
                    )
                else:
                    frames = (
                        (clip.images[:F_].clamp(0, 1) * 255)
                        .byte()
                        .permute(0, 2, 3, 1)
                        .numpy()
                    )
                    render_tracking_video(
                        frames,
                        pred_NF3,
                        vis_NF,
                        K,
                        args.out_dir / f"{stem}.mp4",
                        fps=args.fps,
                    )
                r3d._plot_clip_3d_png(
                    pred_plot,
                    gt_NF3,
                    gt_vis_NF,
                    anchor_n,
                    args.out_dir / f"{stem}_3d.png",
                    title,
                    args.max_tracks,
                )
                r3d._plot_clip_3d_html(
                    pred_plot,
                    gt_NF3,
                    gt_vis_NF,
                    anchor_n,
                    args.out_dir / f"{stem}_3d.html",
                    title,
                    args.max_tracks,
                )
                rst._plot_clip_st_png(
                    pred_plot,
                    gt_NF3,
                    gt_vis_NF,
                    anchor_n,
                    times_s,
                    args.out_dir / f"{stem}_st.png",
                    title,
                    args.max_tracks,
                )
                vid_ext = "html(d4rt)" if args.style == "d4rt" else "mp4"
                print(
                    f"[viz] {stem}: raw={e_raw:.2f}m median={e_med:.2f}m → {vid_ext}/_3d/_st",
                    flush=True,
                )
            except Exception as e:
                print(
                    f"[viz] {sub}/{path.stem}: FAIL {type(e).__name__}: {e}", flush=True
                )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    overall_raw = (
        np.nanmean([v for vs in raw_err.values() for v in vs])
        if raw_err
        else float("nan")
    )
    overall_med = (
        np.nanmean([v for vs in med_err.values() for v in vs])
        if med_err
        else float("nan")
    )
    summary = {
        "method": args.method,
        "scaling": args.scaling,
        "per_subset": {
            s: {
                "metric_err_raw_m": float(np.nanmean(raw_err[s])),
                "metric_err_median_m": float(np.nanmean(med_err[s])),
            }
            for s in raw_err
        },
        "overall": {
            "metric_err_raw_m": float(overall_raw),
            "metric_err_median_m": float(overall_med),
        },
        "per_clip": per_clip_metrics,
    }
    (args.out_dir / "metrics.json").write_text(json.dumps(summary, indent=2))

    print(f"\n[viz] === real-metric 3D error ({args.method}) ===")
    print("subset       raw(m)  median-scaled(m)")
    for s in raw_err:
        print(f"{s:11s}  {np.nanmean(raw_err[s]):6.2f}  {np.nanmean(med_err[s]):6.2f}")
    print(f"{'OVERALL':11s}  {overall_raw:6.2f}  {overall_med:6.2f}")
    print(f"[viz] DONE — artifacts + metrics.json in {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
