"""Absolute-metric 3D-tracking evaluator for the SEA-RAFT+DA3 family.

Runs the full TAPVid-3D minival for one method and reports THREE metric families
per clip so absolute-depth quality (hidden by the leaderboard's scale-invariant
score) is exposed side by side:

  1. median-scaled (reference / leaderboard): 3D-AJ, APD3D, OA
     — TAPVid-3D default: median(‖gt‖/‖pred‖) rescale + depth-relative thresholds.
  2. absolute-metric: metric-AJ, metric-APD3D
     — same official machinery but scaling="none" + fixed metre thresholds
       (1cm..2.56m). Scale-sensitive: punishes DA3's global depth-scale bias.
  3. mean & median real-metric 3D error in metres over visible points
     (median too, since drivetrack ~20 m outliers dominate the mean).

Method:
  --method searaft                          training-free SEA-RAFT chaining + DA3 unproject
  --method v33 --ckpt <path>                Mamba3DepthRefiner (depth-along-ray refiner)

Output: metric_results/<subset>.json (per-clip) + summary.md + metrics.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from mamba3_tracker.data.tapvid3d import SUBSETS, list_clips, load_clip
from mamba3_tracker.eval.tapvid3d_eval import (
    aggregate,
    compute_clip_metrics,
    compute_clip_metrics_absolute,
)
from mamba3_tracker.train.loss import _unproject_with_depth
from searaft_flow import FlowModel, track_clip


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
    bidirectional=False,
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
        bidirectional=bidirectional,
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
        images_b = images.unsqueeze(0).to(device)  # (1,F,3,H,W) in [0,1]
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
        xyz = model(ray, z_raw, vis.unsqueeze(0).to(device)).xyz[0]  # (F,N,3)
    return xyz.transpose(0, 1).cpu().numpy(), vis.transpose(0, 1).numpy()


def _metric_err(pred_NF3, gt_NF3, vis_NF):
    """(mean, median) real-metric 3D error in metres over visible points."""
    m = (vis_NF > 0.5) & np.isfinite(pred_NF3).all(-1) & np.isfinite(gt_NF3).all(-1)
    if not m.any():
        return float("nan"), float("nan")
    d = np.linalg.norm((pred_NF3 - gt_NF3)[m], axis=-1)
    return float(d.mean()), float(np.median(d))


def _load_external(pred_dir: Path, subset: str, clip_id: str):
    """Load a released baseline prediction npz -> (pred (N,F,3), vis (N,F))."""
    p = Path(pred_dir).expanduser() / subset / (clip_id + ".npz")
    with np.load(p) as d:
        tr = np.asarray(d["tracks_XYZ"], dtype=np.float32)  # (F,N,3)
        vis = np.asarray(d["visibility"]).astype(np.float32)  # (F,N)
    return np.transpose(tr, (1, 0, 2)), np.transpose(vis, (1, 0))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--method", choices=["searaft", "v33", "v35", "external"], required=True
    )
    ap.add_argument(
        "--ckpt", type=Path, default=None, help="required for --method v33/v35"
    )
    ap.add_argument(
        "--pred-dir",
        type=Path,
        default=None,
        help="required for --method external: dir with <subset>/<clip>.npz "
        "(keys tracks_XYZ (F,N,3), visibility (F,N))",
    )
    ap.add_argument(
        "--label", type=str, default=None, help="display name (default = method)"
    )
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, default=Path("~/data"))
    ap.add_argument("--da3-depth-root", type=Path, default=Path("~/data/tapvid3d_da3"))
    ap.add_argument("--subsets", nargs="+", default=list(SUBSETS))
    ap.add_argument(
        "--split", choices=["all", "minival", "full_eval"], default="minival"
    )
    ap.add_argument("--max-clips-per-subset", type=int, default=0)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--image-size", type=int, default=896)
    ap.add_argument(
        "--url", type=str, default="MemorySlices/Tartan-C-T-TSKH-spring540x960-M"
    )
    ap.add_argument("--iters", type=int, default=None)
    ap.add_argument("--scale", type=int, default=None)
    ap.add_argument("--fb-alpha", type=float, default=0.05)
    ap.add_argument("--fb-beta", type=float, default=1.0)
    ap.add_argument(
        "--bidirectional",
        action="store_true",
        default=False,
        help="Run SEA-RAFT on reversed video for backward tracking (offline only; doubles flow compute)",
    )
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.data_root = args.data_root.expanduser()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    flow_model = None
    model = None
    if args.method == "external":
        if args.pred_dir is None:
            ap.error("--method external requires --pred-dir")
        print(f"[metric3d] external predictions from {args.pred_dir}")
    else:
        flow_model = FlowModel(device, url=args.url, iters=args.iters, scale=args.scale)
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
        print(f"[metric3d] v33 ckpt {args.ckpt} (step={state.get('step', '?')})")
    if args.method == "v35":
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
        print(f"[metric3d] v35 ckpt {args.ckpt} (step={state.get('step', '?')})")
    if flow_model is not None:
        print(
            f"[metric3d] method={args.method}  SEA-RAFT iters={flow_model.args.iters} scale={flow_model.args.scale}"
        )

    if args.split == "minival":
        from mamba3_tracker.data.tapvid3d_splits import MINIVAL_FILES as ALLOW
    elif args.split == "full_eval":
        from mamba3_tracker.data.tapvid3d_splits import FULL_EVAL_FILES as ALLOW
    else:
        ALLOW = None
    allow = {
        s: (set(ALLOW.get(s, [])) if ALLOW is not None else None) for s in args.subsets
    }

    metrics_root = args.out_dir / "metric_results"
    metrics_root.mkdir(parents=True, exist_ok=True)
    summary: dict[str, dict] = {}
    t_start = time.time()
    n_frames_total = 0
    n_fail = 0

    for sub in args.subsets:
        clips = list_clips(args.data_root, [sub])
        if allow[sub] is not None:
            clips = [p for p in clips if p.name in allow[sub]]
        if args.max_clips_per_subset:
            clips = clips[: args.max_clips_per_subset]
        print(f"[metric3d] {sub}: {len(clips)} clips", flush=True)

        per_clip: list[dict] = []
        for path in clips:
            try:
                clip = load_clip(path)
                if args.method == "external":
                    pred_NF3, pred_vis = _load_external(args.pred_dir, sub, path.stem)
                else:
                    pred_NF3, pred_vis = _infer(
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
                        bidirectional=args.bidirectional,
                    )
                # Align frame/point counts (released preds may truncate frames).
                Fg = int(clip.tracks_XYZ.shape[0])
                F_ = (
                    min(pred_NF3.shape[1], Fg)
                    if not args.max_frames
                    else min(pred_NF3.shape[1], Fg, args.max_frames)
                )
                N_ = min(pred_NF3.shape[0], int(clip.tracks_XYZ.shape[1]))
                pred_NF3 = pred_NF3[:N_, :F_]
                pred_vis = pred_vis[:N_, :F_]
                n_frames_total += F_
                gt_xyz = clip.tracks_XYZ[:F_, :N_].numpy()  # (F,N,3)
                gt_vis = clip.visibility[:F_, :N_].float().numpy()  # (F,N)
                gt_NF3 = np.transpose(gt_xyz, (1, 0, 2))
                gt_vis_NF = np.transpose(gt_vis, (1, 0))
                K = clip.K.numpy()
                # TAPVid-3D defines the depth-relative pixel thresholds relative to
                # 256-px images, so the official evaluator rescales intrinsics by
                # 256/min(H,W) before scoring (matches tapnet evaluate_model.py).
                # Without this, drivetrack (1280x1920) median-AJ reads ~7x low.
                # The absolute metric (fixed-metre thresholds) ignores intrinsics.
                Ho, Wo = int(clip.images.shape[-2]), int(clip.images.shape[-1])
                s256 = 256.0 / min(Ho, Wo)
                intrin = np.array([K[0, 0], K[1, 1], K[0, 2], K[1, 2]]) * s256

                med = compute_clip_metrics(gt_xyz, gt_vis, pred_NF3, pred_vis, intrin)
                ab = compute_clip_metrics_absolute(
                    gt_xyz, gt_vis, pred_NF3, pred_vis, intrin
                )
                mean_m, median_m = _metric_err(pred_NF3, gt_NF3, gt_vis_NF)
                rec = {
                    "clip_id": clip.clip_id,
                    **med,
                    "metric_average_jaccard": ab["metric_average_jaccard"],
                    "metric_average_pts_within_thresh": ab[
                        "metric_average_pts_within_thresh"
                    ],
                    "metric_err_mean_m": mean_m,
                    "metric_err_median_m": median_m,
                }
                per_clip.append(rec)
                print(
                    f"[metric3d] {sub}/{clip.clip_id}: AJ={med['average_jaccard']:.4f} "
                    f"mAJ={ab['metric_average_jaccard']:.4f} "
                    f"mAPD={ab['metric_average_pts_within_thresh']:.4f} "
                    f"err={mean_m:.2f}m(med {median_m:.2f})",
                    flush=True,
                )
            except Exception as e:
                n_fail += 1
                print(
                    f"[metric3d] {sub}/{path.stem}: FAIL {type(e).__name__}: {e}",
                    flush=True,
                )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        (metrics_root / f"{sub}.json").write_text(json.dumps(per_clip, indent=2))
        summary[sub] = aggregate(per_clip)
        print(f"[metric3d] {sub} mean: {summary[sub]}", flush=True)

    overall = aggregate(summary.values())
    elapsed = time.time() - t_start
    fps = n_frames_total / elapsed if elapsed > 0 else float("nan")

    metrics_json = {
        "method": args.method,
        "split": args.split,
        "ckpt": str(args.ckpt) if args.ckpt else None,
        "per_subset": summary,
        "overall": overall,
        "frames": n_frames_total,
        "elapsed_s": elapsed,
        "fps": fps,
        "failures": n_fail,
    }
    (args.out_dir / "metrics.json").write_text(json.dumps(metrics_json, indent=2))

    def _g(d, k):
        return d.get(k, float("nan"))

    rows = [
        f"# Absolute-metric 3D tracking — {args.method} — TAPVid-3D {args.split}",
        f"\nThroughput: {n_frames_total} frames in {elapsed:.0f}s ({fps:.1f} fps on {device}); failures={n_fail}",
        "\nmedian-* = leaderboard (scale-invariant). metric-* = absolute (no median scaling, "
        "fixed-metre thresholds). err = real 3D error in metres over visible points.\n",
        "| subset | 3D-AJ | APD3D | OA | metric-AJ | metric-APD3D | err mean(m) | err median(m) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for sub, mm in summary.items():
        rows.append(
            f"| {sub} | {_g(mm, 'average_jaccard'):.4f} | {_g(mm, 'average_pts_within_thresh'):.4f} | "
            f"{_g(mm, 'occlusion_accuracy'):.4f} | {_g(mm, 'metric_average_jaccard'):.4f} | "
            f"{_g(mm, 'metric_average_pts_within_thresh'):.4f} | "
            f"{_g(mm, 'metric_err_mean_m'):.3f} | {_g(mm, 'metric_err_median_m'):.3f} |"
        )
    rows.append(
        f"| **mean** | **{_g(overall, 'average_jaccard'):.4f}** | "
        f"**{_g(overall, 'average_pts_within_thresh'):.4f}** | "
        f"**{_g(overall, 'occlusion_accuracy'):.4f}** | "
        f"**{_g(overall, 'metric_average_jaccard'):.4f}** | "
        f"**{_g(overall, 'metric_average_pts_within_thresh'):.4f}** | "
        f"**{_g(overall, 'metric_err_mean_m'):.3f}** | **{_g(overall, 'metric_err_median_m'):.3f}** |"
    )
    (args.out_dir / "summary.md").write_text("\n".join(rows) + "\n")
    print(f"\n[metric3d] DONE. {args.out_dir / 'summary.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
