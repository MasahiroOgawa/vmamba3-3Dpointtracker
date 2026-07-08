"""Evaluate Mamba3DepthRefiner (v33) on TAPVid-3D minival.

SEA-RAFT 2D track + FB-consistency visibility are frozen; the SSM refines only
per-track depth along each fixed pixel ray. The reported 3D track is
`z_pred * (ray_x, ray_y, 1)` — its 2D reprojection is identical to SEA-RAFT's,
so the per-clip motion ratio must match the SEA-RAFT baseline exactly.

Output mirrors eval_searaft_tracker.py: per-subset JSONs + summary.md.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from mamba3_tracker.data.tapvid3d import SUBSETS, list_clips, load_clip
from mamba3_tracker.eval.tapvid3d_eval import aggregate, compute_clip_metrics
from mamba3_tracker.model.depth_refined_tracker import Mamba3DepthRefiner
from searaft_flow import FlowModel, track_clip


def _load_depth(da3_depth_root: Path, subset: str, clip_id: str, F_: int) -> np.ndarray:
    depth_path = Path(da3_depth_root).expanduser() / subset / (clip_id + ".npz")
    with np.load(depth_path) as dd:
        if "depth_q" in dd:
            q = np.asarray(dd["depth_q"][:F_]).astype(np.float32)
            d_min, d_max = float(dd["d_min"]), float(dd["d_max"])
            return d_min + q * (max(d_max - d_min, 1e-6) / 65535.0)
        return np.asarray(dd["depth"][:F_], dtype=np.float32)


def _ray_from_uv(uv: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    fx, fy = K[:, 0, 0].view(-1, 1, 1), K[:, 1, 1].view(-1, 1, 1)
    cx, cy = K[:, 0, 2].view(-1, 1, 1), K[:, 1, 2].view(-1, 1, 1)
    return torch.stack([(uv[..., 0] - cx) / fx, (uv[..., 1] - cy) / fy], dim=-1)


@torch.no_grad()
def _infer_clip(model, flow_model, clip, image_size, fb_alpha, fb_beta, da3_depth_root, max_frames):
    F_ = int(clip.images.shape[0]) if not max_frames else min(int(clip.images.shape[0]), max_frames)
    images = clip.images[:F_].clone()
    H_orig, W_orig = images.shape[-2], images.shape[-1]
    if (H_orig, W_orig) != (image_size, image_size):
        images = F.interpolate(images, size=(image_size, image_size), mode="bilinear", align_corners=False)
    images_255 = images * 255.0

    sx, sy = image_size / float(W_orig), image_size / float(H_orig)
    q = clip.queries_xyt.clone()
    queries_xy = torch.stack([q[:, 0] * sx, q[:, 1] * sy], dim=-1)
    anchor_t = q[:, 2].long().clamp(0, F_ - 1)

    device = flow_model.device
    uv, vis = track_clip(flow_model, images_255.to(device), queries_xy, anchor_t, image_size, fb_alpha, fb_beta)

    K = clip.K.clone()
    K[0] *= sx
    K[1] *= sy
    K_t = K.unsqueeze(0).to(device)
    uv_d = uv.unsqueeze(0).to(device)
    vis_d = vis.unsqueeze(0).to(device)
    ray = _ray_from_uv(uv_d, K_t)

    depth_full = _load_depth(da3_depth_root, clip.subset, clip.clip_id, F_)
    depth_t = torch.from_numpy(depth_full).unsqueeze(0).to(device)   # (1,F,Hd,Wd)
    Hd, Wd = depth_t.shape[-2:]
    grid = (2.0 * uv_d / image_size - 1.0).view(F_, 1, -1, 2)
    z_raw = F.grid_sample(depth_t.squeeze(0).unsqueeze(1), grid,
                          mode="bilinear", padding_mode="border", align_corners=False).view(1, F_, -1)

    pred = model(ray, z_raw, vis_d)
    xyz = pred.xyz[0].cpu().numpy()        # (F,N,3)
    pred_vis = vis.numpy()                 # frozen SEA-RAFT FB-consistency (F,N)
    return xyz.transpose(1, 0, 2), pred_vis.T   # (N,F,3), (N,F)


def _motion_ratio(clip, pred_tracks, F_):
    K = clip.K.numpy()
    gt_xyz = clip.tracks_XYZ[:F_].numpy()
    gt_vis = clip.visibility[:F_].float().numpy()
    N = clip.queries_xyt.shape[0]
    a_idx = clip.queries_xyt[:, 2].long().clamp(0, F_ - 1).numpy()
    track_idx = np.arange(N)

    def proj(xyz):
        Z = np.clip(xyz[..., 2:3], 1e-6, None)
        return (xyz[..., :2] / Z) * np.array([K[0, 0], K[1, 1]]) + np.array([K[0, 2], K[1, 2]])

    uv_pred = proj(pred_tracks)
    uv_gt = proj(np.transpose(gt_xyz, (1, 0, 2)))
    travel_pred = np.linalg.norm(uv_pred - uv_pred[track_idx, a_idx][:, None], axis=-1)
    travel_gt = np.linalg.norm(uv_gt - uv_gt[track_idx, a_idx][:, None], axis=-1)
    vis_NT = np.transpose(gt_vis, (1, 0))
    vis_anchor = vis_NT[track_idx, a_idx]
    z_ok = (pred_tracks[..., 2] > 0) & (np.transpose(gt_xyz, (1, 0, 2))[..., 2] > 0)
    finite = np.isfinite(travel_pred) & np.isfinite(travel_gt)
    mask = (vis_NT > 0.5) & (vis_anchor[:, None] > 0.5) & finite & z_ok
    return float((travel_pred * mask).sum()), float((travel_gt * mask).sum())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, default=Path("~/data"))
    ap.add_argument("--da3-depth-root", type=Path, default=Path("~/data/tapvid3d_da3"))
    ap.add_argument("--subsets", nargs="+", default=list(SUBSETS))
    ap.add_argument("--split", choices=["all", "minival", "full_eval"], default="minival")
    ap.add_argument("--max-clips-per-subset", type=int, default=0)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--image-size", type=int, default=896)
    ap.add_argument("--url", type=str, default="MemorySlices/Tartan-C-T-TSKH-spring540x960-M")
    ap.add_argument("--iters", type=int, default=None)
    ap.add_argument("--scale", type=int, default=None)
    ap.add_argument("--fb-alpha", type=float, default=0.05)
    ap.add_argument("--fb-beta", type=float, default=1.0)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.data_root = args.data_root.expanduser()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    mc = state.get("cfg", {}).get("model", {})
    model = Mamba3DepthRefiner(
        dim=int(mc.get("dim", 128)), state_dim=int(mc.get("state_dim", 64)),
        num_heads=int(mc.get("num_heads", 4)), num_layers=int(mc.get("num_layers", 2)),
        max_log_correction=float(mc.get("max_log_correction", 2.0)),
    ).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    print(f"[eval] loaded {args.ckpt} (step={state.get('step', '?')})")

    flow_model = FlowModel(device, url=args.url, iters=args.iters, scale=args.scale)
    print(f"[eval] FlowModel ({args.url}) iters={flow_model.args.iters} scale={flow_model.args.scale}")

    if args.split == "minival":
        from mamba3_tracker.data.tapvid3d_splits import MINIVAL_FILES as ALLOW
    elif args.split == "full_eval":
        from mamba3_tracker.data.tapvid3d_splits import FULL_EVAL_FILES as ALLOW
    else:
        ALLOW = None
    allow = {s: (set(ALLOW.get(s, [])) if ALLOW is not None else None) for s in args.subsets}

    metrics_root = args.out_dir / "metric_results"
    metrics_root.mkdir(parents=True, exist_ok=True)
    summary: dict[str, dict] = {}
    motion_pred: dict[str, float] = defaultdict(float)
    motion_gt: dict[str, float] = defaultdict(float)
    t_start = time.time()
    n_frames_total = 0
    n_fail = 0

    for sub in args.subsets:
        clips = list_clips(args.data_root, [sub])
        if allow[sub] is not None:
            clips = [p for p in clips if p.name in allow[sub]]
        if args.max_clips_per_subset:
            clips = clips[: args.max_clips_per_subset]
        print(f"[eval] {sub}: {len(clips)} clips", flush=True)

        per_clip: list[dict] = []
        for path in clips:
            try:
                clip = load_clip(path)
                pred_tracks, pred_vis = _infer_clip(model, flow_model, clip, args.image_size,
                                                    args.fb_alpha, args.fb_beta,
                                                    args.da3_depth_root, args.max_frames)
                F_ = pred_tracks.shape[1]
                n_frames_total += F_
                gt_xyz = clip.tracks_XYZ[:F_].numpy()
                gt_vis = clip.visibility[:F_].float().numpy()
                K = clip.K.numpy()
                intrin = np.array([K[0, 0], K[1, 1], K[0, 2], K[1, 2]])
                m = compute_clip_metrics(gt_xyz, gt_vis, pred_tracks, pred_vis, intrin)
                m["clip_id"] = clip.clip_id
                p_sum, g_sum = _motion_ratio(clip, pred_tracks, F_)
                motion_pred[sub] += p_sum
                motion_gt[sub] += g_sum
                m["motion_ratio_clip"] = (p_sum / g_sum) if g_sum > 0 else float("nan")
                per_clip.append(m)
                print(f"[eval] {sub}/{clip.clip_id}: AJ={m['average_jaccard']:.4f}  "
                      f"APD3D={m['average_pts_within_thresh']:.4f}  OA={m['occlusion_accuracy']:.4f}  "
                      f"motion={m['motion_ratio_clip']:.2%}", flush=True)
            except Exception as e:
                n_fail += 1
                print(f"[eval] {sub}/{path.stem}: FAIL {type(e).__name__}: {e}", flush=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        (metrics_root / f"{sub}.json").write_text(json.dumps(per_clip, indent=2))
        agg = aggregate(per_clip)
        agg["motion_ratio"] = (motion_pred[sub] / motion_gt[sub]) if motion_gt[sub] > 0 else float("nan")
        summary[sub] = agg
        print(f"[eval] {sub} mean: {agg}", flush=True)

    overall = aggregate(summary.values())
    elapsed = time.time() - t_start
    fps = n_frames_total / elapsed if elapsed > 0 else float("nan")
    rows = [
        "# Mamba3DepthRefiner (v33) — TAPVid-3D evaluation",
        f"\nCheckpoint: `{args.ckpt}` (step={state.get('step', '?')})",
        f"Flow model: `{args.url}` (iters={flow_model.args.iters}, scale={flow_model.args.scale})",
        f"image_size={args.image_size}, fb_alpha={args.fb_alpha}, fb_beta={args.fb_beta}",
        f"\nThroughput: {n_frames_total} frames in {elapsed:.0f}s ({fps:.1f} frames/s on {device})",
        f"Failures: {n_fail}",
        "\n## Per-subset means\n",
        "| subset | 3D-AJ | APD3D | OA | motion ratio |",
        "|---|---|---|---|---|",
    ]
    for sub, mm in summary.items():
        rows.append(f"| {sub} | {mm.get('average_jaccard', float('nan')):.4f} | "
                    f"{mm.get('average_pts_within_thresh', float('nan')):.4f} | "
                    f"{mm.get('occlusion_accuracy', float('nan')):.4f} | "
                    f"{mm.get('motion_ratio', float('nan')):.2%} |")
    rows.append(f"| **mean** | **{overall.get('average_jaccard', float('nan')):.4f}** | "
                f"**{overall.get('average_pts_within_thresh', float('nan')):.4f}** | "
                f"**{overall.get('occlusion_accuracy', float('nan')):.4f}** | |")
    (args.out_dir / "summary.md").write_text("\n".join(rows) + "\n")
    print(f"\n[eval] DONE. Summary at {args.out_dir / 'summary.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
