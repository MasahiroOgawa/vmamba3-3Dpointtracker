"""Evaluate FlowConditionedTracker (v32) on TAPVid-3D minival.

For each clip:
  1. Chain SEA-RAFT flow (frozen) → uv_fwd, vis_fwd, flow_at_uv.
  2. Sample DA3 metric depth at uv_fwd.
  3. FlowConditionedTracker refines uv_fwd → uv_pred.
  4. Unproject uv_pred through DA3 depth → xyz_pred.
  5. Score with compute_clip_metrics (3D-AJ, APD3D, OA) + motion ratio.

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
from mamba3_tracker.model.flow_conditioned_tracker import FlowConditionedTracker
from mamba3_tracker.train.loss import _unproject_with_depth
from searaft_flow import FlowModel, track_clip_with_flow


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
def _infer_clip(
    model: FlowConditionedTracker,
    flow_model: FlowModel,
    clip,
    image_size: int,
    fb_alpha: float,
    fb_beta: float,
    da3_depth_root: Path,
    max_frames: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (pred_tracks (N,F,3), pred_vis (N,F))."""
    F_ = int(clip.images.shape[0]) if not max_frames else min(int(clip.images.shape[0]), max_frames)
    images = clip.images[:F_].clone()
    H_orig, W_orig = images.shape[-2], images.shape[-1]
    if (H_orig, W_orig) != (image_size, image_size):
        images = F.interpolate(images, size=(image_size, image_size),
                               mode="bilinear", align_corners=False)
    images_255 = images * 255.0

    sx, sy = image_size / float(W_orig), image_size / float(H_orig)
    q = clip.queries_xyt.clone()
    queries_xy = torch.stack([q[:, 0] * sx, q[:, 1] * sy], dim=-1)
    anchor_t = q[:, 2].long().clamp(0, F_ - 1)

    device = flow_model.device
    uv, vis, flow_at = track_clip_with_flow(
        flow_model, images_255.to(device), queries_xy, anchor_t, image_size, fb_alpha, fb_beta,
    )

    depth_t = _load_depth(da3_depth_root, clip.subset, clip.clip_id, F_).to(device)
    K = clip.K.clone()
    K[0] *= sx
    K[1] *= sy

    uv_dev  = uv.unsqueeze(0).to(device)       # (1,F,N,2)
    flow_dev = flow_at.unsqueeze(0).to(device) # (1,F,N,2)
    vis_dev  = vis.unsqueeze(0).to(device)     # (1,F,N)
    # Sample depth at tracked positions for model input.
    grid = (2.0 * uv_dev / image_size - 1.0).view(F_, 1, -1, 2)
    depth_at = F.grid_sample(
        depth_t.squeeze(0).unsqueeze(1),       # (F,1,Hd,Wd)
        grid, mode="bilinear", padding_mode="border", align_corners=False,
    ).view(1, F_, -1)                          # (1,F,N)

    pred = model(uv_dev, flow_dev, depth_at, vis_dev, float(image_size))
    xyz = _unproject_with_depth(pred.uv, depth_t, K.unsqueeze(0).to(device), float(image_size))
    # pred.uv: (1,F,N,2); vis_logits: (1,F,N)
    pred_vis = (torch.sigmoid(pred.vis_logits[0]) > 0.5).float().cpu().numpy()   # (F,N)
    return xyz[0].cpu().numpy().transpose(1, 0, 2), pred_vis.T    # (N,F,3), (N,F)


def _motion_ratio(clip, pred_tracks: np.ndarray, F_: int) -> tuple[float, float]:
    K = clip.K.numpy()
    gt_xyz = clip.tracks_XYZ[:F_].numpy()
    gt_vis = clip.visibility[:F_].float().numpy()
    N = clip.queries_xyt.shape[0]
    a_idx = clip.queries_xyt[:, 2].long().clamp(0, F_ - 1).numpy()
    track_idx = np.arange(N)

    def proj(xyz_NT3: np.ndarray) -> np.ndarray:
        Z = np.clip(xyz_NT3[..., 2:3], 1e-6, None)
        return (xyz_NT3[..., :2] / Z) * np.array([K[0, 0], K[1, 1]]) + np.array([K[0, 2], K[1, 2]])

    uv_pred = proj(pred_tracks)
    uv_gt   = proj(np.transpose(gt_xyz, (1, 0, 2)))
    travel_pred = np.linalg.norm(uv_pred - uv_pred[track_idx, a_idx][:, None], axis=-1)
    travel_gt   = np.linalg.norm(uv_gt   - uv_gt[track_idx, a_idx][:, None],   axis=-1)
    vis_NT = np.transpose(gt_vis, (1, 0))
    vis_anchor = vis_NT[track_idx, a_idx]
    z_ok   = (pred_tracks[..., 2] > 0) & (np.transpose(gt_xyz, (1, 0, 2))[..., 2] > 0)
    finite = np.isfinite(travel_pred) & np.isfinite(travel_gt)
    mask   = (vis_NT > 0.5) & (vis_anchor[:, None] > 0.5) & finite & z_ok
    return float((travel_pred * mask).sum()), float((travel_gt * mask).sum())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt",           type=Path, required=True, help="Path to ckpt_<step>.pt")
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="output dir; defaults to <ckpt_parent>/eval_ckpt<step>/",
    )
    ap.add_argument("--data-root",      type=Path, default=Path("~/data"))
    ap.add_argument("--da3-depth-root", type=Path, default=Path("~/data/tapvid3d_da3"))
    ap.add_argument("--subsets",        nargs="+", default=list(SUBSETS))
    ap.add_argument("--split",          choices=["all", "minival", "full_eval"], default="minival")
    ap.add_argument("--max-clips-per-subset", type=int, default=0)
    ap.add_argument("--max-frames",     type=int, default=0)
    ap.add_argument("--image-size",     type=int, default=896)
    ap.add_argument("--url",            type=str, default="MemorySlices/Tartan-C-T-TSKH-spring540x960-M")
    ap.add_argument("--iters",          type=int, default=None)
    ap.add_argument("--scale",          type=int, default=None)
    ap.add_argument("--fb-alpha",       type=float, default=0.05)
    ap.add_argument("--fb-beta",        type=float, default=1.0)
    args = ap.parse_args()

    if args.out_dir is None:
        step = args.ckpt.stem.split("_")[-1] if "_" in args.ckpt.stem else args.ckpt.stem
        args.out_dir = args.ckpt.parent / f"eval_ckpt{step}"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.data_root = args.data_root.expanduser()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    saved_cfg = state.get("cfg", {})
    model_cfg = saved_cfg.get("model", {})
    model = FlowConditionedTracker(
        dim=int(model_cfg.get("dim", 128)),
        state_dim=int(model_cfg.get("state_dim", 64)),
        num_heads=int(model_cfg.get("num_heads", 4)),
        num_layers=int(model_cfg.get("num_layers", 2)),
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
    motion_gt:   dict[str, float] = defaultdict(float)
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
                pred_tracks, pred_vis = _infer_clip(
                    model, flow_model, clip, args.image_size,
                    args.fb_alpha, args.fb_beta, args.da3_depth_root, args.max_frames,
                )
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
                motion_gt[sub]   += g_sum
                m["motion_ratio_clip"] = (p_sum / g_sum) if g_sum > 0 else float("nan")
                per_clip.append(m)
                print(f"[eval] {sub}/{clip.clip_id}: "
                      f"AJ={m['average_jaccard']:.4f}  "
                      f"APD3D={m['average_pts_within_thresh']:.4f}  "
                      f"OA={m['occlusion_accuracy']:.4f}  "
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
        "# FlowConditionedTracker (v32) — TAPVid-3D evaluation",
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
