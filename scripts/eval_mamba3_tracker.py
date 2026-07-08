"""Evaluate a trained Mamba-3 tracker on TAPVid-3D.

For each held-out clip:
  1. Load ground-truth tracks + visibility.
  2. Run inference: feed the full video as one window (or stream in chunks).
  3. Use the model's Hungarian-matching anchor logic against the GT queries
     to assign predicted slots to GT track IDs, so per-track metrics line up.
  4. Compute 3D-AJ / APD3D / OA via `mamba3_tracker.eval.tapvid3d_eval`.

Output: per-clip JSONs + a per-subset roll-up `summary.md`.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from mamba3_tracker.data.tapvid3d import SUBSETS, list_clips, load_clip
from mamba3_tracker.eval.tapvid3d_eval import aggregate, compute_clip_metrics
from mamba3_tracker.model.tracker import Mamba3Tracker


def _build_model(state: dict, device: torch.device) -> Mamba3Tracker:
    cfg = state["cfg"]
    # v8+ uses nested `cfg["model"]`; v6/v7 used flat keys.
    m = cfg.get("model", cfg)
    model = Mamba3Tracker(
        dim=int(m["dim"]), num_heads=int(m["num_heads"]),
        state_dim=int(m["state_dim"]),
        level_sizes=tuple(m.get("level_sizes", [32, 64])),
        num_iters=int(m.get("num_iters", 1)),
        use_correlation=bool(m.get("use_correlation", False)),
        encoder_kind=str(m.get("encoder_kind", "pyramid")),
        dinov2_model=str(m.get("dinov2_model", "facebook/dinov2-small")),
        dinov2_image_size=int(m.get("dinov2_image_size", 448)),
        dinov2_fuse_layers=m.get("dinov2_fuse_layers"),
        predict_scale=bool(m.get("predict_scale", False)),
        head_mode=str(m.get("head_mode", "xyz")),
    ).to(device)
    model.load_state_dict(state["model"], strict=False)   # frozen DINO weights load via from_pretrained
    model.eval()
    return model


@torch.no_grad()
def _infer_clip(
    model, clip, device, amp_dtype, max_frames: int = 0,
    da3_depth_root: Path | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (pred_tracks_NT3, pred_visibility_NT) for one clip.

    v11-v30: the model emits per-frame motion `Δp̂(t, n) = pred.xyz[t, n]`,
    reconstructed by anchor-aligned cumulative summation.
    v31 (head_mode='uv'): the model emits `pred.uv` per (t, n) in pixel
    coords; absolute (X, Y, Z) comes from DA3-cached depth at `pred.uv`
    via pinhole unprojection.
    """
    F = int(clip.images.shape[0]) if max_frames in (0, None) else min(int(clip.images.shape[0]), max_frames)
    images = clip.images[:F].clone()
    H_orig, W_orig = images.shape[-2], images.shape[-1]
    img_size = model.image_size
    if H_orig != img_size or W_orig != img_size:
        images = torch.nn.functional.interpolate(
            images, size=(img_size, img_size),
            mode="bilinear", align_corners=False,
        )
    images = images.unsqueeze(0).to(device)                  # (1, F, 3, S, S)

    # Scale query (x, y) to resized pixel space; clamp anchor frame to [0, F-1].
    sx = img_size / float(W_orig)
    sy = img_size / float(H_orig)
    N_q = clip.queries_xyt.shape[0]
    queries = clip.queries_xyt.clone()
    queries[:, 0] *= sx
    queries[:, 1] *= sy
    queries[:, 2] = queries[:, 2].clamp(max=F - 1)
    queries = queries.unsqueeze(0).to(device)
    qmask = torch.ones(1, N_q, dtype=torch.bool, device=device)

    with torch.autocast(device_type=device.type, dtype=amp_dtype):
        pred = model(images, queries, qmask)

    pred_vis = torch.sigmoid(pred.vis_logits[0].float()).cpu()  # (F, N_q)

    if getattr(model, "head_mode", "xyz") == "uv":
        if da3_depth_root is None:
            raise RuntimeError(
                "v31 eval requires --da3-depth-root with cached depth .npz files"
            )
        uv = pred.uv[0].float().cpu()                            # (F, N_q, 2)
        depth_path = Path(da3_depth_root).expanduser() / clip.subset / (clip.clip_id + ".npz")
        with np.load(depth_path) as dd:
            if "depth_q" in dd:
                q = np.asarray(dd["depth_q"][:F])
                d_min = float(dd["d_min"])
                d_max = float(dd["d_max"])
                scale = max(d_max - d_min, 1e-6)
                depth_full = d_min + q.astype(np.float32) * (scale / 65535.0)
            else:
                depth_full = np.asarray(dd["depth"][:F], dtype=np.float32)
        depth_t = torch.from_numpy(depth_full).unsqueeze(0)              # (1, F, Hd, Wd)
        # Build K scaled to the resized pixel coords (uv lives there).
        K = clip.K.clone()
        K[0, 0] *= sx
        K[1, 1] *= sy
        K[0, 2] *= sx
        K[1, 2] *= sy
        from mamba3_tracker.train.loss import _unproject_with_depth
        xyz = _unproject_with_depth(
            uv.unsqueeze(0), depth_t, K.unsqueeze(0), float(img_size),
        )[0]                                                              # (F, N_q, 3)
        pred_tracks_NT3 = xyz.transpose(0, 1).numpy()
        pred_vis_NT = pred_vis.transpose(0, 1).numpy()
        return pred_tracks_NT3, pred_vis_NT

    delta = pred.xyz[0].float().cpu()                        # (F, N_q, 3)
    if pred.scale is not None:                                # v18: pre-multiply by learned scalar
        delta = float(pred.scale[0].float().cpu().item()) * delta

    # v11–v17 per-anchor bidirectional cumsum with GT init at the anchor;
    # v18 applies the per-clip scale above, then the same cumsum:
    #   p̂(a_n, n) = clip.tracks_XYZ[a_n, n]
    #   p̂(t, n)   = p̂(a_n, n) + (cumsum_{τ=0..t} Δp̂ − cumsum_{τ=0..a_n} Δp̂)
    a_n = clip.queries_xyt[:, 2].long().clamp(min=0, max=F - 1)  # (N_q,)
    init = clip.tracks_XYZ[a_n, torch.arange(N_q)]                # (N_q, 3)
    if F == 1:
        pred_abs = init.unsqueeze(0)
    else:
        delta_zero_init = delta.clone()
        delta_zero_init[0] = 0.0
        cs = delta_zero_init.cumsum(dim=0)                        # (F, N_q, 3)
        cs_at_anchor = cs[a_n, torch.arange(N_q)]                 # (N_q, 3)
        pred_abs = init.unsqueeze(0) + cs - cs_at_anchor.unsqueeze(0)  # (F, N_q, 3)

    pred_tracks_NT3 = pred_abs.transpose(0, 1).numpy()
    pred_vis_NT = pred_vis.transpose(0, 1).numpy()
    return pred_tracks_NT3, pred_vis_NT


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="output dir; defaults to <ckpt_parent>/eval_ckpt<step>/",
    )
    ap.add_argument("--data-root", type=Path, default=Path("~/data"))
    ap.add_argument("--subsets", nargs="+", default=list(SUBSETS))
    ap.add_argument("--split", choices=["all", "minival", "full_eval"], default="all",
                    help="all = every .npz under <data_root>/tapvid3d/<subset>/ (default); "
                         "minival = restrict to the 50-per-subset MINIVAL_FILES held-out set "
                         "(use this when comparing to paper Table 3 minival baselines); "
                         "full_eval = restrict to FULL_EVAL_FILES.")
    ap.add_argument("--max-clips-per-subset", type=int, default=0,
                    help="0 = all clips.")
    ap.add_argument("--max-frames", type=int, default=0,
                    help="0 = full clip. Use a small value for quick smoke runs.")
    ap.add_argument("--amp", choices=["bf16", "fp16", "fp32"], default="bf16")
    ap.add_argument("--da3-depth-root", type=Path, default=None,
                    help="v31 only: root of pre-computed DA3 depth cache "
                         "(<root>/<subset>/<clip>.npz). Required when the "
                         "checkpoint was trained with head_mode='uv'.")
    args = ap.parse_args()

    if args.out_dir is None:
        step = args.ckpt.stem.split("_")[-1] if "_" in args.ckpt.stem else args.ckpt.stem
        args.out_dir = args.ckpt.parent / f"eval_ckpt{step}"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.data_root = args.data_root.expanduser()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.amp]

    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = _build_model(state, device)
    print(f"[eval] loaded {args.ckpt} (step={state.get('step', '?')})")

    # Resolve the named split into a per-subset allow-list of filenames.
    allow: dict[str, set[str] | None] = {}
    if args.split == "minival":
        from mamba3_tracker.data.tapvid3d_splits import MINIVAL_FILES
        allow = {s: set(MINIVAL_FILES.get(s, [])) for s in args.subsets}
    elif args.split == "full_eval":
        from mamba3_tracker.data.tapvid3d_splits import FULL_EVAL_FILES
        allow = {s: set(FULL_EVAL_FILES.get(s, [])) for s in args.subsets}
    else:
        allow = {s: None for s in args.subsets}

    per_subset_clips: dict[str, list] = defaultdict(list)
    for sub in args.subsets:
        clips = list_clips(args.data_root, [sub])
        if allow[sub] is not None:
            clips = [p for p in clips if p.name in allow[sub]]
        if args.max_clips_per_subset:
            clips = clips[: args.max_clips_per_subset]
        per_subset_clips[sub] = clips
        print(f"[eval] {sub}: {len(clips)} clips")

    metrics_root = args.out_dir / "metric_results"
    metrics_root.mkdir(parents=True, exist_ok=True)
    summary: dict[str, dict[str, float]] = {}
    # Per-subset motion-ratio accumulators (Σ predicted 2D pixel travel from
    # each track's anchor / Σ GT 2D pixel travel, over visible (t, n) pairs).
    # Mirrors the in-training _motion_check in scripts/train_mamba3_tracker.py.
    motion_pred_sums: dict[str, float] = defaultdict(float)
    motion_gt_sums:   dict[str, float] = defaultdict(float)

    for sub, clips in per_subset_clips.items():
        per_clip: list[dict[str, float]] = []
        for i, path in enumerate(clips):
            try:
                clip = load_clip(path)
                pred_tracks, pred_vis = _infer_clip(
                    model, clip, device, amp_dtype, args.max_frames,
                    da3_depth_root=args.da3_depth_root,
                )
                F = pred_tracks.shape[1]
                gt_xyz = clip.tracks_XYZ[:F].numpy()
                gt_vis = clip.visibility[:F].float().numpy()
                K = clip.K.numpy()
                intrin = np.array([K[0, 0], K[1, 1], K[0, 2], K[1, 2]])
                m = compute_clip_metrics(gt_xyz, gt_vis, pred_tracks, pred_vis, intrin)
                m["clip_id"] = clip.clip_id
                # Motion ratio (pixel travel from anchor, visible-at-both pairs).
                N_q = clip.queries_xyt.shape[0]
                a_idx = clip.queries_xyt[:, 2].long().clamp(min=0, max=F - 1).numpy()
                track_idx = np.arange(N_q)
                def _proj_uv(xyz_NT3: np.ndarray) -> np.ndarray:
                    Z = np.clip(xyz_NT3[..., 2:3], 1e-6, None)
                    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
                    return (xyz_NT3[..., :2] / Z) * np.array([fx, fy]) + np.array([cx, cy])
                uv_pred = _proj_uv(pred_tracks)                              # (N, F, 2)
                uv_gt   = _proj_uv(np.transpose(gt_xyz, (1, 0, 2)))           # (N, F, 2)
                ref_pred = uv_pred[track_idx, a_idx]                         # (N, 2)
                ref_gt   = uv_gt[track_idx, a_idx]
                travel_pred = np.linalg.norm(uv_pred - ref_pred[:, None, :], axis=-1)   # (N, F)
                travel_gt   = np.linalg.norm(uv_gt   - ref_gt[:,   None, :], axis=-1)
                vis_NT = np.transpose(gt_vis, (1, 0))                        # (N, F)
                vis_anchor = vis_NT[track_idx, a_idx]                        # (N,)
                finite = np.isfinite(travel_pred) & np.isfinite(travel_gt)
                # Exclude predicted-Z<=0 frames: those points are behind the
                # camera and cannot appear on the image, so projecting them
                # is meaningless — they otherwise drag the motion ratio to
                # billions on ADT clips with near-anchor depths.
                pred_z = pred_tracks[..., 2]                                  # (N, F)
                gt_z   = np.transpose(gt_xyz, (1, 0, 2))[..., 2]              # (N, F)
                z_ok   = (pred_z > 0) & (gt_z > 0)
                mask = (vis_NT > 0.5) & (vis_anchor[:, None] > 0.5) & finite & z_ok
                clip_pred_sum = float((travel_pred * mask).sum())
                clip_gt_sum   = float((travel_gt   * mask).sum())
                motion_pred_sums[sub] += clip_pred_sum
                motion_gt_sums[sub]   += clip_gt_sum
                m["motion_ratio_clip"] = (clip_pred_sum / clip_gt_sum) if clip_gt_sum > 0 else float("nan")
                per_clip.append(m)
                print(f"[eval] {sub}/{clip.clip_id}: AJ={m['average_jaccard']:.4f} "
                      f"APD3D={m['average_pts_within_thresh']:.4f} OA={m['occlusion_accuracy']:.4f} "
                      f"motion={m['motion_ratio_clip']:.2%}",
                      flush=True)
            except Exception as e:
                print(f"[eval] {sub}/{path.stem}: FAIL {type(e).__name__}: {e}", flush=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        (metrics_root / f"{sub}.json").write_text(json.dumps(per_clip, indent=2))
        agg = aggregate(per_clip)
        # Subset-level motion ratio = Σ pred / Σ gt (over all clips & pairs).
        agg["motion_ratio"] = (motion_pred_sums[sub] / motion_gt_sums[sub]
                                if motion_gt_sums[sub] > 0 else float("nan"))
        summary[sub] = agg
        print(f"[eval] {sub} mean: {agg}", flush=True)

    # Roll-up
    overall = aggregate(summary.values())
    rows = [
        "# Mamba-3 Tracker — TAPVid-3D evaluation",
        f"\nCheckpoint: `{args.ckpt}` (step {state.get('step', '?')})",
        "\n## Per-subset means\n",
        "| subset | 3D-AJ | APD3D | OA |",
        "|---|---|---|---|",
    ]
    for sub, m in summary.items():
        rows.append(
            f"| {sub} | {m.get('average_jaccard', float('nan')):.4f} | "
            f"{m.get('average_pts_within_thresh', float('nan')):.4f} | "
            f"{m.get('occlusion_accuracy', float('nan')):.4f} |"
        )
    rows.append(
        f"| **mean** | **{overall.get('average_jaccard', float('nan')):.4f}** | "
        f"**{overall.get('average_pts_within_thresh', float('nan')):.4f}** | "
        f"**{overall.get('occlusion_accuracy', float('nan')):.4f}** |"
    )
    (args.out_dir / "summary.md").write_text("\n".join(rows) + "\n")
    print(f"\n[eval] DONE. Summary at {args.out_dir / 'summary.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
