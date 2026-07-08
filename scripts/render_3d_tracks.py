"""Render the spatial 3D trajectory (X, Y, Z all in world metres) —
both static PNG and interactive HTML, one of each per clip.

This is the "xyz" plot. For the three additional space-time projections
(xy-t / yz-t / zx-t) where Z = time, see render_space_time_tracks.py.
Together the two scripts produce 4 plots per clip: xyz + xyt + yzt + zxt.

Plot layout:
  * X axis = world X (metres)
  * Y axis = world Y (metres)
  * Z axis = world Z / depth (metres)

Tracks drawn as 3D polylines:
  * predicted track       : solid line, per-track colour
  * ground-truth track    : dashed line, same track colour, lower alpha
  * anchor frame position : marker dot

Same model + inference path as scripts/render_tracker_video.py.

Usage:
    uv run python scripts/render_3d_tracks.py \\
        --ckpt result/<dt>_track_v18/ckpt_30000.pt \\
        --out-dir result/<dt>_track_v18/viz_step30000 \\
        --subsets pstudio drivetrack \\
        --clips-per-subset 2 \\
        --max-frames 32 \\
        --max-tracks 32
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import torch

from mamba3_tracker.data.dataset import filter_to_split
from mamba3_tracker.data.tapvid3d import SUBSETS, has_images, list_clips, load_clip
from mamba3_tracker.model.tracker import Mamba3Tracker


def _build_model(state: dict, device: torch.device) -> Mamba3Tracker:
    cfg = state["cfg"]
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
    ).to(device)
    model.load_state_dict(state["model"], strict=False)
    model.eval()
    return model


@torch.no_grad()
def _predict_tracks(model, clip, device, amp_dtype, max_frames: int) -> np.ndarray:
    """Returns predicted absolute 3D positions in camera coords, shape (N_q, F, 3)."""
    F = int(clip.images.shape[0]) if not max_frames else min(int(clip.images.shape[0]), max_frames)
    images = clip.images[:F].clone()
    H_orig, W_orig = images.shape[-2], images.shape[-1]
    img_size = model.image_size
    if H_orig != img_size or W_orig != img_size:
        images = torch.nn.functional.interpolate(
            images, size=(img_size, img_size), mode="bilinear", align_corners=False,
        )
    images = images.unsqueeze(0).to(device)
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
    delta = pred.xyz[0].float().cpu()
    if pred.scale is not None:
        delta = float(pred.scale[0].float().cpu().item()) * delta
    a_n = clip.queries_xyt[:, 2].long().clamp(min=0, max=F - 1)
    init = clip.tracks_XYZ[a_n, torch.arange(N_q)]
    if F == 1:
        abs_pred = init.unsqueeze(0)
    else:
        delta_zero = delta.clone()
        delta_zero[0] = 0.0
        cs = delta_zero.cumsum(dim=0)
        cs_at_anchor = cs[a_n, torch.arange(N_q)]
        abs_pred = init.unsqueeze(0) + cs - cs_at_anchor.unsqueeze(0)
    return abs_pred.transpose(0, 1).numpy()                # (N_q, F, 3)


def _pick_tracks(vis_NF: np.ndarray, max_tracks: int, seed: int = 0) -> list[int]:
    """Pick up to max_tracks tracks, preferring longest-visible ones."""
    rng = np.random.default_rng(seed)
    vis_count = vis_NF.sum(axis=1)
    candidate = np.argsort(-vis_count)[: min(max_tracks * 2, vis_NF.shape[0])]
    if len(candidate) > max_tracks:
        candidate = rng.choice(candidate, size=max_tracks, replace=False)
    return sorted(candidate.tolist())


def _plot_clip_3d_png(
    pred_NF3: np.ndarray,         # (N_q, F, 3) predicted absolute 3D
    gt_NF3:   np.ndarray,         # (N_q, F, 3) GT absolute 3D
    vis_NF:   np.ndarray,         # (N_q, F)   bool: GT visibility per (n, t)
    anchor_n: np.ndarray,         # (N_q,)     anchor frame per track
    out_path: Path,
    title: str,
    max_tracks: int = 32,
) -> None:
    N_q, F, _ = pred_NF3.shape
    sel = _pick_tracks(vis_NF, max_tracks)

    fig = plt.figure(figsize=(11, 6))
    ax = fig.add_subplot(111, projection="3d")
    cmap = plt.get_cmap("tab20")
    for i, n in enumerate(sel):
        color = cmap(i % 20)
        m = vis_NF[n].astype(bool)
        if m.sum() < 2:
            continue
        ax.plot(gt_NF3[n, m, 0], gt_NF3[n, m, 1], gt_NF3[n, m, 2],
                linestyle="--", linewidth=1.0, color=color, alpha=0.55)
        ax.plot(pred_NF3[n, m, 0], pred_NF3[n, m, 1], pred_NF3[n, m, 2],
                linestyle="-",  linewidth=1.4, color=color, alpha=0.95)
        a = int(anchor_n[n])
        if 0 <= a < F:
            ax.scatter([gt_NF3[n, a, 0]], [gt_NF3[n, a, 1]], [gt_NF3[n, a, 2]],
                       s=18, color=color, edgecolors="black", linewidths=0.4)

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(f"{title}\nsolid = predicted, dashed = GT, dot = anchor frame", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def _plot_clip_3d_html(
    pred_NF3: np.ndarray,         # (N_q, F, 3)
    gt_NF3:   np.ndarray,         # (N_q, F, 3)
    vis_NF:   np.ndarray,         # (N_q, F)   bool
    anchor_n: np.ndarray,         # (N_q,)
    out_path: Path,
    title: str,
    max_tracks: int = 32,
) -> None:
    """Interactive plotly version. Open the resulting .html in a browser
    to rotate / zoom / pan / hover individual tracks / toggle them via
    the legend. Self-contained — no internet or plotly-server required.
    """
    N_q, F, _ = pred_NF3.shape
    sel = _pick_tracks(vis_NF, max_tracks)
    palette = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
        "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
        "#c49c94", "#f7b6d2", "#c7c7c7", "#dbdb8d", "#9edae5",
    ]
    traces: list[go.Scatter3d] = []
    for i, n in enumerate(sel):
        color = palette[i % len(palette)]
        m = vis_NF[n].astype(bool)
        if m.sum() < 2:
            continue
        legend_group = f"track {n}"
        traces.append(go.Scatter3d(
            x=gt_NF3[n, m, 0], y=gt_NF3[n, m, 1], z=gt_NF3[n, m, 2],
            mode="lines",
            line=dict(color=color, width=2, dash="dash"),
            opacity=0.6,
            name=f"{legend_group} GT", legendgroup=legend_group, showlegend=True,
            hovertemplate=("GT track %d<br>"
                           "(X=%%{x:.2f}, Y=%%{y:.2f}, Z=%%{z:.2f}) m"
                           "<extra></extra>" % n),
        ))
        traces.append(go.Scatter3d(
            x=pred_NF3[n, m, 0], y=pred_NF3[n, m, 1], z=pred_NF3[n, m, 2],
            mode="lines",
            line=dict(color=color, width=4),
            opacity=0.95,
            name=f"{legend_group} pred", legendgroup=legend_group, showlegend=True,
            hovertemplate=("Pred track %d<br>"
                           "(X=%%{x:.2f}, Y=%%{y:.2f}, Z=%%{z:.2f}) m"
                           "<extra></extra>" % n),
        ))
        a = int(anchor_n[n])
        if 0 <= a < F:
            traces.append(go.Scatter3d(
                x=[gt_NF3[n, a, 0]], y=[gt_NF3[n, a, 1]], z=[gt_NF3[n, a, 2]],
                mode="markers",
                marker=dict(color=color, size=4, line=dict(color="black", width=0.5)),
                name=f"{legend_group} anchor", legendgroup=legend_group, showlegend=False,
                hovertemplate=("Anchor track %d (frame %d)<extra></extra>" % (n, a)),
            ))
    fig = go.Figure(data=traces)
    fig.update_layout(
        title=dict(text=f"{title}<br><sub>solid = pred, dashed = GT, dot = anchor</sub>",
                   font=dict(size=12)),
        scene=dict(
            xaxis_title="X (m)",
            yaxis_title="Y (m)",
            zaxis_title="Z (m)",
            aspectmode="data",
        ),
        legend=dict(font=dict(size=9), itemsizing="constant"),
        margin=dict(l=0, r=0, b=0, t=60),
        height=700,
    )
    fig.write_html(out_path, include_plotlyjs="cdn", full_html=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="output dir; defaults to <ckpt_parent>/viz3d_ckpt<step>/",
    )
    ap.add_argument("--data-root", type=Path, default=Path("~/data"))
    ap.add_argument("--subsets", nargs="+", default=list(SUBSETS))
    ap.add_argument("--clips-per-subset", type=int, default=2)
    ap.add_argument("--split", choices=["all", "minival", "full_eval"], default="all",
                    help="Restrict source clips to a named TAPVid-3D split.")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--max-tracks", type=int, default=32)
    ap.add_argument("--amp", choices=["bf16", "fp32"], default="bf16")
    args = ap.parse_args()

    if args.out_dir is None:
        step = args.ckpt.stem.split("_")[-1] if "_" in args.ckpt.stem else args.ckpt.stem
        args.out_dir = args.ckpt.parent / f"viz3d_ckpt{step}"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.data_root = args.data_root.expanduser()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}[args.amp]

    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = _build_model(state, device)
    print(f"[3d] loaded {args.ckpt} (step={state.get('step', '?')})")

    for sub in args.subsets:
        clips = [p for p in filter_to_split(list_clips(args.data_root, [sub]), args.split)
                 if has_images(p)][: args.clips_per_subset]
        print(f"[3d] {sub}: rendering {len(clips)} clips")
        for path in clips:
            try:
                clip = load_clip(path)
                pred = _predict_tracks(model, clip, device, amp_dtype, args.max_frames)
                F = pred.shape[1]
                gt = clip.tracks_XYZ[:F].numpy().transpose(1, 0, 2)         # (N, F, 3)
                vis = clip.visibility[:F].numpy().transpose(1, 0)            # (N, F)
                anchor = clip.queries_xyt[:, 2].long().clamp(0, F - 1).numpy()
                title = f"{sub}/{clip.clip_id}  (step {state.get('step', '?')})  — xyz (Z = world depth)"
                png_path  = args.out_dir / f"{sub}_{clip.clip_id}_3d.png"
                html_path = args.out_dir / f"{sub}_{clip.clip_id}_3d.html"
                _plot_clip_3d_png(pred, gt, vis, anchor, png_path, title, args.max_tracks)
                _plot_clip_3d_html(pred, gt, vis, anchor, html_path, title, args.max_tracks)
                print(f"[3d] wrote {png_path}")
                print(f"[3d] wrote {html_path}")
            except Exception as e:
                print(f"[3d] {sub}/{path.stem}: FAIL {type(e).__name__}: {e}")
    print("[3d] DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
