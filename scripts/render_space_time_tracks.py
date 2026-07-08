"""Render space-time diagrams of predicted vs GT 3D point trajectories.

For each clip we draw THREE 3D plots side-by-side. Each plot's vertical
axis (`z` in the rendered 3D figure) is the frame index (time), and the
two horizontal axes are one of the planar projections of the world-space
trajectory:

    plot 1   X (world)  ─ Y (world)   ─ time
    plot 2   Y (world)  ─ Z (world)   ─ time
    plot 3   Z (world)  ─ X (world)   ─ time

For each sampled track we draw the predicted trajectory as a solid line,
the GT as a dashed line in the same per-track colour, and mark the anchor
frame with a dot.

Outputs per clip (in --out-dir):
    <subset>_<clip_id>_st.png      matplotlib mplot3d 1×3 grid (static)
    <subset>_<clip_id>_st.html     plotly self-contained 1×3 interactive
                                   (open in browser to rotate / zoom each)

Same model + inference path as scripts/render_3d_tracks.py.

Usage:
    uv run python scripts/render_space_time_tracks.py \\
        --ckpt result/<dt>_track_v18/ckpt_30000.pt \\
        --out-dir result/<dt>_track_v18/viz_step30000 \\
        --subsets pstudio drivetrack --clips-per-subset 2 \\
        --max-frames 32 --max-tracks 24
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
from plotly.subplots import make_subplots

from mamba3_tracker.data.dataset import filter_to_split
from mamba3_tracker.data.tapvid3d import SUBSETS, has_images, list_clips, load_clip
from mamba3_tracker.model.tracker import Mamba3Tracker


# Per-subset frame rate from the TAPVid-3D paper (clip-level FPS isn't
# stored in the .npz files). Used to convert frame index → seconds on
# the time axis.
SUBSET_FPS = {"pstudio": 30.0, "drivetrack": 10.0, "adt": 20.0}


# (a_idx, b_idx, a_label, b_label) — order chosen so the planes cycle
# XY → YZ → ZX, matching the user spec.
PROJECTIONS = [
    (0, 1, "X", "Y"),
    (1, 2, "Y", "Z"),
    (2, 0, "Z", "X"),
]

PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
    "#c49c94", "#f7b6d2", "#c7c7c7", "#dbdb8d", "#9edae5",
]


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
    """Return predicted absolute 3D positions (N_q, F, 3) in camera coords."""
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
    return abs_pred.transpose(0, 1).numpy()


def _pick_tracks(vis_NF: np.ndarray, max_tracks: int, seed: int = 0) -> list[int]:
    rng = np.random.default_rng(seed)
    vis_count = vis_NF.sum(axis=1)
    candidate = np.argsort(-vis_count)[: min(max_tracks * 2, vis_NF.shape[0])]
    if len(candidate) > max_tracks:
        candidate = rng.choice(candidate, size=max_tracks, replace=False)
    return sorted(candidate.tolist())


def _plot_clip_st_png(
    pred_NF3: np.ndarray, gt_NF3: np.ndarray, vis_NF: np.ndarray,
    anchor_n: np.ndarray, times_s: np.ndarray,
    out_path: Path, title: str, max_tracks: int = 32,
) -> None:
    N_q, F, _ = pred_NF3.shape
    sel = _pick_tracks(vis_NF, max_tracks)

    fig = plt.figure(figsize=(21, 7))
    cmap = plt.get_cmap("tab20")
    for col, (a_idx, b_idx, a_lab, b_lab) in enumerate(PROJECTIONS):
        ax = fig.add_subplot(1, 3, col + 1, projection="3d")
        for i, n in enumerate(sel):
            color = cmap(i % 20)
            m = vis_NF[n].astype(bool)
            if m.sum() < 2:
                continue
            ax.plot(gt_NF3[n, m, a_idx], gt_NF3[n, m, b_idx], times_s[m],
                    "--", color=color, alpha=0.55, linewidth=1.0)
            ax.plot(pred_NF3[n, m, a_idx], pred_NF3[n, m, b_idx], times_s[m],
                    "-", color=color, alpha=0.95, linewidth=1.4)
            a = int(anchor_n[n])
            if 0 <= a < F:
                ax.scatter([gt_NF3[n, a, a_idx]], [gt_NF3[n, a, b_idx]], [times_s[a]],
                           s=18, color=color, edgecolors="black", linewidths=0.4)
        ax.set_xlabel(f"{a_lab} (m)", fontsize=12, labelpad=6)
        ax.set_ylabel(f"{b_lab} (m)", fontsize=12, labelpad=6)
        ax.set_zlabel("time (s)",     fontsize=12, labelpad=6)
        ax.set_title(f"{a_lab.lower()}{b_lab.lower()}t  —  {a_lab}-{b_lab} plane vs time",
                     fontsize=13, fontweight="bold")
    fig.suptitle(f"{title}\nsolid = predicted, dashed = GT, dot = anchor frame", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def _plot_clip_st_html(
    pred_NF3: np.ndarray, gt_NF3: np.ndarray, vis_NF: np.ndarray,
    anchor_n: np.ndarray, times_s: np.ndarray,
    out_path: Path, title: str, max_tracks: int = 32,
) -> None:
    N_q, F, _ = pred_NF3.shape
    sel = _pick_tracks(vis_NF, max_tracks)

    fig = make_subplots(
        rows=1, cols=3, specs=[[{"type": "scene"}] * 3],
        subplot_titles=[
            f"<b>{a_lab.lower()}{b_lab.lower()}t</b>  —  {a_lab}-{b_lab} plane vs time"
            for _, _, a_lab, b_lab in PROJECTIONS
        ],
        horizontal_spacing=0.02,
    )
    for col, (a_idx, b_idx, a_lab, b_lab) in enumerate(PROJECTIONS):
        for i, n in enumerate(sel):
            color = PALETTE[i % len(PALETTE)]
            m = vis_NF[n].astype(bool)
            if m.sum() < 2:
                continue
            # Show legend only on the first subplot to avoid duplicates;
            # legendgroup keeps tracks linked across all three plots.
            show = col == 0
            grp = f"track {n}"
            fig.add_trace(go.Scatter3d(
                x=gt_NF3[n, m, a_idx], y=gt_NF3[n, m, b_idx], z=times_s[m],
                mode="lines",
                line=dict(color=color, width=2, dash="dash"),
                opacity=0.6,
                name=f"{grp} GT", legendgroup=grp, showlegend=show,
                hovertemplate=("GT track %d<br>t=%%{z:.2f}s<br>"
                               "(%s=%%{x:.2f}, %s=%%{y:.2f}) m<extra></extra>"
                               % (n, a_lab, b_lab)),
            ), row=1, col=col + 1)
            fig.add_trace(go.Scatter3d(
                x=pred_NF3[n, m, a_idx], y=pred_NF3[n, m, b_idx], z=times_s[m],
                mode="lines",
                line=dict(color=color, width=4),
                opacity=0.95,
                name=f"{grp} pred", legendgroup=grp, showlegend=show,
                hovertemplate=("Pred track %d<br>t=%%{z:.2f}s<br>"
                               "(%s=%%{x:.2f}, %s=%%{y:.2f}) m<extra></extra>"
                               % (n, a_lab, b_lab)),
            ), row=1, col=col + 1)
            a = int(anchor_n[n])
            if 0 <= a < F:
                fig.add_trace(go.Scatter3d(
                    x=[gt_NF3[n, a, a_idx]], y=[gt_NF3[n, a, b_idx]], z=[times_s[a]],
                    mode="markers",
                    marker=dict(color=color, size=4, line=dict(color="black", width=0.5)),
                    name=f"{grp} anchor", legendgroup=grp, showlegend=False,
                    hovertemplate=("Anchor track %d (t=%.2fs)<extra></extra>"
                               % (n, float(times_s[a]))),
                ), row=1, col=col + 1)
        scene_id = "scene" if col == 0 else f"scene{col + 1}"
        fig.layout[scene_id].update(
            xaxis_title=f"{a_lab} (m)",
            yaxis_title=f"{b_lab} (m)",
            zaxis_title="time (s)",
            aspectmode="cube",
        )
    fig.update_layout(
        title=dict(text=f"{title}<br><sub>solid = pred, dashed = GT, dot = anchor</sub>",
                   font=dict(size=12)),
        legend=dict(font=dict(size=8), itemsizing="constant"),
        margin=dict(l=0, r=0, b=0, t=70),
        height=620,
    )
    fig.write_html(out_path, include_plotlyjs="cdn", full_html=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="output dir; defaults to <ckpt_parent>/viz_st_ckpt<step>/",
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
        args.out_dir = args.ckpt.parent / f"viz_st_ckpt{step}"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.data_root = args.data_root.expanduser()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}[args.amp]

    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = _build_model(state, device)
    print(f"[st] loaded {args.ckpt} (step={state.get('step', '?')})")

    for sub in args.subsets:
        clips = [p for p in filter_to_split(list_clips(args.data_root, [sub]), args.split)
                 if has_images(p)][: args.clips_per_subset]
        print(f"[st] {sub}: rendering {len(clips)} clips")
        for path in clips:
            try:
                clip = load_clip(path)
                pred = _predict_tracks(model, clip, device, amp_dtype, args.max_frames)
                F = pred.shape[1]
                gt = clip.tracks_XYZ[:F].numpy().transpose(1, 0, 2)
                vis = clip.visibility[:F].numpy().transpose(1, 0)
                anchor = clip.queries_xyt[:, 2].long().clamp(0, F - 1).numpy()
                fps = SUBSET_FPS.get(sub, 30.0)
                times_s = np.arange(F) / fps
                title = (f"{sub}/{clip.clip_id}  (step {state.get('step', '?')}, "
                         f"{fps:g} fps)")
                png_path  = args.out_dir / f"{sub}_{clip.clip_id}_st.png"
                html_path = args.out_dir / f"{sub}_{clip.clip_id}_st.html"
                _plot_clip_st_png(pred, gt, vis, anchor, times_s, png_path, title, args.max_tracks)
                _plot_clip_st_html(pred, gt, vis, anchor, times_s, html_path, title, args.max_tracks)
                print(f"[st] wrote {png_path}")
                print(f"[st] wrote {html_path}")
            except Exception as e:
                print(f"[st] {sub}/{path.stem}: FAIL {type(e).__name__}: {e}")
    print("[st] DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
