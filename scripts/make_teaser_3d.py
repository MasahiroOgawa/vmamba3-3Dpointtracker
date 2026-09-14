"""Render Figure 1(b) teaser: v39 (our best) recovered metric 3D tracks on the
teaser drivetrack clip, with equal x/y/z scale and the shared image-like camera
(X right, Y down, Z 45 deg up-right) that matches the 2D driving image in panel
(a) and the qualitative grid of Fig. 13 -- see `mamba3_tracker.viz.track3d_axes`.

v39 = the v35 depth refiner fed by the WAFT front-end (same clip/caches as the
eval). Inference is cached to an npz so the figure can be re-rendered instantly
via --replot without re-running the model.

  # first run (inference, on CPU to leave the GPU for training):
  CUDA_VISIBLE_DEVICES="" uv run python scripts/make_teaser_3d.py \
      --v35-ckpt ~/data/ckpts/v35_da3l_ckpt_20000.pt \
      --waft-pred-dir ~/data/tapvid3d_baseline_preds/waft \
      --da3-depth-root ~/data/tapvid3d_da3 --out /tmp/teaser_track3d.png
  # re-render from the cached inference only:
  uv run python scripts/make_teaser_3d.py --replot --out /tmp/teaser_track3d.png
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from mamba3_tracker.paths import cache_dir  # noqa: E402
from mamba3_tracker.viz.track3d_axes import (  # noqa: E402
    apply_equal_cube,
    apply_image_like_view,
)

# Fig 1(b) prints at ~2.67 in (single-column fraction), so the canvas is shrunk
# hard by LaTeX; size the lettering up front so it stays >=7pt effective.
plt.rcParams.update({"font.size": 18})

_HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("eval_metric3d", _HERE / "eval_metric3d.py")
_ev = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ev)

# The clip used in Figure 1(a) (identified by matching its GT coord ranges).
TEASER_CLIP = "tapvid3d_1022527355599519580_4866_960_4886_960_2_L58RM2TH_i-3sYbjr6JjQQ"
IMAGE_SIZE = 896
MAX_TRACKS = 32
PAD = 0.02  # tiny: it inflates the longest axis, which sets the cube for all three
# Side of the plotted cube, in metres. Framing the whole clip packs the trajectories into a
# band too dense to read (the review asked for fewer points on this panel); a fixed, smaller
# cube separates them instead. Tracks leaving the cube are clipped, which is the trade.
CUBE_M = 3.0
CACHE = cache_dir() / "teaser_3d.npz"


def _build_v39(ckpt: Path, dev) -> torch.nn.Module:
    from mamba3_tracker.model.depth_refined_tracker import Mamba3V35Refiner

    st = torch.load(ckpt, map_location="cpu", weights_only=False)
    mc = st.get("cfg", {}).get("model", {})
    model = Mamba3V35Refiner(
        dim=int(mc.get("dim", 128)), state_dim=int(mc.get("state_dim", 64)),
        num_heads=int(mc.get("num_heads", 4)), num_layers=int(mc.get("num_layers", 2)),
        max_log_correction=float(mc.get("max_log_correction", 2.0)),
        max_delta_uv=float(mc.get("max_delta_uv", 2.0)),
        patch_size=int(mc.get("patch_size", 5)),
        two_pool=bool(mc.get("two_pool", False)),
        per_frame_scale=bool(mc.get("per_frame_scale", False)),
        within_frame=bool(mc.get("within_frame", False)),
        d_proj=int(mc.get("d_proj", 64)),
        dino_model=str(mc.get("dino_model", "facebook/dinov3-vits16-pretrain-lvd1689m")),
        dino_image_size=int(mc.get("dino_image_size", 448)),
        image_size=int(mc.get("image_size", 896)),
    ).to(dev)
    # Tolerant load: checkpoints written before the refiner gained its own vis_head lack
    # those tensors, and this arm does not use them -- visibility comes from the v94 head.
    _ev._load_ckpt_into(model, st["model"])
    model.eval()
    return model


def _build_vis_head(ckpt: Path, dev):
    """The flow-only visibility head of the reported best arm (v94)."""
    from mamba3_tracker.model.flow_vis_head import FlowVisHead

    st = torch.load(ckpt, map_location="cpu", weights_only=False)
    mc = st.get("cfg", {}).get("model", {})
    head = FlowVisHead(
        dim=int(mc.get("dim", 64)), state_dim=int(mc.get("state_dim", 64)),
        num_heads=int(mc.get("num_heads", 4)), num_layers=int(mc.get("num_layers", 2)),
        bidirectional=bool(mc.get("bidirectional", True)),
    ).to(dev)
    head.load_state_dict(st["model"])
    head.eval()
    return head


def _last_full_frame(uv, vis) -> int:
    """Latest frame at which every visible track is still inside the image.

    Panel (a) should show as much of the motion as possible, but a frame where the target
    has already left the picture shows a trail leading out of an empty scene. This is the
    last frame that still holds the whole target.
    """
    F_ = uv.shape[0]
    best = 0
    for t in range(F_):
        m = vis[:, t] > 0.5
        if not m.any():
            continue
        inside = ((uv[t, :, 0] >= 0) & (uv[t, :, 0] < IMAGE_SIZE)
                  & (uv[t, :, 1] >= 0) & (uv[t, :, 1] < IMAGE_SIZE))
        if (m & inside).sum() == m.sum():
            best = t
    return best


def _plot2d(uv, vis, frame, t_end, out_path: Path) -> None:
    """Figure 1(a): each track drawn as one curve, from its start up to frame `t_end`.

    Two properties the review asked for. Joining each track's per-frame points turns an
    unreadably dense dot cloud into a handful of curves; stopping every curve at the frame
    being displayed makes the picture a snapshot of the tracker's state at that instant,
    rather than a summary of the whole clip drawn over one arbitrary frame.
    """
    uv = np.asarray(uv)          # (F, N, 2) in IMAGE_SIZE pixels
    vis = np.asarray(vis)        # (N, F)
    t_end = int(t_end)
    H, W = frame.shape[:2]
    sx, sy = W / float(IMAGE_SIZE), H / float(IMAGE_SIZE)
    fig, ax = plt.subplots(figsize=(W / 100.0, H / 100.0), dpi=100)
    ax.imshow(frame)
    cmap = plt.get_cmap("hsv")
    sel = _pick_tracks(vis, MAX_TRACKS)
    for i, n in enumerate(sel):
        m = vis[n] > 0.5
        m[t_end + 1:] = False          # trail only, nothing after the displayed frame
        if m.sum() < 2:
            continue
        c = cmap((i / max(len(sel) - 1, 1)) * 0.92)
        ax.plot(uv[m, n, 0] * sx, uv[m, n, 1] * sy, "-", lw=1.6, color=c, alpha=0.95)
        # the head of the trail: where the point actually is in the frame on screen
        ax.plot([uv[t_end, n, 0] * sx], [uv[t_end, n, 1] * sy], "o", ms=3.0,
                color=c, markeredgecolor="black", markeredgewidth=0.4)
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis("off")
    # dpi matches figsize so the frame is written at its own resolution: upsampling the
    # video frame only blurs it, and 1920 px over a half-column is already ~530 dpi in print.
    fig.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    print(f"[teaser] wrote {out_path}")


def _pick_tracks(vis, k):
    return list(np.argsort(-(vis > 0.5).sum(axis=1))[:k])


def _plot(pred, gt, vis, anchor, out_path: Path) -> None:
    # World (X, Y, Z) goes straight onto the plot's (x, y, z): the shared camera
    # already turns them into image directions (X right, Y down, Z into-scene),
    # so no axis permutation -- and no relabelling -- is needed here.
    sel = _pick_tracks(vis, MAX_TRACKS)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    cmap = plt.get_cmap("tab20")
    pts = []
    for i, n in enumerate(sel):
        m = vis[n] > 0.5
        if m.sum() < 2:
            continue
        c = cmap(i % 20)
        ax.plot(gt[n, m, 0], gt[n, m, 1], gt[n, m, 2], "--", lw=1.0, color=c, alpha=0.55)
        ax.plot(pred[n, m, 0], pred[n, m, 1], pred[n, m, 2], "-", lw=1.6, color=c, alpha=0.95)
        a = int(anchor[n])
        if 0 <= a < gt.shape[1] and m[a]:
            ax.scatter([gt[n, a, 0]], [gt[n, a, 1]], [gt[n, a, 2]], s=18,
                       color=c, edgecolors="black", linewidths=0.4)
        pts.append(gt[n, m])
    P = np.concatenate(pts, axis=0)
    # Centre the cube on the anchor-frame points, not on the whole trajectory cloud: the
    # anchors are where every track starts, so they stay in view however far a track runs.
    anchors = np.array([gt[n, int(anchor[n])] for n in sel
                        if 0 <= int(anchor[n]) < gt.shape[1] and vis[n][int(anchor[n])]])
    centre = np.median(anchors if len(anchors) else P, axis=0)
    lo, hi = centre - CUBE_M / 2.0, centre + CUBE_M / 2.0
    # Same framing helper as the Fig. 13 panels: one equal-span cube, sized to the
    # largest extent, so a gridline step means the same number of metres on every
    # axis. Y here is near-planar (~1 m against ~16 m of ground) and so fills only
    # a thin band of its axis -- the honest consequence of equal span.
    apply_equal_cube(ax, [(float(lo[i]), float(hi[i])) for i in range(3)])
    apply_image_like_view(ax)
    ax.set_xlabel("X (m)", labelpad=10)
    ax.set_ylabel("Y (m)", labelpad=14)
    # Z runs diagonally under the box, so its label needs a bigger outward offset
    # than X/Y or it collides with its own tick labels.
    ax.set_zlabel("Z (m)", labelpad=30)
    # pad_inches guards the rotated 3D axis labels, which bbox_inches="tight"
    # under-measures for mplot3d and would otherwise clip at the frame edge.
    fig.savefig(out_path, dpi=450, bbox_inches="tight", pad_inches=0.6)
    plt.close(fig)
    print(f"[teaser] wrote {out_path}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v35-ckpt", type=Path, default=Path("~/data/ckpts/v35_da3l_ckpt_20000.pt"))
    ap.add_argument("--waft-pred-dir", type=Path, default=Path("~/data/tapvid3d_baseline_preds/waft"))
    ap.add_argument("--da3-depth-root", type=Path, default=Path("~/data/tapvid3d_da3"))
    ap.add_argument("--out", type=Path, default=Path("/tmp/teaser_track3d.png"))
    ap.add_argument("--out2d", type=Path, default=None,
                    help="also write Figure 1(a): 2-D tracks drawn as polylines over a frame")
    ap.add_argument("--vis-head-ckpt", type=Path, default=None,
                    help="FlowVisHead checkpoint; selects the flow-only visibility head")
    ap.add_argument("--flowvis-dir", type=Path, default=None,
                    help="dir of cached forward/backward flow, required with --vis-head-ckpt")
    ap.add_argument("--replot", action="store_true", help="re-plot from cached inference only")
    args = ap.parse_args()

    if args.replot and CACHE.exists():
        d = np.load(CACHE)
        _plot(d["pred"], d["gt"], d["vis"], d["anchor"], args.out)
        if args.out2d:
            _plot2d(d["uv"], d["vis"], d["frame"], d["t_end"], args.out2d)
        return 0

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    v39 = _build_v39(args.v35_ckpt.expanduser(), dev)
    path = next(p for p in _ev.list_clips(Path("~/data").expanduser(), ["drivetrack"])
                if p.stem == TEASER_CLIP)
    clip = _ev.load_clip(path)
    gt = clip.tracks_XYZ.numpy().transpose(1, 0, 2)
    anchor = clip.queries_xyt[:, 2].numpy().astype(int)
    vis_head, vis_source = None, "flow"
    if args.vis_head_ckpt:
        if not args.flowvis_dir:
            raise SystemExit("--vis-head-ckpt needs --flowvis-dir")
        vis_head, vis_source = _build_vis_head(args.vis_head_ckpt.expanduser(), dev), "v94"
    extra = {}
    pred, vis = _ev._infer(
        method="v35", flow_model=None, model=v39, clip=clip, image_size=IMAGE_SIZE,
        fb_alpha=0.05, fb_beta=1.0, da3_depth_root=args.da3_depth_root.expanduser(),
        max_frames=0, device=dev, waft_pred_dir=args.waft_pred_dir.expanduser(),
        vis_source=vis_source, vis_head=vis_head,
        flowvis_dir=str(args.flowvis_dir.expanduser()) if args.flowvis_dir else None,
        out=extra,
    )
    t_end = _last_full_frame(extra["uv"], vis)
    frame = (clip.images[t_end].numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
    np.savez(CACHE, pred=pred, gt=gt, vis=vis, anchor=anchor, uv=extra["uv"],
             frame=frame, t_end=t_end)
    print(f"[teaser] panel (a) drawn at frame {t_end} of {gt.shape[1]}")
    print(f"[teaser] clip {TEASER_CLIP}  N={gt.shape[0]} F={gt.shape[1]}  cached -> {CACHE}")
    _plot(pred, gt, vis, anchor, args.out)
    if args.out2d:
        _plot2d(extra["uv"], vis, frame, t_end, args.out2d)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
