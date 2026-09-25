#!/usr/bin/env python3
"""Qualitative 3D-trajectory figures: our best method v39 vs the similar-size SOTA
DELTA+DA3-l, on each subset's BEST-scoring v39 clip.

For each subset we pick the clip where v39 (WAFT flow + the v35 depth refiner) has the
highest per-clip absolute metric-AJ, then render DELTA+DA3-l (left) and v39 (right) 3D
tracks against GT. A red ring marks the track where v39 beats DELTA the most (same
physical region circled in both panels, so the win is directly comparable). Both share
the DA3-l depth backbone; DELTA runs its own DenseTrack3D 2D tracker (no WAFT), which is
intrinsic to the two methods. Fonts are 2x the previous size and titles are short.

Panels use the shared image-like camera (X right, Y down, Z 45 deg up-right) of
`mamba3_tracker.viz.track3d_axes`, the same one as the Fig. 1(b) teaser.

Inference is cached per subset so the panels can be re-rendered instantly via
--replot (no checkpoint, no depth cache, no GPU) after a plotting change.

  uv run python scripts/render_qual_3d.py \
    --v35-ckpt ~/proj/study/largescale3Dreconstruction_using_SSM/result/20260701_v35/ckpt_20000.pt \
    --waft-pred-dir ~/data/tapvid3d_baseline_preds/waft \
    --delta-pred-dir ~/data/tapvid3d_baseline_preds/delta \
    --da3-depth-root ~/data/tapvid3d_da3 \
    --scores-dir ~/proj/study/vmamba3-3Dpointtracker/result/20260710-1056_metric3d_v39_waft_v35/metric_results \
    --out-dir doc/vmamba3_3dpointtrack/figs
  uv run python scripts/render_qual_3d.py --replot --out-dir doc/vmamba3_3dpointtrack/figs
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from mpl_toolkits.mplot3d import proj3d  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from mamba3_tracker.paths import cache_dir  # noqa: E402
from mamba3_tracker.viz.track3d_axes import (  # noqa: E402
    apply_equal_cube,
    apply_image_like_view,
)

plt.rcParams.update({"font.size": 27})  # axis labels; see PRINT_SCALE below
# when these panels print at ~2.4 in (paper Fig 13) / ~3 in (memo Figs 14-16).

_HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "eval_metric3d", _HERE / "eval_metric3d.py"
)
_ev = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ev)  # provides _infer, _load_external, load_clip, list_clips

SUBSETS = ["pstudio", "drivetrack", "adt"]
IMAGE_SIZE = 896
MAX_TRACKS = 32
CACHE_DIR = cache_dir() / "qual_3d"

# Each panel is drawn FIG_W_IN wide and placed at 0.36\linewidth of a 522pt text block, so a
# glyph of native size s renders at s * 0.36 * 522 / (FIG_W_IN * 72) on the page. The paper's
# caption font is \footnotesize = 8pt: in-plot text should match it, neither falling under
# (the previous 24pt on an 11in canvas landed at 5.3pt) nor overshooting it.
FIG_W_IN, FIG_H_IN = 8.0, 7.0
PRINT_SCALE = 0.36 * 522.0 / (FIG_W_IN * 72.0)
FS_TICK = 26
FS_LEGEND = 25
assert 8.0 <= min(FS_TICK, FS_LEGEND) * PRINT_SCALE <= 9.5


def _best_clip(scores_dir: Path, subset: str) -> str:
    clips = json.loads((scores_dir / f"{subset}.json").read_text())
    best = max(
        (c for c in clips if isinstance(c.get("metric_average_jaccard"), (int, float))),
        key=lambda c: c["metric_average_jaccard"],
    )
    return best["clip_id"]


def _pick_tracks(vis_NF: np.ndarray, k: int) -> list[int]:
    order = np.argsort(-vis_NF.sum(axis=1))
    return list(order[:k])


def _win_center(other, ours, gt, vis):
    """Among the plotted tracks, find the one whose per-frame 3D error `ours` reduces
    most vs `other` (the SOTA), and return the (x,y,z) centroid of its GT path (the
    region to circle). Returns None if no track is actually better."""
    best_n, best_gain = None, 0.0
    for n in _pick_tracks(vis, MAX_TRACKS):
        m = vis[n] > 0.5
        if m.sum() < 2:
            continue
        oe = np.linalg.norm(other[n, m] - gt[n, m], axis=1)
        ve = np.linalg.norm(ours[n, m] - gt[n, m], axis=1)
        gain = float((oe - ve).sum())
        if gain > best_gain:
            best_gain, best_n = gain, n
    if best_n is None:
        return None
    m = vis[best_n].astype(bool)
    return gt[best_n, m].mean(axis=0)


# The ring marks a ball of RING_R metres around the winning track, and the inset shows
# exactly that ball -- the review asked for "the same content, enlarged". Both are derived
# from this one number, so they cannot drift apart.
CUBE_M = 3.0         # side of the plotted volume, metres -- see _cube_lims
RING_FRAC = 0.075    # ring radius as a fraction of that side, so every panel matches
RING_R = RING_FRAC * CUBE_M
BUBBLE_MAG = 2.0     # the inset shows the ring's ball at this many times its on-screen size
# A 3-D axes draws its cube inside only part of its own rectangle, so a rect sized to the
# wanted magnification lands well short of it. Measured from a rendered panel: cube width /
# rect width = 0.61. The rect is divided by this, and the ring is kept small enough that the
# result still fits -- at RING_FRAC 0.15 a true 2x inset would have needed 115% of the figure.
AXES_FILL = 0.61
BUBBLE_FILL = 2.15


def _cube_lims(gt, vis, side=CUBE_M):
    """A fixed `side`-metre cube centred on the plotted ground truth.

    Framing each clip to its own extent made the panels incomparable and, on drivetrack,
    unreadable: that clip spans 6.5 m in depth against 1.3 m across, and the equal-scale
    cube the depth forces shrinks the tracks to a clump. A fixed cube gives every panel the
    same scale, so the red ring covers the same fraction of every box. Tracks leaving the
    cube are clipped, which the caption states."""
    pts = np.concatenate([gt[n, vis[n] > 0.5] for n in range(gt.shape[0])
                          if (vis[n] > 0.5).sum() > 1])
    c = np.median(pts, axis=0)
    return [(float(c[i] - side / 2.0), float(c[i] + side / 2.0)) for i in range(3)]


def _ring_radius_px(fig, ax, c, r_world):
    """On-screen radius, in display pixels, of a ball of r_world metres centred at c.

    The camera is orthographic with equal axis scales, so one metre is the same number of
    pixels in every direction and projecting one offset point suffices. The draw() is
    required: before it, a 3-D axes' projection matrix is stale and the transform silently
    returns positions from the previous layout."""
    fig.canvas.draw()
    pts = []
    for q in (c, (c[0] + r_world, c[1], c[2])):
        x, y, _ = proj3d.proj_transform(q[0], q[1], q[2], ax.get_proj())
        pts.append(np.asarray(ax.transData.transform((x, y)), dtype=float))
    return pts[0], float(np.hypot(*(pts[1] - pts[0])))


def _add_zoom_bubble(fig, ax, pred, gt, vis, c, ring_px, r_px) -> None:
    """AC10: magnify the ringed region into an inset drawn inside the same axes box.

    The review asked for the circled region enlarged. Placing the inset inside the figure
    rather than beside it keeps Fig. 9 on one page -- the grid is already 3 rows by 2
    columns -- and a leader line ties it back to the ring so the two read as one object.
    """
    # side of the inset in figure fractions: BUBBLE_MAG x the ring's on-screen diameter
    fw, fh = fig.get_size_inches() * fig.dpi
    side_x = BUBBLE_MAG * 2.0 * r_px / AXES_FILL / fw
    side_y = BUBBLE_MAG * 2.0 * r_px / AXES_FILL / fh
    # Top-left, not bottom-right: mplot3d draws the Z tick labels at the lower right of the
    # cube, and an inset anchored to the figure's bottom-right corner lands on top of them.
    # On the drivetrack row, whose Z labels are two digits wide (9, 10, 11), that clipped
    # "10" to a bare "1". The top-left corner is empty in every panel -- the X ticks start
    # well to its right -- so the inset can sit there without covering any label.
    rect = (0.03, 1.0 - side_y - 0.03, side_x, side_y)
    # The red edge makes the inset read as one object with the ring it magnifies.
    fig.add_artist(Rectangle((rect[0], rect[1]), rect[2], rect[3],
                             transform=fig.transFigure, facecolor="white",
                             edgecolor="red", linewidth=1.5, zorder=10))
    inset = fig.add_axes(rect, projection="3d")
    inset.set_zorder(11)
    inset.patch.set_alpha(0.0)
    inset.set_facecolor("white")
    cmap = plt.get_cmap("tab20")
    near = False
    for i, n in enumerate(_pick_tracks(vis, MAX_TRACKS)):
        m = vis[n] > 0.5
        if m.sum() < 2:
            continue
        # keep only the stretch of this track that passes through the zoomed volume
        inside = m & (np.abs(gt[n, :, :] - c).max(axis=1) <= RING_R)
        if inside.sum() < 2:
            continue
        near = True
        col = cmap(i % 20)
        inset.plot(gt[n, inside, 0], gt[n, inside, 1], gt[n, inside, 2],
                   "--", lw=1.6, color=col, alpha=0.7)
        inset.plot(pred[n, inside, 0], pred[n, inside, 1], pred[n, inside, 2],
                   "-", lw=2.4, color=col, alpha=0.95)
    if not near:
        fig.delaxes(inset)
        return
    apply_equal_cube(inset, [(float(c[i] - RING_R), float(c[i] + RING_R)) for i in range(3)])
    apply_image_like_view(inset)
    inset.set_box_aspect((1, 1, 1), zoom=BUBBLE_FILL)
    # No axis box inside the bubble: the red frame already delimits it, and a second set of
    # panes and ticks at this size is clutter that competes with the tracks.
    inset.set_axis_off()
    # leader line from the ring to the inset, drawn in figure coordinates
    # start at the ring's edge, not its centre, so the line does not cross the marked region
    target_px = np.array([(rect[0] + 0.03) * fw, (rect[1] + rect[3]) * fh])
    d = target_px - np.asarray(ring_px, dtype=float)
    n = float(np.hypot(*d)) or 1.0
    edge_px = np.asarray(ring_px, dtype=float) + d / n * r_px
    p_fig = fig.transFigure.inverted().transform(edge_px)
    fig.add_artist(Line2D([p_fig[0], rect[0] + 0.03],
                          [p_fig[1], rect[1] + rect[3]],
                          color="red", lw=1.8, alpha=0.8, zorder=15))


def _render_3d(
    pred, gt, vis, anchor, out_path: Path, title: str, highlight=None, lims=None,
    legend: bool = False,
) -> None:
    """pred/gt: (N,F,3); vis: (N,F); anchor: (N,). Short 2x-font 3D plot.
    highlight: (3,) world point to ring in red (the region we win most), or None.
    lims: shared [(xlo,xhi),(ylo,yhi),(zlo,zhi)] applied to both panels, or None."""
    N, Fn, _ = pred.shape

    def _clip(p):
        """Blank predicted points outside the displayed box so 3D line drawing
        doesn't streak them across the axes (matplotlib does not clip 3D lines).
        Identical rule for both methods; GT (which defines the box) is untouched."""
        if lims is None:
            return p
        q = p.copy()
        for i in range(3):
            out = (q[:, i] < lims[i][0]) | (q[:, i] > lims[i][1])
            q[out] = np.nan
        return q

    fig = plt.figure(figsize=(FIG_W_IN, FIG_H_IN))
    ax = fig.add_subplot(111, projection="3d")
    cmap = plt.get_cmap("tab20")
    for i, n in enumerate(_pick_tracks(vis, MAX_TRACKS)):
        m = vis[n] > 0.5
        if m.sum() < 2:
            continue
        c = cmap(i % 20)
        ax.plot(
            gt[n, m, 0], gt[n, m, 1], gt[n, m, 2], "--", lw=1.4, color=c, alpha=0.55
        )
        pc = _clip(pred[n])
        ax.plot(
            pc[m, 0],
            pc[m, 1],
            pc[m, 2],
            "-",
            lw=2.0,
            color=c,
            alpha=0.95,
        )
        a = int(anchor[n])
        if 0 <= a < Fn:
            ax.scatter(
                [gt[n, a, 0]],
                [gt[n, a, 1]],
                [gt[n, a, 2]],
                s=30,
                color=c,
                edgecolors="black",
                linewidths=0.6,
            )
    if highlight is not None:
        _ring = np.asarray(highlight, dtype=float)
    # AC10: the solid/dashed convention lived only in the caption; state it in the figure.
    # Only one panel carries it -- the convention is shared by all six, and six legible
    # copies would take more room than the plots.
    if legend:
        ax.plot([], [], [], "-", lw=2.0, color="black", label="predicted")
        ax.plot([], [], [], "--", lw=1.4, color="black", label="ground truth")
        if highlight is not None:
            ax.scatter([], [], [], s=90, facecolors="none", edgecolors="red", linewidths=2.0,
                       label="largest improvement")
        ax.legend(loc="lower left", fontsize=FS_LEGEND, framealpha=0.9, borderpad=0.3,
                  handlelength=1.6, labelspacing=0.25, bbox_to_anchor=(-0.02, -0.02))
    if lims is not None:
        # Equal metric scale with a box that hugs the data, so the tracks reach
        # the axis edges instead of sitting in a small central region.
        # (Point-clipping above still uses `lims`, so the "DELTA leaves the true
        # volume" view holds.)
        apply_equal_cube(ax, lims)
    apply_image_like_view(ax)
    ax.set_xlabel("X (m)", labelpad=12)
    ax.set_ylabel("Y (m)", labelpad=12)
    # Z runs diagonally under the box, so its label needs a bigger outward offset
    # than X/Y or it collides with its own tick labels.
    ax.set_zlabel("Z (m)", labelpad=40)
    ax.tick_params(labelsize=FS_TICK)
    # No in-plot title: the method/subset is stated by the LaTeX sub-caption
    # (paper Fig 13) / figure caption (memo), so a title here is redundant.
    fig.tight_layout()
    if highlight is not None:
        # after tight_layout, so the projection and the axes box are final
        p0, r_px = _ring_radius_px(fig, ax, _ring, RING_R)
        # a square, not a circle: the inset shows a box, and a circle would promise a
        # region the magnified view does not actually correspond to. marker="s" takes an
        # area in points squared, so its side is exactly d_pt.
        d_pt = 2.0 * r_px * 72.0 / fig.dpi          # marker size is in points, not pixels
        ax.scatter([_ring[0]], [_ring[1]], [_ring[2]], s=d_pt ** 2, marker="s",
                   facecolors="none", edgecolors="red", linewidths=3.0, zorder=20)
        _add_zoom_bubble(fig, ax, pred, gt, vis, _ring, p0, r_px)
    # pad_inches leaves whitespace so the rotated 3D "Z (m)" label (which
    # bbox_inches="tight" under-measures for mplot3d) is not clipped at the edge.
    fig.savefig(out_path, dpi=330, bbox_inches="tight", pad_inches=0.5)
    plt.close(fig)
    print(f"[qual] wrote {out_path}")


def _render_st(gt, pred, vis, out_path: Path, title: str) -> None:
    """Space-time: X-t / Y-t / Z-t panels. gt/pred (N,F,3), vis (N,F)."""
    N, Fn, _ = pred.shape
    t = np.arange(Fn)
    fig, axes = plt.subplots(1, 3, figsize=(20, 6.5))
    cmap = plt.get_cmap("tab20")
    picked = _pick_tracks(vis, MAX_TRACKS)
    for ax, ax_i, lab in zip(axes, range(3), ["X", "Y", "Z"]):
        for i, n in enumerate(picked):
            m = vis[n] > 0.5
            if m.sum() < 2:
                continue
            c = cmap(i % 20)
            ax.plot(t[m], gt[n, m, ax_i], "--", lw=1.4, color=c, alpha=0.55)
            ax.plot(t[m], pred[n, m, ax_i], "-", lw=2.0, color=c, alpha=0.95)
        ax.set_xlabel("frame")
        ax.set_ylabel(f"{lab} (m)")
        ax.set_title(f"{lab}-t")
        ax.grid(alpha=0.3)
    # Ring the largest depth error on the Z-t panel: depth (Z) is the hard axis.
    zerr = np.abs(pred[:, :, 2] - gt[:, :, 2]) * vis
    zerr[[n for n in range(N) if n not in picked]] = 0.0
    n_max, f_max = np.unravel_index(np.argmax(zerr), zerr.shape)
    if zerr[n_max, f_max] > 0:
        axes[2].scatter(
            [t[f_max]],
            [pred[n_max, f_max, 2]],
            s=9000,
            facecolors="none",
            edgecolors="red",
            linewidths=3.0,
            zorder=20,
        )
    fig.suptitle(title)
    fig.tight_layout()
    # pad_inches leaves whitespace so the rotated 3D "Z (m)" label (which
    # bbox_inches="tight" under-measures for mplot3d) is not clipped at the edge.
    fig.savefig(out_path, dpi=330, bbox_inches="tight", pad_inches=0.5)
    plt.close(fig)
    print(f"[qual] wrote {out_path}")


_VIS_HEAD = [None]


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


def _build_v39(ckpt: Path, dev) -> torch.nn.Module:
    """v39 = the v35 depth refiner (Mamba3V35Refiner) fed by the WAFT front-end."""
    from mamba3_tracker.model.depth_refined_tracker import Mamba3V35Refiner

    st = torch.load(ckpt, map_location="cpu", weights_only=False)
    mc = st.get("cfg", {}).get("model", {})
    model = Mamba3V35Refiner(
        dim=int(mc.get("dim", 128)),
        state_dim=int(mc.get("state_dim", 64)),
        num_heads=int(mc.get("num_heads", 4)),
        num_layers=int(mc.get("num_layers", 2)),
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
    # Tolerant: checkpoints written before the refiner gained its own vis_head lack those
    # tensors, and this arm takes visibility from the separate flow-only head instead.
    _ev._load_ckpt_into(model, st["model"])
    model.eval()
    return model


def _infer_subset(args, v39, dev, sub: str):
    """Run (or load) the per-subset tracks. Cached to an npz so a plotting-only
    change can be re-rendered with --replot, without the checkpoint or the GPU."""
    cache = CACHE_DIR / f"{sub}.npz"
    if args.replot:
        d = np.load(cache)
        return str(d["clip_id"]), d["gt"], d["anchor"], d["v39"], d["vis"], d["delta"]

    clip_id = _best_clip(args.scores_dir, sub)
    path = next(
        p for p in _ev.list_clips(Path("~/data").expanduser(), [sub]) if p.stem == clip_id
    )
    clip = _ev.load_clip(path)
    gt = clip.tracks_XYZ.numpy().transpose(1, 0, 2)  # (N,F,3)
    anchor = clip.queries_xyt[:, 2].numpy().astype(int)
    v39xyz, vis = _ev._infer(
        method="v35",
        flow_model=None,
        model=v39,
        clip=clip,
        image_size=IMAGE_SIZE,
        fb_alpha=0.05,
        fb_beta=1.0,
        da3_depth_root=args.da3_depth_root,
        max_frames=0,
        device=dev,
        waft_pred_dir=args.waft_pred_dir,
        vis_source="v94" if args.vis_head_ckpt else "flow",
        vis_head=_VIS_HEAD[0],
        flowvis_dir=str(Path(args.flowvis_dir).expanduser()) if args.flowvis_dir else None,
    )
    delta, _ = _ev._load_external(args.delta_pred_dir, sub, clip_id)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(cache, clip_id=clip_id, gt=gt, anchor=anchor, v39=v39xyz, vis=vis, delta=delta)
    return clip_id, gt, anchor, v39xyz, vis, delta


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v35-ckpt", type=Path)
    ap.add_argument("--waft-pred-dir", type=Path)
    ap.add_argument("--delta-pred-dir", type=Path)
    ap.add_argument("--da3-depth-root", type=Path)
    ap.add_argument("--scores-dir", type=Path, help="v39 metric_results")
    ap.add_argument("--vis-head-ckpt", type=Path, default=None,
                    help="FlowVisHead checkpoint; selects the flow-only visibility head")
    ap.add_argument("--flowvis-dir", type=Path, default=None,
                    help="dir of cached forward/backward flow, required with --vis-head-ckpt")
    ap.add_argument(
        "--out-dir", type=Path, default=Path("doc/vmamba3_3dpointtrack/figs")
    )
    ap.add_argument(
        "--replot", action="store_true", help="re-plot from cached inference only"
    )
    args = ap.parse_args()
    needed = ["v35_ckpt", "waft_pred_dir", "delta_pred_dir", "da3_depth_root", "scores_dir"]
    if not args.replot and any(getattr(args, a) is None for a in needed):
        ap.error("without --replot these are required: " + ", ".join("--" + a.replace("_", "-") for a in needed))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.vis_head_ckpt and not args.replot:
        if not args.flowvis_dir:
            raise SystemExit("--vis-head-ckpt needs --flowvis-dir")
        _VIS_HEAD[0] = _build_vis_head(args.vis_head_ckpt, dev)
    v39 = None if args.replot else _build_v39(args.v35_ckpt, dev)

    for sub in SUBSETS:
        clip_id, gt, anchor, v39xyz, vis, delta = _infer_subset(args, v39, dev, sub)
        hl = _win_center(delta, v39xyz, gt, vis)
        lims = _cube_lims(gt, vis)
        print(
            f"[qual] {sub}: best v39 clip {clip_id}  N={gt.shape[0]} F={gt.shape[1]}  "
            f"highlight={'none' if hl is None else hl.round(2)}"
        )

        _render_3d(
            delta,
            gt,
            vis,
            anchor,
            args.out_dir / f"qual_{sub}_delta_3d.png",
            f"{sub}: DELTA+DA3-l (SOTA)",
            highlight=hl,
            lims=lims,
            legend=(sub == "drivetrack"),
        )
        _render_3d(
            v39xyz,
            gt,
            vis,
            anchor,
            args.out_dir / f"qual_{sub}_v39_3d.png",
            f"{sub}: v39 (ours)",
            highlight=hl,
            lims=lims,
        )
        if sub == "drivetrack":
            _render_st(
                gt,
                delta,
                vis,
                args.out_dir / "qual_drivetrack_delta_st.png",
                "drivetrack DELTA+DA3-l (SOTA): position vs time",
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
