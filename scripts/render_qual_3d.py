#!/usr/bin/env python3
"""Qualitative 3D-trajectory figures: our best method v39 vs the similar-size SOTA
DELTA+DA3-l, on each subset's BEST-scoring v39 clip.

For each subset we pick the clip where v39 (WAFT flow + the v35 depth refiner) has the
highest per-clip absolute metric-AJ, then render DELTA+DA3-l (left) and v39 (right) 3D
tracks against GT. A red ring marks the track where v39 beats DELTA the most (same
physical region circled in both panels, so the win is directly comparable). Both share
the DA3-l depth backbone; DELTA runs its own DenseTrack3D 2D tracker (no WAFT), which is
intrinsic to the two methods. Fonts are 2x the previous size and titles are short.

  uv run python scripts/render_qual_3d.py \
    --v35-ckpt ~/proj/study/largescale3Dreconstruction_using_SSM/result/20260701_v35/ckpt_20000.pt \
    --waft-pred-dir ~/data/tapvid3d_baseline_preds/waft \
    --delta-pred-dir ~/data/tapvid3d_baseline_preds/delta \
    --da3-depth-root ~/data/tapvid3d_da3 \
    --scores-dir ~/proj/study/vmamba3-3Dpointtracker/result/20260710-1056_metric3d_v39_waft_v35/metric_results \
    --out-dir doc/vmamba3_3dpointtrack/figs
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

plt.rcParams.update({"font.size": 24})  # sized so lettering stays >=7pt effective
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


def _gt_lims(gt, vis, pad: float = 0.1):
    """Shared (x,y,z) axis limits from the GT of the plotted tracks, so the SOTA and
    ours panels show the exact same volume and the red ring lands in the same place."""
    pts = [
        gt[n, vis[n].astype(bool)]
        for n in _pick_tracks(vis, MAX_TRACKS)
        if vis[n].astype(bool).sum() >= 2
    ]
    P = np.concatenate(pts, axis=0)
    lo, hi = P.min(axis=0), P.max(axis=0)
    rng = np.maximum(hi - lo, 1e-3)
    lo, hi = lo - pad * rng, hi + pad * rng
    return [(float(lo[i]), float(hi[i])) for i in range(3)]


def _win_center(other, ours, gt, vis):
    """Among the plotted tracks, find the one whose per-frame 3D error `ours` reduces
    most vs `other` (the SOTA), and return the (x,y,z) centroid of its GT path (the
    region to circle). Returns None if no track is actually better."""
    best_n, best_gain = None, 0.0
    for n in _pick_tracks(vis, MAX_TRACKS):
        m = vis[n].astype(bool)
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


def _render_3d(
    pred, gt, vis, anchor, out_path: Path, title: str, highlight=None, lims=None
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

    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")
    cmap = plt.get_cmap("tab20")
    for i, n in enumerate(_pick_tracks(vis, MAX_TRACKS)):
        m = vis[n].astype(bool)
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
        ax.scatter(
            [highlight[0]],
            [highlight[1]],
            [highlight[2]],
            s=9000,
            facecolors="none",
            edgecolors="red",
            linewidths=3.0,
            zorder=20,
        )
    if lims is not None:
        # Equal axis scale: give x, y, z a common span (the largest GT extent),
        # each centred on its own midpoint, and force a cubic box so one metre is
        # the same length on every axis. (Point-clipping above still uses the
        # tighter GT box `lims`, so the "DELTA leaves the true volume" view holds.)
        mids = [0.5 * (lo + hi) for (lo, hi) in lims]
        half = max(hi - lo for (lo, hi) in lims) / 2.0
        ax.set_xlim(mids[0] - half, mids[0] + half)
        ax.set_ylim(mids[1] - half, mids[1] + half)
        ax.set_zlim(mids[2] - half, mids[2] + half)
        ax.set_box_aspect((1, 1, 1))
    ax.set_xlabel("X (m)", labelpad=12)
    ax.set_ylabel("Y (m)", labelpad=12)
    ax.set_zlabel("Z (m)", labelpad=12)
    ax.tick_params(labelsize=24)  # ticks are the smallest text -> keep >=7pt effective
    # No in-plot title: the method/subset is stated by the LaTeX sub-caption
    # (paper Fig 13) / figure caption (memo), so a title here is redundant.
    fig.tight_layout()
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
            m = vis[n].astype(bool)
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
        per_frame_scale=bool(mc.get("per_frame_scale", False)),
        within_frame=bool(mc.get("within_frame", False)),
        d_proj=int(mc.get("d_proj", 64)),
        dino_model=str(mc.get("dino_model", "facebook/dinov3-vits16-pretrain-lvd1689m")),
        dino_image_size=int(mc.get("dino_image_size", 448)),
        image_size=int(mc.get("image_size", 896)),
    ).to(dev)
    model.load_state_dict(st["model"])
    model.eval()
    return model


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v35-ckpt", type=Path, required=True)
    ap.add_argument("--waft-pred-dir", type=Path, required=True)
    ap.add_argument("--delta-pred-dir", type=Path, required=True)
    ap.add_argument("--da3-depth-root", type=Path, required=True)
    ap.add_argument("--scores-dir", type=Path, required=True, help="v39 metric_results")
    ap.add_argument(
        "--out-dir", type=Path, default=Path("doc/vmamba3_3dpointtrack/figs")
    )
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    v39 = _build_v39(args.v35_ckpt, dev)

    for sub in SUBSETS:
        clip_id = _best_clip(args.scores_dir, sub)
        path = next(
            p
            for p in _ev.list_clips(Path("~/data").expanduser(), [sub])
            if p.stem == clip_id
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
        )
        delta, _ = _ev._load_external(args.delta_pred_dir, sub, clip_id)
        hl = _win_center(delta, v39xyz, gt, vis)
        lims = _gt_lims(gt, vis)
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
