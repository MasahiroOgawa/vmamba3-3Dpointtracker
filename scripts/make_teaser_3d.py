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


def _pick_tracks(vis, k):
    return list(np.argsort(-vis.sum(axis=1))[:k])


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
        m = vis[n].astype(bool)
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
    lo, hi = P.min(0), P.max(0)
    span = np.maximum(hi - lo, 1e-3)
    lo, hi = lo - PAD * span, hi + PAD * span
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
    ap.add_argument("--replot", action="store_true", help="re-plot from cached inference only")
    args = ap.parse_args()

    if args.replot and CACHE.exists():
        d = np.load(CACHE)
        _plot(d["pred"], d["gt"], d["vis"], d["anchor"], args.out)
        return 0

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    v39 = _build_v39(args.v35_ckpt.expanduser(), dev)
    path = next(p for p in _ev.list_clips(Path("~/data").expanduser(), ["drivetrack"])
                if p.stem == TEASER_CLIP)
    clip = _ev.load_clip(path)
    gt = clip.tracks_XYZ.numpy().transpose(1, 0, 2)
    anchor = clip.queries_xyt[:, 2].numpy().astype(int)
    pred, vis = _ev._infer(
        method="v35", flow_model=None, model=v39, clip=clip, image_size=IMAGE_SIZE,
        fb_alpha=0.05, fb_beta=1.0, da3_depth_root=args.da3_depth_root.expanduser(),
        max_frames=0, device=dev, waft_pred_dir=args.waft_pred_dir.expanduser(),
    )
    np.savez(CACHE, pred=pred, gt=gt, vis=vis, anchor=anchor)
    print(f"[teaser] clip {TEASER_CLIP}  N={gt.shape[0]} F={gt.shape[1]}  cached -> {CACHE}")
    _plot(pred, gt, vis, anchor, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
