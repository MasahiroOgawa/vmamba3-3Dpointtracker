#!/usr/bin/env python3
"""Regenerate the qualitative 3D-trajectory figures (baseline vs v33) for the memo.

For each subset we pick the BEST-scored v33 clip (highest per-clip metric-AJ from the
v33 eval JSONs), then render baseline (WAFT+DA3 unprojection) and v33 (depth-refiner)
3D tracks against GT, plus a space-time (X-t/Y-t/Z-t) view for drivetrack baseline.
Fonts are 2x the previous size and titles are short. Reuses eval_metric3d._infer so
baseline and v33 share the exact same cached WAFT front-end + DA3 depth.

  uv run python scripts/render_qual_3d.py \
    --v33-ckpt ~/proj/study/largescale3Dreconstruction_using_SSM/result/20260617-0001_v33/ckpt_20000.pt \
    --waft-pred-dir ~/data/tapvid3d_baseline_preds/waft \
    --da3-depth-root ~/data/tapvid3d_da3 \
    --scores-dir ~/proj/study/largescale3Dreconstruction_using_SSM/result/20260618-1718_metric3d_v33_fix/metric_results \
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

plt.rcParams.update({"font.size": 20})  # 2x the previous base font (default 10)

_HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "eval_metric3d", _HERE / "eval_metric3d.py"
)
_ev = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ev)  # provides _infer, load_clip, list_clips, MINIVAL_FILES

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


def _most_improved_center(base, v33, gt, vis):
    """Among the plotted tracks, find the one whose per-frame 3D error is reduced
    most by v33 vs the baseline, and return the (x,y,z) centroid of its GT path
    (the region to circle). Returns None if no track actually improves."""
    best_n, best_gain = None, 0.0
    for n in _pick_tracks(vis, MAX_TRACKS):
        m = vis[n].astype(bool)
        if m.sum() < 2:
            continue
        be = np.linalg.norm(base[n, m] - gt[n, m], axis=1)
        ve = np.linalg.norm(v33[n, m] - gt[n, m], axis=1)
        gain = float((be - ve).sum())
        if gain > best_gain:
            best_gain, best_n = gain, n
    if best_n is None:
        return None
    m = vis[best_n].astype(bool)
    return gt[best_n, m].mean(axis=0)


def _render_3d(pred, gt, vis, anchor, out_path: Path, title: str, highlight=None) -> None:
    """pred/gt: (N,F,3); vis: (N,F); anchor: (N,). Short 2x-font 3D plot.
    highlight: (3,) world point to ring in red (the most-improved region), or None."""
    N, Fn, _ = pred.shape
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
        ax.plot(
            pred[n, m, 0],
            pred[n, m, 1],
            pred[n, m, 2],
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
    ax.set_xlabel("X (m)", labelpad=12)
    ax.set_ylabel("Y (m)", labelpad=12)
    ax.set_zlabel("Z (m)", labelpad=12)
    ax.tick_params(labelsize=16)
    ax.set_title(title)  # short
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
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
    # Ring the largest baseline depth error on the Z-t panel: this is the along-ray
    # depth jitter our refiner is built to remove.
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
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[qual] wrote {out_path}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v33-ckpt", type=Path, required=True)
    ap.add_argument("--waft-pred-dir", type=Path, required=True)
    ap.add_argument("--da3-depth-root", type=Path, required=True)
    ap.add_argument("--scores-dir", type=Path, required=True)
    ap.add_argument(
        "--out-dir", type=Path, default=Path("doc/vmamba3_3dpointtrack/figs")
    )
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    from mamba3_tracker.model.depth_refined_tracker import Mamba3DepthRefiner

    st = torch.load(args.v33_ckpt, map_location="cpu", weights_only=False)
    mc = st.get("cfg", {}).get("model", {})
    v33 = Mamba3DepthRefiner(
        dim=int(mc.get("dim", 128)),
        state_dim=int(mc.get("state_dim", 64)),
        num_heads=int(mc.get("num_heads", 4)),
        num_layers=int(mc.get("num_layers", 2)),
        max_log_correction=float(mc.get("max_log_correction", 2.0)),
    ).to(dev)
    v33.load_state_dict(st["model"])
    v33.eval()

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

        common = dict(
            flow_model=None,
            clip=clip,
            image_size=IMAGE_SIZE,
            fb_alpha=0.05,
            fb_beta=1.0,
            da3_depth_root=args.da3_depth_root,
            max_frames=0,
            device=dev,
            waft_pred_dir=args.waft_pred_dir,
        )
        base, vis = _ev._infer(method="searaft", model=None, **common)
        v33xyz, _ = _ev._infer(method="v33", model=v33, **common)
        hl = _most_improved_center(base, v33xyz, gt, vis)
        print(
            f"[qual] {sub}: best clip {clip_id}  N={gt.shape[0]} F={gt.shape[1]}  "
            f"highlight={'none' if hl is None else hl.round(2)}"
        )

        _render_3d(
            base,
            gt,
            vis,
            anchor,
            args.out_dir / f"qual_{sub}_baseline_3d.png",
            f"{sub}: baseline (WAFT+DA3)",
            highlight=hl,
        )
        _render_3d(
            v33xyz,
            gt,
            vis,
            anchor,
            args.out_dir / f"qual_{sub}_v33_3d.png",
            f"{sub}: v33",
            highlight=hl,
        )
        if sub == "drivetrack":
            _render_st(
                gt,
                base,
                vis,
                args.out_dir / "qual_drivetrack_baseline_st.png",
                "drivetrack baseline: position vs time",
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
