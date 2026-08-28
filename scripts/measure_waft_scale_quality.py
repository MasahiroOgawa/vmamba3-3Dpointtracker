"""Does running WAFT at half resolution degrade its 2D track, and by how much?

The matched-protocol arms (v63, v64) run WAFT at scale=-1 for both training and evaluation, while
every cached arm was scored against tracks generated at scale=0. If half-resolution WAFT produces a
worse 2D track, those arms are being scored from worse input, and their metric-AJ is not comparable
to the cached arms' regardless of how the refiner was trained.

This measures the front-end alone -- no refiner, no depth -- as pixel error against ground truth 2D,
obtained by projecting the ground-truth XYZ with the clip's own intrinsics. Only points the dataset
marks visible at that frame are scored, since an occluded point has no meaningful 2D position.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
_cwd = os.getcwd()
import eval_waft as ew  # noqa: E402  (chdirs to the WAFT checkout at import)
os.chdir(_cwd)

from searaft_flow.flow_tracker import track_clip  # noqa: E402


def gt_uv(data):
    """Ground-truth pixel tracks, projected from XYZ with the clip's intrinsics."""
    xyz = data["tracks_XYZ"].astype(np.float32)              # (F,N,3)
    fx, fy, cx, cy = data["fx_fy_cx_cy"].astype(np.float32).tolist()
    z = np.clip(xyz[..., 2], 1e-6, None)
    u = fx * xyz[..., 0] / z + cx
    v = fy * xyz[..., 1] / z + cy
    return np.stack([u, v], axis=-1)


def build(scale, device):
    """One model per scale, reused across clips: construction reloads the DepthAnything weights and
    dominates the run otherwise."""
    os.chdir(ew.WAFT_ROOT)   # DepthAnythingFeature loads its weights by a relative path
    try:
        return ew.build_flow(ew.WAFT_ROOT / "config" / "a1" / "tar-c-t.json",
                             ew.WAFT_ROOT / "ckpts" / "waft_a1_recommended.pth",
                             device, scale=scale, iters=4)
    finally:
        os.chdir(_cwd)


@torch.no_grad()
def track_at(flow, data, image_size, device):
    imgs = ew.decode_images(data["images_jpeg_bytes"])
    F_, _, H, W = imgs.shape
    images = torch.from_numpy(imgs)
    if (H, W) != (image_size, image_size):
        images = F.interpolate(images, size=(image_size, image_size),
                               mode="bilinear", align_corners=False)
    images_255 = (images * 255.0).to(device)
    sx, sy = image_size / float(W), image_size / float(H)
    q = data["queries_xyt"].astype(np.float32)
    queries = torch.from_numpy(np.stack([q[:, 0] * sx, q[:, 1] * sy], -1)).float()
    anchor = torch.from_numpy(q[:, 2]).long().clamp(0, F_ - 1)
    uv, _ = track_clip(flow, images_255, queries, anchor, image_size, 0.05, 1.0,
                       bidirectional=False)
    return uv.cpu().numpy(), (sx, sy)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--per-subset", type=int, default=4)
    ap.add_argument("--image-size", type=int, default=896)
    args = ap.parse_args()
    device = torch.device("cuda")
    models = {s: build(s, device) for s in (0, -1)}

    print(f"{'subset':11s} {'clip':34s} {'scale0 px':>10s} {'scale-1 px':>11s} {'delta':>8s}")
    print("-" * 78)
    agg = {0: [], -1: []}
    for subset in ("drivetrack", "pstudio", "adt"):
        names = ew.MINIVAL_FILES[subset][: args.per_subset]
        for name in names:
            f = ew.TAPVID3D_ROOT / subset / name
            if not f.exists():
                continue
            data = dict(np.load(f, allow_pickle=True))
            g = gt_uv(data)                      # (F,N,2) in ORIGINAL pixels
            vis = data["visibility"].astype(bool)                 # (F,N)
            errs = {}
            for s in (0, -1):
                uv, (sx, sy) = track_at(models[s], data, args.image_size, device)
                # bring prediction back to original pixel units before comparing
                pred = uv / np.array([sx, sy], dtype=np.float32)
                d = np.linalg.norm(pred - g, axis=-1)             # (F,N)
                m = vis & np.isfinite(d)
                errs[s] = float(np.median(d[m])) if m.any() else float("nan")
                agg[s].append(errs[s])
            print(f"{subset:11s} {name[:34]:34s} {errs[0]:10.2f} {errs[-1]:11.2f} "
                  f"{errs[-1]-errs[0]:+8.2f}")
    print("-" * 78)
    a0, a1 = np.array(agg[0]), np.array(agg[-1])
    print(f"{'MEDIAN over clips':46s} {np.median(a0):10.2f} {np.median(a1):11.2f} "
          f"{np.median(a1)-np.median(a0):+8.2f}")
    print(f"{'clips where half-res is WORSE':46s} {int((a1>a0).sum())}/{len(a0)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
