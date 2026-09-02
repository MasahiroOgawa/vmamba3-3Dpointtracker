"""Is SEA-RAFT's 2-D track noisier than WAFT's on the training clips?

The DA3-g arms trained on SEA-RAFT tracks beat the ones trained on WAFT tracks, even though both are
evaluated on WAFT. Two hypotheses were eliminated first: the evaluation inputs are equivalent in
quality (compare_waft_cache_quality.py, -0.0014), and the de-flicker stage's weights are larger, not
smaller, under WAFT training.

What remains is train-on-hard, test-on-easy: if SEA-RAFT's track is noisier, a refiner trained on it
sees a harder problem and is handed an easier one at evaluation. That predicts SEA-RAFT's 2-D error
exceeds WAFT's, measured the same way for both, and it is the premise this checks. If SEA-RAFT is
NOT noisier, the hypothesis is dead and the cause is elsewhere.
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
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
_cwd = os.getcwd()
# SEA-RAFT first: importing eval_waft puts the WAFT checkout on sys.path, and both repos ship a
# top-level `utils` package, so afterwards `utils.utils` resolves to WAFT's and SEA-RAFT's
# InputPadder cannot be found.
from searaft_flow.flow_tracker import track_clip  # noqa: E402
from searaft_flow.model import FlowModel  # noqa: E402
import eval_waft as ew  # noqa: E402
os.chdir(_cwd)


def gt_uv(data):
    xyz = data["tracks_XYZ"].astype(np.float32)
    fx, fy, cx, cy = data["fx_fy_cx_cy"].astype(np.float32).tolist()
    z = np.clip(xyz[..., 2], 1e-6, None)
    return np.stack([fx * xyz[..., 0] / z + cx, fy * xyz[..., 1] / z + cy], axis=-1)


@torch.no_grad()
def run(flow, data, image_size, device):
    imgs = ew.decode_images(data["images_jpeg_bytes"])
    F_, _, H, W = imgs.shape
    images = torch.from_numpy(imgs)
    if (H, W) != (image_size, image_size):
        images = F.interpolate(images, size=(image_size, image_size), mode="bilinear",
                               align_corners=False)
    images_255 = (images * 255.0).to(device)
    sx, sy = image_size / float(W), image_size / float(H)
    q = data["queries_xyt"].astype(np.float32)
    queries = torch.from_numpy(np.stack([q[:, 0] * sx, q[:, 1] * sy], -1)).float()
    anchor = torch.from_numpy(q[:, 2]).long().clamp(0, F_ - 1)
    uv, _ = track_clip(flow, images_255, queries, anchor, image_size, 0.05, 1.0,
                       bidirectional=False)
    return uv.cpu().numpy() / np.array([sx, sy], dtype=np.float32)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--per-subset", type=int, default=4)
    ap.add_argument("--image-size", type=int, default=896)
    args = ap.parse_args()
    device = torch.device("cuda")

    # FlowModel imports SEA-RAFT's `raft` lazily inside __init__, and by then eval_waft has put the
    # WAFT checkout on sys.path. Both repos ship a top-level `utils`, so SEA-RAFT's InputPadder
    # resolves to WAFT's module and the import fails. Build it with the WAFT paths withheld and the
    # cached `utils` purged, then restore.
    waft_paths = [q for q in sys.path if "WAFT" in q]
    saved = list(sys.path)
    for q in waft_paths:
        sys.path.remove(q)
    for mod in [m for m in list(sys.modules) if m == "utils" or m.startswith("utils.")]:
        del sys.modules[mod]
    try:
        searaft = FlowModel(device, iters=4, scale=None)   # spring-M default scale -1
    finally:
        sys.path[:] = saved
    for mod in [m for m in list(sys.modules) if m == "utils" or m.startswith("utils.")]:
        del sys.modules[mod]
    os.chdir(ew.WAFT_ROOT)
    try:
        waft = ew.build_flow(ew.WAFT_ROOT / "config" / "a1" / "tar-c-t.json",
                             ew.WAFT_ROOT / "ckpts" / "waft_a1_recommended.pth",
                             device, scale=-1, iters=4)
    finally:
        os.chdir(_cwd)

    print(f"  {'subset':11s} {'clip':30s} {'SEA-RAFT':>9s} {'WAFT':>8s} {'ratio':>7s}")
    agg = {"s": [], "w": []}
    for subset in ("drivetrack", "pstudio", "adt"):
        for name in ew.MINIVAL_FILES[subset][: args.per_subset]:
            f = ew.TAPVID3D_ROOT / subset / name
            if not f.exists():
                continue
            data = dict(np.load(f, allow_pickle=True))
            g, vis = gt_uv(data), data["visibility"].astype(bool)
            errs = {}
            for tag, model in (("s", searaft), ("w", waft)):
                pred = run(model, data, args.image_size, device)
                d = np.linalg.norm(pred - g, axis=-1)
                m = vis & np.isfinite(d)
                errs[tag] = float(np.median(d[m])) if m.any() else float("nan")
                agg[tag].append(errs[tag])
            print(f"  {subset:11s} {name[:30]:30s} {errs['s']:9.2f} {errs['w']:8.2f} "
                  f"{errs['s']/max(errs['w'],1e-9):7.2f}")
    s, w = np.median(agg["s"]), np.median(agg["w"])
    print(f"\n  MEDIAN over clips: SEA-RAFT {s:.2f} px   WAFT {w:.2f} px   ratio {s/w:.2f}x")
    print(f"  clips where SEA-RAFT is worse: {sum(1 for a,b in zip(agg['s'],agg['w']) if a>b)}/{len(agg['s'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
