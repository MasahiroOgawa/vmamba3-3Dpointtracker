"""What signal does the de-flicker stage actually train against, under each front-end?

De-flicker emits ONE per-frame log-scale correction, so it can only remove the component of the
depth error that is common to every point in a frame. Two quantities therefore matter, and previous
hypotheses conflated them:

  magnitude  -- the spread of that per-frame scale error over time. Injecting 2.72 px of positional
                noise raised the error and made results WORSE (0.2238 -> 0.2200), so magnitude alone
                is not what the stage needs.
  structure  -- the fraction of the depth error that a single per-frame scalar can explain. This is
                the part de-flicker is architecturally able to correct. If SEA-RAFT sampling yields
                a more per-frame-explainable error, the stage has more to learn from, which would
                explain its +0.0074 there against +0.0020 under WAFT.

Compares both front-ends' sampled depth against ground truth on the same clips and the same DA3-g
cache, so the depth model is held fixed and only where it is sampled varies.
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
from searaft_flow.flow_tracker import track_clip  # noqa: E402
from searaft_flow.model import FlowModel  # noqa: E402
import eval_waft as ew  # noqa: E402
os.chdir(_cwd)


def sample_depth(depth, uv, image_size):
    """Bilinear-sample a (F,H,W) depth map at uv (F,N,2) given in image_size pixels."""
    F_, H, W = depth.shape
    d = torch.from_numpy(depth).unsqueeze(1)                       # (F,1,H,W)
    g = torch.from_numpy(uv).unsqueeze(1)                          # (F,1,N,2)
    gx = (g[..., 0] / image_size) * 2 - 1
    gy = (g[..., 1] / image_size) * 2 - 1
    grid = torch.stack([gx, gy], dim=-1)
    out = F.grid_sample(d, grid, mode="bilinear", align_corners=False)
    return out[:, 0, 0, :].numpy()                                 # (F,N)


def analyse(z_pred, z_gt, vis):
    """Return (per-frame scale spread, fraction of log error a per-frame scalar explains)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.log(np.clip(z_pred, 1e-6, None)) - np.log(np.clip(z_gt, 1e-6, None))
    m = vis & np.isfinite(r)
    if m.sum() < 50:
        return None, None
    per_frame = np.array([np.median(r[t][m[t]]) if m[t].any() else np.nan
                          for t in range(r.shape[0])])
    ok = np.isfinite(per_frame)
    if ok.sum() < 4:
        return None, None
    spread = float(np.std(per_frame[ok]))
    total = float(np.var(r[m]))
    resid = float(np.var((r - per_frame[:, None])[m]))
    explained = (total - resid) / total if total > 0 else np.nan
    return spread, float(explained)


@torch.no_grad()
def uv_from(flow, data, image_size, device):
    imgs = ew.decode_images(data["images_jpeg_bytes"])
    F_, _, H, W = imgs.shape
    images = torch.from_numpy(imgs)
    if (H, W) != (image_size, image_size):
        images = F.interpolate(images, size=(image_size, image_size), mode="bilinear",
                               align_corners=False)
    sx, sy = image_size / float(W), image_size / float(H)
    q = data["queries_xyt"].astype(np.float32)
    queries = torch.from_numpy(np.stack([q[:, 0] * sx, q[:, 1] * sy], -1)).float()
    anchor = torch.from_numpy(q[:, 2]).long().clamp(0, F_ - 1)
    uv, _ = track_clip(flow, (images * 255.0).to(device), queries, anchor, image_size, 0.05, 1.0,
                       bidirectional=False)
    return uv.cpu().numpy(), (sx, sy)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--per-subset", type=int, default=8)
    ap.add_argument("--image-size", type=int, default=896)
    args = ap.parse_args()
    device = torch.device("cuda")

    waft_paths = [q for q in sys.path if "WAFT" in q]
    saved = list(sys.path)
    for q in waft_paths:
        sys.path.remove(q)
    for mod in [m for m in list(sys.modules) if m == "utils" or m.startswith("utils.")]:
        del sys.modules[mod]
    try:
        searaft = FlowModel(device, iters=4, scale=None)
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

    agg = {"s": {"sp": [], "ex": []}, "w": {"sp": [], "ex": []}}
    print(f"  {'subset':11s} {'clip':26s} {'SR spread':>10s} {'WF spread':>10s} "
          f"{'SR expl':>8s} {'WF expl':>8s}")
    for subset in ("drivetrack", "pstudio", "adt"):
        for name in ew.MINIVAL_FILES[subset][: args.per_subset]:
            f = ew.TAPVID3D_ROOT / subset / name
            if not f.exists():
                continue
            data = dict(np.load(f, allow_pickle=True))
            F_ = data["tracks_XYZ"].shape[0]
            try:
                depth = ew.load_da3_depth(subset, name, F_)
            except Exception:
                continue
            gt_z = data["tracks_XYZ"].astype(np.float32)[..., 2]
            vis = data["visibility"].astype(bool)
            row = {}
            for tag, model in (("s", searaft), ("w", waft)):
                uv, _ = uv_from(model, data, args.image_size, device)
                zp = sample_depth(depth, uv, args.image_size)
                sp, ex = analyse(zp, gt_z, vis)
                if sp is None:
                    row = {}
                    break
                row[tag] = (sp, ex)
                agg[tag]["sp"].append(sp)
                agg[tag]["ex"].append(ex)
            if len(row) == 2:
                print(f"  {subset:11s} {name[:26]:26s} {row['s'][0]:10.4f} {row['w'][0]:10.4f} "
                      f"{row['s'][1]:8.3f} {row['w'][1]:8.3f}")
    print()
    for k, lab in (("sp", "per-frame scale spread (std of log-scale over frames)"),
                   ("ex", "fraction of log-depth error a per-frame scalar explains")):
        s, w = np.median(agg["s"][k]), np.median(agg["w"][k])
        print(f"  {lab:56s} SEA-RAFT {s:.4f}  WAFT {w:.4f}  ratio {s/max(w,1e-9):.2f}x")
    print(f"\n  n = {len(agg['s']['sp'])} clips. 'expl' is the share of depth error de-flicker can")
    print("  architecturally remove; a higher value under SEA-RAFT would explain its stronger gain.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
