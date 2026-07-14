"""Confirm DA3-Nested-Giant metric-depth quality across ALL subsets (drivetrack/adt/pstudio).

Measures median(DA3-nested depth / GT depth) at tracked pixels, stratified by depth, per subset.
Our current da3metric-large is ~0.55 on drivetrack far-field; check if nested is ~1.0 everywhere.

Run: uv run python scripts/exp_da3nested_depth.py --n-clips 3 --k-frames 8
"""
from __future__ import annotations

import argparse
import glob
import io
import os

import numpy as np
from PIL import Image

try:
    import scripts.exp_worldframe_triangulation as X
except ModuleNotFoundError:
    import exp_worldframe_triangulation as X

ROOT = os.path.expanduser("~/data/tapvid3d")
SUBSETS = ["drivetrack", "adt", "pstudio"]


def decode(arr, idx):
    return [np.array(Image.open(io.BytesIO(bytes(arr[i]))).convert("RGB")) for i in idx]


def run(args):
    import sys
    import types
    import torch
    sys.modules.setdefault("moviepy.editor", types.ModuleType("moviepy.editor"))
    sys.path.insert(0, "third_party/depth-anything-3/src")
    from depth_anything_3.api import DepthAnything3

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"loading {args.model} ...", flush=True)
    model = DepthAnything3.from_pretrained(f"depth-anything/{args.model}").to(dev)
    model.device = dev
    model.eval()

    for sub in SUBSETS:
        clips = sorted(glob.glob(os.path.join(ROOT, sub, "*.npz")))[: args.n_clips]
        ratios, near, far = [], [], []
        for path in clips:
            d = np.load(path, allow_pickle=True)
            if "images_jpeg_bytes" not in d.files:
                continue
            XYZ = d["tracks_XYZ"].astype(float)
            vis = d["visibility"].astype(bool)
            K = X.K_from(d["fx_fy_cx_cy"].astype(float))
            uv = X.project(XYZ, K)
            F = XYZ.shape[0]
            idx = np.unique(np.linspace(0, F - 1, args.k_frames).astype(int))
            rgbs = decode(d["images_jpeg_bytes"], idx)
            h, w = rgbs[0].shape[:2]
            with torch.inference_mode():
                try:
                    pred = model.inference(rgbs, process_res=args.res, export_format="mini_npz",
                                           use_ray_pose=False)
                except TypeError:
                    pred = model.inference(rgbs, process_res=args.res, export_format="mini_npz")
            depth = np.asarray(pred.depth, float)
            del pred
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            for a, fa in enumerate(idx):
                m = vis[fa]
                dd = X.sample_depth(depth[a], uv[fa, m], h, w)
                zz = XYZ[fa, m, 2]
                ok = (dd > 0) & (zz > 0) & np.isfinite(dd)
                r = dd[ok] / zz[ok]
                ratios.extend(r.tolist())
                near.extend((dd[ok & (zz < 10)] / zz[ok & (zz < 10)]).tolist())
                far.extend((dd[ok & (zz > 30)] / zz[ok & (zz > 30)]).tolist())
        def med(x):
            return f"{np.median(x):.3f}" if x else "n/a"
        if ratios:
            print(f"{sub:<11} median(pred/GT): all {med(ratios)}  near<10 {med(near)}  "
                  f"far>30 {med(far)}  (n={len(ratios)})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="DA3NESTED-GIANT-LARGE")
    ap.add_argument("--n-clips", type=int, default=3)
    ap.add_argument("--k-frames", type=int, default=8)
    ap.add_argument("--res", type=int, default=504)
    run(ap.parse_args())
