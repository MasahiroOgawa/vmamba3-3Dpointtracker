"""Check whether DA3's camera-pose output is (a) accurate and (b) METRIC, on drivetrack.

DA3-LARGE (any-view) estimates per-frame extrinsics. We run it on a few sampled frames per clip,
then compare its RELATIVE pose between frame pairs to the GT oracle relative pose (fit from GT
tracks_XYZ via RANSAC-rigid, the reliable reference from the triangulation experiment):
  * rotation error (deg)
  * translation-direction error (deg)
  * translation-SCALE ratio  ||t_da3|| / ||t_gt||   -> is it metric (ratio ~1 and consistent)?

Run: uv run python scripts/exp_da3_pose.py --n-clips 3 --k-frames 10
"""

from __future__ import annotations

import argparse
import glob
import io
import os

import numpy as np
from PIL import Image

try:
    import scripts.exp_worldframe_triangulation as X  # reuse kabsch/ransac_rigid/rot_angle/K_from
except ModuleNotFoundError:  # when run as a file, scripts/ is on sys.path[0]
    import exp_worldframe_triangulation as X


def decode(jpeg_bytes_arr, idx):
    out = []
    for i in idx:
        b = jpeg_bytes_arr[i]
        out.append(np.array(Image.open(io.BytesIO(bytes(b))).convert("RGB")))
    return out


def rel(Ea, Eb):  # 3x4 or 4x4 w2c -> 4x4 relative cam_a->cam_b
    def h(E):
        M = np.eye(4)
        M[:3, :4] = E[:3, :4]
        return M
    return h(Eb) @ np.linalg.inv(h(Ea))


def ang(u, v):
    u = u / (np.linalg.norm(u) + 1e-9)
    v = v / (np.linalg.norm(v) + 1e-9)
    return np.degrees(np.arccos(np.clip(u @ v, -1, 1)))


def run(args):
    import sys
    import types
    import torch
    sys.modules.setdefault("moviepy.editor", types.ModuleType("moviepy.editor"))
    sys.path.insert(0, "third_party/depth-anything-3/src")
    from depth_anything_3.api import DepthAnything3

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"loading {args.model} on {dev} ...", flush=True)
    model = DepthAnything3.from_pretrained(f"depth-anything/{args.model}").to(dev)
    model.device = dev
    model.eval()

    clips = sorted(glob.glob(os.path.join(X.DATA, "*.npz")))[: args.n_clips]
    rot, dird, sc, dratio, dratio_far = [], [], [], [], []
    for path in clips:
        d = np.load(path, allow_pickle=True)
        XYZ = d["tracks_XYZ"].astype(float)
        vis = d["visibility"].astype(bool)
        K = X.K_from(d["fx_fy_cx_cy"].astype(float))
        uv_all = X.project(XYZ, K)
        F = XYZ.shape[0]
        idx = np.unique(np.linspace(0, F - 1, args.k_frames).astype(int))
        rgbs = decode(d["images_jpeg_bytes"], idx)
        h_img, w_img = rgbs[0].shape[:2]
        try:
            with torch.inference_mode():
                try:
                    pred = model.inference(rgbs, process_res=args.res, export_format="mini_npz",
                                           use_ray_pose=True)
                except TypeError:
                    pred = model.inference(rgbs, process_res=args.res, export_format="mini_npz")
            E = np.asarray(pred.extrinsics, dtype=float)  # (K,3,4) or (K,4,4) w2c
            depth = np.asarray(pred.depth, dtype=float)  # (K,Hd,Wd) metres (nested) or model units
        except Exception as e:
            print(f"  {os.path.basename(path)[:24]}: DA3 failed: {e}")
            continue
        # depth metric bias: model depth / GT depth at tracked pixels
        for a in range(len(idx)):
            fa = idx[a]
            m = vis[fa]
            dd = X.sample_depth(depth[a], uv_all[fa, m], h_img, w_img)
            zz = XYZ[fa, m, 2]
            ok = (dd > 0) & (zz > 0) & np.isfinite(dd)
            dratio.extend((dd[ok] / zz[ok]).tolist())
            far = ok & (zz > 30)
            dratio_far.extend((dd[far] / zz[far]).tolist())
        if E is None or len(E) != len(idx):
            print(f"  {os.path.basename(path)[:24]}: no/mismatched extrinsics (shape {getattr(E,'shape',None)})")
            continue
        # pairwise comparison over sampled frames
        for a in range(len(idx)):
            for b in range(a + 1, len(idx)):
                fa, fb = idx[a], idx[b]
                both = np.where(vis[fa] & vis[fb])[0]
                if len(both) < 20:
                    continue
                (Rg, tg), inl = X.ransac_rigid(XYZ[fa, both], XYZ[fb, both], iters=200, thresh=0.3)
                if inl.sum() < 20 or np.linalg.norm(tg) < 2.0:  # need real baseline
                    continue
                T = rel(E[a], E[b])
                Rd, td = T[:3, :3], T[:3, 3]
                rot.append(X.rot_angle(Rd, Rg))
                dird.append(ang(td, tg))
                sc.append(np.linalg.norm(td) / np.linalg.norm(tg))
        print(f"  {os.path.basename(path)[:30]}: {len(rot)} pairs so far", flush=True)

    if not rot:
        print("NO comparable pairs.")
        return
    rot, dird, sc = np.array(rot), np.array(dird), np.array(sc)
    if dratio:
        dr = np.array(dratio)
        print(f"\nDA3 ({args.model}) DEPTH bias  median(pred/GT): all {np.median(dr):.3f}"
              + (f", far>30m {np.median(dratio_far):.3f}" if dratio_far else "")
              + "   (1.0 = perfect metric; our da3metric-large was ~0.55)")
    print(f"\n=== DA3 ({args.model}) pose vs GT oracle, {len(rot)} frame-pairs over {len(clips)} clips ===")
    print(f"rotation error:            median {np.median(rot):6.2f} deg   (p90 {np.percentile(rot,90):.2f})")
    print(f"translation-direction err: median {np.median(dird):6.2f} deg   (p90 {np.percentile(dird,90):.2f})")
    print(f"translation-SCALE ratio |t_da3|/|t_gt|: median {np.median(sc):.3f}, "
          f"IQR [{np.percentile(sc,25):.3f}, {np.percentile(sc,75):.3f}]")
    consistent = (np.percentile(sc, 75) - np.percentile(sc, 25)) / max(np.median(sc), 1e-6)
    print(f"  scale consistency (IQR/median): {consistent:.2f}  "
          f"(low=one fixed gauge scale; ~1.0 metric only if median~1)")
    print("\nInterpretation: metric pose needs scale-ratio median ~1.0 AND consistent. "
          "An any-view model gives good rotation/direction but an arbitrary (non-metric) scale.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="DA3-LARGE")
    ap.add_argument("--n-clips", type=int, default=3)
    ap.add_argument("--k-frames", type=int, default=10)
    ap.add_argument("--res", type=int, default=504)
    run(ap.parse_args())
