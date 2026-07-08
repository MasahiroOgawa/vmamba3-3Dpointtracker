"""Validate our metric against the official tapnet evaluate_model.py invocation.

Reproduces evaluate_model.py EXACTLY (raw GT npz, query_points passed,
order='t n') using our vendored compute_tapvid3d_metrics, on released
SpatialTracker drivetrack predictions. If median 3D-AJ here matches the paper's
drivetrack number (~0.058), our eval_metric3d wrapper invocation (query_points
omitted, order='n t') is what differs and must be fixed; if it stays ~0.008,
the released predictions are simply a weak/example run.

Also reports the official-invocation ABSOLUTE metric (scaling='none',
use_fixed_metric_threshold=True) so the SpatialTracker metric-AJ is computed the
same way as the headline comparison.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from mamba3_tracker.eval import tapvid3d_official_metrics as M


def _jpeg_hw(b: bytes) -> tuple[int, int]:
    import cv2
    arr = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)
    return arr.shape[0], arr.shape[1]


def _avg(dicts, key):
    vals = [float(d[key]) for d in dicts if np.isfinite(d[key])]
    return sum(vals) / max(1, len(vals))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", type=Path,
                    default=Path("~/data/tapvid3d_baseline_preds/spatracker/drivetrack"))
    ap.add_argument("--gt-dir", type=Path, default=Path("~/data/tapvid3d/drivetrack"))
    args = ap.parse_args()
    pred_dir = args.pred_dir.expanduser()
    gt_dir = args.gt_dir.expanduser()

    preds = sorted(pred_dir.glob("*.npz"))
    print(f"[validate] {len(preds)} SpatialTracker drivetrack predictions")

    med, ab = [], []
    n_fail = 0
    for pp in preds:
        gtp = gt_dir / pp.name
        if not gtp.exists():
            n_fail += 1
            continue
        g = np.load(gtp)
        p = np.load(pp)
        visibles = g["visibility"]                 # (F,N)
        tracks_xyz = g["tracks_XYZ"].astype(np.float32)   # (F,N,3)
        queries_xyt = g["queries_xyt"].astype(np.float32)  # (N,3) = x,y,t
        # Official: resize intrinsics so the SMALLEST image side = 256 px
        # (TAPVid-3D defines pixel thresholds relative to 256-px images).
        H, W = _jpeg_hw(g["images_jpeg_bytes"][0])
        scaling_factor = 256.0 / min(H, W)
        intr = g["fx_fy_cx_cy"].astype(np.float32) * scaling_factor   # (4,)
        ptr = p["tracks_XYZ"].astype(np.float32)           # (F,N,3)
        pvis = p["visibility"]                              # (F,N)

        F_ = min(tracks_xyz.shape[0], ptr.shape[0])
        N_ = min(tracks_xyz.shape[1], ptr.shape[1])
        common = dict(
            gt_occluded=np.logical_not(visibles[:F_, :N_]),
            gt_tracks=tracks_xyz[:F_, :N_],
            pred_occluded=np.logical_not(pvis[:F_, :N_]),
            pred_tracks=ptr[:F_, :N_],
            intrinsics_params=intr,
            query_points=queries_xyt[:N_, ::-1],   # x,y,t -> t,y,x (official)
            order="t n",
        )
        m1 = M.compute_tapvid3d_metrics(scaling="median", **common)
        m2 = M.compute_tapvid3d_metrics(scaling="none", use_fixed_metric_threshold=True, **common)
        med.append({k: float(m1[k]) for k in
                    ("average_jaccard", "average_pts_within_thresh", "occlusion_accuracy")})
        ab.append({k: float(m2[k]) for k in
                   ("average_jaccard", "average_pts_within_thresh")})

    print(f"[validate] scored {len(med)} clips ({n_fail} missing GT)")
    print("\n=== OFFICIAL invocation (query_points passed, order='t n') ===")
    print(f"median-scaled 3D-AJ      = {_avg(med, 'average_jaccard'):.4f}   (paper drivetrack SpatialTracker = 0.058)")
    print(f"median-scaled APD3D      = {_avg(med, 'average_pts_within_thresh'):.4f}")
    print(f"occlusion accuracy       = {_avg(med, 'occlusion_accuracy'):.4f}")
    print(f"absolute metric-AJ       = {_avg(ab, 'average_jaccard'):.4f}")
    print(f"absolute metric-APD3D    = {_avg(ab, 'average_pts_within_thresh'):.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
