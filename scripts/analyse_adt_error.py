"""What is the DA3-g error on adt actually made of?

The whole gap to DELTA on DA3-g is adt: -0.0184 metric-AJ, against +0.0087 on pstudio and -0.0009 on
drivetrack. Every attempt so far has attacked a per-frame SCALE error, which is what dominates
drivetrack; on adt an oracle scale gains least (+0.10 against drivetrack's +0.34) and a fitted linear
scale actively hurts (-0.044). So adt's error is something else, and nothing has asked what.

Decomposes the 3-D error on the WAFT track into components a refiner could plausibly address, per
subset so adt can be read against the two that work:

  scale      what a perfect per-frame global scale would remove
  bias       what a perfect per-clip constant offset would remove (a subset of scale)
  per-point  what remains after the best per-frame scale: genuine per-point depth error
  2-D        how much of the error is the TRACK being in the wrong place, not the depth
  visibility how often the predicted visibility disagrees with ground truth

The last two matter because metric-AJ is ~0.63x APD3D: a third of it is visibility agreement, which
no depth correction can touch.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
_cwd = os.getcwd()
from eval_waft import MINIVAL_FILES, TAPVID3D_ROOT  # noqa: E402
os.chdir(_cwd)

THRESHOLDS = (0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28, 2.56)


def frac(d, m):
    return float(np.mean([np.mean(d[m] < t) for t in THRESHOLDS])) if m.any() else np.nan


def main() -> int:
    cache = Path.home() / "data/tapvid3d_baseline_preds/waft_minival_is896_s-1_i4"
    out = {}
    for subset in ("drivetrack", "pstudio", "adt"):
        rows = []
        for name in MINIVAL_FILES[subset]:
            gtf, pf = TAPVID3D_ROOT / subset / name, cache / subset / name
            if not (gtf.exists() and pf.exists()):
                continue
            g = dict(np.load(gtf, allow_pickle=True))
            gt, vis = g["tracks_XYZ"].astype(np.float32), g["visibility"].astype(bool)
            z = np.load(pf)
            pr, pvis = z["tracks_XYZ"].astype(np.float32), z["visibility"]
            if pr.shape != gt.shape:
                continue
            m = vis & np.isfinite(pr).all(-1) & np.isfinite(gt).all(-1)
            if m.sum() < 50:
                continue
            d0 = np.linalg.norm(pr - gt, axis=-1)

            with np.errstate(divide="ignore", invalid="ignore"):
                r = np.log(np.clip(gt[..., 2], 1e-6, None)) - np.log(np.clip(pr[..., 2], 1e-6, None))
            ds = np.array([np.median(r[t][m[t]]) if m[t].any() else 0.0 for t in range(r.shape[0])])
            d_scale = np.linalg.norm(pr * np.exp(ds)[:, None, None] - gt, axis=-1)
            d_bias = np.linalg.norm(pr * np.exp(np.median(ds)) - gt, axis=-1)

            # how much of the residual is the ray being wrong rather than the depth: replace the
            # predicted direction with the ground-truth one, keeping the (scale-corrected) depth
            zc = (pr * np.exp(ds)[:, None, None])[..., 2]
            gdir = gt / np.clip(np.linalg.norm(gt, axis=-1, keepdims=True), 1e-6, None)
            gz = np.clip(gt[..., 2] / np.clip(np.linalg.norm(gt, axis=-1), 1e-6, None), 1e-6, None)
            d_ray = np.linalg.norm(gdir * (zc / gz)[..., None] - gt, axis=-1)

            visagree = float(np.mean((pvis > 0.5) == vis)) if pvis.shape == vis.shape else np.nan
            rows.append((frac(d0, m), frac(d_bias, m), frac(d_scale, m), frac(d_ray, m), visagree))
        if rows:
            a = np.array(rows)
            out[subset] = a.mean(0)

    print(f"  {'subset':11s} {'as-is':>7s} {'+bias':>7s} {'+scale':>7s} {'+GT ray':>8s} "
          f"{'vis agree':>10s}")
    for s, v in out.items():
        print(f"  {s:11s} {v[0]:7.4f} {v[1]:7.4f} {v[2]:7.4f} {v[3]:8.4f} {v[4]:10.4f}")
    print()
    print("  columns are APD3D after applying an ORACLE fix, so each is a ceiling:")
    print("    +bias    one constant scale per clip")
    print("    +scale   one scale per frame (includes bias)")
    print("    +GT ray  scale fixed AND the 2-D track direction made perfect")
    print("  vis agree is the fraction of (frame, point) where predicted visibility matches GT.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
