"""Fit the linear log-depth-quantile -> per-frame-scale map on TRAINING clips only.

The result initialises Mamba3DepthScaleRefiner's linear bypass, so the module starts at a solution
worth +0.0734 of metric-AJ instead of at zero. Fitted on full_eval clips with minival removed: the
same fit on minival would leak the evaluation set into the initialisation.

Writes the coefficients to a .npz the training script loads.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
_cwd = os.getcwd()
from eval_waft import (  # noqa: E402
    DA3_ROOT,
    MINIVAL_FILES,
    TAPVID3D_ROOT,
    get_full_eval_files,
    load_da3_depth,
)
os.chdir(_cwd)

QS = (5, 15, 25, 35, 45, 55, 65, 75, 85, 95)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--per-subset", type=int, default=250,
                    help="clips per subset; the fit has 11 parameters so this is ample")
    ap.add_argument("--out", type=Path, default=Path("result/scale_linear_init.npz"))
    args = ap.parse_args()

    waft = Path.home() / "data/tapvid3d_baseline_preds/waft_full_eval"
    minival = {n for s in MINIVAL_FILES for n in MINIVAL_FILES[s]}
    X, y = [], []
    for subset in ("drivetrack", "pstudio", "adt"):
        used = 0
        for name in get_full_eval_files(subset):
            if used >= args.per_subset:
                break
            if name in minival:
                continue
            gtf, pf = TAPVID3D_ROOT / subset / name, waft / subset / name
            if not (gtf.exists() and pf.exists() and (DA3_ROOT / subset / name).exists()):
                continue
            g = dict(np.load(gtf, allow_pickle=True))
            gt, vis = g["tracks_XYZ"].astype(np.float32), g["visibility"].astype(bool)
            pr = np.load(pf)["tracks_XYZ"].astype(np.float32)
            if pr.shape != gt.shape:
                continue
            try:
                depth = load_da3_depth(subset, name, gt.shape[0])
            except Exception:
                continue
            with np.errstate(divide="ignore", invalid="ignore"):
                r = np.log(np.clip(gt[..., 2], 1e-6, None)) - np.log(np.clip(pr[..., 2], 1e-6, None))
            m = vis & np.isfinite(r)
            for t in range(gt.shape[0]):
                if m[t].sum() < 8:
                    continue
                X.append(np.percentile(np.log(np.clip(depth[t], 1e-6, None)), QS))
                y.append(float(np.median(r[t][m[t]])))
            used += 1
        print(f"  {subset:11s} {used} clips used", flush=True)

    X, y = np.array(X, dtype=np.float64), np.array(y, dtype=np.float64)
    A = np.concatenate([X, np.ones((len(X), 1))], 1)
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    r2 = 1.0 - float(np.var(y - A @ beta) / np.var(y))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, weight=beta[:-1], bias=beta[-1], quantiles=np.array(QS), r2=r2, n=len(y))
    print(f"\n  fitted on {len(y)} frames from TRAINING clips only (minival excluded)")
    print(f"  in-sample R^2 = {r2:.4f}   |weight| max = {np.abs(beta[:-1]).max():.4f}   "
          f"bias = {beta[-1]:+.4f}")
    print(f"  -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
