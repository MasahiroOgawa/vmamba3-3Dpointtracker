"""Is the per-frame scale error predictable from the depth map at all?

Mamba3DepthScaleRefiner reads the depth map and must emit ds*_f = median_n[log z_gt - log z_raw].
That only works if ds* is a function of the map. This checks the premise directly, and should have
been run before the module was built: it regresses ds* on log-depth quantiles of the full frame --
the summary any encoder over that map can extract -- and reports how much variance is explained,
per frame and per clip.

An R^2 near zero means the target is not recoverable from the input, L_dsr is asking the network to
predict noise, and no amount of training or architecture fixes that.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
_cwd = os.getcwd()
from eval_waft import MINIVAL_FILES, TAPVID3D_ROOT, load_da3_depth  # noqa: E402
os.chdir(_cwd)

QS = (5, 15, 25, 35, 45, 55, 65, 75, 85, 95)


def r2(X, y):
    """R^2 of a least-squares fit of y on X with an intercept."""
    A = np.concatenate([X, np.ones((len(X), 1))], 1)
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ beta
    return 1.0 - float(np.var(resid) / np.var(y)) if np.var(y) > 0 else float("nan")


def main() -> int:
    feats, targets, clip_feat, clip_tgt = [], [], [], []
    cache = Path.home() / "data/tapvid3d_baseline_preds/waft_minival_is896_s-1_i4"
    for subset in ("drivetrack", "pstudio", "adt"):
        for name in MINIVAL_FILES[subset]:
            gtf, pf = TAPVID3D_ROOT / subset / name, cache / subset / name
            if not (gtf.exists() and pf.exists()):
                continue
            g = dict(np.load(gtf, allow_pickle=True))
            gt, vis = g["tracks_XYZ"].astype(np.float32), g["visibility"].astype(bool)
            pr = np.load(pf)["tracks_XYZ"].astype(np.float32)
            if pr.shape != gt.shape:
                continue
            F_ = gt.shape[0]
            try:
                depth = load_da3_depth(subset, name, F_)
            except Exception:
                continue
            with np.errstate(divide="ignore", invalid="ignore"):
                r = np.log(np.clip(gt[..., 2], 1e-6, None)) - np.log(np.clip(pr[..., 2], 1e-6, None))
            m = vis & np.isfinite(r)
            per_clip = []
            for t in range(F_):
                if m[t].sum() < 8:
                    continue
                q = np.percentile(np.log(np.clip(depth[t], 1e-6, None)), QS)
                feats.append(q)
                targets.append(float(np.median(r[t][m[t]])))
                per_clip.append((q, targets[-1]))
            if per_clip:
                clip_feat.append(np.mean([p[0] for p in per_clip], 0))
                clip_tgt.append(float(np.median([p[1] for p in per_clip])))

    X, y = np.array(feats), np.array(targets)
    Xc, yc = np.array(clip_feat), np.array(clip_tgt)
    print(f"  per-FRAME  n={len(y):6d}  R^2 of ds* on log-depth quantiles = {r2(X, y):.4f}")
    print(f"  per-CLIP   n={len(yc):6d}  R^2 of the constant offset       = {r2(Xc, yc):.4f}")
    print(f"\n  target spread: per-frame std {y.std():.4f}, per-clip std {yc.std():.4f}")
    print(f"  A single mean-log-depth feature alone: per-frame R^2 = "
          f"{r2(X.mean(1, keepdims=True), y):.4f}")
    print("\n  R^2 near 0 means the depth map does not determine ds*, so L_dsr asks the network")
    print("  to predict something its input cannot express.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
