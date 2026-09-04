"""What does a LINEAR scale predictor on log-depth quantiles actually buy, held out?

ds* is 0.51-R^2 predictable from the frame's log-depth quantiles, so the learned module's weak
result is an optimisation problem rather than a missing signal. This asks the practical question:
fit the simplest possible predictor and score it.

Fitted and evaluated with a clip-disjoint split -- clips are split in half, the predictor is fitted
on one half and scored on the other, then the halves are swapped -- so no clip contributes to both
its own fit and its own score. Reported on the same fixed-metre thresholds as metric-AJ.
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
THRESHOLDS = (0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28, 2.56)


def frac_within(pred, gt, vis):
    d = np.linalg.norm(pred - gt, axis=-1)
    m = vis & np.isfinite(d)
    return float(np.mean([np.mean(d[m] < t) for t in THRESHOLDS])) if m.any() else None


def main() -> int:
    cache = Path.home() / "data/tapvid3d_baseline_preds/waft_minival_is896_s-1_i4"
    clips = []
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
            try:
                depth = load_da3_depth(subset, name, gt.shape[0])
            except Exception:
                continue
            with np.errstate(divide="ignore", invalid="ignore"):
                r = np.log(np.clip(gt[..., 2], 1e-6, None)) - np.log(np.clip(pr[..., 2], 1e-6, None))
            m = vis & np.isfinite(r)
            Q = np.stack([np.percentile(np.log(np.clip(depth[t], 1e-6, None)), QS)
                          for t in range(gt.shape[0])])
            tgt = np.array([np.median(r[t][m[t]]) if m[t].sum() >= 8 else np.nan
                            for t in range(gt.shape[0])])
            clips.append((subset, Q, tgt, pr, gt, vis))

    idx = np.arange(len(clips))
    rng = np.random.default_rng(0)
    rng.shuffle(idx)
    half = len(idx) // 2
    folds = [(idx[:half], idx[half:]), (idx[half:], idx[:half])]

    base, lin, per_sub = [], [], {}
    for fit_i, test_i in folds:
        X = np.concatenate([clips[i][1][np.isfinite(clips[i][2])] for i in fit_i])
        y = np.concatenate([clips[i][2][np.isfinite(clips[i][2])] for i in fit_i])
        A = np.concatenate([X, np.ones((len(X), 1))], 1)
        beta, *_ = np.linalg.lstsq(A, y, rcond=None)
        for i in test_i:
            sub, Q, _, pr, gt, vis = clips[i]
            ds = np.concatenate([Q, np.ones((len(Q), 1))], 1) @ beta
            b = frac_within(pr, gt, vis)
            a = frac_within(pr * np.exp(ds)[:, None, None], gt, vis)
            if b is None or a is None:
                continue
            base.append(b)
            lin.append(a)
            per_sub.setdefault(sub, []).append((b, a))

    print(f"  {'subset':11s} {'n':>4s} {'as-is':>8s} {'linear':>8s} {'gain':>8s}")
    for sub, v in per_sub.items():
        b = np.mean([x[0] for x in v]); a = np.mean([x[1] for x in v])
        print(f"  {sub:11s} {len(v):4d} {b:8.4f} {a:8.4f} {a-b:+8.4f}")
    print(f"  {'ALL':11s} {len(base):4d} {np.mean(base):8.4f} {np.mean(lin):8.4f} "
          f"{np.mean(lin)-np.mean(base):+8.4f}")
    print("\n  held out by clip; the oracle ceiling on the same data is 0.5033 (+0.2549)")
    print("  the trained module v72 reached 0.2099 (+0.0101 over the old stage's 0.1998)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
