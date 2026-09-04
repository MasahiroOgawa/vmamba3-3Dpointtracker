"""How much could a PERFECT de-flicker recover? The ceiling on the whole stage.

The de-flicker emits one log-scale per frame and applies it to every point, so the most any version
of it can do is remove the component of the error that a single per-frame scalar accounts for. This
computes that oracle directly: take the per-frame scale straight from ground truth, apply it, and
score before and after.

The number this produces bounds every de-flicker design -- better input, better loss, more
parameters. If the ceiling is small, the stage is not where DA3-g is won, and no amount of work on
it will change that.

Scoring uses the same fixed-metre thresholds metric-AJ is built from, on the cached WAFT tracks the
rebuilt arms were evaluated against, so it is comparable to those numbers.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
_cwd = os.getcwd()
from eval_waft import MINIVAL_FILES, TAPVID3D_ROOT  # noqa: E402
os.chdir(_cwd)

THRESHOLDS = (0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28, 2.56)


def frac_within(pred, gt, vis):
    d = np.linalg.norm(pred - gt, axis=-1)
    m = vis & np.isfinite(d)
    if not m.any():
        return None
    return float(np.mean([np.mean(d[m] < t) for t in THRESHOLDS]))


def oracle_scale(pred, gt, vis):
    """Per-frame log-scale taken from ground truth: the best a single scalar can do."""
    zp, zg = pred[..., 2], gt[..., 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.log(np.clip(zg, 1e-6, None)) - np.log(np.clip(zp, 1e-6, None))
    m = vis & np.isfinite(r)
    return np.array([np.median(r[t][m[t]]) if m[t].any() else 0.0 for t in range(r.shape[0])])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", type=Path,
                    default=Path.home() / "data/tapvid3d_baseline_preds/waft_minival_is896_s-1_i4")
    args = ap.parse_args()

    rows = {}
    for subset in ("drivetrack", "pstudio", "adt"):
        before, after, constonly, mags = [], [], [], []
        for name in MINIVAL_FILES[subset]:
            gtf = TAPVID3D_ROOT / subset / name
            pf = args.cache / subset / name
            if not (gtf.exists() and pf.exists()):
                continue
            g = dict(np.load(gtf, allow_pickle=True))
            gt, vis = g["tracks_XYZ"].astype(np.float32), g["visibility"].astype(bool)
            pred = np.load(pf)["tracks_XYZ"].astype(np.float32)
            if pred.shape != gt.shape:
                continue
            b = frac_within(pred, gt, vis)
            ds = oracle_scale(pred, gt, vis)
            # scaling depth by s scales the whole ray-parameterised point by s
            a = frac_within(pred * np.exp(ds)[:, None, None], gt, vis)
            # Split the oracle: a de-flicker removes frame-to-frame WOBBLE, not a constant offset.
            # A per-clip constant is a calibration problem and a different fix entirely.
            const = float(np.median(ds))
            c = frac_within(pred * np.exp(const), gt, vis)
            if b is None or a is None or c is None:
                continue
            before.append(b)
            after.append(a)
            constonly.append(c)
            mags.append(float(np.std(ds - const)))
        if before:
            rows[subset] = (np.mean(before), np.mean(constonly), np.mean(after),
                            np.median(mags), len(before))

    print(f"  {'subset':11s} {'n':>4s} {'as-is':>8s} {'per-clip':>9s} {'per-frame':>10s} "
          f"{'flicker only':>13s} {'wobble std':>11s}")
    tb, tc, ta, n = [], [], [], 0
    for s, (b, c, a, m, k) in rows.items():
        print(f"  {s:11s} {k:4d} {b:8.4f} {c:9.4f} {a:10.4f} {a-c:+13.4f} {m:11.4f}")
        tb.append(b * k)
        tc.append(c * k)
        ta.append(a * k)
        n += k
    if n:
        B, C, A = sum(tb) / n, sum(tc) / n, sum(ta) / n
        print(f"  {'ALL':11s} {n:4d} {B:8.4f} {C:9.4f} {A:10.4f} {A-C:+13.4f}")
        print("\n  per-clip  = one constant scale for the whole clip (a CALIBRATION fix)")
        print("  per-frame = the full oracle, constant plus frame-to-frame wobble")
        print("  flicker only = per-frame minus per-clip: what a DE-FLICKER can add on top")
        print("\n  'oracle' applies the ground-truth per-frame scale -- a de-flicker that is exactly")
        print("  right on every frame. The gain is therefore the CEILING for the stage.")
        print("  For scale: de-flicker as trained contributes +0.0020 (v68 0.2218 -> v63 0.2238).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
