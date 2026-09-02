"""Which WAFT track set is the better evaluation input: the old one or the rebuilt one?

v45 (the paper's DA3-g row) was scored against tracks generated with eval_waft's defaults --
image_size 512, the flow network's own scale 0. v63 was scored against image_size 896 with scale -1.
Those are different-quality inputs, so the 0.232-vs-0.2238 gap between them is not necessarily a
statement about how either model was trained.

Both sets are unprojected 3D tracks already on disk, so this compares them against ground truth
without touching the GPU. Reported as the same fixed-metre thresholds the metric-AJ uses, plus raw
3D error, on points the dataset marks visible.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import os
_cwd = os.getcwd()
from eval_waft import MINIVAL_FILES, TAPVID3D_ROOT  # noqa: E402
os.chdir(_cwd)

# TAPVid-3D's fixed-metre thresholds
THRESHOLDS = (0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28, 2.56)


def score(pred_xyz, gt_xyz, vis):
    d = np.linalg.norm(pred_xyz - gt_xyz, axis=-1)
    m = vis & np.isfinite(d)
    if not m.any():
        return None, None
    frac = float(np.mean([np.mean(d[m] < t) for t in THRESHOLDS]))
    return float(np.median(d[m])), frac


def main() -> int:
    old = Path.home() / "data/tapvid3d_baseline_preds/waft_minival_cudnn925"
    new = Path.home() / "data/tapvid3d_baseline_preds/waft_minival_is896_s-1_i4"
    agg = {"old": [], "new": []}
    per_subset = {}
    for subset in ("drivetrack", "pstudio", "adt"):
        rows = {"old": [], "new": []}
        for name in MINIVAL_FILES[subset]:
            gtf = TAPVID3D_ROOT / subset / name
            if not gtf.exists():
                continue
            g = dict(np.load(gtf, allow_pickle=True))
            gt, vis = g["tracks_XYZ"].astype(np.float32), g["visibility"].astype(bool)
            for tag, root in (("old", old), ("new", new)):
                f = root / subset / name
                if not f.exists():
                    continue
                p = np.load(f)["tracks_XYZ"].astype(np.float32)
                if p.shape != gt.shape:
                    continue
                med, frac = score(p, gt, vis)
                if med is not None:
                    rows[tag].append((med, frac))
                    agg[tag].append((med, frac))
        per_subset[subset] = rows

    print(f"  {'subset':11s} {'old med':>9s} {'new med':>9s} | {'old <thr':>9s} {'new <thr':>9s} {'delta':>8s}")
    for subset, rows in per_subset.items():
        if not rows["old"] or not rows["new"]:
            continue
        om, of = np.median([r[0] for r in rows["old"]]), np.mean([r[1] for r in rows["old"]])
        nm, nf = np.median([r[0] for r in rows["new"]]), np.mean([r[1] for r in rows["new"]])
        print(f"  {subset:11s} {om:9.3f} {nm:9.3f} | {of:9.4f} {nf:9.4f} {nf-of:+8.4f}")
    om, of = np.median([r[0] for r in agg["old"]]), np.mean([r[1] for r in agg["old"]])
    nm, nf = np.median([r[0] for r in agg["new"]]), np.mean([r[1] for r in agg["new"]])
    print(f"  {'ALL':11s} {om:9.3f} {nm:9.3f} | {of:9.4f} {nf:9.4f} {nf-of:+8.4f}")
    print("\n  old = image_size 512, scale 0 (eval_waft defaults) -- what v45 was scored against")
    print("  new = image_size 896, scale -1                     -- what v63 was scored against")
    print("  'med' is median 3-D error in metres; '<thr' is the mean fraction within the "
          "fixed-metre thresholds, i.e. the quantity metric-AJ is built from.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
