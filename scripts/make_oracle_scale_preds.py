"""Write oracle-scale-corrected tracks so the OFFICIAL evaluator can score them in metric-AJ.

Every ceiling reported so far was APD3D, computed by a local helper that ignores visibility. AJ is
~0.63x APD3D because the remaining third is visibility agreement, which a per-frame scale cannot
change -- so those ceilings overstate what any scale correction can reach on the metric the paper
reports. This writes predictions in the released-baseline format and lets eval_metric3d score them,
giving the ceiling in metric-AJ instead of an estimate.

Visibility is copied through unchanged: the oracle corrects depth only, which is the point.
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["scale", "bias"], default="scale",
                    help="scale: one factor per frame; bias: one constant per clip")
    ap.add_argument("--src", type=Path,
                    default=Path.home() / "data/tapvid3d_baseline_preds/waft_minival_is896_s-1_i4")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    n = 0
    for subset in ("drivetrack", "pstudio", "adt"):
        for name in MINIVAL_FILES[subset]:
            gtf, pf = TAPVID3D_ROOT / subset / name, args.src / subset / name
            if not (gtf.exists() and pf.exists()):
                continue
            g = dict(np.load(gtf, allow_pickle=True))
            gt, vis = g["tracks_XYZ"].astype(np.float32), g["visibility"].astype(bool)
            z = np.load(pf)
            pr, pvis = z["tracks_XYZ"].astype(np.float32), z["visibility"]
            if pr.shape != gt.shape:
                continue
            m = vis & np.isfinite(pr).all(-1)
            with np.errstate(divide="ignore", invalid="ignore"):
                r = np.log(np.clip(gt[..., 2], 1e-6, None)) - np.log(np.clip(pr[..., 2], 1e-6, None))
            ds = np.array([np.median(r[t][m[t]]) if m[t].any() else 0.0 for t in range(r.shape[0])])
            if args.mode == "bias":
                ds = np.full_like(ds, float(np.median(ds)))
            out_dir = args.out / subset
            out_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(out_dir / name,
                                tracks_XYZ=pr * np.exp(ds)[:, None, None],
                                visibility=pvis)
            n += 1
    print(f"  wrote {n} clips to {args.out} (mode={args.mode})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
