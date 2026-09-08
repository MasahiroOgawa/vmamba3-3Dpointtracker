"""Apply the standalone-trained per-frame scale to minival tracks, for official scoring.

The standalone model reduces scale error by 66.6% on held-out training clips. That is a regression
result; this converts it into the metric the paper reports, by writing corrected tracks in the
released-baseline format so eval_metric3d can score them in metric-AJ.

It is the same substitution the oracle used, with the model's prediction in place of ground truth,
so the two are directly comparable: oracle 0.3298 metric-AJ, raw 0.2238-equivalent, DELTA 0.2270.
Visibility passes through untouched -- a depth scale cannot change it.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
_cwd = os.getcwd()
from eval_waft import MINIVAL_FILES, load_da3_depth, DEPTH_SOURCE  # noqa: E402
os.chdir(_cwd)
from mamba3_tracker.model.depth_refined_tracker import Mamba3DepthScaleRefiner  # noqa: E402

GRID = 64


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", type=Path, default=Path("result/scale_standalone/best.pt"))
    ap.add_argument("--src", type=Path,
                    default=Path.home() / "data/tapvid3d_baseline_preds/waft_minival_is896_s-1_i4")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    st = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    # This writes tracks_XYZ = pr * exp(ds): pr's Z is used DIRECTLY, so the depth that
    # built pr must be the depth the refiner was trained on. Unlike a uv-only use, a
    # mismatch here silently produces a hybrid nobody can interpret.
    print(f"[apply] depth source: {DEPTH_SOURCE}")
    print(f"[apply] waft tracks: {DEPTH_SOURCE.verify(args.waft)}")

    model = Mamba3DepthScaleRefiner(two_pool=True, max_scale_correction=2.5).to(dev)
    model.load_state_dict(st["model"])
    model.eval()
    print(f"  loaded {args.ckpt} (step {st.get('step')})", flush=True)

    n, mags = 0, []
    with torch.no_grad():
        for subset in ("drivetrack", "pstudio", "adt"):
            for name in MINIVAL_FILES[subset]:
                pf = args.src / subset / name
                if not pf.exists():
                    continue
                z = np.load(pf)
                pr, pvis = z["tracks_XYZ"].astype(np.float32), z["visibility"]
                try:
                    depth = load_da3_depth(subset, name, pr.shape[0])
                except Exception:
                    continue
                dt = torch.from_numpy(np.log(np.clip(depth, 1e-6, None))).unsqueeze(1)
                g = torch.nn.functional.adaptive_avg_pool2d(dt, GRID)[:, 0]
                ds = model.per_frame_logscale(g.unsqueeze(0).to(dev))[0, :, 0].cpu().numpy()
                mags.append(float(np.abs(ds).mean()))
                d = args.out / subset
                d.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(d / name,
                                    tracks_XYZ=pr * np.exp(ds)[:, None, None], visibility=pvis)
                n += 1
    print(f"  wrote {n} clips to {args.out}; mean |predicted ds| = {np.mean(mags):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
