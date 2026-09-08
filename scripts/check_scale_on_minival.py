"""Does a depth-scale refiner reduce the scale error ON MINIVAL (not on its training pool)?

Reports, per subset, |ds*| before against |ds* - ds_pred| after, plus the signed means so a
sign inversion is visible. The first DA3-g refiner passed its own held-out objective by 73.3%
while predicting the OPPOSITE SIGN here, because its target came from DA3-l depth.
"""

import argparse
import os
import pathlib
import sys
import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("DA3_ROOT", str(pathlib.Path.home() / "data/tapvid3d_da3nested"))
from mamba3_tracker.model.depth_refined_tracker import Mamba3DepthScaleRefiner  # noqa: E402
from eval_waft import MINIVAL_FILES, load_da3_depth, TAPVID3D_ROOT  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=pathlib.Path, required=True)
    ap.add_argument("--clips", type=int, default=4)
    ap.add_argument("--frames", type=int, default=24)
    a = ap.parse_args()
    m = Mamba3DepthScaleRefiner(two_pool=True, max_scale_correction=2.5)
    st = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    m.load_state_dict(st.get("model", st))
    m.eval()
    waft = (
        pathlib.Path.home() / "data/tapvid3d_baseline_preds/waft_minival_is896_s-1_i4"
    )
    print(f"  checkpoint: {a.ckpt}")
    verdicts = []
    for sub in ("drivetrack", "pstudio", "adt"):
        B, A, T, P = [], [], [], []
        for name in list(MINIVAL_FILES[sub])[: a.clips]:
            try:
                g = np.load(TAPVID3D_ROOT / sub / name, allow_pickle=True)
                pr = np.load(waft / sub / name)["tracks_XYZ"].astype(np.float64)
            except Exception:
                continue
            gxyz = g["tracks_XYZ"].astype(np.float64)
            vis = g["visibility"].astype(bool)
            F_ = min(gxyz.shape[0], pr.shape[0], a.frames)
            try:
                d = load_da3_depth(sub, name, F_)
            except Exception:
                continue
            if d.shape[0] < F_:
                continue
            fx, fy, cx, cy = (float(v) for v in g["fx_fy_cx_cy"])
            Hd, Wd = d.shape[1], d.shape[2]
            zc = np.clip(pr[:F_, :, 2], 1e-6, None)
            ui = np.clip(
                np.round((fx * pr[:F_, :, 0] / zc + cx) * (Wd / (2 * cx))).astype(int),
                0,
                Wd - 1,
            )
            vi = np.clip(
                np.round((fy * pr[:F_, :, 1] / zc + cy) * (Hd / (2 * cy))).astype(int),
                0,
                Hd - 1,
            )
            zr = np.take_along_axis(d.reshape(F_, -1), vi * Wd + ui, axis=1)
            zg = gxyz[:F_, :, 2]
            ok = vis[:F_] & (zg > 1e-3) & (zr > 1e-3)
            ds = np.array(
                [
                    np.median(np.log(zg[f][ok[f]]) - np.log(zr[f][ok[f]]))
                    if ok[f].sum() > 4
                    else np.nan
                    for f in range(F_)
                ]
            )
            dt = torch.from_numpy(np.log(np.clip(d, 1e-6, None))).unsqueeze(1).float()
            gr = torch.nn.functional.adaptive_avg_pool2d(dt, 64)[:, 0]
            with torch.no_grad():
                dp = m.per_frame_logscale(gr.unsqueeze(0))[0, :, 0].numpy()
            k = ~np.isnan(ds)
            if k.sum():
                B.append(np.abs(ds[k]))
                A.append(np.abs(ds[k] - dp[:F_][k]))
                T.append(ds[k])
                P.append(dp[:F_][k])
        if not B:
            print(f"  {sub:11s} no data")
            continue
        Bc, Ac, Tc, Pc = (np.concatenate(x) for x in (B, A, T, P))
        good = Ac.mean() < Bc.mean()
        verdicts.append(good)
        print(
            f"  {sub:11s} n={Bc.size:3d}  before={Bc.mean():.4f}  after={Ac.mean():.4f}  "
            f"{100 * (Ac.mean() - Bc.mean()) / Bc.mean():+.1f}%  "
            f"true={Tc.mean():+.3f} pred={Pc.mean():+.3f}  "
            f"{'IMPROVES' if good else 'WORSENS'}"
        )
    print(f"  VERDICT: improves {sum(verdicts)}/{len(verdicts)} subsets")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
