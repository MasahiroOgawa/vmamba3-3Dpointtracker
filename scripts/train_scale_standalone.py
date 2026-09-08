"""Train the depth-scale refiner on its own: depth map -> one log-scale per frame.

Not part of the tracker loop. The scale is a self-contained regression -- given a clip's depth maps,
predict the per-frame log-scale that would make them metric -- so training it alone removes the two
things that defeated the previous attempts: a tracking loss ~100x larger in gradient competing for
the same parameters, and a closed-form bypass the network could hide behind.

Targets come from ground truth, ds*_f = median_n[log z_gt - log z_raw] over visible points, computed
once and cached. Fitted on minival-DISJOINT training clips and validated on held-out training clips,
so minival stays untouched for the final evaluation.

Reference points, all measured: a linear fit on ten log-depth quantiles reaches R^2 0.51 and is
worth +0.0734 APD3D; a PERFECT per-frame scale reaches 0.3298 metric-AJ against v63's 0.2238 and
DELTA's 0.2270. The question this answers is how much of that a small network captures.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
_cwd = os.getcwd()
from eval_waft import (  # noqa: E402
    DA3_ROOT,
    MINIVAL_FILES,
    TAPVID3D_ROOT,
    get_full_eval_files,
    load_da3_depth,
)
os.chdir(_cwd)
from mamba3_tracker.model.depth_refined_tracker import Mamba3DepthScaleRefiner  # noqa: E402

GRID = 64


def build_cache(n_clips: int, cache_path: Path, waft: Path):
    """(depth grids, ds* targets) per clip, from training clips only."""
    if cache_path.exists():
        d = np.load(cache_path, allow_pickle=True)
        return list(d["grids"]), list(d["targets"])
    minival = {n for s in MINIVAL_FILES for n in MINIVAL_FILES[s]}
    grids, targets, per_sub = [], [], {}
    for subset in ("drivetrack", "pstudio", "adt"):
        used = 0
        for name in get_full_eval_files(subset):
            if used >= n_clips:
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
                # TARGET BUG, fixed. This used pr[...,2] -- the WAFT prediction's own z. But the
                # WAFT prediction set (waft_full_eval, written 2026-08-17) was unprojected with
                # DA3-l depth, while the model's INPUT here is DA3_ROOT (DA3-g). So the refiner was
                # trained to read a DA3-g map and emit the DA3-l correction. Measured consequence on
                # minival: true ds* = -0.927 on drivetrack while it predicted +0.586 -- opposite
                # sign -- and applying it made metric-AJ 0.1435 against 0.208 for no correction.
                # The target must be the residual of the depth the model will actually be applied to,
                # i.e. DA3_ROOT's depth sampled at the WAFT track's uv.
                fx, fy, cx, cy = (float(v) for v in g["fx_fy_cx_cy"])
                W_o, H_o = 2.0 * cx, 2.0 * cy          # principal point is image centre here
                Hd, Wd = depth.shape[1], depth.shape[2]
                zc_ = np.clip(pr[..., 2], 1e-6, None)
                u_ = (fx * pr[..., 0] / zc_ + cx) * (Wd / W_o)
                v_ = (fy * pr[..., 1] / zc_ + cy) * (Hd / H_o)
                ui_ = np.clip(np.round(u_).astype(np.int64), 0, Wd - 1)
                vi_ = np.clip(np.round(v_).astype(np.int64), 0, Hd - 1)
                F_c = min(depth.shape[0], pr.shape[0], gt.shape[0])
                z_here = np.take_along_axis(
                    depth[:F_c].reshape(F_c, -1), (vi_[:F_c] * Wd + ui_[:F_c]), axis=1
                )
                r = np.log(np.clip(gt[:F_c, ..., 2], 1e-6, None)) - np.log(np.clip(z_here, 1e-6, None))
            m = vis[: r.shape[0]] & np.isfinite(r)
            tgt = np.array([np.median(r[t][m[t]]) if m[t].sum() >= 1 else np.nan
                            for t in range(r.shape[0])], dtype=np.float32)
            if np.isnan(tgt).all():
                continue
            dt = torch.from_numpy(np.log(np.clip(depth, 1e-6, None))).unsqueeze(1)
            small = torch.nn.functional.adaptive_avg_pool2d(dt, GRID)[:, 0].numpy().astype(np.float32)
            grids.append(small)
            targets.append(tgt)
            used += 1
        per_sub[subset] = used
    print(f"  cached {len(grids)} clips {per_sub}", flush=True)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, grids=np.array(grids, dtype=object),
                        targets=np.array(targets, dtype=object), allow_pickle=True)
    return grids, targets


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--clips-per-subset", type=int, default=200)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--window", type=int, default=24)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--out", type=Path, default=Path("result/scale_standalone"))
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Cache path keyed on the depth root. A shared path would hand a DA3-l cache to a DA3-g run
    # without a word, which is how the v73-v85 arms ended up trained on the wrong depth.
    TARGET_VERSION = "tgt2"   # tgt2: residual vs DA3_ROOT depth at the WAFT uv (tgt1 used pr z)
    cache = Path(f"result/scale_train_cache_{DA3_ROOT.name}_{TARGET_VERSION}.npz")
    from eval_waft import DEPTH_SOURCE as _DS
    _waft = Path.home() / "data/tapvid3d_baseline_preds/waft_full_eval"
    # The WAFT track is used for uv ONLY -- the target is now the residual against _DS depth
    # sampled at that uv. uv is invariant to scaling along the ray, so a DA3-l-built track set
    # is a legitimate uv source for a DA3-g model. It was taking pr[...,2] as DEPTH that broke
    # this, so the check is reported rather than enforced -- and the distinction is the point.
    print(f"  depth source: {_DS}", flush=True)
    print(f"  waft tracks : {_waft.name} [{_DS.verify(_waft, strict=False)}] -- used for uv only",
          flush=True)
    print(f"  cache: {cache}", flush=True)
    grids, targets = build_cache(args.clips_per_subset, cache,
                                 Path.home() / "data/tapvid3d_baseline_preds/waft_full_eval")
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(grids))
    nval = max(1, int(len(idx) * args.val_frac))
    val_i, tr_i = idx[:nval], idx[nval:]
    print(f"  {len(tr_i)} train / {len(val_i)} val clips", flush=True)

    model = Mamba3DepthScaleRefiner(two_pool=True, max_scale_correction=2.5).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)

    def batch(ids):
        i = int(rng.choice(ids))
        g, t = grids[i], targets[i]
        F_ = g.shape[0]
        s = int(rng.integers(0, max(1, F_ - args.window)))
        gg = torch.from_numpy(g[s:s + args.window]).unsqueeze(0).to(dev)
        tt = torch.from_numpy(t[s:s + args.window]).unsqueeze(0).to(dev)
        return gg, tt

    def evaluate():
        model.eval()
        errs, base = [], []
        with torch.no_grad():
            for i in val_i:
                g = torch.from_numpy(grids[i]).unsqueeze(0).to(dev)
                t = torch.from_numpy(targets[i]).unsqueeze(0).to(dev)
                p = model.per_frame_logscale(g)[..., 0]
                m = torch.isfinite(t)
                if m.any():
                    errs.append(float((p[m] - t[m]).abs().mean()))
                    base.append(float(t[m].abs().mean()))   # what emitting zero would score
        model.train()
        return float(np.mean(errs)), float(np.mean(base))

    t0, best = time.time(), float("inf")
    args.out.mkdir(parents=True, exist_ok=True)
    for step in range(1, args.steps + 1):
        g, t = batch(tr_i)
        p = model.per_frame_logscale(g)[..., 0]
        m = torch.isfinite(t)
        if not m.any():
            continue
        loss = (p[m] - t[m]).abs().mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 500 == 0 or step == args.steps:
            # reported as a percentage reduction in mean |error|, NOT as R^2: these are L1 errors,
            # and 1-(v/b)^2 on them is not 1-SS_res/SS_tot.
            v, b = evaluate()
            flag = ""
            if v < best:
                best = v
                torch.save({"model": model.state_dict(), "step": step,
                            "cfg": {"model": {"two_pool": True, "max_scale_correction": 2.5}}},
                           args.out / "best.pt")
                flag = "  <- saved"
            print(f"  step {step:5d}  train {float(loss):.4f}  val |err| {v:.4f}  "
                  f"(emitting zero = {b:.4f})  {(1-v/b)*100:5.1f}% reduction{flag}", flush=True)
    print(f"  done in {(time.time()-t0)/60:.1f} min; best val |err| {best:.4f} -> {args.out/'best.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
