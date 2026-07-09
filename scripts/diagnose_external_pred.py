#!/usr/bin/env python3
"""Diagnose why an external method's TAPVid-3D predictions score low.

Separates a *coordinate-frame bug* from *genuine tracking error*. Run it whenever a
newly-onboarded baseline scores near-zero AJ, before assuming the model is bad.

For each sampled clip it reports:

  1. query-frame reprojection error (px): pred (and GT) at each query's own frame,
     projected with K. TAPVid-3D GT is PER-FRAME camera coords, so this must be ~0
     for GT; if pred is also ~0 the point ordering + frame are correct.
  2. identity AJ  vs  per-frame Umeyama-aligned AJ. Per-frame Umeyama (rigid+scale
     best-fit of pred→gt at each frame) is the theoretical upper bound for ANY
     frame/axis/rotation/translation/scale error. If aligned AJ >> identity AJ, the
     loss is a coordinate-frame bug (fix it in the inference wrapper, NOT the scorer).
     If aligned AJ is also ~0, the geometry is genuinely wrong (depth / drift).
  3. median 3D error vs temporal distance from the query frame — a rising curve is
     windowing drift (e.g. SpatialTrackerV2 s_wind), not a frame bug.

Usage (main-repo venv):
  uv run python scripts/diagnose_external_pred.py \
      --pred-dir /home/mas/data/tapvid3d_baseline_preds/<method> \
      --subsets adt pstudio --n 3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from mamba3_tracker.data.tapvid3d import load_clip  # noqa: E402
from mamba3_tracker.eval.tapvid3d_eval import compute_clip_metrics  # noqa: E402

TAPVID3D_ROOT = Path("/home/mas/data/tapvid3d")


def umeyama(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Similarity transform (scale+R+t) mapping src→dst, applied to src. (M,3) each."""
    if len(src) < 3:
        return src
    mu_s, mu_d = src.mean(0), dst.mean(0)
    s0, d0 = src - mu_s, dst - mu_d
    cov = d0.T @ s0 / len(src)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    var = (s0**2).sum() / len(src)
    c = (D * np.diag(S)).sum() / max(var, 1e-12)
    return (c * (R @ src.T).T) + (mu_d - c * (R @ mu_s))


def intr_256(clip) -> np.ndarray:
    K = clip.K.numpy()
    Ho, Wo = int(clip.images.shape[-2]), int(clip.images.shape[-1])
    s = 256.0 / min(Ho, Wo)
    return np.array([K[0, 0], K[1, 1], K[0, 2], K[1, 2]]) * s


def diagnose(pred_dir: Path, subset: str, clip_name: str) -> None:
    clip = load_clip(TAPVID3D_ROOT / subset / clip_name)
    with np.load(pred_dir / subset / clip_name) as d:
        tr = d["tracks_XYZ"].astype(np.float32)  # (F,N,3)
        vis = d["visibility"].astype(np.float32)  # (F,N)
    F = min(tr.shape[0], clip.tracks_XYZ.shape[0])
    N = min(tr.shape[1], clip.tracks_XYZ.shape[1])
    gt = clip.tracks_XYZ[:F, :N].numpy()  # (F,N,3)
    gv = clip.visibility[:F, :N].float().numpy()  # (F,N)
    pr = tr[:F, :N]
    pv = vis[:F, :N]
    q = clip.queries_xyt.numpy()[:N]
    K = clip.K.numpy()
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    # 1. query-frame reprojection
    def reproj_err(xyz):
        errs = []
        for n in range(N):
            t = int(q[n, 2])
            if t >= F:
                continue
            X, Y, Z = xyz[t, n]
            if not np.isfinite([X, Y, Z]).all() or Z <= 1e-6:
                continue
            errs.append(np.hypot(fx * X / Z + cx - q[n, 0], fy * Y / Z + cy - q[n, 1]))
        return float(np.median(errs)) if errs else float("nan")

    # 2. identity vs per-frame Umeyama AJ
    intr = intr_256(clip)
    pr_NF = np.transpose(pr, (1, 0, 2))
    pv_NF = np.transpose(pv, (1, 0))
    aj_id = compute_clip_metrics(gt, gv, pr_NF, pv_NF, intr)["average_jaccard"]

    pr_al = pr.copy()
    for t in range(F):
        m = (gv[t] > 0.5) & np.isfinite(pr[t]).all(-1) & np.isfinite(gt[t]).all(-1)
        if m.sum() >= 3:
            pr_al[t, m] = umeyama(pr[t, m], gt[t, m])
    aj_al = compute_clip_metrics(gt, gv, np.transpose(pr_al, (1, 0, 2)), pv_NF, intr)[
        "average_jaccard"
    ]

    # 3. drift vs distance from query frame (after global median scale)
    scale = np.median(np.linalg.norm(gt[gv > 0.5], axis=-1)) / max(
        np.median(np.linalg.norm(pr[gv > 0.5], axis=-1)), 1e-9
    )
    prs = pr * scale
    bins = [(0, 5), (5, 20), (20, 60), (60, 10000)]
    drift = []
    for lo, hi in bins:
        errs = []
        for n in range(N):
            tq = int(q[n, 2])
            for t in range(F):
                if (
                    lo <= abs(t - tq) < hi
                    and gv[t, n] > 0.5
                    and np.isfinite(prs[t, n]).all()
                ):
                    errs.append(np.linalg.norm(prs[t, n] - gt[t, n]))
        drift.append(np.median(errs) if errs else float("nan"))

    pred_rp = reproj_err(pr)
    aligned_helps = aj_al > aj_id + 0.05 and aj_al > 0.05
    if pred_rp > 5.0:
        verdict = "FRAME/ORDERING BUG (pred misreprojects at its own query frame)"
    elif aligned_helps:
        verdict = "per-frame depth-scale/pose drift (frame OK; Umeyama absorbs monocular-depth scale)"
    else:
        verdict = "genuine tracking error (alignment does not help)"
    print(f"\n{subset}/{clip_name}  F={F} N={N}")
    print(f"  query-reproj px:   GT={reproj_err(gt):.2f}  pred={pred_rp:.2f}")
    print(
        f"  AJ:  identity={aj_id:.4f}   per-frame-Umeyama(upper bound)={aj_al:.4f}   -> {verdict}"
    )
    print(
        "  median 3D err (m) vs |t-t_query|:  "
        + "  ".join(
            f"[{lo},{hi if hi < 9999 else '∞'})={d:.2f}"
            for (lo, hi), d in zip(bins, drift)
        )
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pred-dir", type=Path, required=True)
    ap.add_argument("--subsets", nargs="+", default=["adt", "pstudio", "drivetrack"])
    ap.add_argument("--n", type=int, default=3, help="clips per subset to sample")
    args = ap.parse_args()
    for sub in args.subsets:
        avail = sorted((args.pred_dir / sub).glob("*.npz"))[: args.n]
        for p in avail:
            try:
                diagnose(args.pred_dir, sub, p.name)
            except Exception as e:
                print(f"[skip] {sub}/{p.name}: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
