"""Phase-2 gate experiment for the v40 world-frame proposal.

Question: on drivetrack far-field, does parallax TRIANGULATION of a static point beat the monocular
DA3-depth estimate? If yes, the world-frame program (estimate camera pose from static points, then
triangulate) is justified. We test the *load-bearing* claim directly, with exact GT pixel
correspondences, isolating two factors:

  * triangulation with GT pose  (oracle: "is triangulation useful at all?")
  * triangulation with pose ESTIMATED from the static monocular point clouds (RANSAC rigid Kabsch)
    ("does it survive realistic pose estimation?")

Static points are selected using GT extrinsics (a clean test — separates "is triangulation useful"
from "is static detection hard"; the full v40 learns static detection). Errors are stratified by GT
depth. Gate: estimated-pose triangulation cuts far-field (>FAR m) median depth error by >= 25%.

Reads only; no training, no model. Run: uv run python scripts/exp_worldframe_triangulation.py
"""

from __future__ import annotations

import argparse
import glob
import os

import numpy as np

DATA = os.path.expanduser("~/data/tapvid3d/drivetrack")
DA3 = os.path.expanduser("~/data/tapvid3d_da3/drivetrack")


def K_from(fxfycxcy):
    fx, fy, cx, cy = fxfycxcy
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], float)


def project(XYZ, K):  # (...,3) -> (...,2) pixels
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    Z = np.clip(XYZ[..., 2], 1e-6, None)
    return np.stack([fx * XYZ[..., 0] / Z + cx, fy * XYZ[..., 1] / Z + cy], -1)


def unproject(uv, depth, K):  # (...,2),(...) -> (...,3) camera
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    x = (uv[..., 0] - cx) / fx * depth
    y = (uv[..., 1] - cy) / fy * depth
    return np.stack([x, y, depth], -1)


def load_da3(cid):
    e = np.load(os.path.join(DA3, cid), allow_pickle=True)
    dq = e["depth_q"].astype(np.float32)
    depth = float(e["d_min"]) + dq * (float(e["d_max"]) - float(e["d_min"])) / 65535.0
    return depth, int(e["h"]), int(e["w"])  # (F,Hd,Wd), orig h, w


def sample_depth(dmap, uv, h, w):  # dmap (Hd,Wd), uv (N,2) orig px -> (N,) bilinear
    Hd, Wd = dmap.shape
    c = uv[:, 0] / w * Wd - 0.5
    r = uv[:, 1] / h * Hd - 0.5
    c0 = np.clip(np.floor(c).astype(int), 0, Wd - 2)
    r0 = np.clip(np.floor(r).astype(int), 0, Hd - 2)
    fc = np.clip(c - c0, 0, 1)
    fr = np.clip(r - r0, 0, 1)
    d = (
        dmap[r0, c0] * (1 - fc) * (1 - fr)
        + dmap[r0, c0 + 1] * fc * (1 - fr)
        + dmap[r0 + 1, c0] * (1 - fc) * fr
        + dmap[r0 + 1, c0 + 1] * fc * fr
    )
    return d


def cam_center(E):  # world->cam 4x4 -> camera center in world
    R, t = E[:3, :3], E[:3, 3]
    return -R.T @ t


def rel_pose(Et, Es):  # cam_t -> cam_s
    return Es @ np.linalg.inv(Et)


def kabsch(src, dst):  # rigid: dst ~ R@src + t
    sc, dc = src.mean(0), dst.mean(0)
    H = (src - sc).T @ (dst - dc)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return R, dc - R @ sc


def ransac_rigid(src, dst, iters=200, thresh=0.5, rng=None):
    rng = rng or np.random.default_rng(0)
    n = len(src)
    if n < 3:
        return kabsch(src, dst) if n else (np.eye(3), np.zeros(3)), np.zeros(n, bool)
    best_in, best = None, None
    for _ in range(iters):
        idx = rng.choice(n, 3, replace=False)
        try:
            R, t = kabsch(src[idx], dst[idx])
        except np.linalg.LinAlgError:
            continue
        err = np.linalg.norm((src @ R.T + t) - dst, axis=1)
        inl = err < thresh
        if best_in is None or inl.sum() > best_in.sum():
            best_in, best = inl, (R, t)
    if best_in.sum() >= 3:
        best = kabsch(src[best_in], dst[best_in])
    return best, best_in


def triangulate(P1, P2, x1, x2):  # -> 3D in frame 1
    A = np.stack(
        [x1[0] * P1[2] - P1[0], x1[1] * P1[2] - P1[1], x2[0] * P2[2] - P2[0], x2[1] * P2[2] - P2[1]]
    )
    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    return X[:3] / X[3]


def rot_angle(Ra, Rb):
    c = (np.trace(Ra @ Rb.T) - 1) / 2
    return np.degrees(np.arccos(np.clip(c, -1, 1)))


def pick_pair(vis, XYZ, tref, args):
    """Given a reference frame tref, pick the widest-baseline co-visible frame s and the oracle
    relative pose T_{tref->s} + static inlier set, fit by RANSAC-rigid on GT camera clouds."""
    F = XYZ.shape[0]
    best = None
    for s in range(F):
        if s == tref:
            continue
        both = np.where(vis[tref] & vis[s])[0]
        if len(both) < args.min_covis:
            continue
        src, dst = XYZ[tref, both], XYZ[s, both]
        (R, t), inl = ransac_rigid(src, dst, iters=200, thresh=args.rigid_inlier)
        base = np.linalg.norm(t)  # camera translation magnitude between the two frames
        if inl.sum() < args.min_covis:
            continue
        if best is None or base > best[0]:
            best = (base, s, (R, t), both, inl)
    return best  # None or (baseline, s, T_oracle, common_idx, static_inlier_mask)


def run(args):
    clips = sorted(glob.glob(os.path.join(DATA, "*.npz")))[: args.n_clips]
    rows = []  # (Zgt, err_mono, err_oracle, err_est, rot_err, trans_err)
    for ci, path in enumerate(clips):
        cid = os.path.basename(path)
        if not os.path.exists(os.path.join(DA3, cid)):
            continue
        d = np.load(path, allow_pickle=True)
        XYZ = d["tracks_XYZ"].astype(float)  # (F,N,3) camera
        vis = d["visibility"].astype(bool)  # (F,N)
        K = K_from(d["fx_fy_cx_cy"].astype(float))
        F, N = XYZ.shape[:2]
        depth_maps, h, w = load_da3(cid)
        uv = project(XYZ, K)  # (F,N,2) exact GT pixels
        mono = np.stack([sample_depth(depth_maps[t], uv[t], h, w) for t in range(F)])  # (F,N)
        mono_cloud = unproject(uv, mono, K)  # (F,N,3)

        # reference frame: earliest frame with the most visible points
        tref = int(np.argmax(vis[: max(1, F // 3)].sum(1)))
        pair = pick_pair(vis, XYZ, tref, args)
        if pair is None:
            continue
        base, s, (Ro, to), common, static_inl = pair
        static = common[static_inl]  # indices of static (rigid-inlier) points
        if len(static) < args.min_covis:
            continue

        # estimated pose from MONOCULAR clouds over the same common points (realistic)
        src, dst = mono_cloud[tref, common], mono_cloud[s, common]
        good = np.isfinite(src).all(1) & np.isfinite(dst).all(1)
        (Re, te), _ = ransac_rigid(src[good], dst[good], iters=300, thresh=args.ransac_thresh)
        rot_err = rot_angle(Re, Ro)
        trans_err = np.linalg.norm(te - to)

        P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
        P2o = K @ np.hstack([Ro, to[:, None]])
        P2e = K @ np.hstack([Re, te[:, None]])
        for n in static:
            Zgt = XYZ[tref, n, 2]
            if not np.isfinite(Zgt) or Zgt <= 0 or base / Zgt < args.min_parallax:
                continue
            err_mono = abs(mono[tref, n] - Zgt)
            Xo = triangulate(P1, P2o, uv[tref, n], uv[s, n])
            err_oracle = abs(Xo[2] - Zgt) if Xo[2] > 0 else err_mono
            Xe = triangulate(P1, P2e, uv[tref, n], uv[s, n])
            err_est = abs(Xe[2] - Zgt) if Xe[2] > 0 else err_mono
            rows.append((Zgt, err_mono, err_oracle, err_est, rot_err, trans_err))
        if (ci + 1) % 10 == 0:
            print(f"  processed {ci+1}/{len(clips)} clips, {len(rows)} static-point samples")

    R = np.array(rows)
    if len(R) == 0:
        print("NO samples — check data/paths.")
        return
    Z, em, eo, ee, rerr, terr = R.T
    print(f"\n=== {len(R)} static-point samples over {len(clips)} drivetrack clips ===")
    print(f"pose estimate error: rot median {np.median(rerr):.2f} deg, "
          f"trans median {np.median(terr):.2f} m")
    bins = [("near <%d" % args.near, Z < args.near),
            ("mid %d-%d" % (args.near, args.far), (Z >= args.near) & (Z < args.far)),
            ("far >%d" % args.far, Z >= args.far),
            ("all", np.ones(len(Z), bool))]
    hdr = f"{'stratum':<14}{'n':>6}{'mono med':>10}{'oracle med':>12}{'est med':>10}{'est vs mono':>13}"
    print(hdr)
    far_impr = None
    for name, mask in bins:
        if mask.sum() == 0:
            continue
        mm, oo, eem = np.median(em[mask]), np.median(eo[mask]), np.median(ee[mask])
        impr = 100 * (mm - eem) / mm if mm > 0 else 0
        print(f"{name:<14}{mask.sum():>6}{mm:>10.3f}{oo:>12.3f}{eem:>10.3f}{impr:>12.1f}%")
        if name.startswith("far"):
            far_impr = impr
    print("\n(median absolute depth error in metres; 'est vs mono' = % reduction, higher=better)")
    if far_impr is not None:
        verdict = "PASS" if far_impr >= 25 else "FAIL"
        print(f"\nGATE (far-field est-pose triangulation cuts median error >=25%): "
              f"{far_impr:.1f}% -> {verdict}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-clips", type=int, default=50)
    ap.add_argument("--min-covis", type=int, default=20, help="min co-visible points for a frame pair")
    ap.add_argument("--rigid-inlier", type=float, default=0.3, help="GT-cloud rigid inlier / static (m)")
    ap.add_argument("--min-parallax", type=float, default=0.02, help="min baseline/depth (rad)")
    ap.add_argument("--ransac-thresh", type=float, default=0.5, help="mono-cloud rigid RANSAC inlier (m)")
    ap.add_argument("--near", type=float, default=10.0)
    ap.add_argument("--far", type=float, default=30.0)
    run(ap.parse_args())
