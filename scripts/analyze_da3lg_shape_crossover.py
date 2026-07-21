#!/usr/bin/env python
"""At which GT distance does DA3-g's depth SHAPE overtake DA3-l's?

For every TAPVid-3D minival point we sample DA3-l and DA3-g depth at the GT pixel,
remove a per-frame global scale (median GT/pred) to isolate *shape*, and measure the
residual scale-invariant error |log(scale*pred / gt)|. Binning by GT distance gives,
per subset, the distance where DA3-g becomes more accurate than DA3-l -- i.e. the
crossover that sets `thre_meter` for a distance-gated DA3-l/DA3-g fusion.
"""

import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mamba3_tracker.data.tapvid3d import list_clips, load_clip
from mamba3_tracker.data.tapvid3d_splits import MINIVAL_FILES

DATA_ROOT = Path("~/data").expanduser()
DA3 = {
    "DA3-l": Path("~/data/tapvid3d_da3").expanduser(),
    "DA3-g": Path("~/data/tapvid3d_da3nested").expanduser(),
}
SUBSETS = ["drivetrack", "pstudio", "adt"]
BIN_EDGES = np.array(
    [0, 0.5, 1, 1.5, 2, 2.5, 3, 4, 5, 7, 10, 15, 20, 30, 50, 80, 120], float
)
BIN_CENTERS = 0.5 * (BIN_EDGES[:-1] + BIN_EDGES[1:])
MIN_PER_BIN = 50


def load_depth(root: Path, subset: str, clip_id: str, F_: int) -> torch.Tensor:
    with np.load(root / subset / (clip_id + ".npz")) as dd:
        if "depth_q" in dd:
            q = dd["depth_q"][:F_].astype(np.float32)
            dmin, dmax = float(dd["d_min"]), float(dd["d_max"])
            depth = dmin + q * ((dmax - dmin) / 65535.0)
        else:
            depth = dd["depth"][:F_].astype(np.float32)
    return torch.from_numpy(depth)  # (F, Hd, Wd)


def sample_depth(depth: torch.Tensor, uv: torch.Tensor, W: int, H: int) -> np.ndarray:
    gx = 2 * uv[..., 0] / W - 1
    gy = 2 * uv[..., 1] / H - 1
    grid = torch.stack([gx, gy], dim=-1).unsqueeze(1)  # (F,1,N,2)
    z = F.grid_sample(
        depth.unsqueeze(1),
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    )
    return z.view(depth.shape[0], -1).numpy()  # (F, N)


def median_per_bin(gt: np.ndarray, r: np.ndarray):
    idx = np.digitize(gt, BIN_EDGES) - 1
    med = np.full(len(BIN_CENTERS), np.nan)
    mean = np.full(len(BIN_CENTERS), np.nan)
    p90 = np.full(len(BIN_CENTERS), np.nan)
    cnt = np.zeros(len(BIN_CENTERS), int)
    for b in range(len(BIN_CENTERS)):
        sel = idx == b
        cnt[b] = sel.sum()
        if cnt[b] >= MIN_PER_BIN:
            med[b] = np.median(r[sel])
            mean[b] = np.mean(r[sel])
            p90[b] = np.quantile(r[sel], 0.9)
    return med, mean, p90, cnt


def crossover(med_l, med_g):
    """First bin (increasing distance) where DA3-g <= DA3-l, both valid."""
    for b in range(len(BIN_CENTERS)):
        if np.isnan(med_l[b]) or np.isnan(med_g[b]):
            continue
        if med_g[b] <= med_l[b]:
            lo, hi = BIN_EDGES[b], BIN_EDGES[b + 1]
            return b, lo, hi
    return None, None, None


def main() -> int:
    per = {s: {n: {"gt": [], "r": [], "r_clip": []} for n in DA3} for s in SUBSETS}
    for sub in SUBSETS:
        allow = set(MINIVAL_FILES[sub])
        clips = [p for p in list_clips(DATA_ROOT, [sub]) if p.name in allow]
        for path in clips:
            clip = load_clip(path)
            F_ = clip.tracks_XYZ.shape[0]
            H, W = clip.images.shape[-2], clip.images.shape[-1]
            K = clip.K.numpy()
            xyz = clip.tracks_XYZ.numpy()  # (F,N,3)
            vis = clip.visibility.numpy().astype(bool)
            z_gt = xyz[..., 2]
            fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
            zc = np.clip(z_gt, 1e-6, None)
            u = fx * xyz[..., 0] / zc + cx
            v = fy * xyz[..., 1] / zc + cy
            uv = torch.from_numpy(np.stack([u, v], -1)).float()

            preds, ok = {}, True
            for name, root in DA3.items():
                if not (root / sub / (clip.clip_id + ".npz")).exists():
                    ok = False
                    break
                d = load_depth(root, sub, clip.clip_id, F_)
                preds[name] = sample_depth(d, uv[: d.shape[0]], W, H)
            if not ok:
                continue
            Fd = min(preds["DA3-l"].shape[0], preds["DA3-g"].shape[0], F_)
            for name in DA3:
                gt_c, zp_c = [], []
                for f in range(Fd):
                    zp = preds[name][f]
                    mm = vis[f] & (z_gt[f] > 0) & (zp > 0.05)
                    if mm.sum() < 10:
                        continue
                    s = np.median(z_gt[f][mm] / zp[mm])  # per-frame scale
                    r = np.abs(np.log((s * zp[mm]) / z_gt[f][mm]))
                    per[sub][name]["gt"].append(z_gt[f][mm])
                    per[sub][name]["r"].append(r)
                    gt_c.append(z_gt[f][mm])
                    zp_c.append(zp[mm])
                if gt_c:  # one scale for the whole clip -> exposes temporal scale drift
                    gt_c, zp_c = np.concatenate(gt_c), np.concatenate(zp_c)
                    sc = np.median(gt_c / zp_c)
                    per[sub][name]["r_clip"].append(np.abs(np.log((sc * zp_c) / gt_c)))
        print(f"[{sub}] clips scored", flush=True)

    # aggregate + report
    fig, axes = plt.subplots(2, len(SUBSETS) + 1, figsize=(22, 9))
    print("\n=== scale-invariant shape error |log(scale*pred/gt)| per distance bin ===")
    print("    (median = typical shape; mean/p90 = tail/outlier behaviour)")
    for col, sub in enumerate(SUBSETS + ["ALL"]):
        stat = {}
        for name in DA3:
            if sub == "ALL":
                gt = np.concatenate(
                    [np.concatenate(per[s][name]["gt"]) for s in SUBSETS]
                )
                r = np.concatenate([np.concatenate(per[s][name]["r"]) for s in SUBSETS])
            else:
                gt = np.concatenate(per[sub][name]["gt"])
                r = np.concatenate(per[sub][name]["r"])
            med, mean, p90, cnt = median_per_bin(gt, r)
            stat[name] = dict(med=med, mean=mean, p90=p90, cnt=cnt)
        for row, key in enumerate(["med", "mean"]):
            ax = axes[row][col]
            for name in DA3:
                ax.plot(BIN_CENTERS, stat[name][key], "o-", label=name)
            ax.set_xscale("log")
            ax.grid(alpha=0.3)
            ax.legend()
            if row == 0:
                bm, lo, hi = crossover(stat["DA3-l"]["med"], stat["DA3-g"]["med"])
                bt, loT, hiT = crossover(stat["DA3-l"]["mean"], stat["DA3-g"]["mean"])
                ax.set_title(
                    f"{sub}\nmedian cross: {f'{lo:g}-{hi:g}m' if bm is not None else 'none'}"
                )
            else:
                ax.set_title(f"{sub}  mean (tail)")
            ax.set_xlabel("GT distance (m)")
        print(f"\n-- {sub} --")
        print(
            f"   {'dist':>6} {'l_med':>7} {'g_med':>7} {'l_mean':>7} {'g_mean':>7} {'l_p90':>7} {'g_p90':>7} {'n':>9}"
        )
        for i, c in enumerate(BIN_CENTERS):
            if stat["DA3-l"]["cnt"][i] >= MIN_PER_BIN:
                L, G = stat["DA3-l"], stat["DA3-g"]
                mk = " g<l(med)" if G["med"][i] <= L["med"][i] else ""
                mk += " g<l(mean)" if G["mean"][i] <= L["mean"][i] else ""
                print(
                    f"   {c:6.2f} {L['med'][i]:7.4f} {G['med'][i]:7.4f} "
                    f"{L['mean'][i]:7.4f} {G['mean'][i]:7.4f} {L['p90'][i]:7.4f} {G['p90'][i]:7.4f} "
                    f"{L['cnt'][i]:9d}{mk}"
                )
    axes[0][0].set_ylabel("median |log(scale*pred / gt)|")
    axes[1][0].set_ylabel("mean |log(scale*pred / gt)|")

    print("\n=== per-frame vs per-clip scale alignment (mean |log(scale*pred/gt)|) ===")
    print("    per-frame = removes each frame's scale -> pure SPATIAL shape")
    print(
        "    per-clip  = removes ONE scale for the clip -> shape + TEMPORAL scale drift"
    )
    print(
        f"   {'subset':>10} | {'DA3-l frame':>11} {'DA3-l clip':>11} | {'DA3-g frame':>11} {'DA3-g clip':>11}"
    )
    for sub in SUBSETS:
        vals = []
        for name in DA3:
            rf = np.concatenate(per[sub][name]["r"])
            rc = np.concatenate(per[sub][name]["r_clip"])
            vals += [np.mean(rf), np.mean(rc)]
        print(
            f"   {sub:>10} | {vals[0]:11.4f} {vals[1]:11.4f} | {vals[2]:11.4f} {vals[3]:11.4f}"
        )
    out = Path("result/figures/da3lg_shape_crossover.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"\n[saved] {out}")

    # ---- clean 2-panel figure for the memo ----
    fig2, (axA, axB) = plt.subplots(1, 2, figsize=(13, 4.5))
    colors = {"drivetrack": "C0", "pstudio": "C1", "adt": "C2"}
    for sub in SUBSETS:
        m_l, _, _, _ = median_per_bin(
            np.concatenate(per[sub]["DA3-l"]["gt"]),
            np.concatenate(per[sub]["DA3-l"]["r"]),
        )
        m_g, _, _, _ = median_per_bin(
            np.concatenate(per[sub]["DA3-g"]["gt"]),
            np.concatenate(per[sub]["DA3-g"]["r"]),
        )
        axA.plot(BIN_CENTERS, m_l, "--", color=colors[sub], alpha=0.9)
        axA.plot(BIN_CENTERS, m_g, "-", color=colors[sub], label=sub)
    axA.set_xscale("log")
    axA.set_xlabel("GT distance (m)")
    axA.set_ylabel("median |log(scale*pred/gt)|")
    axA.set_title("(a) spatial shape vs distance  (dashed=DA3-l, solid=DA3-g)")
    axA.grid(alpha=0.3)
    axA.legend(title="subset")

    x = np.arange(len(SUBSETS))
    w = 0.2
    lf = [np.mean(np.concatenate(per[s]["DA3-l"]["r"])) for s in SUBSETS]
    lc = [np.mean(np.concatenate(per[s]["DA3-l"]["r_clip"])) for s in SUBSETS]
    gf = [np.mean(np.concatenate(per[s]["DA3-g"]["r"])) for s in SUBSETS]
    gc = [np.mean(np.concatenate(per[s]["DA3-g"]["r_clip"])) for s in SUBSETS]
    axB.bar(x - 1.5 * w, lf, w, label="DA3-l per-frame", color="#9ecae1")
    axB.bar(x - 0.5 * w, lc, w, label="DA3-l per-clip", color="#3182bd")
    axB.bar(x + 0.5 * w, gf, w, label="DA3-g per-frame", color="#fdae6b")
    axB.bar(x + 1.5 * w, gc, w, label="DA3-g per-clip", color="#e6550d")
    axB.set_xticks(x)
    axB.set_xticklabels(SUBSETS)
    axB.set_ylabel("mean |log(scale*pred/gt)|")
    axB.set_title("(b) per-frame vs per-clip scale  (per-clip gap = temporal drift)")
    axB.legend(fontsize=8)
    axB.grid(axis="y", alpha=0.3)
    fig2.tight_layout()
    out2 = Path("result/figures/da3lg_error_analysis.png")
    fig2.savefig(out2, dpi=140, bbox_inches="tight")
    print(f"[saved] {out2}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
