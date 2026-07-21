#!/usr/bin/env python
"""Re-anchor DA3-g's per-frame scale to test the temporal-scale-drift hypothesis (pstudio).

Two rescaled DA3-g caches, each depth map multiplied by ONE scalar per frame:
  oracle : a(f) = median(z_gt / DA3-g) at GT pixels  -> removes per-frame scale error (needs GT; ceiling).
  lprof  : DA3-g gets DA3-l's temporal scale PROFILE (GT-free realizable fix): per-frame cross-model
           median ratio, normalised to preserve DA3-g's clip-average scale (so no DA3-l bias imported).

Usage: uv run python scripts/generate_da3g_reanchor.py
Then eval each with:  eval_metric3d.py --subsets pstudio --da3-depth-root <out_dir> ...
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from mamba3_tracker.data.tapvid3d import list_clips, load_clip
from mamba3_tracker.data.tapvid3d_splits import MINIVAL_FILES

DATA_ROOT = Path("~/data").expanduser()
DA3G = Path("~/data/tapvid3d_da3nested").expanduser()
DA3L = Path("~/data/tapvid3d_da3").expanduser()
OUT = {
    "oracle": Path("~/data/tapvid3d_da3nested_oracle").expanduser(),
    "lprof": Path("~/data/tapvid3d_da3nested_lprof").expanduser(),
}
SUB = "pstudio"


def load_depth(root: Path, clip_id: str, F_: int) -> np.ndarray:
    with np.load(root / SUB / (clip_id + ".npz")) as dd:
        if "depth_q" in dd:
            q = dd["depth_q"][:F_].astype(np.float32)
            dmin, dmax = float(dd["d_min"]), float(dd["d_max"])
            return dmin + q * ((dmax - dmin) / 65535.0)
        return dd["depth"][:F_].astype(np.float32)


def sample(depth: np.ndarray, uv: torch.Tensor, W: int, H: int) -> np.ndarray:
    d = torch.from_numpy(depth).unsqueeze(1)  # (F,1,Hd,Wd)
    gx = 2 * uv[..., 0] / W - 1
    gy = 2 * uv[..., 1] / H - 1
    grid = torch.stack([gx, gy], -1).unsqueeze(1)  # (F,1,N,2)
    z = F.grid_sample(
        d, grid, mode="bilinear", padding_mode="border", align_corners=False
    )
    return z.view(depth.shape[0], -1).numpy()


def main() -> int:
    for d in OUT.values():
        (d / SUB).mkdir(parents=True, exist_ok=True)
    clips = [
        p for p in list_clips(DATA_ROOT, [SUB]) if p.name in set(MINIVAL_FILES[SUB])
    ]
    print(f"[reanchor] {len(clips)} pstudio minival clips")
    for path in clips:
        clip = load_clip(path)
        F_ = clip.tracks_XYZ.shape[0]
        H, W = clip.images.shape[-2], clip.images.shape[-1]
        K = clip.K.numpy()
        xyz = clip.tracks_XYZ.numpy()
        vis = clip.visibility.numpy().astype(bool)
        z_gt = xyz[..., 2]
        zc = np.clip(z_gt, 1e-6, None)
        u = K[0, 0] * xyz[..., 0] / zc + K[0, 2]
        v = K[1, 1] * xyz[..., 1] / zc + K[1, 2]
        uv = torch.from_numpy(np.stack([u, v], -1)).float()

        Dg = load_depth(DA3G, clip.clip_id, F_)  # (F,Hd,Wd)
        Dl = load_depth(DA3L, clip.clip_id, F_)
        Fd = min(Dg.shape[0], Dl.shape[0], F_)
        Dg, Dl = Dg[:Fd], Dl[:Fd]
        zg = sample(Dg, uv[:Fd], W, H)  # (F,N) at GT pixels

        # oracle per-frame scale from GT
        a = np.ones(Fd, np.float32)
        for f in range(Fd):
            m = vis[f] & (z_gt[f] > 0) & (zg[f] > 0.05)
            if m.sum() >= 10:
                a[f] = np.median(z_gt[f][m] / zg[f][m])
        Dg_oracle = Dg * a[:, None, None]

        # lprof: DA3-l temporal profile, GT-free, preserve DA3-g clip-average scale
        rl = np.array([np.median(Dl[f][Dl[f] > 0.05]) for f in range(Fd)], np.float32)
        rg = np.array([np.median(Dg[f][Dg[f] > 0.05]) for f in range(Fd)], np.float32)
        ratio = np.clip(
            rl / np.clip(rg, 1e-6, None), 1e-6, None
        )  # per-frame cross-model scale
        s = ratio / np.exp(
            np.mean(np.log(ratio))
        )  # geomean-normalised -> mean correction = 1
        Dg_lprof = Dg * s[:, None, None]

        np.savez_compressed(
            OUT["oracle"] / SUB / (clip.clip_id + ".npz"),
            depth=Dg_oracle.astype(np.float32),
        )
        np.savez_compressed(
            OUT["lprof"] / SUB / (clip.clip_id + ".npz"),
            depth=Dg_lprof.astype(np.float32),
        )
    print("[reanchor] done:", {k: str(v) for k, v in OUT.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
