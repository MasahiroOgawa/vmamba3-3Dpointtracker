"""Bake DA3-g raw depth.npz into TAPIP3D per-sequence HDF5 annotations.

DA3-g counterpart of TAPIP3D's make_da3_annotations_minival.py, kept in THIS
repo so no external repo is edited. Reads the DA3-g raw output written by
run_da3_raw_tapvid3d.py and writes one gzip HDF5 per minival sequence, indexed
by position in the sorted MINIVAL_FILES[subset] list (seq_id 0..49) — the layout
TAPIP3D's provider expects.

Input:  <da3g-out>/<subset>/<clip_stem>/depth.npz   (depth, intrinsics, extrinsics)
Output: <out-root>/<subset>_da3g_minival/da3/<seq_id>.h5  (depths/intrinsics/extrinsics)
        + <seq_id>.confirm sentinel

Run:
    uv run python scripts/make_da3g_annotations.py \\
      --da3g-out ~/data/tapvid3d_da3g_out --out-root ~/data/tapip3d_annotations_da3g
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import h5py
import numpy as np

from mamba3_tracker.data.tapvid3d_splits import MINIVAL_FILES

TAPVID_ROOT = Path("~/data/tapvid3d").expanduser()


def rgb_hw(npz) -> tuple[int, int]:
    b = npz["images_jpeg_bytes"][0]
    img = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)
    return img.shape[0], img.shape[1]


def scale_intrinsics(K: np.ndarray, src_hw, dst_hw) -> np.ndarray:
    """Scale per-frame 3x3 intrinsics from depth resolution to RGB resolution."""
    (sh, sw), (dh, dw) = src_hw, dst_hw
    K = K.copy().astype(np.float32)
    K[:, 0, 0] *= dw / sw
    K[:, 0, 2] *= dw / sw
    K[:, 1, 1] *= dh / sh
    K[:, 1, 2] *= dh / sh
    return K


def build_subset(subset: str, da3g_out: Path, out_root: Path, overwrite: bool) -> None:
    seqs = sorted(MINIVAL_FILES[subset])
    out_dir = out_root / f"{subset}_da3g_minival" / "da3"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[{subset}] {len(seqs)} minival seqs -> {out_dir}", flush=True)

    for seq_id, fname in enumerate(seqs):
        confirm = out_dir / f"{seq_id}.confirm"
        if confirm.exists() and not overwrite:
            continue
        stem = fname[:-4]
        da3_path = da3g_out / subset / stem / "depth.npz"
        if not da3_path.exists():
            print(f"  MISSING DA3-g output: {da3_path}", flush=True)
            continue

        # allow_pickle: official TAPVid-3D npz stores images_jpeg_bytes as object array
        d = np.load(TAPVID_ROOT / subset / fname, allow_pickle=True)
        e = np.load(da3_path)
        Hr, Wr = rgb_hw(d)
        depth = e["depth"].astype(np.float32)  # (T, Hd, Wd)
        Hd, Wd = depth.shape[1], depth.shape[2]
        intr = scale_intrinsics(e["intrinsics"], (Hd, Wd), (Hr, Wr))
        extr = e["extrinsics"].astype(np.float32)  # (T, 4, 4)
        T = int(d["visibility"].shape[0])
        d.close()
        e.close()

        assert depth.shape[0] == T, f"{stem}: DA3-g T={depth.shape[0]} != gt T={T}"
        with h5py.File(out_dir / f"{seq_id}.h5", "w") as f:
            for k, v in dict(depths=depth, intrinsics=intr, extrinsics=extr).items():
                f.create_dataset(k, data=v, compression="gzip")
        confirm.touch()
        if seq_id % 10 == 0:
            print(f"  [{subset}] {seq_id}/{len(seqs)} {stem} depth{depth.shape} rgb=({Hr},{Wr})",
                  flush=True)
    print(f"[{subset}] done.", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subsets", nargs="+", default=["drivetrack", "pstudio", "adt"])
    ap.add_argument("--da3g-out", type=Path, default=Path("~/data/tapvid3d_da3g_out"))
    ap.add_argument("--out-root", type=Path, default=Path("~/data/tapip3d_annotations_da3g"))
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    for s in a.subsets:
        build_subset(s, a.da3g_out.expanduser(), a.out_root.expanduser(), a.overwrite)


if __name__ == "__main__":
    main()
