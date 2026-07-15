"""Run DepthAnything-3 on TAPVid-3D clips and save raw depth.npz per clip
(depth + intrinsics + extrinsics), the format TAPIP3D's h5 baker consumes.

Unlike precompute_da3_depths.py (which stores only uint16 depth_q for the
vmamba3 trackers), this keeps DA3's predicted intrinsics/extrinsics so TAPIP3D
can be re-evaluated with DA3-g using DA3-g's OWN predicted intrinsics — the same
methodology used for the DA3-l TAPIP3D baseline (Decision 1 = A).

Output layout (mirrors monorepo run_da3_tapvid3d.py, which the baker expects):
    <out-root>/<subset>/<clip_stem>/depth.npz
      depth      (T, Hd, Wd)  float32 metric metres
      intrinsics (T, 3, 3)    float32 at depth resolution
      extrinsics (T, 4, 4)    float32 (identity when chunked)

Usage (DA3-g, 150 minival clips):
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run python \\
      scripts/run_da3_raw_tapvid3d.py --split minival \\
      --da3-model DA3NESTED-GIANT-LARGE --chunk-size 4 \\
      --out-root ~/data/tapvid3d_da3g_out
"""

from __future__ import annotations

import argparse
import sys
import types
from pathlib import Path

import cv2
import numpy as np
import torch

# DA3 needs the moviepy stub before any import that pulls it in.
sys.modules.setdefault("moviepy.editor", types.ModuleType("moviepy.editor"))
sys.path.insert(
    0, str(Path(__file__).resolve().parent.parent / "third_party" / "depth-anything-3" / "src")
)

from depth_anything_3.api import DepthAnything3  # noqa: E402
from PIL import Image  # noqa: E402

TAPVID_ROOT = Path("~/data/tapvid3d").expanduser()


def list_seqs(subset: str, split: str) -> list[str]:
    from mamba3_tracker.data.tapvid3d_splits import FULL_EVAL_FILES, MINIVAL_FILES

    files = MINIVAL_FILES[subset] if split == "minival" else FULL_EVAL_FILES[subset]
    return sorted(files)


def decode_jpeg(b: bytes) -> np.ndarray:
    arr = np.frombuffer(b, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def run_da3_clip(da3: DepthAnything3, npz_path: Path, process_res: int, chunk_size: int) -> dict:
    """Return dict(depth, intrinsics, extrinsics) at depth resolution.

    chunk_size > 0 processes windows of that many frames (identity extrinsics),
    which is required for the 1.4 B nested-giant model at 504² on a 12 GB GPU.
    """
    npz = np.load(npz_path, allow_pickle=True)  # official TAPVid-3D; trusted local data
    frames = [Image.fromarray(decode_jpeg(b)) for b in npz["images_jpeg_bytes"]]
    npz.close()
    T = len(frames)

    def _intr(pred, n, Hd, Wd):
        if pred.intrinsics is not None:
            ki = np.asarray(pred.intrinsics, dtype=np.float32)
            return np.broadcast_to(ki[None], (n, 3, 3)).copy() if ki.ndim == 2 else ki
        f = float(max(Hd, Wd))
        K = np.array([[f, 0, Wd / 2], [0, f, Hd / 2], [0, 0, 1]], np.float32)
        return np.broadcast_to(K[None], (n, 3, 3)).copy()

    if chunk_size <= 0 or T <= chunk_size:
        pred = da3.inference(frames, process_res=process_res, export_format="mini_npz")
        depth = np.asarray(pred.depth, dtype=np.float32)
        Hd, Wd = depth.shape[1], depth.shape[2]
        ki = _intr(pred, T, Hd, Wd)
        if pred.extrinsics is not None:
            ei = np.asarray(pred.extrinsics, dtype=np.float32)
            if ei.ndim == 2:
                ei = np.broadcast_to(ei[None], (T, 4, 4)).copy()
        else:
            ei = np.tile(np.eye(4, dtype=np.float32), (T, 1, 1))
        return {"depth": depth, "intrinsics": ki, "extrinsics": ei}

    depths, intrinsics = [], []
    for start in range(0, T, chunk_size):
        chunk = frames[start : start + chunk_size]
        pred = da3.inference(chunk, process_res=process_res, export_format="mini_npz")
        d = np.asarray(pred.depth, dtype=np.float32)
        depths.append(d)
        intrinsics.append(_intr(pred, len(chunk), d.shape[1], d.shape[2]))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    depth_all = np.concatenate(depths, axis=0)
    intr_all = np.concatenate(intrinsics, axis=0)
    extr_all = np.tile(np.eye(4, dtype=np.float32), (depth_all.shape[0], 1, 1))
    return {"depth": depth_all, "intrinsics": intr_all, "extrinsics": extr_all}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default="minival", choices=["minival", "full_eval"])
    ap.add_argument("--subsets", nargs="+", default=["drivetrack", "pstudio", "adt"])
    ap.add_argument("--da3-model", default="DA3NESTED-GIANT-LARGE")
    ap.add_argument("--process-res", type=int, default=504)
    ap.add_argument("--chunk-size", type=int, default=4,
                    help="frames per DA3 call; 4 fits the nested-giant model in 12 GB")
    ap.add_argument("--out-root", type=Path, default=Path("~/data/tapvid3d_da3g_out"))
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    out_root = args.out_root.expanduser()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[da3-raw] device={device}  model={args.da3_model}  split={args.split}  out={out_root}")

    da3 = DepthAnything3.from_pretrained(f"depth-anything/{args.da3_model}").to(device)
    da3.eval()

    for subset in args.subsets:
        seqs = list_seqs(subset, args.split)
        print(f"[da3-raw] {subset}: {len(seqs)} clips", flush=True)
        for i, fname in enumerate(seqs):
            stem = fname[:-4]
            out_path = out_root / subset / stem / "depth.npz"
            if out_path.exists() and not args.overwrite:
                print(f"  [{i + 1}/{len(seqs)}] skip (exists): {stem}", flush=True)
                continue
            npz_path = TAPVID_ROOT / subset / fname
            if not npz_path.exists():
                print(f"  [{i + 1}/{len(seqs)}] MISSING: {npz_path}", flush=True)
                continue
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with torch.no_grad():
                result = run_da3_clip(da3, npz_path, args.process_res, args.chunk_size)
            np.savez_compressed(out_path, **result)
            print(f"  [{i + 1}/{len(seqs)}] {stem}: depth{result['depth'].shape}", flush=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    print("[da3-raw] done.")


if __name__ == "__main__":
    main()
