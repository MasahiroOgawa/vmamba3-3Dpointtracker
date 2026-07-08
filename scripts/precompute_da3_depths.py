"""Pre-compute DA3-Large metric-depth maps for every TAPVid-3D clip.

v31 uses these depths as a frozen scale source: at train/eval time the 2D
tracker emits (u, v) per (frame, query), the cached depth is bilinearly
sampled at that pixel, and the (X, Y, Z) is recovered via pinhole
unprojection with the clip's intrinsics.

Saving once per clip avoids ~2-3 s of DA3 forward in every training step
(would add ~24h to a 30k-step v31 run). One-time cost is ~45-90 min on
a single GPU; disk ~5 MB per clip × 4569 clips = ~22 GB total.

Output layout (one .npz per clip — uint16 quantized for ~4× disk savings
vs float32; per-clip min/max gives ~0.1 mm precision at 10 m range):
    ~/data/tapvid3d_da3/<subset>/<clip_stem>.npz
        depth_q      : (F, Hd, Wd) uint16   — quantized depth (0..65535)
        d_min        : float32              — per-clip min metres (skips Z=0)
        d_max        : float32              — per-clip max metres
        process_res  : int                  — DA3 working resolution (504)
        clip_path    : str                  — original .npz path
        h, w         : int, int             — original frame H, W

Decode: depth = d_min + (depth_q / 65535) * (d_max - d_min).

Run:
    uv run python scripts/precompute_da3_depths.py
    uv run python scripts/precompute_da3_depths.py --clips 3 --dry-run
"""

from __future__ import annotations

import argparse
import io
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.modules.setdefault("moviepy.editor", types.ModuleType("moviepy.editor"))
sys.path.insert(0, "third_party/depth-anything-3/src")

from depth_anything_3.api import DepthAnything3  # noqa: E402

from mamba3_tracker.data.dataset import official_train_test_split  # noqa: E402


def _which_subset(path: Path) -> str:
    for s in ("pstudio", "drivetrack", "adt"):
        if f"/{s}/" in str(path) or s in path.parts:
            return s
    return "unknown"


def _decode_all_frames(jpeg_bytes_arr: np.ndarray) -> list[np.ndarray]:
    return [
        np.asarray(Image.open(io.BytesIO(bytes(jb))).convert("RGB"))
        for jb in jpeg_bytes_arr
    ]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", type=Path, default=Path.home() / "data")
    ap.add_argument("--out-root", type=Path, default=Path.home() / "data" / "tapvid3d_da3")
    ap.add_argument("--model-name", default="da3metric-large")
    ap.add_argument("--process-res", type=int, default=504)
    ap.add_argument("--subsets", nargs="+", default=["pstudio", "drivetrack", "adt"])
    ap.add_argument("--clips", type=int, default=0,
                    help="Limit total clips for smoke testing (0 = all).")
    ap.add_argument("--chunk-frames", type=int, default=16,
                    help="DA3-Large at 504² won't fit a full ADT clip (300 frames) in "
                         "<=12 GiB. Process this many frames per DA3 forward and "
                         "concatenate depth outputs along the frame axis.")
    ap.add_argument("--skip-existing", action="store_true", default=True)
    ap.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[da3-precompute] device={device}, model={args.model_name}, "
          f"process_res={args.process_res}", flush=True)

    train_clips, test_clips = official_train_test_split(
        args.data_root, subsets=args.subsets,
    )
    all_clips: list[Path] = sorted({*train_clips, *test_clips})
    if args.clips > 0:
        all_clips = all_clips[: args.clips]
    print(f"[da3-precompute] {len(all_clips)} clips "
          f"({len(train_clips)} train + {len(test_clips)} test, deduped)", flush=True)

    print("[da3-precompute] loading DA3 ...", flush=True)
    model = DepthAnything3.from_pretrained(f"depth-anything/{args.model_name}").to(device)
    model.device = device
    model.eval()
    print("[da3-precompute] loaded", flush=True)

    args.out_root.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    n_done = n_skipped = n_failed = 0

    for i, clip_path in enumerate(all_clips):
        subset = _which_subset(clip_path)
        out_dir = args.out_root / subset
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / (clip_path.stem + ".npz")
        if args.skip_existing and out_path.exists():
            n_skipped += 1
            continue
        try:
            with np.load(clip_path, allow_pickle=True) as d:
                if "images_jpeg_bytes" not in d.files:
                    print(f"  [{i:5d}] {subset:<11} {clip_path.name}: SKIP labels-only", flush=True)
                    n_skipped += 1
                    continue
                rgbs = _decode_all_frames(d["images_jpeg_bytes"])
            H, W = rgbs[0].shape[:2]
            chunk = max(1, int(args.chunk_frames))
            depth_chunks: list[np.ndarray] = []
            with torch.inference_mode():
                for cs in range(0, len(rgbs), chunk):
                    ce = min(cs + chunk, len(rgbs))
                    pred = model.inference(
                        rgbs[cs:ce], process_res=args.process_res, export_format="mini_npz",
                    )
                    depth_chunks.append(np.asarray(pred.depth, dtype=np.float32))
                    del pred
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            depth = np.concatenate(depth_chunks, axis=0)            # (F, Hd, Wd)
            # uint16 quantization with per-clip min/max. Ignore Z<=0 when
            # computing the range so a few stray bad pixels don't blow up
            # the dequantization scale (DA3 occasionally emits negatives).
            valid = depth[depth > 0]
            if valid.size == 0:
                print(f"  [{i:5d}] {subset:<11} {clip_path.name}: SKIP no positive depth",
                      flush=True)
                n_skipped += 1
                continue
            d_min = float(valid.min())
            d_max = float(valid.max())
            scale = max(d_max - d_min, 1e-6)
            depth_q = np.clip(
                np.round((depth - d_min) / scale * 65535.0), 0, 65535,
            ).astype(np.uint16)
            np.savez_compressed(
                out_path,
                depth_q=depth_q,
                d_min=np.float32(d_min), d_max=np.float32(d_max),
                process_res=np.int32(args.process_res),
                clip_path=str(clip_path),
                h=np.int32(H), w=np.int32(W),
            )
            n_done += 1
            if (i + 1) % 10 == 0 or i == 0:
                elapsed = time.time() - t_start
                rate = max(n_done, 1) / max(elapsed, 1e-3)
                eta = (len(all_clips) - i - 1) / max(rate, 1e-6)
                print(f"  [{i+1:5d}/{len(all_clips)}] {subset:<11} {clip_path.name}: "
                      f"depth={depth.shape} done={n_done} skipped={n_skipped} fail={n_failed} "
                      f"rate={rate:.2f}/s eta={eta/60:.1f}min", flush=True)
        except Exception as e:
            print(f"  [{i:5d}] {subset:<11} {clip_path.name}: FAIL ({type(e).__name__}: {e})",
                  flush=True)
            n_failed += 1

    elapsed = time.time() - t_start
    print(f"\n[da3-precompute] DONE: {n_done} written, {n_skipped} skipped, {n_failed} failed "
          f"in {elapsed/60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
