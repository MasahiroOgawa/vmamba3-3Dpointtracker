"""Run MegaSAM on TAPVid-3D minival clips and save TAPIP3D H5 annotations.

Runs on drivetrack and pstudio by default; pass --subsets adt to run on ADT.
Use --frame-stride 2 for ADT (300 frames → 150) to avoid DROID-SLAM OOM on 12 GB GPUs.

Outputs:
    result/tapip3d_annotations/<subset>_megasam[_s<stride>]_minival/megasam/<seq_id>.h5
    keys: depths (T,H,W) float32, intrinsics (T,3,3), extrinsics (T,4,4)

Usage:
    uv run python scripts/make_megasam_annotations.py [--subsets drivetrack pstudio] [--start 0] [--end -1]
    uv run python scripts/make_megasam_annotations.py --subsets adt --frame-stride 2
"""

from __future__ import annotations

import argparse
import io
import math
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
TAPIP3D_ROOT = Path("/home/mas/proj/study/TAPIP3D")
MEGASAM_ROOT = TAPIP3D_ROOT / "third_party" / "megasam"
TAPVID_ROOT = Path("~/data/tapvid3d").expanduser()

TAPIP3D_ANNO_ROOT = REPO_ROOT / "result" / "tapip3d_annotations"
TAPIP3D_ANNO_CFG_ROOT = TAPIP3D_ROOT / "configs" / "annotation"

VENV_PYTHON = str(TAPIP3D_ROOT / ".venv" / "bin" / "python")
CU13 = str(
    TAPIP3D_ROOT / ".venv" / "lib" / "python3.11" / "site-packages" / "nvidia" / "cu13"
)

sys.path.insert(0, str(TAPIP3D_ROOT))
from evaluation.tapvid3d_splits import MINIVAL_FILES  # noqa: E402


def fov_x_degrees(fx: float, W: int) -> float:
    return math.degrees(2.0 * math.atan(W / (2.0 * fx)))


def run_megasam(
    frames_dir: Path, output_npz: Path, fov_x: float, resolution: int = 196608
) -> bool:
    """Run MegaSAM inference.py on a directory of JPEG frames."""
    inference_script = MEGASAM_ROOT / "inference.py"
    env = os.environ.copy()
    venv_bin = str(TAPIP3D_ROOT / ".venv" / "bin")
    env["PATH"] = venv_bin + ":" + env.get("PATH", "")  # so subprocesses find "python"
    env["CUDA_HOME"] = CU13
    env["LD_LIBRARY_PATH"] = ":".join(
        [
            str(
                TAPIP3D_ROOT
                / ".venv"
                / "lib"
                / "python3.11"
                / "site-packages"
                / "torch"
                / "lib"
            ),
            CU13 + "/lib",
            env.get("LD_LIBRARY_PATH", ""),
        ]
    )
    env["PYTHONPATH"] = str(MEGASAM_ROOT)

    cmd = [
        VENV_PYTHON,
        str(inference_script),
        "--input_dir",
        str(frames_dir),
        "--output_path",
        str(output_npz),
        "--fov_x",
        f"{fov_x:.4f}",
        "--resolution",
        str(resolution),
        "--depth_model",
        "dav1",
    ]
    result = subprocess.run(cmd, env=env, capture_output=False)
    if result.returncode != 0:
        print(f"    MegaSAM FAILED (exit {result.returncode})")
        return False
    return True


def extract_frames(npz_path: Path, out_dir: Path, stride: int = 1) -> tuple[int, int]:
    """Extract JPEG frames from TAPVid-3D NPZ to out_dir. Returns (W, H).

    stride=2 halves the frame count (every other frame), which avoids DROID-SLAM
    OOM on 12 GB GPUs when clips have ≥300 frames (e.g. ADT).
    """
    # allow_pickle=True: images_jpeg_bytes is a Python-bytes object array (trusted local files).
    npz = np.load(npz_path, allow_pickle=True)
    imgs = npz["images_jpeg_bytes"][::stride]
    out_dir.mkdir(parents=True, exist_ok=True)
    W, H = 0, 0
    for i, jpeg_bytes in enumerate(imgs):
        img = Image.open(io.BytesIO(bytes(jpeg_bytes)))
        W, H = img.size
        img.save(out_dir / f"{i:06d}.jpg", quality=95)
    return W, H


def megasam_npz_to_h5(megasam_npz_path: Path, h5_path: Path) -> bool:
    """Convert MegaSAM output NPZ to TAPIP3D H5 annotation."""
    try:
        npz = np.load(megasam_npz_path)
    except Exception as e:
        print(f"    Failed to load MegaSAM NPZ: {e}")
        return False

    depths = npz["depths"].astype(np.float32)  # (T, H, W)
    intrinsic = npz["intrinsic"]  # (3, 3) at output resolution
    cam_c2w = npz["cam_c2w"].astype(np.float32)  # (T, 4, 4) camera-to-world

    T = depths.shape[0]
    # w2c = inv(c2w) — TAPIP3D uses w2c convention for extrinsics
    extrinsics = np.linalg.inv(cam_c2w).astype(np.float32)  # (T, 4, 4)
    intrinsics = np.tile(intrinsic.astype(np.float32)[None], (T, 1, 1))  # (T, 3, 3)

    with h5py.File(h5_path, "w") as f:
        f.create_dataset("depths", data=depths, compression="gzip")
        f.create_dataset("intrinsics", data=intrinsics, compression="gzip")
        f.create_dataset("extrinsics", data=extrinsics, compression="gzip")
    return True


def process_clip(
    subset: str, seq_id: int, fname: str, h5_dir: Path, frame_stride: int = 1
) -> bool:
    """Run MegaSAM on one clip and save H5. Returns True if newly processed."""
    confirm = h5_dir / f"{seq_id}.confirm"
    h5_path = h5_dir / f"{seq_id}.h5"

    if confirm.exists() and h5_path.exists():
        return False  # cached

    npz_path = TAPVID_ROOT / subset / fname
    # allow_pickle=True: images_jpeg_bytes is a Python-bytes object array.
    # Files are trusted local TAPVid-3D dataset files, not from the network.
    npz_meta = np.load(npz_path, allow_pickle=True)
    fx, fy, cx, cy = npz_meta["fx_fy_cx_cy"].astype(float)

    # Extract one image to get W, H
    img0 = Image.open(io.BytesIO(bytes(npz_meta["images_jpeg_bytes"][0])))
    W, H = img0.size

    fov_x = fov_x_degrees(fx, W)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        frames_dir = tmp_path / "frames"
        extract_frames(npz_path, frames_dir, stride=frame_stride)

        megasam_out = tmp_path / "output.npz"
        t0 = time.time()
        ok = run_megasam(frames_dir, megasam_out, fov_x)
        elapsed = time.time() - t0

        if not ok or not megasam_out.exists():
            print(f"    [{subset}] seq {seq_id}: MegaSAM failed after {elapsed:.0f}s")
            return False

        print(f"    [{subset}] seq {seq_id}: MegaSAM OK in {elapsed:.0f}s")
        if not megasam_npz_to_h5(megasam_out, h5_path):
            return False

    confirm.touch()
    return True


def write_tapip3d_configs(subsets: list[str], stride_tag: str = "") -> None:
    """Write TAPIP3D annotation + dataset configs for MegaSAM annotations."""
    tag = f"megasam{stride_tag}"
    for subset in subsets:
        h5_dir = TAPIP3D_ANNO_ROOT / f"{subset}_{tag}_minival" / "megasam"
        anno_dir = TAPIP3D_ANNO_CFG_ROOT / f"{subset}_{tag}_minival"
        anno_dir.mkdir(parents=True, exist_ok=True)
        (anno_dir / "megasam.yaml").write_text(
            f"overrides:\n"
            f"  depths: {h5_dir}\n"
            f"  extrinsics: {h5_dir}\n"
            f"  intrinsics: {h5_dir}\n"
        )

        ref_cfg = (
            TAPIP3D_ROOT / "configs" / "dataset" / "eval" / f"{subset}_da3_minival.yaml"
        )
        if ref_cfg.exists():
            new_text = ref_cfg.read_text().replace(
                f"override_anno: {subset}_da3_minival/da3",
                f"override_anno: {subset}_{tag}_minival/megasam",
            )
            out_cfg = (
                TAPIP3D_ROOT
                / "configs"
                / "dataset"
                / "eval"
                / f"{subset}_{tag}_minival.yaml"
            )
            out_cfg.write_text(new_text)

    dataset_defaults = "\n".join(
        f"  - dataset/eval/{s}_{tag}_minival@test_datasets.{s}_{tag}_minival"
        for s in subsets
    )
    top = TAPIP3D_ROOT / "configs" / f"tapip3d_{tag}_minival_eval.yaml"
    top.write_text(
        f"# TAPVid-3D MINIVAL eval with MegaSAM depths ({', '.join(subsets)}).\n"
        "defaults:\n"
        "  - tapip3d\n"
        f"{dataset_defaults}\n"
    )
    print(f"Wrote TAPIP3D configs → {TAPIP3D_ROOT}/configs/")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--subsets",
        nargs="+",
        default=["drivetrack", "pstudio"],
        choices=["adt", "drivetrack", "pstudio"],
    )
    ap.add_argument("--start", type=int, default=0, help="start clip index")
    ap.add_argument("--end", type=int, default=-1, help="end clip index (-1=all)")
    ap.add_argument(
        "--frame-stride",
        type=int,
        default=1,
        help="subsample frames: 2 = every other frame (use for ADT to avoid OOM)",
    )
    args = ap.parse_args()

    stride_tag = f"_s{args.frame_stride}" if args.frame_stride != 1 else ""

    processed_subsets = []
    for subset in args.subsets:
        h5_dir = TAPIP3D_ANNO_ROOT / f"{subset}_megasam{stride_tag}_minival" / "megasam"
        h5_dir.mkdir(parents=True, exist_ok=True)

        clips = sorted(MINIVAL_FILES[subset])
        end = len(clips) if args.end < 0 else min(args.end, len(clips))
        clips_to_run = list(enumerate(clips))[args.start : end]

        print(
            f"\n[{subset}] {len(clips_to_run)} clips (indices {args.start}–{end - 1})"
        )
        n_ok, n_skip, n_fail = 0, 0, 0
        for seq_id, fname in clips_to_run:
            clip_id = fname.removesuffix(".npz")
            confirm = h5_dir / f"{seq_id}.confirm"
            if confirm.exists():
                n_skip += 1
                continue
            print(f"  [{subset}] {seq_id:3d}/{len(clips)} {clip_id[:50]}")
            ok = process_clip(
                subset, seq_id, fname, h5_dir, frame_stride=args.frame_stride
            )
            if ok:
                n_ok += 1
            else:
                n_fail += 1

        print(
            f"[{subset}] done: {n_ok} new, {n_skip} cached, {n_fail} failed → {h5_dir}"
        )
        n_done = len(list(h5_dir.glob("*.confirm")))
        if n_done > 0:
            processed_subsets.append(subset)

    if processed_subsets:
        write_tapip3d_configs(processed_subsets, stride_tag=stride_tag)
    else:
        print("No clips processed — no configs written.")


if __name__ == "__main__":
    main()
