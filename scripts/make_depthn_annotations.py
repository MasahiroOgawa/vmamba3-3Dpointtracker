"""Extract depth_preds from TAPVid-3D minival NPZ files.

TAPVid-3D NPZ files contain a built-in `depth_preds` field — metric depth maps
at original image resolution, higher quality than our per-frame DA3 estimates.

This script creates TWO outputs from those depths (minival only):

1. Our eval format (for eval_metric3d.py with --da3-depth-root):
     result/tapvid3d_depthn/<subset>/<clip_id>.npz   (flat file)
     keys: depth (T,H,W) float32

2. TAPIP3D annotation H5 format (for TAPIP3D evaluator):
     result/tapip3d_annotations/<subset>_depthn_minival/da3/<seq_id>.h5
     keys: depths, intrinsics, extrinsics

Also writes the TAPIP3D annotation YAML configs to TAPIP3D/configs/annotation/.

Usage:
    uv run python scripts/make_depthn_annotations.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
TAPIP3D_ROOT = Path("/home/mas/proj/study/TAPIP3D")
TAPVID_ROOT = Path("~/data/tapvid3d").expanduser()

# outputs
# eval_metric3d.py reads flat <root>/<subset>/<clip_id>.npz with key "depth"
OUR_DEPTH_ROOT = REPO_ROOT / "result" / "tapvid3d_depthn"
TAPIP3D_ANNO_ROOT = REPO_ROOT / "result" / "tapip3d_annotations"
TAPIP3D_ANNO_CFG_ROOT = TAPIP3D_ROOT / "configs" / "annotation"

sys.path.insert(0, str(TAPIP3D_ROOT))
from evaluation.tapvid3d_splits import MINIVAL_FILES


def build_K(fx: float, fy: float, cx: float, cy: float, T: int) -> np.ndarray:
    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)
    return np.tile(K[None], (T, 1, 1))  # (T, 3, 3)


def identity_extrinsics(T: int) -> np.ndarray:
    return np.tile(np.eye(4, dtype=np.float32)[None], (T, 1, 1))  # (T, 4, 4)


def process_subset(subset: str) -> None:
    minival_fnames = sorted(MINIVAL_FILES[subset])
    our_dir = OUR_DEPTH_ROOT / subset  # flat NPZ files land here
    h5_dir = TAPIP3D_ANNO_ROOT / f"{subset}_depthn_minival" / "da3"
    our_dir.mkdir(parents=True, exist_ok=True)
    h5_dir.mkdir(parents=True, exist_ok=True)
    print(f"[{subset}] {len(minival_fnames)} minival clips")

    for seq_id, fname in enumerate(minival_fnames):
        clip_id = fname.removesuffix(".npz")
        confirm = h5_dir / f"{seq_id}.confirm"
        our_npz = our_dir / (clip_id + ".npz")  # flat file, not subdirectory

        if confirm.exists() and our_npz.exists():
            print(
                f"  [{subset}] {seq_id:3d}/{len(minival_fnames)} {clip_id[:40]} — skip (cached)"
            )
            continue

        # allow_pickle=True required for images_jpeg_bytes (object array);
        # files are trusted local TAPVid-3D dataset files, not from the network.
        npz = np.load(TAPVID_ROOT / subset / fname, allow_pickle=True)
        if "depth_preds" not in npz:
            if seq_id == 0:
                print(f"  [{subset}] no depth_preds field — skipping subset")
            return
        depth = np.asarray(npz["depth_preds"], dtype=np.float32)  # (T, H, W)
        T, H, W = depth.shape
        fx, fy, cx, cy = npz["fx_fy_cx_cy"].astype(np.float32)
        K = build_K(fx, fy, cx, cy, T)  # (T, 3, 3) at depth (=RGB) resolution
        ext = identity_extrinsics(T)  # (T, 4, 4)

        # --- Our eval format (flat <subset>/<clip_id>.npz, key "depth") ---
        np.savez_compressed(our_npz, depth=depth)

        # --- TAPIP3D H5 format ---
        h5_path = h5_dir / f"{seq_id}.h5"
        with h5py.File(h5_path, "w") as f:
            f.create_dataset("depths", data=depth, compression="gzip")
            f.create_dataset("intrinsics", data=K, compression="gzip")
            f.create_dataset("extrinsics", data=ext, compression="gzip")
        confirm.touch()

        if seq_id % 10 == 0 or seq_id == len(minival_fnames) - 1:
            print(
                f"  [{subset}] {seq_id:3d}/{len(minival_fnames)} {clip_id[:40]} "
                f"depth{depth.shape} fx={fx:.1f}"
            )

    print(f"[{subset}] done → {our_dir}  |  {h5_dir}")


def write_tapip3d_configs(subsets_with_depth: list[str]) -> None:
    """Write TAPIP3D annotation + dataset configs for subsets that have depth_preds."""
    for subset in subsets_with_depth:
        # 1. annotation config (absolute path to H5 dir)
        h5_dir = TAPIP3D_ANNO_ROOT / f"{subset}_depthn_minival" / "da3"
        anno_dir = TAPIP3D_ANNO_CFG_ROOT / f"{subset}_depthn_minival"
        anno_dir.mkdir(exist_ok=True)
        (anno_dir / "da3.yaml").write_text(
            f"overrides:\n"
            f"  depths: {h5_dir}\n"
            f"  extrinsics: {h5_dir}\n"
            f"  intrinsics: {h5_dir}\n"
        )

        # 2. dataset eval config (same structure as *_da3_minival.yaml)
        ref_cfg = (
            TAPIP3D_ROOT / "configs" / "dataset" / "eval" / f"{subset}_da3_minival.yaml"
        )
        ref_text = ref_cfg.read_text()
        new_text = ref_text.replace(
            f"override_anno: {subset}_da3_minival/da3",
            f"override_anno: {subset}_depthn_minival/da3",
        )
        out_cfg = (
            TAPIP3D_ROOT
            / "configs"
            / "dataset"
            / "eval"
            / f"{subset}_depthn_minival.yaml"
        )
        out_cfg.write_text(new_text)

    # 3. top-level TAPIP3D eval config (only subsets with depth_preds)
    dataset_defaults = "\n".join(
        f"  - dataset/eval/{s}_depthn_minival@test_datasets.{s}_depthn_minival"
        for s in subsets_with_depth
    )
    top = TAPIP3D_ROOT / "configs" / "tapip3d_depthn_minival_eval.yaml"
    top.write_text(
        f"# TAPVid-3D MINIVAL eval with built-in depth_preds ({', '.join(subsets_with_depth)}).\n"
        "# Note: drivetrack/pstudio minival NPZ files don't contain depth_preds.\n"
        "defaults:\n"
        "  - tapip3d\n"
        f"{dataset_defaults}\n"
    )
    print(f"Wrote TAPIP3D configs → {TAPIP3D_ROOT}/configs/")


if __name__ == "__main__":
    processed = []
    for subset in ["adt", "drivetrack", "pstudio"]:
        before = (
            len(list((OUR_DEPTH_ROOT / subset).glob("*.npz")))
            if (OUR_DEPTH_ROOT / subset).exists()
            else 0
        )
        process_subset(subset)
        after = (
            len(list((OUR_DEPTH_ROOT / subset).glob("*.npz")))
            if (OUR_DEPTH_ROOT / subset).exists()
            else 0
        )
        if after > 0:
            processed.append(subset)
    write_tapip3d_configs(processed)
    print("\nDone. Next steps:")
    print("  # SEA-RAFT + depthn eval:")
    print("  uv run python scripts/eval_metric3d.py --method searaft \\")
    print(f"    --da3-depth-root result/tapvid3d_depthn --split minival \\")
    print(f"    --out-dir result/$(date +%Y%m%d-%H%M)_metric3d_searaft_depthn_minival")
    print()
    print("  # TAPIP3D + depthn eval:")
    print("  See scripts/run_tapip3d_depthn_eval.sh")
