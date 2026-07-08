"""Create hybrid MegaSAM H5: MegaSAM depths+intrinsics, GT extrinsics (GT coordinate frame).

This fixes the coordinate frame mismatch in TAPIP3D+MegaSAM evaluation:
MegaSAM extrinsics are in MegaSAM's world frame (cam0=identity), not TAPVid-3D's frame.
By replacing extrinsics with GT w2c (first T_msam GT frames), TAPIP3D can evaluate
in the correct coordinate frame with median-AJ.
"""

import h5py
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, "/home/mas/proj/study/TAPIP3D")
from evaluation.tapvid3d_splits import MINIVAL_FILES

DATA_ROOT = Path("/home/mas/data/tapvid3d")
ANNO_ROOT = Path(
    "/home/mas/proj/study/largescale3Dreconstruction_using_SSM/result/tapip3d_annotations"
)
TAPIP3D_ROOT = Path("/home/mas/proj/study/TAPIP3D")


def make_hybrid_h5(subset: str):
    src_dir = ANNO_ROOT / f"{subset}_megasam_minival" / "megasam"
    dst_dir = ANNO_ROOT / f"{subset}_megasam_gtpose_minival" / "megasam"
    dst_dir.mkdir(parents=True, exist_ok=True)

    clips = sorted(MINIVAL_FILES[subset])
    n_done = 0
    for seq_id, fname in enumerate(clips):
        src_h5 = src_dir / f"{seq_id}.h5"
        dst_h5 = dst_dir / f"{seq_id}.h5"
        confirm = dst_dir / f"{seq_id}.confirm"
        if confirm.exists():
            continue
        if not src_h5.exists():
            print(f"  [{subset}] {seq_id}: src H5 missing — skip")
            continue

        # Load MegaSAM depths + intrinsics
        with h5py.File(src_h5, "r") as f:
            depths = f["depths"][:]  # (T_msam, H, W)
            intrinsics = f["intrinsics"][:]  # (T_msam, 3, 3)

        T_msam = depths.shape[0]

        # Load GT extrinsics_w2c (T_gt, 4, 4) — only drivetrack has this field
        npz_path = DATA_ROOT / subset / fname
        # allow_pickle=True required for TAPVid-3D's object-array field (images_jpeg_bytes);
        # these are local trusted dataset files.
        with np.load(npz_path, allow_pickle=True) as d:
            if "extrinsics_w2c" not in d:
                print(f"  [{subset}] {seq_id}: no GT extrinsics — skip")
                continue
            gt_extr = d["extrinsics_w2c"].astype(np.float32)  # (T_gt, 4, 4)

        T_gt = gt_extr.shape[0]
        # Use first min(T_msam, T_gt) GT frames (images are sequential from t=0)
        T_use = min(T_msam, T_gt)
        extrinsics = gt_extr[:T_use]  # (T_use, 4, 4)
        depths_use = depths[:T_use]
        intrinsics_use = intrinsics[:T_use]

        with h5py.File(dst_h5, "w") as f:
            f.create_dataset("depths", data=depths_use, compression="gzip")
            f.create_dataset("intrinsics", data=intrinsics_use, compression="gzip")
            f.create_dataset("extrinsics", data=extrinsics, compression="gzip")
        confirm.touch()
        n_done += 1

    print(f"[{subset}] {n_done} hybrid H5 created → {dst_dir}")

    # Write TAPIP3D annotation override config
    anno_cfg_dir = (
        TAPIP3D_ROOT / "configs" / "annotation" / f"{subset}_megasam_gtpose_minival"
    )
    anno_cfg_dir.mkdir(parents=True, exist_ok=True)
    anno_cfg = anno_cfg_dir / "megasam_gtpose.yaml"
    h5_path_str = str(dst_dir)
    anno_cfg.write_text(
        f"overrides:\n"
        f"  depths: {h5_path_str}\n"
        f"  extrinsics: {h5_path_str}\n"
        f"  intrinsics: {h5_path_str}\n"
    )

    # Write TAPIP3D dataset eval config
    ds_cfg_dir = TAPIP3D_ROOT / "configs" / "dataset" / "eval"
    ds_cfg_dir.mkdir(parents=True, exist_ok=True)
    ds_cfg = ds_cfg_dir / f"{subset}_megasam_gtpose_minival.yaml"
    ds_cfg.write_text(
        f"defaults:\n"
        f"  - eval_base\n\n"
        f"provider_config:\n"
        f"  name: tapvid3d\n"
        f"  stride: 1\n"
        f"  override_anno: {subset}_megasam_gtpose_minival/megasam_gtpose\n"
        f"  config:\n"
        f'    data_root: "/home/mas/data/tapvid3d"\n'
        f'    split: "minival"\n'
        f'    subset: "{subset}"\n\n'
        f"transform:\n"
        f"  - name: resize\n"
        f"    kwargs:\n"
        f"      target_hw: [384, 512]\n"
        f"  - name: filter_edge\n"
        f"    kwargs:\n"
        f"      depth_rtol: 0.3\n"
        f"      normal_tol: 10.0\n"
        f"  - name: set_roi\n"
        f"    kwargs: {{}}\n\n"
        f"query_mode: pass_through\n"
    )
    print(f"  → wrote TAPIP3D configs for {subset}_megasam_gtpose_minival")


for subset in ["drivetrack", "pstudio"]:
    make_hybrid_h5(subset)

# Write top-level eval config
cfg_path = TAPIP3D_ROOT / "configs" / "tapip3d_megasam_gtpose_minival_eval.yaml"
cfg_path.write_text(
    "# TAPVid-3D MINIVAL eval with MegaSAM depths + GT poses (drivetrack, pstudio).\n"
    "defaults:\n"
    "  - tapip3d\n"
    "  - dataset/eval/drivetrack_megasam_gtpose_minival@test_datasets.drivetrack_megasam_gtpose_minival\n"
    "  - dataset/eval/pstudio_megasam_gtpose_minival@test_datasets.pstudio_megasam_gtpose_minival\n"
)
print(f"Wrote top-level config: {cfg_path}")
