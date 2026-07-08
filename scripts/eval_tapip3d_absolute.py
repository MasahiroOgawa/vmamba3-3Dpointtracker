"""Run TAPIP3D inference on TAPVid-3D minival and evaluate with absolute metric.

Saves per-clip 3D predictions as npz files, then evaluates with our
absolute-metric evaluator (scaling="none", fixed-metre thresholds).

Usage:
    cd /home/mas/proj/study/largescale3Dreconstruction_using_SSM
    TAPIP3D=/home/mas/proj/study/TAPIP3D
    VENV=$TAPIP3D/.venv
    CU13=$VENV/lib/python3.11/site-packages/nvidia/cu13
    LD_LIBRARY_PATH=$VENV/lib/python3.11/site-packages/torch/lib:$CU13/lib:$LD_LIBRARY_PATH \\
    PYTHONPATH=$TAPIP3D:$PYTHONPATH \\
    $VENV/bin/python scripts/eval_tapip3d_absolute.py \\
        --subsets drivetrack pstudio adt \\
        --out-dir result/$(date +%Y%m%d-%H%M)_tapip3d_absolute_eval
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

TAPIP3D = Path("/home/mas/proj/study/TAPIP3D")
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TAPIP3D))

import models
from datasets.base_dataset import BaseDataset
from datasets.datatypes import SliceData
from utils.inference_utils import _inference_with_grid

# import our absolute metric evaluator
sys.path.insert(0, str(REPO_ROOT / "src"))
from mamba3_tracker.eval.tapvid3d_eval import (
    aggregate,
    compute_clip_metrics,
    compute_clip_metrics_absolute,
)

ANNO_ROOT = REPO_ROOT / "result" / "tapip3d_annotations"
TAPVID_ROOT = Path("~/data/tapvid3d").expanduser()
CKPT = TAPIP3D / "checkpoints" / "tapip3d_final.pth"

SUBSET_CONFIG_NAMES = {
    "drivetrack": "eval/drivetrack_da3_minival",
    "pstudio": "eval/pstudio_da3_minival",
    "adt": "eval/adt_da3_minival",
}


def build_dataset(subset: str) -> BaseDataset:
    cfg = BaseDataset.load_config(SUBSET_CONFIG_NAMES[subset])
    return BaseDataset.from_config(cfg)


@torch.no_grad()
def run_subset(
    model, device, subset: str, pred_dir: Path, overwrite: bool
) -> list[dict]:
    pred_dir.mkdir(parents=True, exist_ok=True)
    ds = build_dataset(subset)
    loader = DataLoader(
        ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=SliceData.collate
    )

    clip_results = []
    for sample in tqdm(loader, desc=f"TAPIP3D {subset}"):
        seq_name = sample.seq_name[0]  # e.g. "tapvid3d_xxxx.npz" or "basketball_5.npz"
        clip_id = seq_name.removesuffix(".npz")  # strip extension
        out_path = pred_dir / f"{clip_id}.npz"

        # cache GT on CPU before moving sample to device
        gt_s = sample.with_annot_mode("gt")
        gt_xyz = gt_s.gt_trajs_3d[0].numpy().astype(np.float32)  # (T, N, 3)
        gt_vis = gt_s.visibs[0].numpy().astype(bool)  # (T, N)
        orig_HW = gt_s.orig_resolution[0].numpy()  # (2,) [H, W]
        K_np = (
            gt_s.intrinsics[0, 0].numpy().astype(np.float64)
        )  # (3,3) at processed res
        orig_H, orig_W = int(orig_HW[0]), int(orig_HW[1])
        H_proc = int(sample.rgbs.shape[-2])
        W_proc = int(sample.rgbs.shape[-1])
        # Undo the processed-resolution scale and apply isotropic 256-px scale
        # (matches TAPIP3D evaluation/metrics.py gt_resized_intrinsics_iso formula)
        scaling_factor = 256.0 / min(orig_H, orig_W)
        sx = ((orig_W - 1) / (W_proc - 1)) * scaling_factor
        sy = ((orig_H - 1) / (H_proc - 1)) * scaling_factor
        K_256 = K_np.copy()
        K_256[0, 0] *= sx
        K_256[0, 2] *= sx
        K_256[1, 1] *= sy
        K_256[1, 2] *= sy
        intr_params = np.array([K_256[0, 0], K_256[1, 1], K_256[0, 2], K_256[1, 2]])

        if out_path.exists() and not overwrite:
            npz = np.load(out_path)
            tracks = npz["tracks_XYZ"]  # (T, N, 3)
            vis = npz["visibility"]  # (T, N)
        else:
            sample = sample.to(device)
            est = sample.with_annot_mode("est")

            preds, _ = _inference_with_grid(
                grid_size=8,
                model=model,
                video=est.rgbs,
                depths=est.depths,
                num_iters=6,
                query_point=est.query_point,
                intrinsics=est.intrinsics,
                extrinsics=est.extrinsics,
                flags=est.flags,
                depth_roi=est.depth_roi,
            )
            # preds.coords: (1, T, N, 3) camera-frame metric XYZ
            tracks = preds.coords[0].cpu().numpy().astype(np.float32)  # (T, N, 3)
            vis = preds.visibs[0].cpu().numpy() > 0.5  # (T, N)
            np.savez_compressed(out_path, tracks_XYZ=tracks, visibility=vis)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        F_ = min(tracks.shape[0], gt_xyz.shape[0])
        N_ = min(tracks.shape[1], gt_xyz.shape[1])
        g = gt_xyz[:F_, :N_]  # (T, N, 3)
        gv = gt_vis[:F_, :N_]  # (T, N)
        t = tracks[:F_, :N_]  # (T, N, 3)
        pv = vis[:F_, :N_]  # (T, N)

        # metric functions expect pred in (N, T, 3) / (N, T)
        t_NT3 = np.transpose(t, (1, 0, 2))  # (N, T, 3)
        pv_NT = np.transpose(pv, (1, 0))  # (N, T)
        med = compute_clip_metrics(g, gv, t_NT3, pv_NT, intr_params)
        abs_ = compute_clip_metrics_absolute(g, gv, t_NT3, pv_NT, intr_params)

        # 3D error in metres over visible GT points (use TN3 formats)
        mask = (gv > 0.5) & np.isfinite(t).all(-1) & np.isfinite(g).all(-1)
        if mask.any():
            d = np.linalg.norm((t - g)[mask], axis=-1)
            err_mean, err_med = float(d.mean()), float(np.median(d))
        else:
            err_mean = err_med = float("nan")

        clip_results.append(
            {
                "clip_id": clip_id,
                "average_jaccard": med["average_jaccard"],
                "metric_average_jaccard": abs_["metric_average_jaccard"],
                "metric_err_mean_m": err_mean,
                "metric_err_median_m": err_med,
            }
        )

    return clip_results


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subsets", nargs="+", default=["drivetrack", "pstudio", "adt"])
    ap.add_argument(
        "--out-dir", type=Path, default=REPO_ROOT / "result" / "tapip3d_absolute_eval"
    )
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[tapip3d-abs] Loading model from {CKPT}")
    model, _ = models.from_pretrained(CKPT)
    model = model.to(device).eval()

    pred_dir = args.out_dir / "predictions"
    results_dir = args.out_dir / "metric_results"
    results_dir.mkdir(parents=True, exist_ok=True)

    all_clips = []
    for subset in args.subsets:
        sub_pred_dir = pred_dir / subset
        clips = run_subset(model, device, subset, sub_pred_dir, args.overwrite)
        (results_dir / f"{subset}.json").write_text(json.dumps(clips, indent=2))
        all_clips.extend(clips)
        import numpy as np

        med_aj = np.mean([c["average_jaccard"] for c in clips]) * 100
        metric_aj = np.mean([c["metric_average_jaccard"] for c in clips]) * 100
        err_mean = np.mean([c["metric_err_mean_m"] for c in clips])
        print(
            f"[tapip3d-abs] {subset}: median-AJ={med_aj:.2f}%, metric-AJ={metric_aj:.2f}%, err_mean={err_mean:.2f}m"
        )

    print("[tapip3d-abs] done.")


if __name__ == "__main__":
    main()
