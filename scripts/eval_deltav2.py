#!/usr/bin/env python3
"""Run DELTAv2 (DenseTrack3Dv2, snap-research) on TAPVid-3D minival and
save predictions in the eval_metric3d.py --method external format.

Must be run with DELTAv2's own uv venv (torch 2.5.1/cu121, isolated from this repo):

  cd ~/proj/study/DenseTrack3Dv2
  .venv/bin/python ~/proj/study/vmamba3-3Dpointtracker/scripts/eval_deltav2.py \\
    --subsets drivetrack pstudio adt \\
    --out-dir /home/mas/data/tapvid3d_baseline_preds/deltav2_da3l

Output per clip: <out-dir>/<subset>/<clip_name>.npz
  tracks_XYZ  (F, N, 3)  float32  per-frame camera-space XYZ (matches GT convention)
  visibility  (F, N)     float32  0/1

Depth: by default the precomputed DA3-l (da3metric-large) metric depth
(/home/mas/data/tapvid3d_da3), so this is the "DELTAv2+DA3-l" data point — consistent
with the other +DA3-l baselines in the comparison figure. Override the DA3_ROOT env
var with ~/data/tapvid3d_da3nested to produce the "DELTAv2+DA3-g" (nested-giant) point. DELTAv2's per-frame camera
XYZ come from xyz = inv(K) @ [u,v,1] * depth (see convert_trajs_uvd_to_trajs_3d),
so passing the true intrinsics K as predefined_intrs is what makes the output
metric and in the same frame as the TAPVid-3D GT.
"""

import argparse
import io
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as Fn
from PIL import Image
from tqdm import tqdm

# ── DELTA repo root (must run with its venv) ─────────────────────────────────
DELTA_ROOT = Path(__file__).resolve().parent.parent / "third_party" / "DenseTrack3Dv2"
assert DELTA_ROOT.exists(), f"DenseTrack3Dv2 not found at {DELTA_ROOT}"
sys.path.insert(0, str(DELTA_ROOT.resolve()))

# canonical minival split from the main project (no external deps in that file)
_PROJ_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJ_ROOT / "src"))

from densetrack3d.models.densetrack3d.densetrack3dv2 import DenseTrack3DV2  # noqa: E402
from densetrack3d.models.predictor.predictor import Predictor3D  # noqa: E402
from densetrack3d.utils.io import load_checkpoint  # noqa: E402
from mamba3_tracker.data.tapvid3d_splits import MINIVAL_FILES  # noqa: E402

TAPVID3D_ROOT = Path("/home/mas/data/tapvid3d")
# DA3-l (da3metric-large) by default; override with DA3_ROOT env var to swap in
# DA3-g (nested-giant, ~/data/tapvid3d_da3nested) for the DA3-g re-eval.
DA3_ROOT = Path(os.environ.get("DA3_ROOT", "/home/mas/data/tapvid3d_da3"))


def load_da3_depth(subset: str, clip_name: str, F: int) -> np.ndarray:
    """Load and decode DA3 depth → float32 (F, H_da3, W_da3) in metres."""
    p = DA3_ROOT / subset / clip_name
    with np.load(p) as d:
        q = d["depth_q"].astype(np.float32)[:F]
        d_min, d_max = float(d["d_min"]), float(d["d_max"])
    return d_min + q * ((d_max - d_min) / 65535.0)


def decode_images(jpeg_bytes_arr) -> np.ndarray:
    """Decode JPEG byte array → uint8 (F, H, W, 3)."""
    return np.stack(
        [
            np.array(Image.open(io.BytesIO(bytes(b))).convert("RGB"))
            for b in jpeg_bytes_arr
        ]
    )


# Descending ladder of model-input resolutions. The DenseTrack3D pos_emb is a fixed
# sincos buffer that the model crops when input reso < the (384,512) it was built
# with (see densetrack3d.py "cut pos_emb"), so feeding a smaller interp_shape is
# equivalent to building the model at that size — it just costs less memory. Feature
# maps are computed for ALL frames at once, so memory ~ F * h * w; long clips
# (adt=300f, drivetrack up to 198f) must run at a lower rung to fit 12 GB VRAM.
INTERP_LADDER = [(384, 512), (320, 448), (320, 320), (256, 320), (224, 288), (192, 256)]


def pick_interp_idx(F: int) -> int:
    """Starting rung on the ladder for a clip of F frames (validated to fit 12 GB)."""
    if F <= 160:
        return 0
    if F <= 210:
        return 1
    return 2


@torch.inference_mode()
def infer_clip(
    predictor: Predictor3D,
    data: dict,
    subset: str,
    clip_name: str,
    device: str,
    use_fp16: bool,
    max_side: int,
    interp_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Run DELTA on one clip → (tracks_XYZ (F,N,3), visibility (F,N))."""
    predictor.interp_shape = interp_shape
    video_hwc = decode_images(data["images_jpeg_bytes"])  # (F, H, W, 3) uint8
    F, H, W, _ = video_hwc.shape

    queries_xyt = data["queries_xyt"].astype(np.float32)  # (N, 3) = (x, y, t)
    fx, fy, cx, cy = data["fx_fy_cx_cy"].astype(np.float32).tolist()

    depth_np = load_da3_depth(subset, clip_name, F)  # (F, Hd, Wd)

    # Optional isotropic downscale to bound VRAM. Metric XYZ = inv(K)@[u,v,1]*d is
    # invariant to this rescale (ray direction preserved), so it only trades a
    # little tracking quality for memory, never the coordinate frame.
    scale = min(max_side / max(H, W), 1.0) if max_side > 0 else 1.0
    if scale < 1.0:
        newW, newH = (int(W * scale) // 2) * 2, (int(H * scale) // 2) * 2
        sx, sy = newW / W, newH / H
        video_hwc = np.stack(
            [
                np.array(Image.fromarray(f).resize((newW, newH), Image.BILINEAR))
                for f in video_hwc
            ]
        )
        fx, cx, fy, cy = fx * sx, cx * sx, fy * sy, cy * sy
        queries_xyt = queries_xyt.copy()
        queries_xyt[:, 0] *= sx
        queries_xyt[:, 1] *= sy
        H, W = newH, newW

    # Tensors: video (1,F,3,H,W) in [0,255]; depth (1,F,1,H,W) metres, resized to (H,W)
    video = torch.from_numpy(video_hwc.transpose(0, 3, 1, 2)).float()[None].to(device)
    depth_t = torch.from_numpy(depth_np).unsqueeze(1)  # (F,1,Hd,Wd)
    if depth_t.shape[-2:] != (H, W):
        depth_t = Fn.interpolate(
            depth_t, size=(H, W), mode="bilinear", align_corners=False
        )
    videodepth = depth_t[None].to(device).float()  # (1,F,1,H,W)

    # Queries: (x,y,t) → DELTA wants (t,x,y) in native pixels
    queries = (
        torch.from_numpy(queries_xyt[:, [2, 0, 1]]).float()[None].to(device)
    )  # (1,N,3)

    K = torch.tensor(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], device=device
    ).float()

    backward = bool((queries_xyt[:, 2] > 0).any())
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_fp16):
        out = predictor(
            video,
            videodepth,
            queries=queries,
            grid_size=0,
            grid_query_frame=0,
            backward_tracking=backward,
            predefined_intrs=K,
        )

    xyz = out["trajs_3d_dict"]["coords"][0].float().cpu().numpy()  # (F, N, 3)
    vis = out["vis"][0].float().cpu().numpy()  # (F, N)
    return xyz.astype(np.float32), vis.astype(np.float32)


def _sanity_reproj(xyz: np.ndarray, data: dict, subset: str, clip_name: str) -> float:
    """Median reprojection error (px) of pred at each query's own frame — should be ~0."""
    q = data["queries_xyt"].astype(np.float32)
    fx, fy, cx, cy = data["fx_fy_cx_cy"].astype(np.float32).tolist()
    errs = []
    for n in range(q.shape[0]):
        t = int(q[n, 2])
        X, Y, Z = xyz[t, n]
        if not np.isfinite([X, Y, Z]).all() or Z <= 1e-6:
            continue
        u, v = fx * X / Z + cx, fy * Y / Z + cy
        errs.append(np.hypot(u - q[n, 0], v - q[n, 1]))
    return float(np.median(errs)) if errs else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subsets", nargs="+", default=["drivetrack", "pstudio", "adt"])
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/home/mas/data/tapvid3d_baseline_preds/deltav2"),
    )
    ap.add_argument(
        "--ckpt", type=Path, default=DELTA_ROOT / "checkpoints" / "densetrack3dv2.pth"
    )
    ap.add_argument("--device", default="cuda")
    ap.add_argument(
        "--use-fp16", action="store_true", help="bf16 autocast (saves VRAM)"
    )
    ap.add_argument(
        "--max-side",
        type=int,
        default=512,
        help="downscale longest image side to this (bounds the full-res video tensor; 0=native)",
    )
    ap.add_argument(
        "--limit", type=int, default=0, help="first N clips per subset (0=all)"
    )
    args = ap.parse_args()

    # Always build at the trained (384,512); infer_clip lowers the fed interp_shape
    # per clip (the model crops its fixed sincos pos_emb accordingly).
    model = DenseTrack3DV2(
        stride=4,
        window_len=16,
        add_space_attn=True,
        num_virtual_tracks=64,
        model_resolution=(384, 512),
        coarse_to_fine_dense=True,
    )
    # load_checkpoint handles DELTAv2's {"model": ...}/{"state_dict": ...} wrappers.
    sd = load_checkpoint(str(args.ckpt))
    model.load_state_dict(sd, strict=True)
    predictor = Predictor3D(model=model).eval().to(args.device)
    print(f"[deltav2] model loaded from {args.ckpt}")

    for subset in args.subsets:
        if subset not in MINIVAL_FILES:
            print(f"[warn] unknown subset {subset!r}, skip")
            continue
        out_sub = args.out_dir / subset
        out_sub.mkdir(parents=True, exist_ok=True)
        clips = MINIVAL_FILES[subset]
        if args.limit:
            clips = clips[: args.limit]

        for clip_name in tqdm(clips, desc=subset):
            out_path = out_sub / clip_name
            if out_path.exists():
                continue
            # allow_pickle: images_jpeg_bytes is an object array; these are our own
            # local TAPVid-3D benchmark files, not untrusted input.
            data = dict(np.load(TAPVID3D_ROOT / subset / clip_name, allow_pickle=True))
            F = len(data["images_jpeg_bytes"])
            idx = pick_interp_idx(F)
            while idx < len(INTERP_LADDER):
                interp = INTERP_LADDER[idx]
                try:
                    xyz, vis = infer_clip(
                        predictor,
                        data,
                        subset,
                        clip_name,
                        args.device,
                        args.use_fp16,
                        args.max_side,
                        interp,
                    )
                    rerr = _sanity_reproj(xyz, data, subset, clip_name)
                    np.savez_compressed(out_path, tracks_XYZ=xyz, visibility=vis)
                    tqdm.write(
                        f"[delta] {subset}/{clip_name}: N={xyz.shape[1]} F={xyz.shape[0]} "
                        f"interp={interp[0]}x{interp[1]} query-reproj={rerr:.2f}px"
                    )
                    torch.cuda.empty_cache()
                    break
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    idx += 1
                    tqdm.write(
                        f"[OOM]   {subset}/{clip_name} F={F}: retry at rung {idx}"
                    )
                except Exception as e:
                    torch.cuda.empty_cache()
                    tqdm.write(f"[error] {subset}/{clip_name}: {type(e).__name__}: {e}")
                    break

    print(f"[delta] done → {args.out_dir}")


if __name__ == "__main__":
    main()
