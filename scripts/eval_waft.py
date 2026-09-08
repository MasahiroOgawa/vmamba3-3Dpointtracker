#!/usr/bin/env python3
"""WAFT optical flow as the 2D front-end (replacing SEA-RAFT) + DA3 depth
unprojection on TAPVid-3D minival. Saves per-frame camera XYZ predictions in the
eval_metric3d.py --method external format, so v37/v38 are scored by the exact same
pipeline as the SEA-RAFT+DA3 baseline (only the flow network differs).

  v37 = WAFT + DA3            (forward-only flow chaining)
  v38 = WAFT + DA3, bidir     (--bidir-fuse; forward+backward flow fusion per hop)

Run from WAFT's own venv (torch 2.7 / cu128):
  cd ~/proj/study/WAFT
  .venv/bin/python ~/proj/study/vmamba3-3Dpointtracker/scripts/eval_waft.py \\
    --subsets drivetrack pstudio adt \\
    --out-dir /home/mas/data/tapvid3d_baseline_preds/waft            # v37
  ... add --bidir-fuse and --out-dir .../waft_fuse                  # v38

Mirrors scripts/eval_metric3d.py `_infer` (searaft branch): square image_size
resize, anisotropic intrinsic scaling, flow chaining via the shared
searaft_flow.track_clip, and _unproject_with_depth — reused verbatim below.
"""

import argparse
import io
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

WAFT_ROOT = Path("/home/mas/proj/study/WAFT")
# DepthAnythingFeature loads 'depth-anything-ckpts/...' via a relative path at
# construction, so the process CWD must be the WAFT repo root.
os.chdir(WAFT_ROOT)
sys.path.insert(0, str(WAFT_ROOT))
PROJ = Path("/home/mas/proj/study/vmamba3-3Dpointtracker")
sys.path.insert(0, str(PROJ / "src"))

from config.parser import json_to_args  # noqa: E402
from model import fetch_model  # noqa: E402
from utils.utils import load_ckpt  # noqa: E402
from inference_tools import InferenceWrapper  # noqa: E402
from searaft_flow.flow_tracker import track_clip, track_clip_fuse  # noqa: E402
from mamba3_tracker.data.tapvid3d_splits import (  # noqa: E402
    MINIVAL_FILES,
    get_full_eval_files,
)

TAPVID3D_ROOT = Path("/home/mas/data/tapvid3d")
# Env-overridable so the DA3-g cache can be selected without editing code, the same mechanism
# doc/plan_da3g_reeval.md documents for the external-baseline scripts. Default stays DA3-l for
# backward compatibility with every number already published from this script.
# Depth selection goes through the single registry (src/mamba3_tracker/data/depth_source.py).
# DA3_ROOT is still honoured so existing callers keep working, but whatever it names is IDENTIFIED
# against the registry rather than trusted, and every output directory is stamped with the source.
# Unstamped track sets are exactly how a DA3-l cache silently became the target for a DA3-g model.
_here = Path(__file__).resolve().parent
if str(_here.parent / "src") not in sys.path:
    sys.path.insert(0, str(_here.parent / "src"))
from mamba3_tracker.data.depth_source import resolve as _resolve_depth  # noqa: E402

DEPTH_SOURCE = _resolve_depth(os.environ.get("DA3_ROOT"), default="da3l")
DA3_ROOT = DEPTH_SOURCE.root


class WAFTFlow:
    """Adapts WAFT to the searaft_flow FlowModel interface: flow(img1,img2)->(B,2,H,W)."""

    def __init__(self, wrapped: InferenceWrapper, device: torch.device):
        self.wrapped = wrapped
        self.device = device

    @torch.no_grad()
    def flow(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        # img1/img2: (B,3,H,W) float RGB in [0,255] on device (WAFT normalizes internally)
        out = self.wrapped.calc_flow(img1, img2)
        return out["flow"][-1]


def load_da3_depth(subset: str, clip_name: str, F_: int) -> np.ndarray:
    """Depth for a clip, from the resolved DEPTH_SOURCE. One implementation, not six copies."""
    return DEPTH_SOURCE.load(subset, clip_name, F_ if F_ else None)


def decode_images(jpeg_bytes_arr) -> np.ndarray:
    """JPEG bytes -> (F,3,H,W) float32 in [0,1]."""
    frames = [
        np.asarray(Image.open(io.BytesIO(bytes(b))).convert("RGB"), dtype=np.float32)
        / 255.0
        for b in jpeg_bytes_arr
    ]
    return np.stack(frames).transpose(0, 3, 1, 2)


def unproject(
    uv: torch.Tensor, depth: torch.Tensor, K: torch.Tensor, image_size: int
) -> torch.Tensor:
    """Replicates mamba3_tracker.train.loss._unproject_with_depth for (F,N,2) uv.

    uv (F,N,2) in square image_size pixels; depth (F,Hd,Wd) metres; K (3,3). -> (F,N,3).
    """
    F_, N, _ = uv.shape
    Hd, Wd = depth.shape[-2:]
    grid = (2.0 * uv / image_size - 1.0).view(F_, 1, N, 2)
    z = F.grid_sample(
        depth.unsqueeze(1),
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    ).view(F_, N)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    u, v = uv[..., 0], uv[..., 1]
    x = (u - cx) / fx * z
    y = (v - cy) / fy * z
    return torch.stack([x, y, z], dim=-1)


@torch.no_grad()
def infer_clip(
    flow_model, data, subset, clip_name, image_size, fb_alpha, fb_beta, bidir, device
):
    imgs = decode_images(data["images_jpeg_bytes"])  # (F,3,H,W) in [0,1]
    F_, _, H, W = imgs.shape
    images = torch.from_numpy(imgs)
    if (H, W) != (image_size, image_size):
        images = F.interpolate(
            images, size=(image_size, image_size), mode="bilinear", align_corners=False
        )
    images_255 = (images * 255.0).to(device)

    sx, sy = image_size / float(W), image_size / float(H)
    q = data["queries_xyt"].astype(np.float32)
    queries_xy = torch.from_numpy(
        np.stack([q[:, 0] * sx, q[:, 1] * sy], axis=-1)
    ).float()
    anchor_t = torch.from_numpy(q[:, 2]).long().clamp(0, F_ - 1)

    if bidir:  # v38: forward+backward flow fusion per hop (real bidirectional)
        uv, vis = track_clip_fuse(
            flow_model, images_255, queries_xy, anchor_t, image_size, fb_alpha, fb_beta
        )
    else:  # v37: forward-only chaining (track_clip already fills both time directions)
        uv, vis = track_clip(
            flow_model,
            images_255,
            queries_xy,
            anchor_t,
            image_size,
            fb_alpha,
            fb_beta,
            bidirectional=False,
        )  # uv (F,N,2), vis (F,N) on CPU

    depth = torch.from_numpy(load_da3_depth(subset, clip_name, F_)).float().to(device)
    fx, fy, cx, cy = data["fx_fy_cx_cy"].astype(np.float32).tolist()
    K = torch.tensor(
        [[fx * sx, 0.0, cx * sx], [0.0, fy * sy, cy * sy], [0.0, 0.0, 1.0]],
        device=device,
    )
    xyz = unproject(uv.to(device), depth, K, image_size)  # (F,N,3) per-frame camera
    return xyz.cpu().numpy().astype(np.float32), vis.numpy().astype(np.float32)


def build_flow(cfg_path: Path, ckpt: Path, device: torch.device,
               scale: int | None = None, iters: int | None = None) -> WAFTFlow:
    """`scale` is a log2 resolution factor, the same convention SEA-RAFT's wrapper uses.

    It must be passed explicitly for any SEA-RAFT/WAFT comparison. tar-c-t.json sets scale=0,
    i.e. full input resolution, while SEA-RAFT's spring-M.json sets -1, i.e. half. Left at their
    defaults the two front-ends run at 448x448 and 896x896 -- 4x the pixels and 4x the ViT
    tokens -- which is not a comparison of the front-ends but of two operating points.
    """
    args = json_to_args(str(cfg_path))
    # Refinement steps are a protocol setting, not a per-method default: tar-c-t.json asks for 5
    # while our runs give SEA-RAFT 4, and an unmatched count gives one arm more refinement than
    # the other.
    if iters is not None:
        args.iters = int(iters)
    model = fetch_model(args)
    load_ckpt(model, str(ckpt))
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    wrapped = InferenceWrapper(
        model,
        scale=args.scale if scale is None else int(scale),
        train_size=args.image_size,
        pad_to_train_size=False,
        tiling=False,
    )
    return WAFTFlow(wrapped, device)


def main():
    # This host's system cuDNN mismatches torch's bundled one, and every convolution raises
    # "cudnnFinalize failed" without this. eval_metric3d.py has always called it; eval_waft.py
    # never did, so full_eval track generation failed on all 4419 clips while the progress bar
    # advanced normally and the run looked healthy.
    from mamba3_tracker.cudnn_guard import survive_cudnn_mismatch
    survive_cudnn_mismatch()
    ap = argparse.ArgumentParser()
    ap.add_argument("--subsets", nargs="+", default=["drivetrack", "pstudio", "adt"])
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument(
        "--scale", type=int, default=None,
        help="log2 resolution factor; pass -1 to match SEA-RAFT's spring-M setting",
    )
    ap.add_argument(
        "--iters", type=int, default=None,
        help="refinement steps; must match the run this track set will be evaluated against, "
             "which is what the manifest records",
    )
    ap.add_argument(
        "--cfg", type=Path, default=WAFT_ROOT / "config" / "a1" / "tar-c-t.json"
    )
    ap.add_argument(
        "--ckpt", type=Path, default=WAFT_ROOT / "ckpts" / "waft_a1_recommended.pth"
    )
    # 512 ≈ WAFT's [432,960] training resolution: near-identical accuracy to 896
    # but ~3x faster (896 is ~14h/run on this GPU, infeasible for the 150-clip minival).
    ap.add_argument("--image-size", type=int, default=512)
    ap.add_argument("--fb-alpha", type=float, default=0.05)
    ap.add_argument("--fb-beta", type=float, default=1.0)
    ap.add_argument(
        "--bidir-fuse",
        action="store_true",
        help="v38: forward+backward flow fusion per hop (d=0.5*(d_fwd-d_bwd)), "
        "with reject-on-inconsistency. Default off = v37 forward chaining.",
    )
    ap.add_argument(
        "--split",
        choices=["minival", "full_eval"],
        default="minival",
        help="Which clips to generate tracks for. minival (150 clips) is the evaluation set and "
        "is what the published WAFT predictions cover. full_eval (4419 clips) is the TRAINING "
        "set: tracks for it must exist before a refiner can be trained on WAFT flow rather than "
        "merely evaluated with it.",
    )
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    flow_model = build_flow(args.cfg, args.ckpt, device, scale=args.scale, iters=args.iters)
    # A track set is only usable by a run whose flow settings match it, and a directory name is not
    # a record -- this one was generated at iters 5 while its name said 4, and the resulting 0.0113
    # discrepancy looked like a modelling result. The manifest is what eval_metric3d checks.
    import json as _json
    args.out_dir.mkdir(parents=True, exist_ok=True)
    _eff_scale = args.scale if args.scale is not None else 0
    _eff_iters = args.iters if args.iters is not None else 5
    (args.out_dir / "manifest.json").write_text(_json.dumps(
        {"scale": _eff_scale, "iters": _eff_iters, "image_size": args.image_size,
         "fb_alpha": args.fb_alpha, "fb_beta": args.fb_beta,
         "bidir": bool(args.bidir_fuse), "split": args.split,
         "depth_source": DEPTH_SOURCE.name}, indent=2))
    # The 3-D tracks are unprojected with THIS depth, so the track set inherits the depth choice.
    # Leaving that unrecorded is how waft_full_eval (DA3-l) became the training target for a DA3-g
    # model: the input was DA3-g, the target came from these tracks, and nothing could compare them.
    DEPTH_SOURCE.stamp(args.out_dir, produced_by="eval_waft.py", split=args.split,
                       image_size=args.image_size, iters=_eff_iters, scale=_eff_scale)
    print(f"[waft] manifest: scale={_eff_scale} iters={_eff_iters} "
          f"image_size={args.image_size} fb=({args.fb_alpha},{args.fb_beta})")
    print(f"[waft] depth source: {DEPTH_SOURCE}")
    print(
        f"[waft] model from {args.ckpt} (cfg {args.cfg.name}); bidir_fuse={args.bidir_fuse}"
    )

    n_missing = 0
    for subset in args.subsets:
        if subset not in MINIVAL_FILES:
            print(f"[warn] unknown subset {subset!r}, skip")
            continue
        out_sub = args.out_dir / subset
        out_sub.mkdir(parents=True, exist_ok=True)
        clips = (MINIVAL_FILES[subset] if args.split == "minival"
                 else get_full_eval_files(subset))
        if args.limit:
            clips = clips[: args.limit]
        for clip_name in tqdm(clips, desc=subset):
            out_path = out_sub / clip_name
            if out_path.exists():
                continue
            src = TAPVID3D_ROOT / subset / clip_name
            # A clip listed in full_eval but absent locally used to abort the whole sweep: this
            # load sat outside the try below, so one missing file of 4419 threw FileNotFoundError
            # and discarded four hours of completed work. Missing input is now the same kind of
            # per-clip skip as a failed inference.
            if not src.exists():
                n_missing += 1
                tqdm.write(f"[waft] {subset}/{clip_name}: SKIPPED, not present locally")
                continue
            # allow_pickle: images_jpeg_bytes is an object array; our own local
            # TAPVid-3D benchmark files, not untrusted input.
            data = dict(np.load(src, allow_pickle=True))
            try:
                xyz, vis = infer_clip(
                    flow_model,
                    data,
                    subset,
                    clip_name,
                    args.image_size,
                    args.fb_alpha,
                    args.fb_beta,
                    args.bidir_fuse,
                    device,
                )
                np.savez_compressed(out_path, tracks_XYZ=xyz, visibility=vis)
                tqdm.write(
                    f"[waft] {subset}/{clip_name}: N={xyz.shape[1]} F={xyz.shape[0]}"
                )
                torch.cuda.empty_cache()
            except Exception as e:
                torch.cuda.empty_cache()
                tqdm.write(f"[error] {subset}/{clip_name}: {type(e).__name__}: {e}")

    if n_missing:
        print(f"[waft] {n_missing} clip(s) listed but absent locally; skipped")
    print(f"[waft] done -> {args.out_dir}")


if __name__ == "__main__":
    main()
