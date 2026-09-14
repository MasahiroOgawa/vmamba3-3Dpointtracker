"""Absolute-metric 3D-tracking evaluator for the SEA-RAFT+DA3 family.

Runs the full TAPVid-3D minival for one method and reports THREE metric families
per clip so absolute-depth quality (hidden by the leaderboard's scale-invariant
score) is exposed side by side:

  1. median-scaled (reference / leaderboard): 3D-AJ, APD3D, OA
     — TAPVid-3D default: median(‖gt‖/‖pred‖) rescale + depth-relative thresholds.
  2. absolute-metric: metric-AJ, metric-APD3D
     — same official machinery but scaling="none" + fixed metre thresholds
       (1cm..2.56m). Scale-sensitive: punishes DA3's global depth-scale bias.
  3. mean & median real-metric 3D error in metres over visible points
     (median too, since drivetrack ~20 m outliers dominate the mean).

Method:
  --method searaft                          training-free SEA-RAFT chaining + DA3 unproject
  --method v33 --ckpt <path>                Mamba3DepthRefiner (depth-along-ray refiner)

Output: metric_results/<subset>.json (per-clip) + summary.md + metrics.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from mamba3_tracker.data.tapvid3d import SUBSETS, list_clips, load_clip
from mamba3_tracker.eval.tapvid3d_eval import (
    aggregate,
    compute_clip_metrics,
    compute_clip_metrics_absolute,
)
from mamba3_tracker.train.loss import _unproject_with_depth
from searaft_flow import FlowModel, track_clip, track_clip_fuse


def _load_depth(
    da3_depth_root: Path, subset: str, clip_id: str, F_: int
) -> torch.Tensor:
    depth_path = Path(da3_depth_root).expanduser() / subset / (clip_id + ".npz")
    with np.load(depth_path) as dd:
        if "depth_q" in dd:
            q = np.asarray(dd["depth_q"][:F_]).astype(np.float32)
            d_min, d_max = float(dd["d_min"]), float(dd["d_max"])
            depth_full = d_min + q * (max(d_max - d_min, 1e-6) / 65535.0)
        else:
            depth_full = np.asarray(dd["depth"][:F_], dtype=np.float32)
    return torch.from_numpy(depth_full).unsqueeze(0)  # (1, F, Hd, Wd)


def _ray_from_uv(uv: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    fx, fy = K[:, 0, 0].view(-1, 1, 1), K[:, 1, 1].view(-1, 1, 1)
    cx, cy = K[:, 0, 2].view(-1, 1, 1), K[:, 1, 2].view(-1, 1, 1)
    return torch.stack([(uv[..., 0] - cx) / fx, (uv[..., 1] - cy) / fy], dim=-1)


@torch.no_grad()
def _infer(
    method,
    flow_model,
    model,
    clip,
    image_size,
    fb_alpha,
    fb_beta,
    da3_depth_root,
    max_frames,
    device,
    bidirectional=False,
    bidir_fuse=False,
    waft_pred_dir=None,
    z_source="depth_map",
    vis_source="flow",
    vis_head=None,
    flowvis_dir=None,
    out=None,
):
    """Return (pred_tracks (N,F,3) camera-frame XYZ, pred_vis (N,F)).

    vis_source "flow" returns the front-end's forward-backward mask, which is what every number
    in this repo up to now was scored with. "model" returns the refiner's own vis_head; "v94"
    returns the standalone flow-only head, which reads cached forward/backward flow and never
    touches the refiner, so the 3-D positions are identical across all three.
    """
    out_vis_logits = None
    F_ = (
        int(clip.images.shape[0])
        if not max_frames
        else min(int(clip.images.shape[0]), max_frames)
    )
    images = clip.images[:F_].clone()
    z_waft = None
    H_orig, W_orig = images.shape[-2], images.shape[-1]
    if (H_orig, W_orig) != (image_size, image_size):
        images = F.interpolate(
            images, size=(image_size, image_size), mode="bilinear", align_corners=False
        )
    images_255 = images * 255.0

    sx, sy = image_size / float(W_orig), image_size / float(H_orig)
    q = clip.queries_xyt.clone()
    queries_xy = torch.stack([q[:, 0] * sx, q[:, 1] * sy], dim=-1)
    anchor_t = q[:, 2].long().clamp(0, F_ - 1)

    K = clip.K.clone()
    K[0] *= sx
    K[1] *= sy

    if waft_pred_dir is not None:
        # Use WAFT's 2D track as the flow front-end for the refiner: recover uv by
        # projecting the saved WAFT per-frame camera XYZ with K (at this image_size).
        # Projection is exact & resolution-invariant, so WAFT's native (512) track
        # maps directly onto this 896 grid. The refiner then re-samples/corrects
        # DA3 depth at uv — i.e. WAFT flow + v35 depth refinement.
        wp = Path(waft_pred_dir).expanduser() / clip.subset / (clip.clip_id + ".npz")
        with np.load(wp) as wd:
            xyz_w = np.asarray(wd["tracks_XYZ"][:F_], dtype=np.float32)  # (F,N,3)
            vis_w = np.asarray(wd["visibility"][:F_]).astype(np.float32)  # (F,N)
        fx, fy = float(K[0, 0]), float(K[1, 1])
        cx, cy = float(K[0, 2]), float(K[1, 2])
        zc = np.clip(xyz_w[..., 2], 1e-6, None)
        u = fx * xyz_w[..., 0] / zc + cx
        v = fy * xyz_w[..., 1] / zc + cy
        uv = torch.from_numpy(np.stack([u, v], axis=-1)).float()  # (F,N,2)
        vis = torch.from_numpy(vis_w).float()
        # ray*z reconstructs xyz exactly, so keeping WAFT's own z lets z_source="waft" reproduce
        # `tracks_XYZ * exp(ds)` -- the 0.2385 arm -- rather than resampling the DA3 map.
        z_waft = torch.from_numpy(zc).float()
    elif bidir_fuse:  # forward+backward flow fusion per hop (real bidirectional)
        uv, vis = track_clip_fuse(
            flow_model,
            images_255.to(device),
            queries_xy,
            anchor_t,
            image_size,
            fb_alpha,
            fb_beta,
        )
    else:
        uv, vis = track_clip(
            flow_model,
            images_255.to(device),
            queries_xy,
            anchor_t,
            image_size,
            fb_alpha,
            fb_beta,
            bidirectional=bidirectional,
        )
    depth_t = _load_depth(da3_depth_root, clip.subset, clip.clip_id, F_).to(device)
    K_t = K.unsqueeze(0).to(device)  # K computed above (scaled to image_size)
    uv_d = uv.unsqueeze(0).to(device)
    # Figure scripts need the 2-D track as well as the 3-D one. Handing it back through an
    # optional dict keeps the return signature -- and every published number -- unchanged.
    if out is not None:
        out["uv"] = uv.numpy()

    if method == "searaft":
        xyz = _unproject_with_depth(uv_d, depth_t, K_t, float(image_size))[0]  # (F,N,3)
    elif method in ("v35", "v45", "v46", "v47", "v73", "v88"):
        ray = _ray_from_uv(uv_d, K_t)
        if z_source == "waft":
            if z_waft is None:
                raise ValueError("z_source='waft' requires --waft-pred-dir (needs tracks_XYZ)")
            z_raw = z_waft.unsqueeze(0).to(device)
        else:
            grid = (2.0 * uv_d / image_size - 1.0).view(F_, 1, -1, 2)
            z_raw = F.grid_sample(
                depth_t.squeeze(0).unsqueeze(1),
                grid,
                mode="bilinear",
                padding_mode="border",
                align_corners=False,
            ).view(1, F_, -1)
        images_b = images.unsqueeze(0).to(device)  # (1,F,3,H,W) in [0,1]
        _out = model(
            ray, z_raw, vis.unsqueeze(0).to(device), uv_d, depth_t, images_b, K_t
        )
        xyz = _out.xyz[0]
        out_vis_logits = _out.vis_logits[0]
    else:  # v33
        ray = _ray_from_uv(uv_d, K_t)
        grid = (2.0 * uv_d / image_size - 1.0).view(F_, 1, -1, 2)
        z_raw = F.grid_sample(
            depth_t.squeeze(0).unsqueeze(1),
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        ).view(1, F_, -1)
        if method == "v72":
            # reads the depth MAP, not just the sampled points
            xyz = model(ray, z_raw, vis.unsqueeze(0).to(device), depth_t).xyz[0]
        else:
            xyz = model(ray, z_raw, vis.unsqueeze(0).to(device)).xyz[0]  # (F,N,3)
    if vis_source == "model" and out_vis_logits is not None:
        vis = torch.sigmoid(out_vis_logits).float().cpu()
    elif vis_source == "v94":
        # Flow cached by cache_flow_vis.py with v93's flow settings; its mask was verified
        # identical to the WAFT predictions this eval scores, so the point order matches.
        fp = Path(flowvis_dir).expanduser() / clip.subset / (clip.clip_id + ".npz")
        with np.load(fp) as fd:
            ff = torch.from_numpy(fd["flow_fwd"][:F_]).float()
            fb_ = torch.from_numpy(fd["flow_bwd"][:F_]).float()
        if ff.shape[1] != vis.shape[1]:
            raise ValueError(
                f"{clip.subset}/{clip.clip_id}: flow cache has {ff.shape[1]} points, "
                f"track has {vis.shape[1]}"
            )
        logits = vis_head(ff.unsqueeze(0).to(device), fb_.unsqueeze(0).to(device))
        vis = torch.sigmoid(logits)[0].float().cpu()
    return xyz.transpose(0, 1).cpu().numpy(), vis.transpose(0, 1).numpy()


def _load_ckpt_into(model, sd: dict) -> None:
    """Load a checkpoint that may predate a head this model now has.

    Not strict: `vis_head` was added after every checkpoint under result/ was written, and it is
    zero-initialised, so a checkpoint that lacks it reproduces the previous behaviour exactly.
    Anything else missing is a real mismatch and is reported rather than swallowed.
    """
    missing, unexpected = model.load_state_dict(sd, strict=False)
    unrelated = [k for k in missing if "vis_head" not in k]
    if unrelated or unexpected:
        print(f"[eval] checkpoint mismatch: missing={unrelated[:4]} unexpected={unexpected[:4]}",
              flush=True)
    elif missing:
        print(f"[eval] checkpoint predates vis_head ({len(missing)} tensors); it stays "
              "zero-initialised, so visibility is unchanged", flush=True)


def _metric_err(pred_NF3, gt_NF3, vis_NF):
    """(mean, median) real-metric 3D error in metres over visible points."""
    m = (vis_NF > 0.5) & np.isfinite(pred_NF3).all(-1) & np.isfinite(gt_NF3).all(-1)
    if not m.any():
        return float("nan"), float("nan")
    d = np.linalg.norm((pred_NF3 - gt_NF3)[m], axis=-1)
    return float(d.mean()), float(np.median(d))


def _load_external(pred_dir: Path, subset: str, clip_id: str):
    """Load a released baseline prediction npz -> (pred (N,F,3), vis (N,F))."""
    p = Path(pred_dir).expanduser() / subset / (clip_id + ".npz")
    with np.load(p) as d:
        tr = np.asarray(d["tracks_XYZ"], dtype=np.float32)  # (F,N,3)
        vis = np.asarray(d["visibility"]).astype(np.float32)  # (F,N)
    return np.transpose(tr, (1, 0, 2)), np.transpose(vis, (1, 0))


def main() -> int:
    from mamba3_tracker.cudnn_guard import survive_cudnn_mismatch
    survive_cudnn_mismatch()
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--method",
        choices=["searaft", "v33", "v35", "v44", "v45", "v46", "v47", "v72", "v73", "v88", "external"],
        required=True,
    )
    ap.add_argument(
        "--ckpt", type=Path, default=None, help="required for --method v33/v35"
    )
    ap.add_argument(
        "--pred-dir",
        type=Path,
        default=None,
        help="required for --method external: dir with <subset>/<clip>.npz "
        "(keys tracks_XYZ (F,N,3), visibility (F,N))",
    )
    ap.add_argument(
        "--label", type=str, default=None, help="display name (default = method)"
    )
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, default=Path("~/data"))
    ap.add_argument("--da3-depth-root", type=Path, default=Path("~/data/tapvid3d_da3"),
                    help="path form; identified against the depth registry. Prefer --depth.")
    ap.add_argument("--depth", default=None,
                    help="da3l | da3g. Overrides --da3-depth-root and is what the run reports.")
    ap.add_argument("--subsets", nargs="+", default=list(SUBSETS))
    ap.add_argument(
        "--split", choices=["all", "minival", "full_eval"], default="minival"
    )
    ap.add_argument("--max-clips-per-subset", type=int, default=0)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--image-size", type=int, default=896)
    ap.add_argument(
        "--vis-source", choices=("flow", "model", "v94"), default="flow",
        help="which visibility to score: the front-end's forward-backward mask (flow, the "
             "default, and what every existing number used), the refiner's vis_head (model), "
             "or the standalone flow-only head (v94; needs --vis-head-ckpt and --flowvis-dir)",
    )
    ap.add_argument("--vis-head-ckpt", type=str, default=None,
                    help="required for --vis-source v94: FlowVisHead checkpoint")
    ap.add_argument("--flowvis-dir", type=str, default=None,
                    help="required for --vis-source v94: dir with <subset>/<clip>.npz holding "
                         "flow_fwd/flow_bwd (built by cache_flow_vis.py)")
    ap.add_argument(
        "--oracle-vis", action="store_true",
        help="score the predicted positions against ground-truth visibility (a ceiling, not a "
             "result): our visibility comes from the flow's forward-backward check, unlearned",
    )
    ap.add_argument(
        "--url", type=str, default="MemorySlices/Tartan-C-T-TSKH-spring540x960-M"
    )
    ap.add_argument("--iters", type=int, default=None)
    ap.add_argument("--scale", type=int, default=None)
    ap.add_argument(
        "--run-cfg", type=Path, default=None,
        help="cfg.json written beside the checkpoint. The flow front-end and its scale are read "
             "from it so evaluation cannot diverge from the training it is scoring; if omitted, "
             "the cfg.json next to --ckpt is used when present.",
    )
    ap.add_argument("--fb-alpha", type=float, default=0.05)
    ap.add_argument("--fb-beta", type=float, default=1.0)
    ap.add_argument(
        "--bidirectional",
        action="store_true",
        default=False,
        help="Run SEA-RAFT on reversed video for backward tracking (offline only; doubles flow compute)",
    )
    ap.add_argument(
        "--bidir-fuse",
        action="store_true",
        default=False,
        help="Real bidirectional 2D track: fuse forward+backward flow per hop "
        "(d=0.5*(d_fwd-d_bwd)) with reject-on-inconsistency. Use for v36/v38-style variants.",
    )
    ap.add_argument("--z-source", choices=["depth_map", "waft"], default="depth_map",
                    help="depth_map: sample DA3 depth at the tracked uv (default). "
                         "waft: use WAFT's own tracks_XYZ z, so a zero-init refiner "
                         "reproduces tracks_XYZ*exp(ds) exactly.")
    ap.add_argument(
        "--waft-pred-dir",
        type=Path,
        default=None,
        help="Use a WAFT prediction dir (tapvid3d_baseline_preds/waft) as the 2D flow "
        "front-end instead of SEA-RAFT: uv is recovered by projecting the saved WAFT XYZ. "
        "Combine with --method v35 for 'WAFT flow + v35 depth refiner' (v39).",
    )
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.data_root = args.data_root.expanduser()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # The front-end must match the run being scored. Reading it from the run's own cfg.json makes
    # that automatic: a WAFT-trained checkpoint is scored with WAFT at the same resolution, and no
    # precomputed track can silently disagree with what training saw.
    run_cfg_path = args.run_cfg
    if run_cfg_path is None and args.ckpt is not None:
        cand = Path(args.ckpt).parent / "cfg.json"
        run_cfg_path = cand if cand.exists() else None
    flow_source, flow_scale, flow_iters = "searaft", args.scale, args.iters
    if run_cfg_path is not None and Path(run_cfg_path).exists():
        with open(run_cfg_path) as fh:
            rc = json.load(fh).get("flow", {})
        flow_source = str(rc.get("source", "searaft"))
        if args.scale is None and rc.get("scale") is not None:
            flow_scale = int(rc["scale"])
        if args.iters is None and rc.get("iters") is not None:
            flow_iters = int(rc["iters"])
        print(f"[metric3d] front-end from {run_cfg_path}: source={flow_source} scale={flow_scale}")

    flow_model = None
    model = None
    vis_head = None
    if args.vis_source == "v94":
        if not (args.vis_head_ckpt and args.flowvis_dir):
            raise SystemExit("--vis-source v94 requires --vis-head-ckpt and --flowvis-dir")
        from mamba3_tracker.model.flow_vis_head import FlowVisHead

        ck = torch.load(args.vis_head_ckpt, map_location="cpu", weights_only=False)
        mc = ck["cfg"]["model"]
        vis_head = FlowVisHead(
            dim=int(mc["dim"]), state_dim=int(mc["state_dim"]),
            num_heads=int(mc["num_heads"]), num_layers=int(mc["num_layers"]),
            bidirectional=bool(mc["bidirectional"]),
        ).to(device)
        vis_head.load_state_dict(ck["model"])
        vis_head.eval()
        print(f"[eval] v94 visibility head from {args.vis_head_ckpt} "
              f"(step {ck.get('step')}, bidirectional={mc['bidirectional']})", flush=True)
    if args.method == "external":
        if args.pred_dir is None:
            ap.error("--method external requires --pred-dir")
        print(f"[metric3d] external predictions from {args.pred_dir}")
    elif args.waft_pred_dir is not None:
        # 2D track comes from WAFT preds; no SEA-RAFT flow model needed (saves VRAM).
        print(f"[metric3d] WAFT 2D front-end from {args.waft_pred_dir}")
        # A precomputed track set is equivalent to running live ONLY at the same scale and iteration
        # count -- there is no augmentation at evaluation, so nothing else differs. Silently mixing
        # settings produced a 0.0113 swing that read as a modelling result, so it is checked.
        _man = Path(args.waft_pred_dir).expanduser() / "manifest.json"
        if _man.exists() and flow_scale is not None:
            _m = json.loads(_man.read_text())
            # image_size is the tracking resolution and is NOT the same knob as scale, which is
            # the flow network's internal downscale. A set generated at the default 512 while the
            # run tracks at 896 differs by -0.0110, which is the size of a real modelling effect.
            _checks = [("scale", flow_scale), ("iters", flow_iters),
                       ("image_size", args.image_size)]
            _bad = [(k, _m.get(k), w) for k, w in _checks
                    if w is not None and k in _m and int(_m[k]) != int(w)]
            if _bad:
                raise SystemExit(
                    f"[metric3d] track set at {args.waft_pred_dir} does not match this run: "
                    + "; ".join(f"{k} is {g} but the run needs {w}" for k, g, w in _bad)
                    + ". Regenerate it with those settings, or evaluate live.")
            print("[metric3d] track manifest matches: "
                  + " ".join(f"{k}={_m[k]}" for k in ("scale", "iters", "image_size") if k in _m))
        elif not _man.exists():
            print(f"[metric3d] WARNING: {args.waft_pred_dir} has no manifest.json; "
                  f"its scale and iters cannot be verified against this run")
    elif flow_source == "waft_live":
        # Same track_clip as SEA-RAFT, only the flow network differs -- and run live, so there is no
        # cache that could have been built at another resolution or from other images.
        import importlib
        import os as _os
        _cwd = _os.getcwd()
        try:
            ew = importlib.import_module("eval_waft")
            flow_model = ew.build_flow(ew.WAFT_ROOT / "config" / "a1" / "tar-c-t.json",
                                       ew.WAFT_ROOT / "ckpts" / "waft_a1_recommended.pth",
                                       device, scale=flow_scale, iters=flow_iters)
        finally:
            _os.chdir(_cwd)
        print(f"[metric3d] WAFT flow model, run LIVE at scale={flow_scale}")
    else:
        flow_model = FlowModel(device, url=args.url, iters=flow_iters, scale=flow_scale)
    if args.method == "v33":
        if args.ckpt is None:
            ap.error("--method v33 requires --ckpt")
        from mamba3_tracker.model.depth_refined_tracker import Mamba3DepthRefiner

        state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        mc = state.get("cfg", {}).get("model", {})
        model = Mamba3DepthRefiner(
            dim=int(mc.get("dim", 128)),
            state_dim=int(mc.get("state_dim", 64)),
            num_heads=int(mc.get("num_heads", 4)),
            num_layers=int(mc.get("num_layers", 2)),
            max_log_correction=float(mc.get("max_log_correction", 2.0)),
            two_pool=bool(mc.get("two_pool", False)),
        ).to(device)
        _load_ckpt_into(model, state["model"])
        model.eval()
        print(f"[metric3d] v33 ckpt {args.ckpt} (step={state.get('step', '?')})")
    if args.method == "v44":
        if args.ckpt is None:
            ap.error("--method v44 requires --ckpt")
        from mamba3_tracker.model.depth_refined_tracker import Mamba3DeflickerRefiner

        state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        mc = state.get("cfg", {}).get("model", {})
        model = Mamba3DeflickerRefiner(
            dim=int(mc.get("dim", 128)),
            state_dim=int(mc.get("state_dim", 64)),
            num_heads=int(mc.get("num_heads", 4)),
            num_layers=int(mc.get("num_layers", 2)),
            max_scale_correction=float(mc.get("max_scale_correction", 0.5)),
            two_pool=bool(mc.get("two_pool", False)),
        ).to(device)
        _load_ckpt_into(model, state["model"])
        model.eval()
        print(f"[metric3d] v44 ckpt {args.ckpt} (step={state.get('step', '?')})")
    import sys as _sys
    _src_dir = str(Path(__file__).resolve().parent.parent / "src")
    if _src_dir not in _sys.path:
        _sys.path.insert(0, _src_dir)
    from mamba3_tracker.data.depth_source import resolve as _resolve_depth
    _depth_src = _resolve_depth(args.depth or args.da3_depth_root)
    args.da3_depth_root = _depth_src.root
    print(f"[metric3d] depth source: {_depth_src}")
    if args.waft_pred_dir:
        # uv only -- the refiner re-samples depth itself, and uv is invariant to scaling
        # along the ray, so a DA3-l-built track set is a valid uv source for a DA3-g run.
        print(f"[metric3d] waft tracks: "
              f"[{_depth_src.verify(Path(args.waft_pred_dir).expanduser(), strict=False)}]"
              f" -- used for uv only")

    if args.method == "v88":
        if args.ckpt is None:
            ap.error("--method v88 requires --ckpt")
        from mamba3_tracker.model.depth_refined_tracker import Mamba3V88

        state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        mc = state.get("cfg", {}).get("model", {})
        model = Mamba3V88(
            dim=int(mc.get("dim", 128)),
            state_dim=int(mc.get("state_dim", 64)),
            num_heads=int(mc.get("num_heads", 4)),
            num_layers=int(mc.get("num_layers", 2)),
            max_log_correction=float(mc.get("max_log_correction", 2.0)),
            max_delta_uv=float(mc.get("max_delta_uv", 2.0)),
            patch_size=int(mc.get("patch_size", 5)),
            max_scale_correction=float(mc.get("max_scale_correction", 0.5)),
            scale_stage_correction=float(mc.get("scale_stage_correction", 2.5)),
            d_proj=int(mc.get("d_proj", 64)),
            dino_model=str(mc.get("dino_model", "facebook/dinov3-vits16-pretrain-lvd1689m")),
            dino_image_size=int(mc.get("dino_image_size", 448)),
            image_size=int(mc.get("image_size", 896)),
            two_pool=bool(mc.get("two_pool", False)),
            grid=int(mc.get("grid", 64)),
            log_ref=float(mc.get("log_ref", 2.0)),
            log_std=float(mc.get("log_std", 1.5)),
        ).to(device)
        _load_ckpt_into(model, state["model"])
        model.eval()
        print(f"[metric3d] v88 ckpt {args.ckpt} (step={state.get('step', '?')}, "
              f"scale_gate={float(model.scale_gate.detach()):.4f})")
    if args.method == "v73":
        if args.ckpt is None:
            ap.error("--method v73 requires --ckpt")
        from mamba3_tracker.model.depth_refined_tracker import Mamba3V73

        state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        mc = state.get("cfg", {}).get("model", {})
        model = Mamba3V73(
            dim=int(mc.get("dim", 128)),
            state_dim=int(mc.get("state_dim", 64)),
            num_heads=int(mc.get("num_heads", 4)),
            num_layers=int(mc.get("num_layers", 2)),
            max_scale_correction=float(mc.get("max_scale_correction", 2.5)),
            two_pool=bool(mc.get("two_pool", False)),
            grid=int(mc.get("grid", 64)),
            log_ref=float(mc.get("log_ref", 2.0)),
            log_std=float(mc.get("log_std", 1.5)),
            max_log_correction=float(mc.get("max_log_correction", 2.0)),
            max_delta_uv=float(mc.get("max_delta_uv", 2.0)),
            patch_size=int(mc.get("patch_size", 5)),
            d_proj=int(mc.get("d_proj", 64)),
        ).to(device)
        _load_ckpt_into(model, state["model"])
        model.eval()
        print(f"[metric3d] v73 ckpt {args.ckpt} (step={state.get('step', '?')})")
    if args.method == "v72":
        if args.ckpt is None:
            ap.error("--method v72 requires --ckpt")
        from mamba3_tracker.model.depth_refined_tracker import Mamba3DepthScaleRefiner

        state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        mc = state.get("cfg", {}).get("model", {})
        model = Mamba3DepthScaleRefiner(
            dim=int(mc.get("dim", 128)),
            state_dim=int(mc.get("state_dim", 64)),
            num_heads=int(mc.get("num_heads", 4)),
            num_layers=int(mc.get("num_layers", 2)),
            max_scale_correction=float(mc.get("max_scale_correction", 2.5)),
            two_pool=bool(mc.get("two_pool", False)),
            grid=int(mc.get("grid", 64)),
            log_ref=float(mc.get("log_ref", 2.0)),
            log_std=float(mc.get("log_std", 1.5)),
        ).to(device)
        _load_ckpt_into(model, state["model"])
        model.eval()
        print(f"[metric3d] v72 ckpt {args.ckpt} (step={state.get('step', '?')})")
    if args.method == "v45":
        if args.ckpt is None:
            ap.error("--method v45 requires --ckpt")
        from mamba3_tracker.model.depth_refined_tracker import Mamba3V45

        state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        mc = state.get("cfg", {}).get("model", {})
        model = Mamba3V45(
            dim=int(mc.get("dim", 128)),
            state_dim=int(mc.get("state_dim", 64)),
            num_heads=int(mc.get("num_heads", 4)),
            num_layers=int(mc.get("num_layers", 2)),
            max_log_correction=float(mc.get("max_log_correction", 2.0)),
            max_delta_uv=float(mc.get("max_delta_uv", 2.0)),
            patch_size=int(mc.get("patch_size", 5)),
            max_scale_correction=float(mc.get("max_scale_correction", 0.5)),
            d_proj=int(mc.get("d_proj", 64)),
            dino_model=str(
                mc.get("dino_model", "facebook/dinov3-vits16-pretrain-lvd1689m")
            ),
            dino_image_size=int(mc.get("dino_image_size", 448)),
            image_size=int(mc.get("image_size", 896)),
            two_pool=bool(mc.get("two_pool", False)),
        ).to(device)
        _load_ckpt_into(model, state["model"])
        model.eval()
        print(f"[metric3d] v45 ckpt {args.ckpt} (step={state.get('step', '?')})")
    if args.method == "v47":
        if args.ckpt is None:
            ap.error("--method v47 requires --ckpt")
        from mamba3_tracker.model.depth_refined_tracker import Mamba3V45

        state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        mc = state.get("cfg", {}).get("model", {})
        model = Mamba3V45(
            dim=int(mc.get("dim", 128)),
            state_dim=int(mc.get("state_dim", 64)),
            num_heads=int(mc.get("num_heads", 4)),
            num_layers=int(mc.get("num_layers", 2)),
            max_log_correction=float(mc.get("max_log_correction", 2.0)),
            max_delta_uv=float(mc.get("max_delta_uv", 2.0)),
            patch_size=int(mc.get("patch_size", 5)),
            max_scale_correction=float(mc.get("max_scale_correction", 0.5)),
            d_proj=int(mc.get("d_proj", 64)),
            dino_model=str(
                mc.get("dino_model", "facebook/dinov3-vits16-pretrain-lvd1689m")
            ),
            dino_image_size=int(mc.get("dino_image_size", 448)),
            image_size=int(mc.get("image_size", 896)),
            pose_head=True,
            two_pool=bool(mc.get("two_pool", False)),
        ).to(device)
        _load_ckpt_into(model, state["model"])
        model.eval()
        print(f"[metric3d] v47 ckpt {args.ckpt} (step={state.get('step', '?')})")
    if args.method in ("v35", "v46"):
        if args.ckpt is None:
            ap.error(f"--method {args.method} requires --ckpt")
        from mamba3_tracker.model.depth_refined_tracker import Mamba3V35Refiner

        state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        mc = state.get("cfg", {}).get("model", {})
        model = Mamba3V35Refiner(
            dim=int(mc.get("dim", 128)),
            state_dim=int(mc.get("state_dim", 64)),
            num_heads=int(mc.get("num_heads", 4)),
            num_layers=int(mc.get("num_layers", 2)),
            max_log_correction=float(mc.get("max_log_correction", 2.0)),
            max_delta_uv=float(mc.get("max_delta_uv", 2.0)),
            patch_size=int(mc.get("patch_size", 5)),
            per_frame_scale=bool(mc.get("per_frame_scale", False)),
            within_frame=bool(mc.get("within_frame", False)),
            d_proj=int(mc.get("d_proj", 64)),
            dino_model=str(
                mc.get("dino_model", "facebook/dinov3-vits16-pretrain-lvd1689m")
            ),
            dino_image_size=int(mc.get("dino_image_size", 448)),
            image_size=int(mc.get("image_size", 896)),
            feat_encoder=str(mc.get("feat_encoder", "dinov3")),
            vmamba3_dim=int(mc.get("vmamba3_dim", 384)),
            vmamba3_heads=int(mc.get("vmamba3_heads", 6)),
            vmamba3_blocks=int(mc.get("vmamba3_blocks", 2)),
            vmamba3_patch=int(mc.get("vmamba3_patch", 14)),
            vmamba3_grid=int(mc.get("vmamba3_grid", 32)),
            two_pool=bool(mc.get("two_pool", False)),
        ).to(device)
        _load_ckpt_into(model, state["model"])
        model.eval()
        print(
            f"[metric3d] {args.method} ckpt {args.ckpt} (step={state.get('step', '?')})"
        )
    if flow_model is not None and hasattr(flow_model, "args"):
        print(
            f"[metric3d] method={args.method}  SEA-RAFT iters={flow_model.args.iters} scale={flow_model.args.scale}"
        )

    if args.split == "minival":
        from mamba3_tracker.data.tapvid3d_splits import MINIVAL_FILES as ALLOW
    elif args.split == "full_eval":
        from mamba3_tracker.data.tapvid3d_splits import FULL_EVAL_FILES as ALLOW
    else:
        ALLOW = None
    allow = {
        s: (set(ALLOW.get(s, [])) if ALLOW is not None else None) for s in args.subsets
    }

    metrics_root = args.out_dir / "metric_results"
    metrics_root.mkdir(parents=True, exist_ok=True)
    summary: dict[str, dict] = {}
    t_start = time.time()
    n_frames_total = 0
    n_fail = 0

    for sub in args.subsets:
        clips = list_clips(args.data_root, [sub])
        if allow[sub] is not None:
            clips = [p for p in clips if p.name in allow[sub]]
        if args.max_clips_per_subset:
            clips = clips[: args.max_clips_per_subset]
        print(f"[metric3d] {sub}: {len(clips)} clips", flush=True)

        per_clip: list[dict] = []
        for path in clips:
            try:
                clip = load_clip(path)
                if args.method == "external":
                    pred_NF3, pred_vis = _load_external(args.pred_dir, sub, path.stem)
                else:
                    pred_NF3, pred_vis = _infer(
                        args.method,
                        flow_model,
                        model,
                        clip,
                        args.image_size,
                        args.fb_alpha,
                        args.fb_beta,
                        args.da3_depth_root,
                        args.max_frames,
                        device,
                        bidirectional=args.bidirectional,
                        bidir_fuse=args.bidir_fuse,
                        waft_pred_dir=args.waft_pred_dir,
                        z_source=args.z_source,
                        vis_source=args.vis_source,
                        vis_head=vis_head,
                        flowvis_dir=args.flowvis_dir,
                    )
                # Align frame/point counts (released preds may truncate frames).
                Fg = int(clip.tracks_XYZ.shape[0])
                F_ = (
                    min(pred_NF3.shape[1], Fg)
                    if not args.max_frames
                    else min(pred_NF3.shape[1], Fg, args.max_frames)
                )
                N_ = min(pred_NF3.shape[0], int(clip.tracks_XYZ.shape[1]))
                pred_NF3 = pred_NF3[:N_, :F_]
                pred_vis = pred_vis[:N_, :F_]
                n_frames_total += F_
                gt_xyz = clip.tracks_XYZ[:F_, :N_].numpy()  # (F,N,3)
                gt_vis = clip.visibility[:F_, :N_].float().numpy()  # (F,N)
                gt_NF3 = np.transpose(gt_xyz, (1, 0, 2))
                gt_vis_NF = np.transpose(gt_vis, (1, 0))
                if args.oracle_vis:
                    # Ceiling measurement, not a result: our visibility is the flow's
                    # forward-backward check and is never learned, so this asks what the same 3D
                    # positions would score if that one channel were perfect.
                    pred_vis = gt_vis_NF.copy()
                K = clip.K.numpy()
                # TAPVid-3D defines the depth-relative pixel thresholds relative to
                # 256-px images, so the official evaluator rescales intrinsics by
                # 256/min(H,W) before scoring (matches tapnet evaluate_model.py).
                # Without this, drivetrack (1280x1920) median-AJ reads ~7x low.
                # The absolute metric (fixed-metre thresholds) ignores intrinsics.
                Ho, Wo = int(clip.images.shape[-2]), int(clip.images.shape[-1])
                s256 = 256.0 / min(Ho, Wo)
                intrin = np.array([K[0, 0], K[1, 1], K[0, 2], K[1, 2]]) * s256

                med = compute_clip_metrics(gt_xyz, gt_vis, pred_NF3, pred_vis, intrin)
                ab = compute_clip_metrics_absolute(
                    gt_xyz, gt_vis, pred_NF3, pred_vis, intrin
                )
                mean_m, median_m = _metric_err(pred_NF3, gt_NF3, gt_vis_NF)
                rec = {
                    "clip_id": clip.clip_id,
                    **med,
                    "metric_average_jaccard": ab["metric_average_jaccard"],
                    "metric_average_pts_within_thresh": ab[
                        "metric_average_pts_within_thresh"
                    ],
                    "metric_err_mean_m": mean_m,
                    "metric_err_median_m": median_m,
                }
                per_clip.append(rec)
                print(
                    f"[metric3d] {sub}/{clip.clip_id}: AJ={med['average_jaccard']:.4f} "
                    f"mAJ={ab['metric_average_jaccard']:.4f} "
                    f"mAPD={ab['metric_average_pts_within_thresh']:.4f} "
                    f"err={mean_m:.2f}m(med {median_m:.2f})",
                    flush=True,
                )
            except Exception as e:
                n_fail += 1
                print(
                    f"[metric3d] {sub}/{path.stem}: FAIL {type(e).__name__}: {e}",
                    flush=True,
                )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        (metrics_root / f"{sub}.json").write_text(json.dumps(per_clip, indent=2))
        summary[sub] = aggregate(per_clip)
        print(f"[metric3d] {sub} mean: {summary[sub]}", flush=True)

    overall = aggregate(summary.values())
    elapsed = time.time() - t_start
    fps = n_frames_total / elapsed if elapsed > 0 else float("nan")

    metrics_json = {
        "method": args.method,
        "split": args.split,
        "ckpt": str(args.ckpt) if args.ckpt else None,
        "per_subset": summary,
        "overall": overall,
        "frames": n_frames_total,
        "elapsed_s": elapsed,
        "fps": fps,
        "failures": n_fail,
    }
    (args.out_dir / "metrics.json").write_text(json.dumps(metrics_json, indent=2))

    def _g(d, k):
        return d.get(k, float("nan"))

    rows = [
        f"# Absolute-metric 3D tracking — {args.method} — TAPVid-3D {args.split}",
        f"\nThroughput: {n_frames_total} frames in {elapsed:.0f}s ({fps:.1f} fps on {device}); failures={n_fail}",
        "\nmedian-* = leaderboard (scale-invariant). metric-* = absolute (no median scaling, "
        "fixed-metre thresholds). err = real 3D error in metres over visible points.\n",
        "| subset | 3D-AJ | APD3D | OA | metric-AJ | metric-APD3D | err mean(m) | err median(m) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for sub, mm in summary.items():
        rows.append(
            f"| {sub} | {_g(mm, 'average_jaccard'):.4f} | {_g(mm, 'average_pts_within_thresh'):.4f} | "
            f"{_g(mm, 'occlusion_accuracy'):.4f} | {_g(mm, 'metric_average_jaccard'):.4f} | "
            f"{_g(mm, 'metric_average_pts_within_thresh'):.4f} | "
            f"{_g(mm, 'metric_err_mean_m'):.3f} | {_g(mm, 'metric_err_median_m'):.3f} |"
        )
    rows.append(
        f"| **mean** | **{_g(overall, 'average_jaccard'):.4f}** | "
        f"**{_g(overall, 'average_pts_within_thresh'):.4f}** | "
        f"**{_g(overall, 'occlusion_accuracy'):.4f}** | "
        f"**{_g(overall, 'metric_average_jaccard'):.4f}** | "
        f"**{_g(overall, 'metric_average_pts_within_thresh'):.4f}** | "
        f"**{_g(overall, 'metric_err_mean_m'):.3f}** | **{_g(overall, 'metric_err_median_m'):.3f}** |"
    )
    (args.out_dir / "summary.md").write_text("\n".join(rows) + "\n")
    print(f"\n[metric3d] DONE. {args.out_dir / 'summary.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
