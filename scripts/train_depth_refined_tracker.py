"""Train Mamba3DepthRefiner (v33) or Mamba3V35Refiner (v35) on TAPVid-3D.

v33: SEA-RAFT 2D track frozen; SSM refines per-track depth only. Loss is 3D-only.
v35: Adds DINOv3 image features + depth patch; outputs Δuv and Δlog_z jointly.

Usage:
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \\
        uv run python scripts/train_depth_refined_tracker.py \\
            --config configs/v33.yaml --out-dir result/YYYYMMDD-HHMM_v33

    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \\
        uv run python scripts/train_depth_refined_tracker.py \\
            --config configs/v35.yaml --out-dir result/YYYYMMDD-HHMM_v35

Smoke (50 steps):
    uv run python scripts/train_depth_refined_tracker.py \\
        --config configs/v35.yaml --out-dir result/v35_smoke \\
        --steps 50 --window 4 --batch 1 --val-every 25 --log-every 5
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as Fn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from mamba3_tracker.data.dataset import (
    TAPVid3DDataset,
    collate_tracking,
    default_train_val,
    minival_split,
    official_train_test_split,
)
from mamba3_tracker.data.tapvid3d import load_clip
from mamba3_tracker.model.depth_refined_tracker import (
    Mamba3DepthScaleRefiner,
    Mamba3V73,
    Mamba3V88,
    Mamba3DeflickerRefiner,
    Mamba3DepthRefiner,
    Mamba3V35Refiner,
    Mamba3V45,
)
from mamba3_tracker.train.config import dump_resolved, load_config
from mamba3_tracker.train.loss import (
    TrackingLossOutput,
    TrackingLossV33,
    depth_scale_refiner_loss,
    TrackingLossV35,
    TrackingLossV47,
)
from mamba3_tracker.train.schedule import wsd
from searaft_flow import FlowModel, track_clip


_LOSS_KEYS = ("total", "pos_3D", "pos_2D", "vis")


def _sample_depth(
    depth: torch.Tensor, uv: torch.Tensor, image_size: float
) -> torch.Tensor:
    """Bilinear-sample depth (B,F,Hd,Wd) at uv (B,F,N,2) -> (B,F,N)."""
    B, F_, N, _ = uv.shape
    grid = (2.0 * uv / image_size - 1.0).view(B * F_, 1, N, 2)
    d = depth.view(B * F_, 1, depth.shape[-2], depth.shape[-1])
    out = Fn.grid_sample(
        d, grid, mode="bilinear", padding_mode="border", align_corners=False
    )
    return out.view(B, F_, N)


def _ray_from_uv(uv: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    """Pixel ray (u-cx)/fx, (v-cy)/fy. uv (B,F,N,2), K (B,3,3) -> (B,F,N,2)."""
    fx = K[:, 0, 0].view(-1, 1, 1)
    fy = K[:, 1, 1].view(-1, 1, 1)
    cx = K[:, 0, 2].view(-1, 1, 1)
    cy = K[:, 1, 2].view(-1, 1, 1)
    rx = (uv[..., 0] - cx) / fx
    ry = (uv[..., 1] - cy) / fy
    return torch.stack([rx, ry], dim=-1)


def _save_ckpt(out_dir: Path, step: int, model, optim, sched, history, cfg) -> Path:
    path = out_dir / f"ckpt_{step}.pt"
    tmp = out_dir / f"ckpt_{step}.pt.tmp"
    torch.save(
        {
            "step": step,
            "model": model.state_dict(),
            "optim": optim.state_dict(),
            "sched": sched.state_dict() if sched is not None else None,
            "history": history,
            "cfg": cfg,
        },
        tmp,
    )
    tmp.replace(path)
    for old in out_dir.glob("ckpt_*.pt"):
        if old != path:
            try:
                old.unlink()
            except OSError:
                pass
    return path


def _find_latest_ckpt(out_dir: Path) -> Path | None:
    cands = []
    for p in out_dir.glob("ckpt_*.pt"):
        try:
            cands.append((int(p.stem.split("_", 1)[1]), p))
        except ValueError:
            continue
    return max(cands, key=lambda x: x[0])[1] if cands else None


def _loss_to_dict(out: TrackingLossOutput) -> dict[str, float]:
    return {
        "total": float(out.total.item()),
        "pos_3D": float(out.pos_3D.item()),
        "pos_2D": float(out.pos_2D.item()),
        "vis": float(out.vis.item()),
    }


def _fmt_loss_row(d: dict, dsr: float = float("nan")) -> str:
    row = (
        f"loss={d['total']:.4f}  p3D={d['pos_3D']:.4f}  "
        f"p2D={d['pos_2D']:.4f}  vis={d['vis']:.4f}"
    )
    # L_dsr is the whole point of the v72 arm: watching it fall is how we know the scale head is
    # learning the correction rather than being drowned by the per-point term.
    return row if dsr != dsr else f"{row}  Ldsr={dsr:.4f}"


def _per_head_grad_norm(model: Mamba3DepthRefiner) -> dict[str, float]:
    # v45 is composite (.deflicker + .v35); report the refiner stage's parts plus
    # the deflicker's scale head. Other models expose embed/layers/heads directly.
    base = getattr(model, "v35", model)
    groups = {}
    if hasattr(base, "embed"):
        groups["embed"] = list(base.embed.parameters())
    if hasattr(base, "layers"):
        groups["layers"] = list(base.layers.parameters())
    for hname in ("dz_head", "duv_head", "scale_head"):
        head = getattr(base, hname, None)
        if head is not None:
            groups[hname] = list(head.parameters())
    defl = getattr(model, "deflicker", None)
    if defl is not None and hasattr(defl, "scale_head"):
        groups["defl_scale"] = list(defl.scale_head.parameters())
    out = {}
    for name, params in groups.items():
        sq = sum(
            float((p.grad * p.grad).sum().item()) for p in params if p.grad is not None
        )
        out[name] = sq**0.5
    return out


def _fmt_grad_row(g: dict) -> str:
    return "  ".join(f"{k}={v:.2e}" for k, v in g.items())


def _build_waft_flow(device, scale=None, iters=None):
    """WAFT as a drop-in for SEA-RAFT's FlowModel: both are consumed by the same track_clip.

    eval_waft chdirs to the WAFT checkout at import time, because DepthAnythingFeature loads its
    weights by a relative path. That would leave this process there and break every relative path
    the trainer uses -- --out-dir first among them -- so the working directory is restored.
    """
    import importlib
    import os
    cwd = os.getcwd()
    try:
        ew = importlib.import_module("eval_waft")
        return ew.build_flow(ew.WAFT_ROOT / "config" / "a1" / "tar-c-t.json",
                             ew.WAFT_ROOT / "ckpts" / "waft_a1_recommended.pth", device,
                             scale=scale, iters=iters)
    finally:
        os.chdir(cwd)


@torch.no_grad()
def _perturb_uv(uv, sigma_px, rho, generator=None):
    """Add temporally-correlated positional noise to a 2-D track, in pixels.

    Purpose: WAFT's track is measurably cleaner than SEA-RAFT's (median 4.10 px against 5.20 px over
    36 minival clips), and the DA3-g de-flicker stage trains better on the noisier one -- it exists
    to correct instability, so a harder training signal teaches a stronger correction, which still
    helps when applied to the cleaner track at evaluation. This hardens WAFT's training track to
    match, so the arm stays self-consistent while facing the same difficulty.

    The noise is AR(1) along time rather than independent per frame, because flow error drifts along
    a track rather than resampling itself each frame; independent noise would be a different
    distribution with the same variance, and the point is to reproduce the mechanism.
    """
    if sigma_px <= 0:
        return uv
    eps = torch.randn(uv.shape, device=uv.device, dtype=uv.dtype, generator=generator)
    # AR(1) with unit stationary variance, walked along the frame axis (dim -3 of (...,F,N,2))
    out = torch.empty_like(eps)
    out[..., 0, :, :] = eps[..., 0, :, :]
    scale = (1.0 - rho * rho) ** 0.5
    for t in range(1, uv.shape[-3]):
        out[..., t, :, :] = rho * out[..., t - 1, :, :] + scale * eps[..., t, :, :]
    return uv + sigma_px * out


def _run_flow_batch(flow_model, batch, device, image_size, fb_alpha, fb_beta,
                    waft_pred_dir=None, noise_px=0.0, noise_rho=0.9,
                    z_source="depth_map"):
    """Return (ray, z_raw, vis, uv, images, K) — all on device.

    images: (B,F,3,H,W) in [0,1] (needed by v35 DINOv3 encoder)

    With ``waft_pred_dir`` the 2-D track comes from precomputed WAFT predictions instead of
    running SEA-RAFT, so the refiner is TRAINED on the front-end it will be evaluated with.
    Without it the refiner learns SEA-RAFT's error characteristics and is then scored on WAFT,
    which is what every published row here actually does.

    The saved tracks are full-clip, so each batch item is sliced to its own window
    (``batch.frame_start``) and to the query columns that survived subsampling and the
    anchor-in-window filter (``batch.query_idx``). Projection uses the batch's K, which is
    already scaled to image_size, and is resolution-invariant -- the same reasoning
    eval_metric3d relies on.
    """
    B, F_ = batch.images.shape[:2]
    all_uv, all_vis = [], []
    all_zw: list = []
    if waft_pred_dir is not None:
        root = Path(waft_pred_dir).expanduser()
        K_cpu = batch.K
        for b in range(B):
            wp = root / batch.subsets[b] / (batch.clip_ids[b] + ".npz")
            start = batch.frame_start[b]
            idx = batch.query_idx[b].numpy()
            if not wp.exists():
                raise FileNotFoundError(
                    f"no WAFT track for {batch.subsets[b]}/{batch.clip_ids[b]} at {wp}. "
                    "The published prediction set covers minival (150 clips) only, which is the "
                    "EVALUATION split. Generate tracks for the training clips first:\n"
                    "  uv run python scripts/eval_waft.py --split full_eval "
                    "--out-dir ~/data/tapvid3d_baseline_preds/waft_full_eval"
                )
            with np.load(wp) as wd:
                xyz = np.asarray(wd["tracks_XYZ"][start:start + F_], dtype=np.float32)
                vw = np.asarray(wd["visibility"][start:start + F_]).astype(np.float32)
            xyz, vw = xyz[:, idx], vw[:, idx]
            Kb = K_cpu[b]
            fx, fy = float(Kb[0, 0]), float(Kb[1, 1])
            cx, cy = float(Kb[0, 2]), float(Kb[1, 2])
            zc = np.clip(xyz[..., 2], 1e-6, None)
            u = fx * xyz[..., 0] / zc + cx
            v = fy * xyz[..., 1] / zc + cy
            n_pad = batch.queries_xyt.shape[1] - u.shape[1]
            uv_b = torch.from_numpy(np.stack([u, v], -1)).float()
            vis_b = torch.from_numpy(vw).float()
            if n_pad > 0:   # collate padded N_q to the batch max; pad to match
                uv_b = torch.cat([uv_b, uv_b.new_zeros(F_, n_pad, 2)], dim=1)
                vis_b = torch.cat([vis_b, vis_b.new_zeros(F_, n_pad)], dim=1)
            all_uv.append(uv_b.to(device))
            all_vis.append(vis_b.to(device))
            # ray*z reconstructs xyz exactly, so keeping WAFT's own z lets the pipeline reproduce
            # `tracks_XYZ * exp(ds)` -- the 0.2385 arm -- instead of resampling the DA3-g map.
            z_b = torch.from_numpy(np.clip(xyz[..., 2], 1e-6, None)).float()
            if n_pad > 0:
                z_b = torch.cat([z_b, z_b.new_zeros(F_, n_pad)], dim=1)
            all_zw.append(z_b.to(device))
    else:
        for b in range(B):
            imgs = batch.images[b].to(device) * 255.0
            q = batch.queries_xyt[b].to(device)
            anchor_t = q[:, 2].long().clamp(0, F_ - 1)
            uv, vis = track_clip(
                flow_model, imgs, q[:, :2], anchor_t, image_size, fb_alpha, fb_beta
            )
            all_uv.append(uv)
            all_vis.append(vis)
    uv = torch.stack(all_uv).to(device)  # (B,F,N,2)
    vis = torch.stack(all_vis).to(device)  # (B,F,N)
    # Applied before ray and depth are derived, so the perturbation reaches the depth patch and the
    # de-flicker stage exactly as a real tracking error would, not just the coordinates.
    uv = _perturb_uv(uv, noise_px, noise_rho)
    K = batch.K.to(device)
    ray = _ray_from_uv(uv, K)
    if z_source == "waft":
        if not all_zw:
            raise ValueError("z_source='waft' requires flow.source=waft_cached (needs tracks_XYZ)")
        z_raw = torch.stack(all_zw)
    elif z_source == "depth_map":
        z_raw = _sample_depth(batch.depth.to(device), uv, float(image_size))
    else:
        raise ValueError(f"unknown z_source {z_source!r}; expected 'waft' or 'depth_map'")
    images = batch.images.to(device)  # (B,F,3,H,W) in [0,1]
    return ray, z_raw, vis, uv, images, K


def _model_forward(
    model, version, ray, z_raw, vis, uv, images, depth, K, amp_dtype, device, use_amp
):
    """Dispatch model forward for v33 vs v35."""
    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
        if version in ("v35", "v45", "v46", "v47", "v73", "v88"):
            return model(ray, z_raw, vis, uv, depth, images, K)
        if version == "v72":
            return model(ray, z_raw, vis, depth)
        return model(ray, z_raw, vis)


_VAL_RNG_SEED = 12345


@torch.no_grad()
def _validate(
    model,
    version,
    flow_model,
    val_ds,
    loss_fn,
    device,
    amp_dtype,
    image_size,
    fb_alpha,
    fb_beta,
    n_clips=5,
    waft_pred_dir=None,
    z_source="depth_map",
):
    model.eval()
    # Re-seed the dataset RNG on EVERY call. TAPVid3DDataset draws a random temporal window
    # (dataset.py:80) and a random query subset (dataset.py:114) on each __getitem__, and
    # augment=False disables only photometric augmentation -- so without this, consecutive
    # validations measure DIFFERENT windows and queries. The resulting curve is resampling noise,
    # not model change: it swung 5x across the v73-v82 arms independently of learning rate, and
    # step 0 matched to four decimals across runs only because it always drew first from Random(0).
    # Early stopping and best-checkpoint selection on an unseeded signal select on that noise.
    if hasattr(val_ds, "_rng"):
        val_ds._rng = random.Random(_VAL_RNG_SEED)
    totals: dict[str, list[float]] = defaultdict(list)
    for i in range(min(n_clips, len(val_ds))):
        batch = collate_tracking([val_ds[i]])
        queries = batch.queries_xyt.to(device)
        qmask = batch.query_mask.to(device)
        ray, z_raw, vis, uv, images, K = _run_flow_batch(
            flow_model, batch, device, image_size, fb_alpha, fb_beta,
            waft_pred_dir=waft_pred_dir, z_source=z_source,
        )
        pred = _model_forward(
            model,
            version,
            ray,
            z_raw,
            vis,
            uv,
            images,
            batch.depth.to(device),
            K,
            amp_dtype,
            device,
            use_amp=True,
        )
        out = loss_fn(
            pred,
            batch.tracks_XYZ.to(device),
            batch.visibility.to(device),
            qmask,
            queries[..., 2].long(),
            K,
        )
        for k, v in _loss_to_dict(out).items():
            totals[k].append(v)
    model.train()
    return {k: sum(v) / len(v) for k, v in totals.items()}


_SUBSETS_KNOWN = ("pstudio", "drivetrack", "adt")


def _which_subset(path: Path) -> str:
    for s in _SUBSETS_KNOWN:
        if s in path.parts:
            return s
    return "unknown"


@torch.no_grad()
def _motion_check(
    model,
    version,
    flow_model,
    val_paths,
    device,
    amp_dtype,
    image_size,
    fb_alpha,
    fb_beta,
    max_frames=32,
    n_clips=5,
):
    """Pixel travel ratio per subset."""
    model.eval()
    pred_sums: dict[str, float] = defaultdict(float)
    gt_sums: dict[str, float] = defaultdict(float)
    for path in val_paths[:n_clips]:
        sub = _which_subset(path)
        clip = load_clip(path)
        F_ = min(max_frames, clip.F)
        sx = image_size / float(clip.W)
        sy = image_size / float(clip.H)
        imgs_01 = Fn.interpolate(
            clip.images[:F_],
            size=(image_size, image_size),
            mode="bilinear",
            align_corners=False,
        ).to(device)
        N_q = clip.queries_xyt.shape[0]
        q = clip.queries_xyt.clone()
        queries_xy = torch.stack([q[:, 0] * sx, q[:, 1] * sy], dim=-1).to(device)
        anchor_t = q[:, 2].long().clamp(0, F_ - 1).to(device)
        uv, vis = track_clip(
            flow_model,
            imgs_01 * 255.0,
            queries_xy,
            anchor_t,
            image_size,
            fb_alpha,
            fb_beta,
        )
        uv_d = uv.unsqueeze(0).to(device)
        vis_d = vis.unsqueeze(0).to(device)

        K = clip.K.numpy()
        Ks = K.copy()
        Ks[0] *= sx
        Ks[1] *= sy
        K_t = torch.from_numpy(Ks).float().unsqueeze(0).to(device)
        ray = _ray_from_uv(uv_d, K_t)
        depth_path = (
            Path("~/data/tapvid3d_da3").expanduser()
            / clip.subset
            / (clip.clip_id + ".npz")
        )
        with np.load(depth_path) as dd:
            if "depth_q" in dd:
                qd = np.asarray(dd["depth_q"][:F_]).astype(np.float32)
                d_min, d_max = float(dd["d_min"]), float(dd["d_max"])
                depth_full = d_min + qd * (max(d_max - d_min, 1e-6) / 65535.0)
            else:
                depth_full = np.asarray(dd["depth"][:F_], dtype=np.float32)
        depth_t = torch.from_numpy(depth_full).unsqueeze(0).to(device)
        z_raw = _sample_depth(depth_t, uv_d, float(image_size))
        images_b = imgs_01.unsqueeze(0)  # (1,F,3,H,W) in [0,1]
        pred = _model_forward(
            model,
            version,
            ray,
            z_raw,
            vis_d,
            uv_d,
            images_b,
            depth_t,
            K_t,
            amp_dtype,
            device,
            use_amp=True,
        )
        pred_xyz = pred.xyz[0].float().cpu().numpy()  # (F,N,3)

        gt_xyz = clip.tracks_XYZ[:F_].numpy()
        gt_vis = clip.visibility[:F_].float().numpy()
        a_n = q[:, 2].long().clamp(0, F_ - 1).numpy()
        track_idx = np.arange(N_q)

        def _proj(xyz):
            Z = np.clip(xyz[..., 2:3], 1e-6, None)
            return (xyz[..., :2] / Z) * np.array([Ks[0, 0], Ks[1, 1]]) + np.array(
                [Ks[0, 2], Ks[1, 2]]
            )

        uv_pred = _proj(pred_xyz)
        uv_gt = _proj(gt_xyz)
        travel_pred = np.linalg.norm(uv_pred - uv_pred[a_n, track_idx][None], axis=-1)
        travel_gt = np.linalg.norm(uv_gt - uv_gt[a_n, track_idx][None], axis=-1)
        vis_anchor = gt_vis[a_n, track_idx]
        finite = np.isfinite(travel_pred) & np.isfinite(travel_gt)
        mask = (gt_vis > 0.5) & (vis_anchor[None] > 0.5) & finite
        pred_sums[sub] += float((travel_pred * mask).sum())
        gt_sums[sub] += float((travel_gt * mask).sum())
    model.train()
    return {sub: pred_sums[sub] / max(gt_sums[sub], 1.0) for sub in pred_sums}


def _fmt_motion_row(m: dict) -> str:
    return (
        "  ".join(f"{s}={r * 100:5.1f}%" for s, r in sorted(m.items()))
        if m
        else "(no clips)"
    )


def _build_overrides(args: argparse.Namespace) -> dict:
    train_o = {
        k: getattr(args, k)
        for k in (
            "steps",
            "warmup",
            "decay",
            "ckpt_every",
            "val_every",
            "log_every",
            "lr",
            "weight_decay",
            "grad_clip",
            "batch",
            "window",
            "amp",
            "num_workers",
            "seed",
        )
    }
    data_o = {
        "subsets": args.subsets,
        "image_size": args.image_size,
        "num_tracks": args.num_tracks,
    }
    return {
        k: v
        for k, v in {
            "train": {k: v for k, v in train_o.items() if v is not None},
            "data": {k: v for k, v in data_o.items() if v is not None},
        }.items()
        if v
    }


def main() -> int:
    from mamba3_tracker.cudnn_guard import survive_cudnn_mismatch
    survive_cudnn_mismatch()
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="Debug override. Experiments set train.out_dir in the config; "
                         "cfg.json is then a complete record of the run.")
    ap.add_argument("--data-root", type=Path, default=Path("~/data"))
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--warmup", type=int, default=None)
    ap.add_argument("--decay", type=int, default=None)
    ap.add_argument("--ckpt-every", dest="ckpt_every", type=int, default=None)
    ap.add_argument("--val-every", dest="val_every", type=int, default=None)
    ap.add_argument("--log-every", dest="log_every", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--weight-decay", dest="weight_decay", type=float, default=None)
    ap.add_argument("--grad-clip", dest="grad_clip", type=float, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--window", type=int, default=None)
    ap.add_argument("--amp", choices=["bf16", "fp16", "fp32"], default=None)
    ap.add_argument("--num-workers", dest="num_workers", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--subsets", nargs="+", default=None)
    ap.add_argument("--image-size", dest="image_size", type=int, default=None)
    ap.add_argument("--num-tracks", dest="num_tracks", type=int, default=None)
    args = ap.parse_args()

    args.data_root = args.data_root.expanduser()

    cfg = load_config(args.config, overrides=_build_overrides(args))
    model_cfg, data_cfg, train_cfg, loss_cfg = (
        cfg["model"],
        cfg["data"],
        cfg["train"],
        cfg["loss"],
    )
    flow_cfg = cfg.get("flow", {})
    # Every experimental setting comes from the config, so the run is reproducible from it alone
    # and cfg.json below is a complete record of what produced the numbers.
    if args.out_dir is None:
        od = train_cfg.get("out_dir")
        if not od:
            raise SystemExit("set train.out_dir in the config (or pass --out-dir for a debug run)")
        args.out_dir = Path(str(od)).expanduser()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Positional noise added to the training track only, never to validation or evaluation.
    # Validation noise scales as 1/sqrt(clips); at the old default of 5 the median change between
    # consecutive checks was 0.103 against a whole-run spread of 0.096, so two numbers said nothing.
    val_clips_n = int(train_cfg.get("val_clips", 5))
    # Validate BEFORE training moves anything, so every run records the score of its own
    # starting point. Without it a warm-started or zero-init-head arm has no baseline in its own
    # log, and "training made this worse" cannot be distinguished from "this was always bad":
    # the v78 arms were tuned for a morning against a starting point nobody had measured.
    val_at_step0 = bool(train_cfg.get("val_at_step0", True))
    # Early stopping: patience on the BEST value, not flatness between neighbours, with a tolerance
    # above the noise floor. Keeps the best checkpoint rather than the last.
    es_patience = int(train_cfg.get("early_stop_patience", 0))
    es_min_delta = float(train_cfg.get("early_stop_min_delta", 0.01))
    es_best, es_since, es_best_step = float("inf"), 0, -1
    es_raw_best = float("inf")
    lambda_dsr = float(loss_cfg.get("lambda_dsr", 0.0) or 0.0)
    last_dsr = float("nan")
    if lambda_dsr > 0:
        print(f"[train] L_dsr enabled, lambda_dsr={lambda_dsr:.3f}")
    track_noise_px = float(data_cfg.get("track_noise_px", 0.0) or 0.0)
    track_noise_rho = float(data_cfg.get("track_noise_rho", 0.9))
    if track_noise_px > 0:
        print(f"[train] track noise: sigma={track_noise_px:.2f}px AR(1) rho={track_noise_rho:.2f} "
              f"(training only)")

    print(f"[train] config {args.config}  version={cfg['version']}")
    print(f"[train] loss weights (norm): {loss_cfg['weights']}")

    torch.manual_seed(int(train_cfg["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[
        train_cfg["amp"]
    ]
    use_amp = train_cfg["amp"] != "fp32"
    image_size = int(data_cfg["image_size"])

    split_cfg = data_cfg.get("split", {"source": "official"})
    source = split_cfg.get("source", "official")
    if source == "official":
        train_clips, test_clips = official_train_test_split(
            args.data_root, subsets=data_cfg["subsets"]
        )
        n_val_monitor = int(split_cfg.get("n_val_monitor", 15))
        rng = random.Random(int(split_cfg.get("seed", 42)))
        val_clips = rng.sample(train_clips, min(n_val_monitor, len(train_clips)))
        # The monitor clips MUST leave the training set. Without this they are training data, so the
        # "VAL" loss measures fitting rather than generalisation -- it falls while held-out metric-AJ
        # falls with it, which reads as an anti-correlated objective but is plain overfitting. Every
        # lr choice, early stop and "best checkpoint" in the v73-v82 line was selected on that
        # signal. holdout_val=False reproduces the old behaviour for arms already in flight.
        holdout = bool(split_cfg.get("holdout_val", True))
        if holdout:
            _vs = {str(c) for c in val_clips}
            train_clips = [c for c in train_clips if str(c) not in _vs]
        print(
            f"[train] split=official  {len(train_clips)} train / {n_val_monitor} val-monitor "
            f"({'HELD OUT' if holdout else 'IN TRAIN -- not a generalisation signal'}) / "
            f"{len(test_clips)} test  subsets={data_cfg['subsets']}"
        )
    elif source == "minival":
        train_clips, val_clips, test_clips = minival_split(
            args.data_root,
            subsets=data_cfg["subsets"],
            n_train=int(split_cfg.get("n_train", 40)),
            n_val=int(split_cfg.get("n_val", 5)),
            n_test=int(split_cfg.get("n_test", 5)),
            seed=int(split_cfg.get("seed", 42)),
        )
        print(f"[train] split=minival  {len(train_clips)} train / {len(val_clips)} val")
    else:
        train_clips, val_clips = default_train_val(
            args.data_root, subsets=data_cfg["subsets"], val_frac=0.1, seed=42
        )
        print(f"[train] split=legacy  {len(train_clips)} train / {len(val_clips)} val")

    version = cfg["version"]
    # data.depth ("da3l"/"da3g") is the preferred spelling; data.da3_depth_root still works and is
    # IDENTIFIED against the registry rather than trusted, so the log always names which depth this
    # run actually used. Three separate bugs came from that being implicit.
    from mamba3_tracker.data.depth_source import resolve as _resolve_depth
    _spec = data_cfg.get("depth") or data_cfg.get("da3_depth_root")
    if _spec is None:
        raise ValueError("data.depth (da3l|da3g) or data.da3_depth_root is required")
    _depth_src = _resolve_depth(_spec)
    da3_depth_root = _depth_src.root
    print(f"[train] depth source: {_depth_src}")

    # Oversample under-represented (near-range) subsets by replicating their clips.
    # Each replica draws a different augmented window, so this is effective oversampling,
    # not identical repeats. val_clips are sampled above from the un-oversampled list.
    oversample = data_cfg.get("oversample", {})
    if oversample:
        from collections import Counter

        expanded: list = []
        for p in train_clips:
            expanded.extend([p] * int(oversample.get(p.parent.name, 1)))
        train_clips = expanded
        print(
            f"[train] oversampled train clips -> {len(train_clips)} "
            f"{dict(Counter(p.parent.name for p in train_clips))} via {oversample}"
        )

    train_ds = TAPVid3DDataset(
        train_clips,
        window_size=int(train_cfg["window"]),
        augment=True,
        seed=int(train_cfg["seed"]),
        max_queries=int(data_cfg["num_tracks"]),
        image_size=image_size,
        da3_depth_root=da3_depth_root,
    )
    val_ds = TAPVid3DDataset(
        val_clips,
        window_size=int(train_cfg["window"]),
        augment=False,
        seed=0,
        max_queries=int(data_cfg["num_tracks"]),
        image_size=image_size,
        da3_depth_root=da3_depth_root,
    )
    loader = DataLoader(
        train_ds,
        batch_size=int(train_cfg["batch"]),
        shuffle=True,
        num_workers=int(train_cfg["num_workers"]),
        collate_fn=collate_tracking,
        pin_memory=True,
        persistent_workers=False,
    )

    # The front-end lives in the config, never on the command line: cfg.json is the record of what
    # a run actually did, and a setting passed as a flag leaves no trace in it.
    #   flow.source: searaft (default) | waft_live | waft_cached
    #   flow.waft_pred_dir: required by waft_cached, the full_eval track directory
    flow_source = str(flow_cfg.get("source", "searaft"))
    # depth_map (default) samples the DA3 map at the tracked uv. waft uses WAFT's own
    # tracks_XYZ z, which makes a zero-init refiner reproduce tracks_XYZ*exp(ds) exactly.
    z_source = str(flow_cfg.get("z_source", "depth_map"))
    if flow_source not in ("searaft", "waft_live", "waft_cached"):
        raise SystemExit(f"flow.source must be searaft, waft_live or waft_cached; got {flow_source!r}")
    waft_pred_dir = None
    if flow_source == "waft_cached":
        wp = flow_cfg.get("waft_pred_dir")
        if not wp:
            raise SystemExit("flow.source: waft_cached requires flow.waft_pred_dir in the config")
        waft_pred_dir = Path(str(wp)).expanduser()
    if flow_source == "waft_live":
        # Same track_clip, different flow model: this is the only difference between the two arms
        # once the cached path is out of the picture.
        # Same log2 scale SEA-RAFT is given, so the two front-ends run at one resolution and the
        # comparison is of the flow network rather than of two operating points.
        flow_model = _build_waft_flow(device, scale=flow_cfg.get("scale"),
                                      iters=flow_cfg.get("iters"))
        print("[train] WAFT flow model, run LIVE on the augmented images (matches the SEA-RAFT path)")
    else:
        flow_model = FlowModel(
            device,
            url=flow_cfg.get("url", "MemorySlices/Tartan-C-T-TSKH-spring540x960-M"),
            iters=flow_cfg.get("iters"),
            scale=flow_cfg.get("scale"),
        )
    fb_alpha = float(flow_cfg.get("fb_alpha", 0.05))
    fb_beta = float(flow_cfg.get("fb_beta", 1.0))
    if flow_source != "waft_live":
        print(
            f"[train] FlowModel loaded (iters={flow_model.args.iters} scale={flow_model.args.scale})"
        )

    if version in ("v35", "v46"):
        model = Mamba3V35Refiner(
            dim=int(model_cfg["dim"]),
            state_dim=int(model_cfg["state_dim"]),
            num_heads=int(model_cfg["num_heads"]),
            num_layers=int(model_cfg["num_layers"]),
            max_log_correction=float(model_cfg.get("max_log_correction", 2.0)),
            max_delta_uv=float(model_cfg.get("max_delta_uv", 2.0)),
            patch_size=int(model_cfg.get("patch_size", 5)),
            per_frame_scale=bool(model_cfg.get("per_frame_scale", False)),
            within_frame=bool(model_cfg.get("within_frame", False)),
            d_proj=int(model_cfg.get("d_proj", 64)),
            dino_model=str(
                model_cfg.get("dino_model", "facebook/dinov3-vits16-pretrain-lvd1689m")
            ),
            dino_image_size=int(model_cfg.get("dino_image_size", 448)),
            image_size=int(model_cfg.get("image_size", 896)),
            feat_encoder=str(model_cfg.get("feat_encoder", "dinov3")),
            vmamba3_dim=int(model_cfg.get("vmamba3_dim", 384)),
            vmamba3_heads=int(model_cfg.get("vmamba3_heads", 6)),
            vmamba3_blocks=int(model_cfg.get("vmamba3_blocks", 2)),
            vmamba3_patch=int(model_cfg.get("vmamba3_patch", 14)),
            vmamba3_grid=int(model_cfg.get("vmamba3_grid", 32)),
            two_pool=bool(model_cfg.get("two_pool", False)),
            gate_by_vis=bool(model_cfg.get("gate_by_vis", True)),
        ).to(device)
        loss_fn = TrackingLossV35(
            weights=loss_cfg["weights"], image_size=image_size
        ).to(device)
        print("[train] Mamba3V35Refiner  loss: TrackingLossV35")
    elif version == "v45":
        model = Mamba3V45(
            dim=int(model_cfg["dim"]),
            state_dim=int(model_cfg["state_dim"]),
            num_heads=int(model_cfg["num_heads"]),
            num_layers=int(model_cfg["num_layers"]),
            max_log_correction=float(model_cfg.get("max_log_correction", 2.0)),
            max_delta_uv=float(model_cfg.get("max_delta_uv", 2.0)),
            patch_size=int(model_cfg.get("patch_size", 5)),
            max_scale_correction=float(model_cfg.get("max_scale_correction", 0.5)),
            d_proj=int(model_cfg.get("d_proj", 64)),
            dino_model=str(
                model_cfg.get("dino_model", "facebook/dinov3-vits16-pretrain-lvd1689m")
            ),
            dino_image_size=int(model_cfg.get("dino_image_size", 448)),
            image_size=int(model_cfg.get("image_size", 896)),
            two_pool=bool(model_cfg.get("two_pool", False)),
            gate_by_vis=bool(model_cfg.get("gate_by_vis", True)),
        ).to(device)
        loss_fn = TrackingLossV35(
            weights=loss_cfg["weights"], image_size=image_size
        ).to(device)
        print("[train] Mamba3V45 (v44 deflicker + v35 refiner)  loss: TrackingLossV35")
    elif version == "v88":
        model = Mamba3V88(
            dim=int(model_cfg["dim"]),
            state_dim=int(model_cfg["state_dim"]),
            num_heads=int(model_cfg["num_heads"]),
            num_layers=int(model_cfg["num_layers"]),
            max_log_correction=float(model_cfg.get("max_log_correction", 2.0)),
            max_delta_uv=float(model_cfg.get("max_delta_uv", 2.0)),
            patch_size=int(model_cfg.get("patch_size", 5)),
            max_scale_correction=float(model_cfg.get("max_scale_correction", 0.5)),
            scale_stage_correction=float(model_cfg.get("scale_stage_correction", 2.5)),
            d_proj=int(model_cfg.get("d_proj", 64)),
            dino_model=str(
                model_cfg.get("dino_model", "facebook/dinov3-vits16-pretrain-lvd1689m")
            ),
            dino_image_size=int(model_cfg.get("dino_image_size", 448)),
            image_size=int(model_cfg.get("image_size", 896)),
            two_pool=bool(model_cfg.get("two_pool", False)),
            gate_by_vis=bool(model_cfg.get("gate_by_vis", True)),
            grid=int(model_cfg.get("grid", 64)),
            log_ref=float(model_cfg.get("log_ref", 2.0)),
            log_std=float(model_cfg.get("log_std", 1.5)),
        ).to(device)
        loss_fn = TrackingLossV35(
            weights=loss_cfg["weights"], image_size=image_size
        ).to(device)
        print(
            "[train] Mamba3V88 (gated depth-map scale stage + v44 deflicker + v35 refiner)  "
            "loss: TrackingLossV35 + L_dsr"
        )
    elif version == "v47":
        model = Mamba3V45(
            dim=int(model_cfg["dim"]),
            state_dim=int(model_cfg["state_dim"]),
            num_heads=int(model_cfg["num_heads"]),
            num_layers=int(model_cfg["num_layers"]),
            max_log_correction=float(model_cfg.get("max_log_correction", 2.0)),
            max_delta_uv=float(model_cfg.get("max_delta_uv", 2.0)),
            patch_size=int(model_cfg.get("patch_size", 5)),
            max_scale_correction=float(model_cfg.get("max_scale_correction", 0.5)),
            d_proj=int(model_cfg.get("d_proj", 64)),
            dino_model=str(
                model_cfg.get("dino_model", "facebook/dinov3-vits16-pretrain-lvd1689m")
            ),
            dino_image_size=int(model_cfg.get("dino_image_size", 448)),
            image_size=int(model_cfg.get("image_size", 896)),
            two_pool=bool(model_cfg.get("two_pool", False)),
            gate_by_vis=bool(model_cfg.get("gate_by_vis", True)),
            pose_head=True,
        ).to(device)
        loss_fn = TrackingLossV47(
            weights=loss_cfg["weights"], image_size=image_size
        ).to(device)
        print(
            "[train] Mamba3V45+pose_head (v47 shared ego-motion)  loss: TrackingLossV47"
        )
    elif version == "v73":
        model = Mamba3V73(
            dim=int(model_cfg["dim"]),
            state_dim=int(model_cfg["state_dim"]),
            num_heads=int(model_cfg["num_heads"]),
            num_layers=int(model_cfg["num_layers"]),
            max_scale_correction=float(model_cfg.get("max_scale_correction", 2.5)),
            two_pool=bool(model_cfg.get("two_pool", False)),
            grid=int(model_cfg.get("grid", 64)),
            log_ref=float(model_cfg.get("log_ref", 2.0)),
            log_std=float(model_cfg.get("log_std", 1.5)),
            max_log_correction=float(model_cfg["max_log_correction"]),
            max_delta_uv=float(model_cfg["max_delta_uv"]),
            patch_size=int(model_cfg["patch_size"]),
            d_proj=int(model_cfg["d_proj"]),
        ).to(device)
        loss_fn = TrackingLossV35(loss_cfg["weights"]).to(device)
        # The scale stage is trained standalone and loaded here as a fixed depth pre-processor:
        # composing an UNtrained one with the refiner is what produced v74's 0.1792, because the
        # stage began as an identity and then had to learn jointly against the refiner correcting
        # the same error. Frozen, the refiner adapts to depth that is already corrected.
        init_path = model_cfg.get("scale_init")
        if init_path:
            st = torch.load(Path(str(init_path)).expanduser(), map_location="cpu",
                            weights_only=False)["model"]
            own = model.scale_refiner.state_dict()
            bad = [k for k in st if k not in own or own[k].shape != st[k].shape]
            if bad:
                raise SystemExit(
                    f"[train] scale_init {init_path} does not fit scale_refiner: "
                    f"{len(bad)} mismatched, first {bad[0]}")
            model.scale_refiner.load_state_dict(st)
            print(f"[train] scale refiner loaded from {init_path} ({len(st)} tensors)")
            if bool(model_cfg.get("freeze_scale", True)):
                for prm in model.scale_refiner.parameters():
                    prm.requires_grad_(False)
                model.scale_refiner.eval()
                print("[train] scale refiner FROZEN; only the vmamba3 refiner trains")
        print("[train] Mamba3V73 (depth-scale refiner + v35 refiner)  loss: TrackingLossV35 + L_dsr")
    elif version == "v72":
        model = Mamba3DepthScaleRefiner(
            dim=int(model_cfg["dim"]),
            state_dim=int(model_cfg["state_dim"]),
            num_heads=int(model_cfg["num_heads"]),
            num_layers=int(model_cfg["num_layers"]),
            max_scale_correction=float(model_cfg.get("max_scale_correction", 2.5)),
            two_pool=bool(model_cfg.get("two_pool", False)),
            grid=int(model_cfg.get("grid", 64)),
            log_ref=float(model_cfg.get("log_ref", 2.0)),
            log_std=float(model_cfg.get("log_std", 1.5)),
        ).to(device)
        loss_fn = TrackingLossV33(loss_cfg["weights"]).to(device)
        init_path = model_cfg.get("linear_init")
        if init_path:
            n_set = model.load_linear_init(Path(str(init_path)).expanduser())
            print(f"[train] scale bypass seeded from {init_path} ({n_set} tensors)")
        print("[train] Mamba3DepthScaleRefiner (v72)  loss: TrackingLossV33 + L_dsr")
    elif version == "v44":
        model = Mamba3DeflickerRefiner(
            dim=int(model_cfg["dim"]),
            state_dim=int(model_cfg["state_dim"]),
            num_heads=int(model_cfg["num_heads"]),
            num_layers=int(model_cfg["num_layers"]),
            max_scale_correction=float(model_cfg.get("max_scale_correction", 0.5)),
        ).to(device)
        loss_fn = TrackingLossV33(
            weights=loss_cfg["weights"], image_size=image_size
        ).to(device)
        print("[train] Mamba3DeflickerRefiner (v44)  loss: TrackingLossV33 (3D-only)")
    else:
        model = Mamba3DepthRefiner(
            dim=int(model_cfg["dim"]),
            state_dim=int(model_cfg["state_dim"]),
            num_heads=int(model_cfg["num_heads"]),
            num_layers=int(model_cfg["num_layers"]),
            max_log_correction=float(model_cfg.get("max_log_correction", 2.0)),
        ).to(device)
        loss_fn = TrackingLossV33(
            weights=loss_cfg["weights"], image_size=image_size
        ).to(device)
        print("[train] Mamba3DepthRefiner  loss: TrackingLossV33 (3D-only)")

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"[train] trainable={n_trainable:.3f}M params")

    optim = AdamW(
        model.parameters(),
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    n_steps = int(train_cfg["steps"])
    sched = LambdaLR(
        optim,
        lr_lambda=lambda s: wsd(
            s, int(train_cfg["warmup"]), int(train_cfg["decay"]), n_steps
        ),
    )

    history: list[dict] = []
    motion_history: list[dict] = []
    # Read from the config, never a CLI flag. v75's YAML set train.init_ckpt, the run ignored it
    # because only --init-ckpt was consulted, and cfg.json then recorded a warm start that never
    # happened -- v75 silently duplicated v76.
    init_ckpt_cfg = train_cfg.get("init_ckpt")

    cfg_snapshot = {
        **cfg,
        "_launch": {
            "config_path": str(args.config),
            "out_dir": str(args.out_dir),
            "data_root": str(args.data_root),
            "init_ckpt": str(init_ckpt_cfg) if init_ckpt_cfg else None,
        },
    }
    dump_resolved(cfg_snapshot, args.out_dir / "cfg.json")

    start_step = 0
    if init_ckpt_cfg is not None and _find_latest_ckpt(args.out_dir) is None:
        # weights-only warm-start (no optim/sched/step); a resume ckpt in out_dir overrides.
        st = torch.load(
            Path(init_ckpt_cfg).expanduser(), map_location=device, weights_only=False
        )
        # v72's checkpoint stores stage-1 keys at top level while v73 nests them under
        # scale_refiner., so a direct load matches ZERO keys -- and strict=False reports that as
        # success. Remap when the shapes line up under the prefix.
        sd = st["model"]
        own = model.state_dict()
        remap_ok = all(f"scale_refiner.{k}" in own and own[f"scale_refiner.{k}"].shape == v.shape
                       for k, v in sd.items())
        if not (set(sd) & set(own)) and remap_ok:
            sd = {f"scale_refiner.{k}": v for k, v in sd.items()}
            print(f"[train] warm start: remapped {len(sd)} keys under 'scale_refiner.'", flush=True)
        bad = [k for k, v in sd.items() if k in own and own[k].shape != v.shape]
        if bad:
            raise SystemExit(
                f"[train] warm start from {init_ckpt_cfg} is a DIFFERENT architecture: "
                f"{len(bad)} tensors differ in shape, first {bad[0]} "
                f"{tuple(sd[bad[0]].shape)} vs {tuple(own[bad[0]].shape)}. "
                f"Delete the stale output directory or point init_ckpt at a matching run."
            )
        missing, unexpected = model.load_state_dict(sd, strict=False)
        n_loaded = len(own) - len(missing)
        if n_loaded == 0:
            raise SystemExit(
                f"[train] warm start from {init_ckpt_cfg} loaded NOTHING: "
                f"{len(sd)} checkpoint keys matched none of {len(own)} model keys. "
                f"Refusing to train from a silently-ignored initialisation."
            )
        print(f"[train] warm start loaded {n_loaded}/{len(own)} tensors", flush=True)
        print(
            f"[train] warm-started from {init_ckpt_cfg} "
            f"(missing={len(missing)} unexpected={len(unexpected)})",
            flush=True,
        )
    # freeze_scale applies HOWEVER stage 1's weights arrived. It used to live inside the
    # model.scale_init branch, so an arm that warm-started the whole composite through
    # train.init_ckpt got no freeze at all and the flag was a silent no-op -- the same failure as
    # the dead train.init_ckpt key. Placed after the warm start and after optim construction:
    # AdamW skips params whose grad is None, so freezing later is safe.
    # Which submodules to hold fixed, by attribute name. model.freeze_scale stays honoured as a
    # shorthand for ["scale_refiner"]. Applies HOWEVER the weights arrived (scale_init OR
    # train.init_ckpt): the freeze used to live inside the scale_init branch, so an arm warm-started
    # through init_ckpt got no freeze and the flag was a silent no-op.
    _freeze = list(model_cfg.get("freeze_modules", []) or [])
    if bool(model_cfg.get("freeze_scale", False)) and "scale_refiner" not in _freeze:
        _freeze.append("scale_refiner")
    for _name in _freeze:
        _mod = getattr(model, _name, None)
        if _mod is None:
            raise SystemExit(
                f"[train] model.freeze_modules names {_name!r}, which this model does not have. "
                f"Available: {[n for n, _ in model.named_children()]}"
            )
        _held = 0
        for prm in _mod.parameters():
            prm.requires_grad_(False)
            _held += prm.numel()
        _mod.eval()
        print(f"[train] FROZEN {_name}: {_held/1e6:.3f}M params held", flush=True)
    _n_train = sum(q.numel() for q in model.parameters() if q.requires_grad)
    _trainable_names = sorted({n.split(".")[0] for n, q in model.named_parameters() if q.requires_grad})
    print(f"[train] trainable after freezing: {_n_train/1e6:.3f}M across {_trainable_names}", flush=True)

    latest = _find_latest_ckpt(args.out_dir)
    if latest is not None:
        st = torch.load(latest, map_location=device, weights_only=False)
        model.load_state_dict(st["model"], strict=False)
        optim.load_state_dict(st["optim"])
        if st.get("sched") is not None:
            sched.load_state_dict(st["sched"])
        start_step = int(st["step"])
        history = list(st.get("history", []))
        mh_path = args.out_dir / "motion_history.json"
        if mh_path.exists():
            try:
                motion_history = json.loads(mh_path.read_text())
            except json.JSONDecodeError:
                pass
        print(f"[train] RESUMED from {latest} at step {start_step}", flush=True)

    grad_clip_val = float(train_cfg["grad_clip"])
    log_every, val_every, ckpt_every = (
        int(train_cfg["log_every"]),
        int(train_cfg["val_every"]),
        int(train_cfg["ckpt_every"]),
    )

    model.train()
    t0 = time.perf_counter()
    step = start_step
    loader_iter = iter(loader)
    while step < n_steps:
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            batch = next(loader_iter)

        queries = batch.queries_xyt.to(device, non_blocking=True)
        qmask = batch.query_mask.to(device, non_blocking=True)
        depth_d = batch.depth.to(device, non_blocking=True)
        ray, z_raw, vis, uv, images, K_d = _run_flow_batch(
            flow_model, batch, device, image_size, fb_alpha, fb_beta,
            waft_pred_dir=waft_pred_dir, z_source=z_source,
            noise_px=track_noise_px, noise_rho=track_noise_rho,
        )

        pred = _model_forward(
            model,
            version,
            ray,
            z_raw,
            vis,
            uv,
            images,
            depth_d,
            K_d,
            amp_dtype,
            device,
            use_amp,
        )
        loss_out = loss_fn(
            pred,
            batch.tracks_XYZ.to(device, non_blocking=True),
            batch.visibility.to(device, non_blocking=True),
            qmask,
            queries[..., 2].long(),
            K_d,
        )
        # The depth-scale refiner emits one scalar per frame against ~900*F points of tracking loss,
        # so without its own term its gradient is dominated by error it cannot act on.
        total = loss_out.total
        if lambda_dsr > 0 and pred.log_scale is not None:
            l_dsr = depth_scale_refiner_loss(
                pred.log_scale,
                z_raw,
                batch.tracks_XYZ.to(device, non_blocking=True),
                batch.visibility.to(device, non_blocking=True),
            )
            total = total + lambda_dsr * l_dsr
            last_dsr = float(l_dsr.detach())
        total.backward()
        head_grad = _per_head_grad_norm(model)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_val)
        if not torch.isfinite(grad_norm):
            print(f"[train] step {step:6d}: non-finite grad_norm — skip", flush=True)
        elif not torch.isfinite(total):
            print(f"[train] step {step:6d}: non-finite loss — skip", flush=True)
        else:
            optim.step()
        sched.step()
        optim.zero_grad(set_to_none=True)

        if step % log_every == 0:
            lr = sched.get_last_lr()[0]
            dt = time.perf_counter() - t0
            row = _loss_to_dict(loss_out)
            gn = float(grad_norm.item()) if torch.isfinite(grad_norm) else float("nan")
            duv_str = ""
            if pred.delta_uv is not None:
                duv_str = f"  |Δuv|={float(pred.delta_uv.abs().mean().item()):.3f}px"
            print(
                f"[train] step {step:6d}/{n_steps}  {_fmt_loss_row(row, last_dsr)}  lr={lr:.2e}  "
                f"|grad|={gn:.2e}  {_fmt_grad_row(head_grad)}{duv_str}  elapsed={dt:.0f}s",
                flush=True,
            )
            history.append(
                {"step": step, "lr": lr, "grad_norm": gn, "head_grad": head_grad, **row}
            )

        if (step > 0 and step % val_every == 0) or (step == 0 and val_at_step0):
            v = _validate(
                model,
                version,
                flow_model,
                val_ds,
                loss_fn,
                device,
                amp_dtype,
                image_size,
                fb_alpha,
                fb_beta,
                n_clips=val_clips_n,
                waft_pred_dir=waft_pred_dir,
                z_source=z_source,
            )
            print(f"[train] step {step:6d}  VAL     {_fmt_loss_row(v)}", flush=True)
            cur = float(v["total"])
            # The RAW best is saved unconditionally, NOT only when early stopping is on. _save_ckpt
            # prunes every other ckpt_*.pt, so the periodic checkpoints keep just the newest step and
            # the best point on the curve is otherwise unrecoverable: v78a reached its lowest
            # validation (0.0951) at step 750 with early_stop_patience=0, and those weights were lost.
            if cur < es_raw_best:
                es_raw_best = cur
                best_dir = args.out_dir / "best"
                best_dir.mkdir(parents=True, exist_ok=True)
                _save_ckpt(best_dir, step, model, optim, sched, history, cfg_snapshot)
                print(f"[train] best checkpoint: {cur:.4f} at step {step}", flush=True)
            if es_patience > 0:
                # Two separate decisions, deliberately. The checkpoint above follows the RAW best, so
                # a value better by less than the tolerance is still kept; the patience counter uses
                # the tolerance, so noise-sized gains do not postpone stopping forever. Conflating
                # them discards a genuinely better checkpoint for being better by too little.
                if cur < es_best - es_min_delta:
                    es_best, es_since, es_best_step = cur, 0, step
                    print(f"[train] early-stop: improved to {cur:.4f} at step {step}", flush=True)
                else:
                    es_since += 1
                    print(f"[train] early-stop: no improvement ({cur:.4f} vs best "
                          f"{es_best:.4f} @ {es_best_step}), {es_since}/{es_patience}", flush=True)
                    if es_since >= es_patience:
                        print(f"[train] EARLY STOP at step {step}: best {es_best:.4f} was at step "
                              f"{es_best_step}, not beaten by {es_min_delta} in "
                              f"{es_patience} checks", flush=True)
                        _save_ckpt(args.out_dir, step, model, optim, sched, history, cfg_snapshot)
                        break
            history.append({"step": step, "val": v})
            m = _motion_check(
                model,
                version,
                flow_model,
                val_clips,
                device,
                amp_dtype,
                image_size,
                fb_alpha,
                fb_beta,
            )
            print(f"[train] step {step:6d}  MOTION  {_fmt_motion_row(m)}", flush=True)
            motion_history.append(
                {"step": step, **{f"{k}_ratio": r for k, r in m.items()}}
            )
            (args.out_dir / "motion_history.json").write_text(
                json.dumps(motion_history, indent=2)
            )

        if step > 0 and step % ckpt_every == 0:
            p = _save_ckpt(
                args.out_dir, step, model, optim, sched, history, cfg_snapshot
            )
            print(f"[train] saved {p}", flush=True)

        (args.out_dir / "loss_history.json").write_text(json.dumps(history, indent=2))
        step += 1

    # `step`, not n_steps: with early stopping the loop can exit below the ceiling, and labelling
    # those weights ckpt_<n_steps> made a stopped-at-8000 model look like a completed 20000-step run
    # -- which then got evaluated in place of the best checkpoint.
    final = _save_ckpt(
        args.out_dir, step, model, optim, sched, history, cfg_snapshot
    )
    print(f"[train] DONE — {final}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
