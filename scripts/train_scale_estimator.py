"""Train the standalone scene-scale estimator on TAPVid-3D.

Single output: per-clip s = median anchor-frame Z. Loss: L1 |s_pred − s_gt|.
The whole point is to verify scale-prediction trains cleanly on its own —
if this converges to a small per-clip MAE, scale isn't the joint tracker's
bottleneck. See `src/mamba3_tracker/model/scale_estimator.py` for the model.

Usage:
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \\
        uv run python scripts/train_scale_estimator.py \\
            --config configs/scale_est_v1.yaml \\
            --out-dir result/<datetime>_scale_est_v1
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import torch
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from mamba3_tracker.data.dataset import (
    TAPVid3DDataset, collate_tracking, official_train_test_split,
)
from mamba3_tracker.model.scale_estimator import ScaleEstimator, gt_scale_from_batch
from mamba3_tracker.train.schedule import wsd


def _which_subset(path: Path) -> str:
    p = str(path)
    for s in ("pstudio", "drivetrack", "adt"):
        if f"/{s}/" in p:
            return s
    return "unknown"


@torch.no_grad()
def _validate(model, val_ds, val_clips, device, amp_dtype) -> dict[str, float]:
    """MAE + relative-error stats over val clips, both overall and per-subset.

    `val_clips` is the list of clip paths used to build `val_ds`; we pair them
    by index so per-subset labels work (TAPVid3DDataset doesn't expose the
    clip path on each returned sample).
    """
    model.eval()
    per_sub_abs: dict[str, list[float]] = defaultdict(list)
    per_sub_rel: dict[str, list[float]] = defaultdict(list)
    for i in range(len(val_ds)):
        sample = val_ds[i]
        sub = _which_subset(Path(val_clips[i])) if i < len(val_clips) else "unknown"
        batch = collate_tracking([sample])
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=(device.type == "cuda")):
            s_pred = model(batch.images.to(device, non_blocking=True))     # (1,)
        s_gt = gt_scale_from_batch(
            batch.tracks_XYZ.to(device), batch.queries_xyt.to(device),
            batch.query_mask.to(device),
        )
        sp = float(s_pred[0].item())
        sg = float(s_gt[0].item())
        per_sub_abs[sub].append(abs(sp - sg))
        per_sub_rel[sub].append(abs(sp - sg) / max(sg, 1e-3))
    out = {}
    all_abs, all_rel = [], []
    for sub in sorted(per_sub_abs):
        out[f"mae_{sub}"] = sum(per_sub_abs[sub]) / max(1, len(per_sub_abs[sub]))
        out[f"rel_{sub}"] = sum(per_sub_rel[sub]) / max(1, len(per_sub_rel[sub]))
        all_abs += per_sub_abs[sub]; all_rel += per_sub_rel[sub]
    out["mae"] = sum(all_abs) / max(1, len(all_abs))
    out["rel"] = sum(all_rel) / max(1, len(all_rel))
    model.train()
    return out


def _save_ckpt(out_dir: Path, step: int, model, optim, sched, history) -> Path:
    """Atomic ckpt save; keeps only the latest to avoid disk bloat."""
    p_tmp = out_dir / f".ckpt_{step}.pt.tmp"
    p_final = out_dir / f"ckpt_{step}.pt"
    torch.save({
        "step": step,
        "model": model.state_dict(),
        "optim": optim.state_dict(),
        "sched": sched.state_dict(),
        "history": history,
    }, p_tmp)
    p_tmp.rename(p_final)
    # Rotate: keep only the current step ckpt + ckpt_best.pt (the separate
    # best-MAE snapshot saved by the val loop).
    for old in out_dir.glob("ckpt_*.pt"):
        if old.name == p_final.name or old.name == "ckpt_best.pt":
            continue
        old.unlink(missing_ok=True)
    return p_final


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, default=Path.home() / "data")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    supported = ("scale_est_v1", "scale_est_v2", "scale_est_v3", "scale_est_v4", "scale_est_v5", "scale_est_v6", "scale_est_v7", "scale_est_v8", "scale_est_v9")
    if cfg.get("version") not in supported:
        raise ValueError(f"version must be one of {supported}, got {cfg.get('version')!r}")
    if args.steps is not None: cfg["train"]["steps"] = args.steps
    if args.batch is not None: cfg["train"]["batch"] = args.batch

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "cfg.json").write_text(json.dumps(cfg, indent=2))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = str(cfg["train"].get("amp", "bf16"))
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[amp]

    model = ScaleEstimator(
        dinov2_model=cfg["model"]["dinov2_model"],
        dinov2_image_size=int(cfg["model"]["dinov2_image_size"]),
        num_heads=int(cfg["model"].get("num_heads", 6)),
        state_dim=int(cfg["model"].get("state_dim", 64)),
        head_hidden=int(cfg["model"].get("head_hidden", 384)),
        param=str(cfg["model"].get("param", "softplus")),
        use_patches=bool(cfg["model"].get("use_patches", False)),
        fuse_layers=cfg["model"].get("fuse_layers", None),
    ).to(device)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"[scale-est] params: trainable={n_trainable/1e6:.2f}M  total={n_total/1e6:.2f}M  "
          f"image_size={model.image_size}", flush=True)

    # Train on FULL_EVAL (4418 clips), validate on MINIVAL (150 held-out clips).
    # Earlier setup used 15 random clips FROM TRAIN as val — that measured
    # training-distribution fit, not generalisation, and val MAE was 4× too
    # optimistic vs broader train sample. Per user 2026-06-01.
    train_clips, test_clips = official_train_test_split(args.data_root, subsets=cfg["data"]["subsets"])
    val_clips = list(test_clips)
    print(f"[scale-est] {len(train_clips)} train clips (FULL_EVAL), "
          f"{len(val_clips)} val=test clips (MINIVAL, held-out)", flush=True)

    train_ds = TAPVid3DDataset(
        train_clips, window_size=int(cfg["train"]["window"]),
        augment=True, seed=int(cfg["train"]["seed"]),
        max_queries=int(cfg["data"]["num_tracks"]),
        image_size=int(cfg["data"]["image_size"]),
    )
    val_ds = TAPVid3DDataset(
        val_clips, window_size=int(cfg["train"]["window"]),
        augment=False, seed=0,
        max_queries=int(cfg["data"]["num_tracks"]),
        image_size=int(cfg["data"]["image_size"]),
    )
    loader = DataLoader(
        train_ds, batch_size=int(cfg["train"]["batch"]),
        num_workers=int(cfg["train"]["num_workers"]),
        collate_fn=collate_tracking, shuffle=True, pin_memory=True,
        persistent_workers=False,
    )

    optim = AdamW(model.parameters(),
                  lr=float(cfg["train"]["lr"]),
                  weight_decay=float(cfg["train"]["weight_decay"]))
    n_steps = int(cfg["train"]["steps"])
    warmup = int(cfg["train"]["warmup"])
    decay = int(cfg["train"]["decay"])
    sched = LambdaLR(optim, lr_lambda=lambda s: wsd(s, warmup, decay, n_steps))
    grad_clip = float(cfg["train"]["grad_clip"])

    log_every = int(cfg["train"]["log_every"])
    val_every = int(cfg["train"]["val_every"])
    ckpt_every = int(cfg["train"]["ckpt_every"])
    loss_form = str(cfg["train"].get("loss_form", "L1"))
    if loss_form not in ("L1", "log_L1"):
        raise ValueError(f"train.loss_form must be 'L1' or 'log_L1', got {loss_form!r}")

    model.train()
    step = 0
    t0 = time.perf_counter()
    history: list = []
    best_mae = float("inf")
    best_step = -1
    while step < n_steps:
        for batch in loader:
            if step >= n_steps: break
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=(device.type == "cuda")):
                s_pred = model(batch.images.to(device, non_blocking=True))
            s_gt = gt_scale_from_batch(
                batch.tracks_XYZ.to(device), batch.queries_xyt.to(device),
                batch.query_mask.to(device),
            )
            if loss_form == "log_L1":
                # v3: equalises relative error across subsets — drivetrack
                # (s_gt ~ 20-50m) no longer dominates over adt (~1m) in the
                # gradient. matches the "% target across all depths" geometry.
                loss = (torch.log(s_pred) - torch.log(s_gt)).abs().mean()
            else:
                loss = (s_pred - s_gt).abs().mean()
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            if torch.isfinite(gn) and torch.isfinite(loss):
                optim.step()
            sched.step()
            optim.zero_grad(set_to_none=True)

            if step % log_every == 0:
                lr = sched.get_last_lr()[0]
                dt = time.perf_counter() - t0
                row = {
                    "step": step, "lr": lr,
                    "loss": float(loss.item()),
                    "s_pred_mean": float(s_pred.mean().item()),
                    "s_gt_mean":   float(s_gt.mean().item()),
                    "grad_norm":   float(gn.item()) if torch.isfinite(gn) else float("nan"),
                    "elapsed_s": dt,
                }
                history.append(row)
                print(f"[scale-est] step {step:6d}/{n_steps}  "
                      f"L1={row['loss']:.4f}  "
                      f"s_pred={row['s_pred_mean']:.3f} s_gt={row['s_gt_mean']:.3f}  "
                      f"lr={lr:.2e}  |grad|={row['grad_norm']:.2e}  "
                      f"elapsed={dt:.0f}s", flush=True)
                (args.out_dir / "loss_history.json").write_text(json.dumps(history, indent=2))

            if step > 0 and step % val_every == 0:
                v = _validate(model, val_ds, val_clips, device, amp_dtype)
                history.append({"step": step, "val": v})
                vs = "  ".join(f"{k}={v[k]:.3f}" for k in sorted(v))
                print(f"[scale-est] step {step:6d}  VAL  {vs}", flush=True)
                (args.out_dir / "loss_history.json").write_text(json.dumps(history, indent=2))
                # Save the best-MAE ckpt separately so we can eval the model at
                # its best val rather than the noisy final step (v4 drifted from
                # 1.79 m at step 3500 to 3.08 m at step 4500).
                if v["mae"] < best_mae:
                    best_mae = v["mae"]
                    best_step = step
                    torch.save({
                        "step": step, "val": v, "model": model.state_dict(),
                    }, args.out_dir / "ckpt_best.pt")
                    print(f"[scale-est] step {step:6d}  NEW BEST mae={v['mae']:.3f}  → ckpt_best.pt",
                          flush=True)

            if step > 0 and step % ckpt_every == 0:
                p = _save_ckpt(args.out_dir, step, model, optim, sched, history)
                print(f"[scale-est] saved {p}", flush=True)
            step += 1

    final = _save_ckpt(args.out_dir, n_steps, model, optim, sched, history)
    print(f"[scale-est] DONE — {final}  (best val mae={best_mae:.3f} at step {best_step})",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
