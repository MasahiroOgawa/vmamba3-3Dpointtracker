#!/usr/bin/env python
"""Build a v45 warm-start checkpoint = v44 de-flicker (stage 1) + v42 v35 refiner (stage 2).

Both submodules are loaded from their trained checkpoints so v45 fine-tunes the
two-stage pipeline rather than learning it from scratch. Saves result/v45_init.pt
in the trainer's checkpoint format ({"model", "step", "cfg"}).
"""

import sys
from pathlib import Path

import torch

from mamba3_tracker.model.depth_refined_tracker import Mamba3V45
from mamba3_tracker.train.config import load_config

V44_CKPT = Path("result/v44_deflicker/ckpt_20000.pt")
V42_CKPT = Path("result/v42_near/ckpt_20000.pt")
OUT = Path("result/v45_init.pt")


def main() -> int:
    for p in (V44_CKPT, V42_CKPT):
        if not p.exists():
            print(f"[build_v45_init] missing {p}", file=sys.stderr)
            return 1
    cfg = load_config("configs/v45.yaml")
    mc = cfg["model"]
    model = Mamba3V45(
        dim=int(mc["dim"]),
        state_dim=int(mc["state_dim"]),
        num_heads=int(mc["num_heads"]),
        num_layers=int(mc["num_layers"]),
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
    )
    v44 = torch.load(V44_CKPT, map_location="cpu", weights_only=False)
    v42 = torch.load(V42_CKPT, map_location="cpu", weights_only=False)
    md, ud = model.deflicker.load_state_dict(v44["model"], strict=False)
    mv, uv = model.v35.load_state_dict(v42["model"], strict=False)
    print(f"[build_v45_init] deflicker: missing={len(md)} unexpected={len(ud)}")
    print(f"[build_v45_init] v35:       missing={len(mv)} unexpected={len(uv)}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "step": 0, "cfg": cfg}, OUT)
    print(f"[build_v45_init] saved {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
