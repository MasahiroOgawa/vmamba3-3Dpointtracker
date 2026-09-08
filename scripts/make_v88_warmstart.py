"""Assemble a Mamba3V88 checkpoint: v63's trained v45 + the DA3-g scale refiner, gate at zero.

  v45.*            <- a trained Mamba3V45 run (v63): its de-flicker AND its v35 refiner
  scale_refiner.*  <- the standalone DA3-g Mamba3DepthScaleRefiner
  scale_gate       <- stays 0, so ds_total == ds_deflicker and the model is numerically v63

Scoring the result is therefore a MEASUREMENT of the floor, not an assumption: it is v63's weights
evaluated under whatever protocol this checkpoint declares. Training can only add stage 1's
contribution on top, because the gate starts closed.
"""

import argparse
import sys
import pathlib
import json
import torch

sys.path.insert(0, "src")
from mamba3_tracker.model.depth_refined_tracker import Mamba3V88  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v45-ckpt", type=pathlib.Path, required=True)
    ap.add_argument("--scale-ckpt", type=pathlib.Path, required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument(
        "--iters",
        type=int,
        default=4,
        help="flow iters this arm is trained AND scored with; must match the cache manifest",
    )
    a = ap.parse_args()

    v45 = torch.load(a.v45_ckpt, map_location="cpu", weights_only=False)
    mc = dict(v45.get("cfg", {}).get("model", {}))
    kw = dict(
        dim=int(mc.get("dim", 128)),
        state_dim=int(mc.get("state_dim", 64)),
        num_heads=int(mc.get("num_heads", 4)),
        num_layers=int(mc.get("num_layers", 2)),
        max_log_correction=float(mc.get("max_log_correction", 2.0)),
        max_delta_uv=float(mc.get("max_delta_uv", 2.0)),
        patch_size=int(mc.get("patch_size", 5)),
        max_scale_correction=float(mc.get("max_scale_correction", 0.5)),
        scale_stage_correction=2.5,
        d_proj=int(mc.get("d_proj", 64)),
        dino_model=str(
            mc.get("dino_model", "facebook/dinov3-vits16-pretrain-lvd1689m")
        ),
        dino_image_size=int(mc.get("dino_image_size", 448)),
        image_size=int(mc.get("image_size", 896)),
        two_pool=bool(mc.get("two_pool", False)),
    )
    model = Mamba3V88(**kw)
    tgt = model.state_dict()
    new = {}
    n45 = n_sc = 0
    bad = []
    for k, v in v45["model"].items():
        kk = f"v45.{k}"
        if kk in tgt and tgt[kk].shape == v.shape:
            new[kk] = v
            n45 += 1
        else:
            bad.append(
                ("v45", kk, tuple(v.shape), tuple(tgt[kk].shape) if kk in tgt else None)
            )
    sc = torch.load(a.scale_ckpt, map_location="cpu", weights_only=False)
    for k, v in sc.get("model", sc).items():
        kk = k if k.startswith("scale_refiner.") else f"scale_refiner.{k}"
        if kk in tgt and tgt[kk].shape == v.shape:
            new[kk] = v
            n_sc += 1
        else:
            bad.append(
                (
                    "scale",
                    kk,
                    tuple(v.shape),
                    tuple(tgt[kk].shape) if kk in tgt else None,
                )
            )
    print(f"  v45.*           : {n45} tensors from {a.v45_ckpt}")
    print(f"  scale_refiner.* : {n_sc} tensors from {a.scale_ckpt}")
    if bad:
        print(f"  {len(bad)} tensors did NOT map -- warm start incomplete:")
        for w, k, sh, want in bad[:8]:
            print(f"      [{w}] {k}: have {sh} want {want}")
        return 1
    want45 = sum(1 for k in tgt if k.startswith("v45."))
    wantsc = sum(1 for k in tgt if k.startswith("scale_refiner."))
    if n45 != want45 or n_sc != wantsc:
        print(f"  INCOMPLETE: {n45}/{want45} v45 and {n_sc}/{wantsc} scale_refiner")
        return 1
    new["scale_gate"] = torch.zeros(1)
    model.load_state_dict(new, strict=True)
    if float(model.scale_gate.abs().max()) != 0.0:
        print("  scale_gate is not zero; the floor would not be v63")
        return 1

    cfg = {
        **v45.get("cfg", {}),
        "version": "v88",
        "model": {
            **mc,
            "scale_stage_correction": 2.5,
            "grid": 64,
            "log_ref": 2.0,
            "log_std": 1.5,
        },
    }
    cfg["flow"] = {**cfg.get("flow", {}), "iters": int(a.iters), "source": "waft_live"}
    cfg["flow"].pop("_iters_note", None)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "step": 0, "cfg": cfg}, a.out)
    (a.out.parent / "cfg.json").write_text(json.dumps(cfg, indent=1, sort_keys=True))
    n = sum(q.numel() for q in model.parameters() if q.requires_grad)
    print(
        f"  wrote {a.out} (+cfg.json, flow.iters={a.iters})  trainable={n / 1e6:.3f}M  gate=0"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
