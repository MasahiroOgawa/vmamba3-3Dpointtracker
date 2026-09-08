"""Assemble a Mamba3V73 checkpoint from two separately-trained parts.

  stage 1  scale_refiner.*  <- a standalone Mamba3DepthScaleRefiner (train_scale_standalone.py)
  stage 2  v35.*            <- the v35 refiner already trained inside a Mamba3V45 run (e.g. v63)

Both must come from the SAME depth root. Mixing them is what invalidated v73-v85: those arms loaded
a DA3-l-trained scale refiner and fed it DA3-g depth, which doc/plan_da3g_reeval.md had already
ruled out for the v33/v35/v39 refiners.
"""

import argparse
import sys
import pathlib
import torch

sys.path.insert(0, "src")
from mamba3_tracker.model.depth_refined_tracker import Mamba3V73  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale-ckpt", type=pathlib.Path, required=True)
    ap.add_argument(
        "--v45-ckpt",
        type=pathlib.Path,
        required=True,
        help="a Mamba3V45 run; its v35.* tensors become stage 2",
    )
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument(
        "--iters",
        type=int,
        default=4,
        help="flow iters this composite is trained AND scored with; "
        "must match the scoring cache's manifest",
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
        d_proj=int(mc.get("d_proj", 64)),
        dino_model=str(
            mc.get("dino_model", "facebook/dinov3-vits16-pretrain-lvd1689m")
        ),
        dino_image_size=int(mc.get("dino_image_size", 448)),
        image_size=int(mc.get("image_size", 896)),
        two_pool=bool(mc.get("two_pool", False)),
        max_scale_correction=2.5,  # stage 1's own bound, wide enough not to clip
    )
    model = Mamba3V73(**kw)
    tgt = model.state_dict()

    sc = torch.load(a.scale_ckpt, map_location="cpu", weights_only=False)
    sc_sd = sc.get("model", sc)
    new, loaded_scale, loaded_v35, skipped = {}, 0, 0, []
    for k, v in sc_sd.items():
        kk = k if k.startswith("scale_refiner.") else f"scale_refiner.{k}"
        if kk in tgt and tgt[kk].shape == v.shape:
            new[kk] = v
            loaded_scale += 1
        else:
            skipped.append(
                (
                    "scale",
                    k,
                    tuple(v.shape),
                    tuple(tgt[kk].shape) if kk in tgt else None,
                )
            )
    for k, v in v45["model"].items():
        if not k.startswith("v35."):
            continue
        if k in tgt and tgt[k].shape == v.shape:
            new[k] = v
            loaded_v35 += 1
        else:
            skipped.append(
                ("v35", k, tuple(v.shape), tuple(tgt[k].shape) if k in tgt else None)
            )

    print(f"  stage1 scale_refiner.*: {loaded_scale} tensors from {a.scale_ckpt}")
    print(f"  stage2 v35.*          : {loaded_v35} tensors from {a.v45_ckpt}")
    if skipped:
        print(
            f"  SKIPPED {len(skipped)} tensors -- shape or name mismatch, warm start is INCOMPLETE:"
        )
        for w, k, sh, want in skipped[:8]:
            print(f"      [{w}] {k}: have {sh} want {want}")
        return 1
    missing = [k for k in tgt if k not in new]
    if missing:
        print(f"  {len(missing)} target tensors unfilled (kept at init): {missing[:6]}")
    n_scale = sum(1 for k in tgt if k.startswith("scale_refiner."))
    n_v35 = sum(1 for k in tgt if k.startswith("v35."))
    if loaded_scale != n_scale or loaded_v35 != n_v35:
        print(
            f"  INCOMPLETE: filled {loaded_scale}/{n_scale} scale and {loaded_v35}/{n_v35} v35"
        )
        return 1
    model.load_state_dict(new, strict=True)
    cfg = {**v45.get("cfg", {}), "model": {**mc, "max_scale_correction": 2.5}}
    # The flow protocol must be the one this composite is TRAINED AND SCORED under, not the one the
    # donor checkpoint used. v63 ran iters=5 (accidentally); v87 runs iters=4 to match the manifest
    # of waft_minival_is896_s-1_i4. eval_metric3d reads <ckpt dir>/cfg.json and refuses a cache whose
    # manifest disagrees, which is why the donor's value cannot be carried over.
    cfg["flow"] = {**cfg.get("flow", {}), "iters": int(a.iters), "source": "waft_live"}
    cfg["flow"].pop("_iters_note", None)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "step": 0, "cfg": cfg}, a.out)
    import json as _json

    (a.out.parent / "cfg.json").write_text(_json.dumps(cfg, indent=1, sort_keys=True))
    print(f"  wrote {a.out} and {a.out.parent / 'cfg.json'} (flow.iters={a.iters})")
    print("  both stages fully warm-started")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
