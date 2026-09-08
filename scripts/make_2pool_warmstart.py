"""Build a two_pool Mamba3V45 from a 1-pool checkpoint.

pool2_gate is zero-init and the second pool enters as y += pool2_gate * (...), so the result is
numerically identical to the source model at step 0. Scoring it must therefore reproduce the
source's metric-AJ exactly; if it does not, the warm start or the eval protocol is wrong.
"""

import argparse
import sys
import pathlib
import torch

sys.path.insert(0, "src")
from mamba3_tracker.model.depth_refined_tracker import Mamba3V45  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=pathlib.Path, required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    a = ap.parse_args()

    st = torch.load(a.src, map_location="cpu", weights_only=False)
    mc = st.get("cfg", {}).get("model", {})
    kw = dict(
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
    )
    model = Mamba3V45(**kw, two_pool=True)
    missing, unexpected = model.load_state_dict(st["model"], strict=False)
    gates = [k for k in missing if "pool2_gate" in k]
    print(f"  loaded from {a.src}")
    print(
        f"    missing={len(missing)} (expected: the 2nd pool only)  unexpected={len(unexpected)}"
    )
    if unexpected:
        print(f"    UNEXPECTED KEYS -- architecture mismatch: {unexpected[:5]}")
        return 1
    # The 2nd pool is q_proj2 (query proj), m2 (mask proj) and pool2_gate. m2 being randomly
    # initialised is harmless: the gate multiplies the WHOLE second-pool term
    # (cross_attention.py:198 `y = y + pool2_gate * ...`), so step 0 is unchanged.
    POOL2 = ("q_proj2", "m2.", "pool2_gate")
    if not all(any(t in k for t in POOL2) for k in missing):
        bad = [k for k in missing if not any(t in k for t in POOL2)]
        print(
            f"    MISSING KEYS OUTSIDE THE 2ND POOL -- warm start is incomplete: {bad[:8]}"
        )
        return 1
    for k, v in model.state_dict().items():
        if "pool2_gate" in k and float(v.abs().max()) != 0.0:
            print(
                f"    {k} is NOT zero ({float(v)}); step 0 would not equal the source"
            )
            return 1
    print(
        f"    pool2 gates: {len(gates)}, all zero -> step 0 is numerically identical to the source"
    )
    new_cfg = dict(st.get("cfg", {}))
    new_cfg["model"] = {**mc, "two_pool": True}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "step": 0, "cfg": new_cfg}, a.out)
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  wrote {a.out}  trainable={n / 1e6:.3f}M")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
