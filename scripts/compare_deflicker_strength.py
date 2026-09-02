"""Does the de-flicker stage learn a weaker correction when trained on WAFT tracks?

De-flicker contributes +0.0074 to a SEA-RAFT-trained DA3-g arm and only +0.0020 to a WAFT-trained
one, and that 0.0054 is most of the gap between them. The stage emits a per-frame log-scale
correction, so "how strong a correction did it learn" is directly measurable: run both checkpoints
over the same clips with the same input and compare the corrections they produce.

If the WAFT-trained stage emits systematically smaller corrections, the mechanism is that WAFT's
more accurate sampling makes the depth look less flickery during training, so the stage is trained
against a weaker signal than the one it faces at evaluation.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--clips", type=int, default=12)
    args = ap.parse_args()

    ckpts = {
        "SEA-RAFT-trained (v45)": "result/v45_scale/ckpt_20000.pt",
        "WAFT-live-trained (v63)": "result/v63_waft_live_2pool/ckpt_20000.pt",
    }
    for tag, c in ckpts.items():
        p = Path(c)
        if not p.exists():
            print(f"  MISSING {tag}: {c}")
            continue
        d = torch.load(p, map_location="cpu", weights_only=False)
        sd = d["model"]
        # the de-flicker head's output layer: its weight and bias scale set the correction range
        keys = [k for k in sd if "deflick" in k.lower() or "scale_head" in k.lower()]
        if not keys:
            keys = [k for k in sd if k.endswith("weight") and sd[k].ndim == 2][-2:]
        print(f"  {tag}")
        for k in keys[:6]:
            t = sd[k].float()
            print(f"      {k[:56]:56s} |w|mean={t.abs().mean():.5f} |w|max={t.abs().max():.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
