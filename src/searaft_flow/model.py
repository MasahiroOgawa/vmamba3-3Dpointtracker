"""Frozen-inference adapter around the SEA-RAFT submodule.

`third_party/SEA-RAFT` is read-only upstream code (see CLAUDE.md): we only put
its package dirs on `sys.path` and call its public `RAFT` class — never edit it.
This mirrors how `mamba3_attn` wraps Depth-Anything-3 at runtime rather than
forking it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor

_SEARAFT_ROOT = Path(__file__).resolve().parents[2] / "third_party" / "SEA-RAFT"
_DEFAULT_CFG = _SEARAFT_ROOT / "config" / "eval" / "spring-M.json"
# SEA-RAFT's recommended general-purpose checkpoint (Spring 540x960, "M" size).
_DEFAULT_URL = "MemorySlices/Tartan-C-T-TSKH-spring540x960-M"


def _ensure_on_path() -> None:
    # custom.py does `sys.path.append('core')` + bare `from raft import RAFT`,
    # plus `from config.parser import ...` off the repo root. Replicate both.
    for p in (_SEARAFT_ROOT, _SEARAFT_ROOT / "core"):
        s = str(p)
        if s not in sys.path:
            sys.path.insert(0, s)


class FlowModel:
    """Wraps SEA-RAFT for forward dense optical flow.

    `flow(img1, img2)` returns the flow field mapping img1 pixels to img2,
    at the input resolution. Reproduces upstream `custom.calc_flow`, which runs
    the network at `2**scale` of the input then rescales the flow back (spring-M
    uses scale=-1, i.e. half-resolution inference).
    """

    def __init__(
        self,
        device: torch.device,
        cfg_path: Path = _DEFAULT_CFG,
        url: str = _DEFAULT_URL,
        iters: int | None = None,
        scale: int | None = None,
    ) -> None:
        _ensure_on_path()
        from config.parser import json_to_args
        from raft import RAFT

        args = json_to_args(str(cfg_path))
        if iters is not None:
            args.iters = int(iters)
        if scale is not None:
            args.scale = int(scale)
        self.args = args
        self.device = device
        model = RAFT.from_pretrained(url, args=args)
        self.model = model.to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def flow(self, img1: Tensor, img2: Tensor) -> Tensor:
        """Dense flow img1 -> img2.

        Args:
            img1, img2: (B, 3, H, W) float RGB in [0, 255] on `self.device`.
        Returns:
            flow: (B, 2, H, W) pixel displacement at input resolution.
        """
        s = self.args.scale
        i1 = F.interpolate(img1, scale_factor=2 ** s, mode="bilinear", align_corners=False)
        i2 = F.interpolate(img2, scale_factor=2 ** s, mode="bilinear", align_corners=False)
        out = self.model(i1, i2, iters=self.args.iters, test_mode=True)
        flow = out["flow"][-1]
        return F.interpolate(
            flow, scale_factor=0.5 ** s, mode="bilinear", align_corners=False,
        ) * (0.5 ** s)
