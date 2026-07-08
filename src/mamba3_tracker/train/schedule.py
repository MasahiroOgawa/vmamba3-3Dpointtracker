"""LR schedule helpers (small wrapper around the WSD schedule from
`mamba3_attn.train.train_super._wsd_lambda`).
"""

from __future__ import annotations

import math


def wsd(step: int, warmup: int, decay: int, total: int, floor: float = 0.1) -> float:
    """Warmup-Stable-Decay: linear up to `warmup`, constant 1.0 until
    `total - decay`, then cosine down to `floor`."""
    if step < warmup:
        return step / max(1, warmup)
    stable_end = total - decay
    if step < stable_end:
        return 1.0
    prog = (step - stable_end) / max(1, decay)
    return floor + 0.5 * (1 - floor) * (1 + math.cos(math.pi * prog))
