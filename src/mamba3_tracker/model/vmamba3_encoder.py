"""Trainable Vision-Mamba-3 (NC-SSD) image feature encoder.

Drop-in replacement for :class:`DINOv2Encoder` inside :class:`Mamba3V35Refiner`
(v48/v49): exposes ``.dim`` and ``forward_video(video) -> [(B, F, D, g, g)]`` with
the same contract, so ``_sample_dino`` is unchanged. Unlike the frozen DINOv3
backbone, this encoder is trained from scratch, so the refiner's per-track
appearance features are computed by a genuine Vision-Mamba-3 encoder — letting
the evaluation report the standalone ability of Vision-Mamba-3 features rather
than borrowing an off-the-shelf backbone.

Internally it is a single-level :class:`PyramidEncoder` (Conv2d patch embed +
2-D RoPE + non-causal VSSD blocks), which already emits a dense
``(B, F, D, g, g)`` feature grid.
"""

from __future__ import annotations

from torch import Tensor, nn

from .encoder import PyramidEncoder


class VMamba3Encoder(nn.Module):
    """Vision-Mamba-3 (NC-SSD) dense feature encoder, trained from scratch.

    Args:
        dim: feature dimension D of the emitted grid (matches DINOv3-S's 384 by
            default so ``feat_proj`` and warm-started refiner weights line up).
        num_heads: heads per VSSD block (D must divide evenly).
        state_dim: NC-SSD state dimension per head.
        patch: Conv2d patch size; input is resized to ``grid * patch`` px.
        grid: side length of the single-level token grid (grid x grid tokens).
        blocks: number of VSSD blocks.
    """

    def __init__(
        self,
        dim: int = 384,
        num_heads: int = 6,
        state_dim: int = 64,
        patch: int = 14,
        grid: int = 32,
        blocks: int = 2,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.encoder = PyramidEncoder(
            dim=dim,
            num_heads=num_heads,
            state_dim=state_dim,
            patch=patch,
            level_sizes=(grid,),
            blocks_per_level=blocks,
        )

    @staticmethod
    def _normalize(video: Tensor) -> Tensor:
        # Images arrive in [0, 1]; map to ~[-1, 1] for a stable from-scratch stem.
        return video * 2.0 - 1.0

    def forward_video(self, video: Tensor) -> list[Tensor]:
        """(B, F, 3, H, W) -> single-element list [(B, F, D, grid, grid)]."""
        return self.encoder.forward_video(self._normalize(video))
