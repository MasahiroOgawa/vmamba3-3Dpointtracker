"""Per-frame pyramid VSSD encoder.

Image → patch embed at the coarsest pyramid level → stack of VSSD self-attention
blocks → bilinear upsample + residual VSSD blocks at successively finer
resolutions. Produces a list of feature grids at multiple scales:

    F_l ∈ ℝ^(B, D, H_l, W_l)  for l = 0..L-1,
    H_l = base * 2**l,  W_l = base * 2**l   (with `base` typically 32)

The propagator (Step 3) reads the *highest-resolution* level as the kv side.

The pyramid is built in feature space — the same VSSD operator runs at every
scale on its own 2-D token grid, and each finer-resolution feature map is
expressed as `up(coarser) + residual` per the user's requested style.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn

from visionmamba3.rope2d import RoPE2D
from visionmamba3.vssd_attention import Mamba3VSSDAttention


def _build_pos2d(h: int, w: int, device: torch.device) -> Tensor:
    """Integer (y, x) per token, ready for RoPE2D. Shape: (1, h*w, 2)."""
    ys = torch.arange(h, device=device)
    xs = torch.arange(w, device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    pos = torch.stack([yy.flatten(), xx.flatten()], dim=-1)
    return pos.unsqueeze(0)


class VSSDBlock(nn.Module):
    """One pre-norm VSSD self-attention block on a flat token sequence.

    Equivalent to a residual `LayerNorm → Mamba3VSSDAttention → +` block. The
    optional FFN is intentionally skipped — empirically the SSD/VSSD attention
    blocks in `mamba3_attn` are powerful enough without a separate MLP, and
    we want the per-frame encoder to stay compact.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 6,
        state_dim: int = 64,
        rope: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = Mamba3VSSDAttention(
            dim=dim, num_heads=num_heads, state_dim=state_dim, rope=rope,
        )

    def forward(self, x: Tensor, pos: Tensor) -> Tensor:
        return x + self.attn(self.norm(x), pos=pos)


class PatchEmbed(nn.Module):
    """Conv2d-based patch embedding: (B, 3, H, W) → (B, D, H/P, W/P)."""

    def __init__(self, in_channels: int = 3, dim: int = 384, patch: int = 14) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_channels, dim, kernel_size=patch, stride=patch)
        self.patch = patch

    def forward(self, image: Tensor) -> Tensor:
        return self.proj(image)


class PyramidEncoder(nn.Module):
    """Build a feature pyramid in token space.

    Coarsest level: `level_sizes[0] × level_sizes[0]` token grid produced by
    patch-embedding the input image at the right resolution (resized to
    `level_sizes[0] * patch`).

    Each finer level upsamples the previous feature map by 2× via bilinear
    interpolation, then adds a residual VSSD block computed on the upsampled
    grid:

        F_{l+1} = up(F_l) + VSSDBlock(up(F_l))

    The same VSSDBlock weights are *not* shared across levels — one fresh
    block per pyramid level. Each block uses its own RoPE2D so position
    information is rebuilt at the new resolution.
    """

    def __init__(
        self,
        dim: int = 384,
        num_heads: int = 6,
        state_dim: int = 64,
        patch: int = 14,
        level_sizes: tuple[int, ...] = (32, 64, 128),
        blocks_per_level: int = 2,
    ) -> None:
        super().__init__()
        if len(level_sizes) < 1:
            raise ValueError("need at least 1 pyramid level")
        self.dim = dim
        self.patch = patch
        self.level_sizes = tuple(level_sizes)
        self.coarse_size = level_sizes[0]

        # Each pyramid level uses its own RoPE (cached buffers depend on width).
        self.ropes = nn.ModuleList(
            [RoPE2D(base_frequency=100.0) for _ in level_sizes]
        )
        self.patch_embed = PatchEmbed(in_channels=3, dim=dim, patch=patch)
        self.coarse_blocks = nn.ModuleList(
            [VSSDBlock(dim=dim, num_heads=num_heads, state_dim=state_dim,
                       rope=self.ropes[0])
             for _ in range(blocks_per_level)]
        )
        self.refine_blocks = nn.ModuleList(
            [VSSDBlock(dim=dim, num_heads=num_heads, state_dim=state_dim,
                       rope=self.ropes[l])
             for l in range(1, len(level_sizes))]
        )

    @property
    def coarse_image_size(self) -> int:
        return self.coarse_size * self.patch

    def _to_tokens(self, feat: Tensor) -> Tensor:
        """(B, D, h, w) → (B, h*w, D)."""
        B, D, h, w = feat.shape
        return feat.flatten(2).transpose(1, 2)

    def _to_grid(self, tokens: Tensor, h: int, w: int) -> Tensor:
        """(B, h*w, D) → (B, D, h, w)."""
        return tokens.transpose(1, 2).reshape(tokens.shape[0], self.dim, h, w)

    def forward(self, image: Tensor) -> list[Tensor]:
        """
        Args:
            image: (B, 3, H, W). Will be resized to `coarse_image_size`² before
                patch embedding.

        Returns:
            features: list of `len(level_sizes)` tensors, each (B, D, H_l, W_l).
        """
        if image.shape[-1] != self.coarse_image_size or image.shape[-2] != self.coarse_image_size:
            image = torch.nn.functional.interpolate(
                image, size=(self.coarse_image_size, self.coarse_image_size),
                mode="bilinear", align_corners=False,
            )

        feat = self.patch_embed(image)                     # (B, D, h0, w0)
        h, w = feat.shape[-2:]
        pos = _build_pos2d(h, w, feat.device)
        tokens = self._to_tokens(feat)                     # (B, T, D)
        pos_b = pos.expand(tokens.shape[0], -1, -1)
        for blk in self.coarse_blocks:
            tokens = blk(tokens, pos_b)
        feat = self._to_grid(tokens, h, w)

        out: list[Tensor] = [feat]
        for level_idx, refine in enumerate(self.refine_blocks, start=1):
            new_h = self.level_sizes[level_idx]
            new_w = self.level_sizes[level_idx]
            feat_up = torch.nn.functional.interpolate(
                feat, size=(new_h, new_w), mode="bilinear", align_corners=False,
            )
            pos = _build_pos2d(new_h, new_w, feat_up.device)
            pos_b = pos.expand(feat_up.shape[0], -1, -1)
            tokens_up = self._to_tokens(feat_up)
            delta = refine(tokens_up, pos_b)               # residual built inside
            feat = self._to_grid(delta, new_h, new_w)
            out.append(feat)
        return out

    def forward_video(self, video: Tensor) -> list[Tensor]:
        """Run the encoder independently on every frame of (B, F, 3, H, W).

        Returns a list of pyramid levels, each (B, F, D, H_l, W_l).
        """
        B, F, C, H, W = video.shape
        flat = video.reshape(B * F, C, H, W)
        pyr = self.forward(flat)
        out: list[Tensor] = []
        for level in pyr:
            _, D, h, w = level.shape
            out.append(level.reshape(B, F, D, h, w))
        return out
