"""Frozen DINOv2 backbone for v14.

Drop-in replacement for `PyramidEncoder` when the upstream patch-embed +
multi-block VSSD pyramid was the model-quality bottleneck (v6–v13 failure
mode: encoder couldn't resolve sub-patch motion because it never saw the
image at finer than 14-px granularity, and the multi-level "feature
pyramid" was just bilinear-upsampled coarse features).

Design:

  * One frozen forward pass through DINOv2 per frame.
  * Output: a single feature grid per frame at the native DINO patch
    resolution (32×32 for 448×448 input at patch=14).
  * Output dim = 384 — same as the propagator's `dim`, so no projection
    layer.
  * No gradients flow through DINO. The encoder's `.parameters()` are
    marked `requires_grad=False`.

Compare to v13's `PyramidEncoder`:
  * v13 trained ~1.4M params from scratch on 80 clips → could not learn
    drivetrack motion (motion ratio ~2-8% across 6000 steps).
  * v14 uses ~22M pretrained-on-1.4B-images DINO params, frozen → strong
    per-patch appearance features without us spending gradient on them.

Memory at inference is unchanged from v13 since we ditched DINO's
gradient bookkeeping (no `backward()` through it).
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class DINOv2Encoder(nn.Module):
    """Frozen DINOv2 image encoder.

    Args:
        model_name: HuggingFace hub id. Default `facebook/dinov2-small`
            (22M params, hidden_size=384, patch=14). For 448×448 input
            this produces a 32×32 patch token grid plus a CLS token.
        image_size: side length the image gets resized to before feeding
            DINO. Must be a multiple of `patch=14`. Default 448 → 32×32
            grid.

    Output of `.forward_video(video)`:
        list of one tensor with shape (B, F, D, h, w) where D = 384,
        h = w = image_size // patch_size. Returned as a single-element
        list so it matches the existing propagator interface.
    """

    def __init__(
        self,
        model_name: str = "facebook/dinov2-small",
        image_size: int = 448,
        fuse_layers: list[int] | None = None,
    ) -> None:
        super().__init__()
        from transformers import AutoModel
        self.backbone = AutoModel.from_pretrained(model_name)
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad = False

        self.patch_size = int(self.backbone.config.patch_size)
        if image_size % self.patch_size != 0:
            raise ValueError(
                f"image_size={image_size} must be a multiple of patch_size={self.patch_size}"
            )
        self.image_size = int(image_size)
        self.grid_size = self.image_size // self.patch_size
        self.dim = int(self.backbone.config.hidden_size)
        # Number of non-patch tokens prepended to the patch sequence in
        # `last_hidden_state`. DINOv2 has 1 CLS token. DINOv3 has 1 CLS
        # token + N register tokens ("Vision Transformers Need Registers",
        # Darcet et al. 2024). Token order: [CLS, REG_1, ..., REG_N,
        # patch_1, ..., patch_P]. We slice off all of them.
        n_reg = int(getattr(self.backbone.config, "num_register_tokens", 0) or 0)
        self._n_prefix_tokens = 1 + n_reg

        # DPT-style multi-layer fusion. v16: fuse the outputs of several
        # DINOv2 transformer blocks into one feature map, so the propagator
        # sees both early-layer local detail (sharp per-patch appearance —
        # critical for localising small objects like balls) and late-layer
        # semantic content (cross-frame identity). Free in encoder compute
        # — DINOv2 already computes every layer; we just read more outputs.
        # `fuse_layers` is a list of 0-indexed block indices in [0, num_blocks).
        # `None` (the default) means v14/v15 behaviour: use only the last block.
        n_blocks = int(self.backbone.config.num_hidden_layers)
        if fuse_layers is not None and len(fuse_layers) > 1:
            for l in fuse_layers:
                if l < 0 or l >= n_blocks:
                    raise ValueError(
                        f"fuse_layer {l} out of range [0, {n_blocks})"
                    )
            self.fuse_layers = list(fuse_layers)
            # Per-layer LayerNorm — different DINO layers have very different
            # output scales; without normalising, the late layers (which
            # have larger magnitude) would dominate the fusion.
            self.fuse_norms = nn.ModuleList(
                [nn.LayerNorm(self.dim) for _ in self.fuse_layers]
            )
            self.fuse_proj = nn.Linear(
                len(self.fuse_layers) * self.dim, self.dim,
            )
        else:
            self.fuse_layers = None
            self.fuse_norms = None
            self.fuse_proj = None

        # DINOv2 was trained with ImageNet mean/std normalisation.
        self.register_buffer("imagenet_mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("imagenet_std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    # `coarse_image_size` exposed for compatibility with PyramidEncoder.
    @property
    def coarse_image_size(self) -> int:
        return self.image_size

    # Frames are independent at the DINO level (no cross-frame attention),
    # so a 150-frame clip at 896 input fits perfectly fine as 5 chunks of
    # 32 frames each — but not as one batch of 150 (would OOM at ~9 GB).
    # Training runs with window <= 8 so chunking is a no-op there.
    ENC_CHUNK = 32

    def _forward_one_image_batch(self, image: Tensor) -> tuple[Tensor, Tensor]:
        """Args:  image (B, 3, H, W) in [0, 1].
        Returns: tuple (feat, cls) where
            feat: (B, dim, grid, grid)   — patch feature map
            cls:  (B, dim)               — CLS token (DINO global summary)

        DINO backbone runs under `torch.no_grad()` (frozen). The fusion
        projection (v16+, when `fuse_proj` is not None) DOES need
        gradient — it's trainable — so the no_grad context is scoped to
        the backbone call only.

        Large `B` (e.g. full-clip eval, B=150-300 frames) is chunked over
        the batch dim into sub-batches of `ENC_CHUNK` frames so peak
        memory stays bounded; outputs are concatenated.
        """
        if image.shape[-1] != self.image_size or image.shape[-2] != self.image_size:
            image = F.interpolate(
                image, size=(self.image_size, self.image_size),
                mode="bilinear", align_corners=False,
            )
        image = (image - self.imagenet_mean) / self.imagenet_std

        want_hidden_states = self.fuse_proj is not None
        feat_chunks: list[Tensor] = []
        cls_chunks: list[Tensor] = []
        total = image.shape[0]
        for start in range(0, total, self.ENC_CHUNK):
            sub = image[start:start + self.ENC_CHUNK]
            with torch.no_grad():
                out = self.backbone(
                    pixel_values=sub,
                    interpolate_pos_encoding=True,
                    output_hidden_states=want_hidden_states,
                )
            cls_chunks.append(out.last_hidden_state[:, 0, :])
            if not want_hidden_states:
                tokens = out.last_hidden_state[:, self._n_prefix_tokens:, :]
            else:
                parts = []
                for norm, l in zip(self.fuse_norms, self.fuse_layers):
                    h_l = out.hidden_states[l + 1][:, self._n_prefix_tokens:, :]
                    parts.append(norm(h_l))
                tokens = self.fuse_proj(torch.cat(parts, dim=-1))
            B_sub, P, D = tokens.shape
            if P != self.grid_size * self.grid_size:
                raise RuntimeError(
                    f"DINOv2 returned {P} patch tokens; expected "
                    f"{self.grid_size * self.grid_size} for image_size={self.image_size}, "
                    f"patch={self.patch_size}"
                )
            feat_chunks.append(
                tokens.transpose(1, 2).reshape(B_sub, D, self.grid_size, self.grid_size)
            )
            del out, tokens
        feat = torch.cat(feat_chunks, dim=0)                       # (B, D, h, w)
        cls = torch.cat(cls_chunks, dim=0)                         # (B, D)
        return feat, cls

    def forward(self, image: Tensor) -> list[Tensor]:
        """Single-image forward (matches PyramidEncoder.forward signature).
        Returns a one-level "pyramid".
        """
        feat, _ = self._forward_one_image_batch(image)
        return [feat]

    def forward_video(self, video: Tensor) -> list[Tensor]:
        """Args:  video (B, F, 3, H, W).
        Returns: one-element list of (B, F, D, grid, grid).
        """
        pyramid, _ = self.forward_video_with_cls(video)
        return pyramid

    def forward_video_with_cls(self, video: Tensor) -> tuple[list[Tensor], Tensor]:
        """Args:  video (B, F, 3, H, W).
        Returns: (pyramid, cls_per_frame) where
            pyramid:        one-element list of (B, F, D, grid, grid)
            cls_per_frame:  (B, F, D) — DINO CLS token at each frame.

        Used by v18+ tracker (clip-level scale head pools over CLS-per-frame).
        """
        B, F_, C, H, W = video.shape
        flat = video.reshape(B * F_, C, H, W)
        feat, cls = self._forward_one_image_batch(flat)
        D, h, w = feat.shape[1], feat.shape[2], feat.shape[3]
        pyramid = [feat.reshape(B, F_, D, h, w)]
        cls_per_frame = cls.reshape(B, F_, D)
        return pyramid, cls_per_frame

    def forward_frame(self, image: Tensor) -> Tensor:
        """Streaming-inference path: one frame in, one feature grid out.
        Args:  image (B, 3, H, W).
        Returns: (B, D, grid, grid).
        """
        feat, _ = self._forward_one_image_batch(image)
        return feat
