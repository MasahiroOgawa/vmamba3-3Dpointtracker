"""Top-level Mamba-3 tracker (v2 — query-conditioned bank).

Pipeline (matches `doc/attention/mamba3_attention.tex §8` after the
2026-05-18 update):

  video        (B, F, 3, H, W)
  queries_xyt  (B, N, 3) in input-image pixel coords
  query_mask   (B, N) bool
        ↓
  PyramidEncoder.forward_video → list of (B, F, D, h_l, w_l)
        ↓                                                             ↓
  CausalCrossPropagator(pyramid, queries_xyt, mask, image_size)
        ↓
  q_history    (B, F, N, D)
        ↓
  TrackHeads
        ↓
  TrackerOutputs(delta_xyz, vis_logits, spawn_logits)

`delta_xyz` is motion since the query anchor; absolute positions are
recovered as `delta_xyz + p_query`.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .dino_encoder import DINOv2Encoder
from .encoder import PyramidEncoder
from .heads import ScaleHead, TrackerOutputs, TrackHeads
from .propagator import CausalCrossPropagator


class Mamba3Tracker(nn.Module):
    def __init__(
        self,
        dim: int = 384,
        num_heads: int = 6,
        state_dim: int = 64,
        level_sizes: tuple[int, ...] = (32, 64),
        blocks_per_level: int = 2,
        patch: int = 14,
        num_iters: int = 1,
        use_correlation: bool = False,
        encoder_kind: str = "pyramid",         # "pyramid" (v6–v13) or "dinov2" (v14+)
        dinov2_model: str = "facebook/dinov2-small",
        dinov2_image_size: int = 448,
        dinov2_fuse_layers: list[int] | None = None,  # v16+: e.g. [2, 5, 8, 11]
        predict_scale: bool = False,           # v18+: emit per-clip scalar `s`
        scale_param: str = "softplus",         # v20+: "exp" for log-scale parameterisation
        frozen_scale_estimator: nn.Module | None = None,  # v28+: external pretrained scale source
        freeze_scale_estimator: bool = True,   # v29: set False in stage-2 to fine-tune
        head_mode: str = "xyz",                # v31: "uv" for 2D-tracking + external depth
    ) -> None:
        super().__init__()
        if encoder_kind == "pyramid":
            self.encoder = PyramidEncoder(
                dim=dim, num_heads=num_heads, state_dim=state_dim, patch=patch,
                level_sizes=level_sizes, blocks_per_level=blocks_per_level,
            )
            n_pyramid_levels = len(level_sizes)
        elif encoder_kind == "dinov2":
            self.encoder = DINOv2Encoder(
                model_name=dinov2_model, image_size=dinov2_image_size,
                fuse_layers=dinov2_fuse_layers,
            )
            n_pyramid_levels = 1               # DINOv2 emits a single feature grid
        else:
            raise ValueError(f"encoder_kind={encoder_kind!r} must be 'pyramid' or 'dinov2'")
        self.propagator = CausalCrossPropagator(
            dim=dim, num_pyramid_levels=n_pyramid_levels,
            num_heads=num_heads, state_dim=state_dim,
            num_iters=num_iters, use_correlation=use_correlation,
        )
        if head_mode not in ("xyz", "uv"):
            raise ValueError(f"head_mode must be 'xyz' or 'uv', got {head_mode!r}")
        self.head_mode = head_mode
        self.heads = TrackHeads(dim=dim, output_mode=head_mode)
        self.image_size = self.encoder.coarse_image_size

        self.predict_scale = bool(predict_scale)
        # v28: an external pretrained scale-estimator can replace the trained
        # scale_head. When supplied, its forward(video) is used as `pred.scale`.
        # v29: the freeze_scale_estimator flag controls whether the external
        # module's parameters are frozen (True, default — stage-1 shape-only
        # training) or trainable (False — stage-2 fine-tuning).
        self.frozen_scale_estimator = frozen_scale_estimator
        self.freeze_scale_estimator = bool(freeze_scale_estimator)
        if self.frozen_scale_estimator is not None and self.freeze_scale_estimator:
            for p in self.frozen_scale_estimator.parameters():
                p.requires_grad_(False)
            self.frozen_scale_estimator.eval()
        if self.predict_scale:
            if encoder_kind != "dinov2":
                raise ValueError("predict_scale=True requires encoder_kind='dinov2' (needs CLS tokens)")
            if self.frozen_scale_estimator is None:
                self.scale_head = ScaleHead(dim=dim, param=scale_param)
            else:
                self.scale_head = None  # replaced by external frozen module
            # Zero-init xyz_head's final layer so Δp̃ = 0 at startup. With
            # v18's path-length-normalised 3D loss, a random initial
            # Δp̃ ~ O(0.1 m) divided by a 1 mm GT step (e.g. pstudio
            # near the anchor) explodes to a 10⁴ squared residual.
            # Δp̃ = 0 caps the startup loss at the collapse-floor
            # value (≈ 1 per visible (t, n)) and lets the optimiser
            # ramp up cleanly.  (v31 head_mode='uv' has no xyz_head; the
            # uv_head's zero-init lives inside TrackHeads instead.)
            if self.head_mode == "xyz":
                with torch.no_grad():
                    final_xyz = self.heads.xyz_head[-1]
                    final_xyz.weight.zero_()
                    final_xyz.bias.zero_()
        else:
            self.scale_head = None

    def forward(
        self,
        video: Tensor,
        queries_xyt: Tensor,
        query_mask: Tensor,
    ) -> TrackerOutputs:
        """
        Args:
            video: (B, F, 3, H, W). Resized to encoder.coarse_image_size² inside.
            queries_xyt: (B, N, 3) — (x, y, t) per track, in the SAME pixel coord
                system as `video` (i.e. already scaled by the dataset to match
                the encoder's input resolution).
            query_mask: (B, N) bool — True at real query slots, False at padding.
        Returns:
            TrackerOutputs with `xyz` = per-frame Δp̃ (v18: pre-scale relative
            deltas, recovered as Σ_τ s·Δp̃; v11–v17: Δp relative to query
            anchor, recovered via per-anchor cumsum).
        """
        if self.predict_scale:
            pyramid, cls_per_frame = self.encoder.forward_video_with_cls(video)
            if self.frozen_scale_estimator is not None:
                # v28: external pretrained scale estimator. v29: gradient flows
                # through it iff freeze_scale_estimator is False (stage-2
                # joint fine-tune); otherwise no-grad save memory and prevent
                # updates (stage-1 shape-only training).
                if self.freeze_scale_estimator:
                    with torch.no_grad():
                        scale = self.frozen_scale_estimator(video)     # (B,)
                else:
                    scale = self.frozen_scale_estimator(video)         # (B,)
            else:
                scale = self.scale_head(cls_per_frame)                 # (B,)
        else:
            pyramid = self.encoder.forward_video(video)
            scale = None
        q_history = self.propagator(
            pyramid, queries_xyt, query_mask, image_size=self.image_size,
        )
        if self.head_mode == "uv":
            anchor_uv = queries_xyt[..., :2]                   # (B, N, 2)
            out = self.heads(q_history, anchor_uv=anchor_uv)
        else:
            out = self.heads(q_history)
        out.scale = scale
        return out
