"""Standalone scene-scale estimator (per-clip median anchor depth).

Built as an isolated experiment to test whether scale prediction works
cleanly on its own — far simpler than per-pixel monocular depth, just a
single scalar regression per clip from DINO features. If this trains well,
it confirms scale estimation isn't the bottleneck in our joint tracker;
remaining loss noise is on the position/shape side.

Architecture (deliberately minimal):
    video  (B, F, 3, H, W)
      → DINOv3 (frozen)                 → cls_per_frame (B, F, D)
      → Mamba3 cross-attention from a learnable query token over the
        temporal CLS sequence            → q (B, 1, D)
      → MLP                              → z (B,)
      → softplus(z) + eps                → s_pred ∈ ℝ⁺   (metres)

Loss:
    L_scale = mean_clip |s_pred − s_gt|
where s_gt = median anchor-frame Z over query tracks (same per-clip
"scene depth" target used by TrackingLossV20 with scale_source =
'anchor_depth_median').
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from visionmamba3.cross_attention import Mamba3CrossAttention

from .dino_encoder import DINOv2Encoder


class ScaleEstimator(nn.Module):
    def __init__(
        self,
        dinov2_model: str = "facebook/dinov3-vits16-pretrain-lvd1689m",
        dinov2_image_size: int = 896,
        num_heads: int = 6,
        state_dim: int = 64,
        head_hidden: int = 384,
        param: str = "softplus",
        use_patches: bool = False,
        fuse_layers: list[int] | None = None,
    ) -> None:
        super().__init__()
        self.encoder = DINOv2Encoder(
            model_name=dinov2_model,
            image_size=dinov2_image_size,
            fuse_layers=fuse_layers,
        )
        D = self.encoder.dim
        self.dim = D
        # v4: optionally enrich each per-frame token with the spatial
        # mean-pool of patch features (ground plane, object-size priors that
        # CLS alone discards). When enabled, patch_proj fuses CLS + patch_mean
        # back to D before the temporal cross-attention.
        self.use_patches = bool(use_patches)
        if self.use_patches:
            self.patch_proj = nn.Linear(2 * D, D)
        # Learnable "scale query" — one token that cross-attends to the
        # per-frame sequence to produce a single per-clip representation.
        self.query = nn.Parameter(torch.randn(1, 1, D) * 0.02)
        self.q_norm = nn.LayerNorm(D)
        self.kv_norm = nn.LayerNorm(D)
        self.cross_attn = Mamba3CrossAttention(
            dim_q=D, dim_kv=D, num_heads=num_heads, state_dim=state_dim,
        )
        self.out_norm = nn.LayerNorm(D)
        self.head = nn.Sequential(
            nn.Linear(D, head_hidden),
            nn.GELU(),
            nn.Linear(head_hidden, 1),
        )
        if param not in ("softplus", "exp"):
            raise ValueError(f"param must be 'softplus' or 'exp', got {param!r}")
        self.param = param
        # v2: zero-bias init on final linear so s_init = exp(0) = 1 m.
        # Aligned with ScaleHead's v20+ exp pattern (see model/heads.py).
        # Softplus path keeps its default init for back-compat with v1.
        if self.param == "exp":
            nn.init.zeros_(self.head[-1].bias)

    @property
    def image_size(self) -> int:
        return int(self.encoder.coarse_image_size)

    def forward(self, video: Tensor) -> Tensor:
        """video: (B, F, 3, H, W). Returns s_pred (B,) in metres."""
        pyramid, cls_per_frame = self.encoder.forward_video_with_cls(video)
        if self.use_patches:
            # pyramid[0]: (B, F, D, h, w). Spatial mean-pool per frame.
            patch_mean = pyramid[0].mean(dim=(-1, -2))               # (B, F, D)
            kv_in = self.patch_proj(
                torch.cat([cls_per_frame, patch_mean], dim=-1)
            )                                                         # (B, F, D)
        else:
            kv_in = cls_per_frame
        B = kv_in.shape[0]
        q = self.query.expand(B, 1, self.dim)
        kv = self.kv_norm(kv_in)
        qn = self.q_norm(q)
        delta = self.cross_attn(qn, kv)
        q_out = self.out_norm(q + delta)             # (B, 1, D)
        z = self.head(q_out).squeeze(-1).squeeze(-1) # (B,)
        # v2: exp(z) keeps ∂s/∂z = s nonzero everywhere, no v18/v19-style
        # gradient-decay trap. Output is unbounded above; AdamW + grad-clip
        # keep it numerically tame.
        if self.param == "exp":
            return torch.exp(z)
        return F.softplus(z) + 1e-3                  # legacy v1 path


def gt_scale_from_batch(
    tracks_XYZ: Tensor,        # (B, F, N, 3)
    queries_xyt: Tensor,       # (B, N, 3) — (x, y, t_anchor)
    query_mask: Tensor,        # (B, N) bool
    eps: float = 1e-2,
) -> Tensor:
    """GT per-clip scene-scale = median anchor-frame Z over query tracks.

    Matches v23+ `_per_clip_anchor_depth_scale` exactly so the standalone
    estimator and the joint tracker's ScaleHead see the same target.
    """
    B, F_, N, _ = tracks_XYZ.shape
    a = queries_xyt[..., 2].clamp(min=0, max=F_ - 1).long()       # (B, N)
    init_xyz = tracks_XYZ.gather(
        dim=1, index=a.view(B, 1, N, 1).expand(B, 1, N, 3),
    ).squeeze(1)                                                  # (B, N, 3)
    z = init_xyz[..., 2]                                          # (B, N)
    valid = query_mask.float() * torch.isfinite(z).float() * (z > 0).float()
    z_sentinel = torch.where(valid > 0, z, torch.full_like(z, float("inf")))
    sorted_z, _ = torch.sort(z_sentinel, dim=-1)
    n_valid = valid.sum(dim=-1).long().clamp_min(1)
    mid = ((n_valid - 1) // 2).clamp_min(0)
    med = sorted_z.gather(1, mid.unsqueeze(-1)).squeeze(-1)
    med = torch.where(torch.isfinite(med), med, torch.full_like(med, eps))
    return med.clamp_min(eps).detach()
