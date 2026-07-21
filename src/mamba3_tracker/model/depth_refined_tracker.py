"""Mamba-3 depth-along-ray refiner (v33).

Lesson from v32: refining the 2D position (a `delta_uv` residual on top of an
already-good SEA-RAFT flow track) *degrades* 3D accuracy — small nudges push
points across DA3 depth discontinuities and out of the AJ threshold band.

v33 obeys the constraint "the SSM only treats 3D positions, never touches the
SEA-RAFT 2D track". With the pixel position `(u, v)` frozen, the only 3D degree
of freedom that leaves the 2D projection invariant is the depth `z` along the
pixel ray:

    xyz = z * ((u - cx)/fx, (v - cy)/fy, 1) = z * (ray_x, ray_y, 1)

So a small causal Mamba-3 SSM ingests the per-track sequence of
`[ray_x, ray_y, z_raw/z_ref, vis]` and emits a multiplicative depth correction
`z = z_raw * exp(Δlog z)`. The reprojection of `xyz` is exactly `(u, v)` for
every frame — the 2D track is mathematically untouched.

The Δlog-z head is zero-initialised, so at step 0 `z = z_raw` and the model
reproduces the training-free SEA-RAFT+DA3 baseline exactly.

Notation follows doc/vmamba3_3dpointtrack/vmamba3_3dpointtrack.tex.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

import torch.nn.functional as F
from visionmamba3.cross_attention import Mamba3CrossAttention
from .heads import TrackerOutputs


def _mlp(in_dim: int, hidden: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, out_dim)
    )


class Mamba3DepthRefiner(nn.Module):
    def __init__(
        self,
        dim: int = 128,
        state_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        max_log_correction: float = 2.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.max_log_correction = float(max_log_correction)
        # Input: [ray_x, ray_y, z_raw/z_ref, vis] = 4
        self.embed = _mlp(4, dim, dim)
        self.layers = nn.ModuleList(
            [
                Mamba3CrossAttention(
                    dim_q=dim,
                    dim_kv=dim,
                    num_heads=num_heads,
                    state_dim=state_dim,
                    variant="B",
                    bidirectional_mask=False,
                )
                for _ in range(num_layers)
            ]
        )
        self.pre_norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(num_layers)])
        self.post_norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(num_layers)])
        self.out_norm = nn.LayerNorm(dim)

        self.dz_head = _mlp(dim, 64, 1)
        # Zero-init: Δlog z = 0 at step 0 → z = z_raw → SEA-RAFT+DA3 baseline.
        with torch.no_grad():
            self.dz_head[-1].weight.zero_()
            self.dz_head[-1].bias.zero_()

    def forward(
        self,
        ray: Tensor,  # (B, F, N, 2)  fixed pixel ray (u-cx)/fx, (v-cy)/fy
        z_raw: Tensor,  # (B, F, N)     DA3 depth sampled at the frozen uv
        vis: Tensor,  # (B, F, N)     SEA-RAFT FB-consistency flag (frozen)
        z_ref: float | None = None,
    ) -> TrackerOutputs:
        B, F_, N, _ = ray.shape
        zr = (
            z_ref
            if z_ref is not None
            else float(z_raw.flatten().median().item()) + 1e-6
        )
        feat = torch.cat(
            [
                ray,  # (B,F,N,2)
                (z_raw / zr).unsqueeze(-1),  # (B,F,N,1)
                vis.unsqueeze(-1),  # (B,F,N,1)
            ],
            dim=-1,
        )  # (B,F,N,4)

        x = self.embed(feat)  # (B,F,N,D)
        x = x.permute(0, 2, 1, 3).reshape(B * N, F_, self.dim)  # (B*N, F, D)
        for pre_n, layer, post_n in zip(self.pre_norms, self.layers, self.post_norms):
            xn = pre_n(x)
            x = post_n(x + layer(xn, xn))
        x = self.out_norm(x)
        x = x.reshape(B, N, F_, self.dim).permute(0, 2, 1, 3)  # (B,F,N,D)

        dlog = self.dz_head(x).squeeze(-1)  # (B,F,N)
        dlog = dlog.clamp(-self.max_log_correction, self.max_log_correction)
        z_pred = z_raw * torch.exp(dlog)  # (B,F,N)

        xyz = torch.stack([ray[..., 0] * z_pred, ray[..., 1] * z_pred, z_pred], dim=-1)

        # uv / visibility are frozen SEA-RAFT outputs handled outside the model;
        # echo a zero vis_logits placeholder for TrackerOutputs structural compat.
        vis_logits = x.new_zeros(B, F_, N)
        return TrackerOutputs(
            xyz=xyz,
            uv=None,
            vis_logits=vis_logits,
            spawn_logits=vis_logits,
        )


class Mamba3V35Refiner(nn.Module):
    """v35: VMamba3 tracker with image conditioning and joint 2D+depth correction.

    Extends v33 by adding DINOv3 per-track appearance features and a local depth
    patch, and outputs a bounded Δuv correction in addition to Δlog_z.

    At step 0 both heads are zero-init → z = z_raw, uv = SEA-RAFT uv (baseline).

    Forward signature is different from v33:
        model(ray, z_raw, vis, uv, depth_map, images, K)
    """

    def __init__(
        self,
        dim: int = 128,
        state_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        max_log_correction: float = 2.0,
        max_delta_uv: float = 2.0,
        patch_size: int = 5,
        d_proj: int = 64,
        dino_model: str = "facebook/dinov3-vits16-pretrain-lvd1689m",
        dino_image_size: int = 448,
        image_size: int = 896,
        per_frame_scale: bool = False,
        max_scale_correction: float = 0.5,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.max_log_correction = float(max_log_correction)
        self.max_delta_uv = float(max_delta_uv)
        self.patch_size = int(patch_size)
        self.image_size = float(image_size)
        self.per_frame_scale = bool(per_frame_scale)
        self.max_scale_correction = float(max_scale_correction)

        from .dino_encoder import DINOv2Encoder

        self.dino = DINOv2Encoder(model_name=dino_model, image_size=dino_image_size)
        self.feat_proj = nn.Linear(self.dino.dim, d_proj)

        # Input: [ray_x, ray_y, z/z_ref, vis] + depth_patch(k²) + dino_feat(d_proj)
        input_dim = 4 + patch_size * patch_size + d_proj
        self.embed = _mlp(input_dim, dim, dim)
        self.layers = nn.ModuleList(
            [
                Mamba3CrossAttention(
                    dim_q=dim,
                    dim_kv=dim,
                    num_heads=num_heads,
                    state_dim=state_dim,
                    variant="B",
                    bidirectional_mask=False,
                )
                for _ in range(num_layers)
            ]
        )
        self.pre_norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(num_layers)])
        self.post_norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(num_layers)])
        self.out_norm = nn.LayerNorm(dim)

        self.dz_head = _mlp(dim, 64, 1)
        self.duv_head = _mlp(dim, 64, 2)
        zero_heads = [self.dz_head, self.duv_head]
        # v43: per-frame global-scale head fed by within-frame pooling. Corrects
        # DA3-g's per-frame scale drift (sec:v43); zero-init → starts at v42.
        if self.per_frame_scale:
            self.scale_head = _mlp(dim, 64, 1)
            zero_heads.append(self.scale_head)
        with torch.no_grad():
            for head in zero_heads:
                head[-1].weight.zero_()
                head[-1].bias.zero_()

    def _extract_depth_patch(self, depth_map: Tensor, uv: Tensor) -> Tensor:
        """Sample k×k depth patch at uv. Step = image_size/14 (one DA3 patch).

        depth_map: (B, F, Hd, Wd)
        uv: (B, F, N, 2) in image_size pixel coords
        Returns: (B, F, N, k*k) — ratioed to center depth
        """
        B, F_, N, _ = uv.shape
        k = self.patch_size
        step = 2.0 / 14  # normalized step matching DA3 1/14 feature stride
        offs = (torch.arange(k, device=uv.device, dtype=uv.dtype) - k // 2) * step
        dy, dx = torch.meshgrid(offs, offs, indexing="ij")
        d_offsets = torch.stack([dx.reshape(-1), dy.reshape(-1)], dim=-1)  # (k², 2)

        uv_norm = 2.0 * uv / self.image_size - 1.0  # (B, F, N, 2)
        grid = (uv_norm.unsqueeze(-2) + d_offsets.view(1, 1, 1, k * k, 2)).reshape(
            B * F_, 1, N * k * k, 2
        )
        z_patch = F.grid_sample(
            depth_map.reshape(B * F_, 1, depth_map.shape[-2], depth_map.shape[-1]),
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        ).reshape(B, F_, N, k * k)
        z_center = z_patch[..., k * k // 2].unsqueeze(-1).clamp_min(1e-6)
        return z_patch / z_center

    def _sample_dino(self, images: Tensor, uv: Tensor) -> Tensor:
        """Run DINOv3 and sample features at track positions.

        images: (B, F, 3, H, W) in [0, 1]
        uv: (B, F, N, 2) in image_size pixel coords
        Returns: (B, F, N, d_proj)
        """
        B, F_, N, _ = uv.shape
        feat_map = self.dino.forward_video(images)[0]  # (B, F, D, g, g)
        D, g = feat_map.shape[2], feat_map.shape[3]
        uv_norm = 2.0 * uv / self.image_size - 1.0
        grid = uv_norm.reshape(B * F_, 1, N, 2)
        feats = (
            F.grid_sample(
                feat_map.reshape(B * F_, D, g, g).float(),
                grid,
                mode="bilinear",
                padding_mode="border",
                align_corners=False,
            )
            .reshape(B, F_, D, N)
            .permute(0, 1, 3, 2)
        )  # (B, F, N, D)
        return self.feat_proj(feats.to(feat_map.dtype))  # (B, F, N, d_proj)

    def forward(
        self,
        ray: Tensor,  # (B, F, N, 2)  (u-cx)/fx, (v-cy)/fy
        z_raw: Tensor,  # (B, F, N)     DA3 depth at SEA-RAFT uv
        vis: Tensor,  # (B, F, N)     SEA-RAFT FB-consistency flag
        uv: Tensor,  # (B, F, N, 2)  SEA-RAFT pixel coords (image_size px)
        depth_map: Tensor,  # (B, F, Hd, Wd)
        images: Tensor,  # (B, F, 3, H, W) in [0, 1]
        K: Tensor,  # (B, 3, 3)
        z_ref: float | None = None,
    ) -> TrackerOutputs:
        B, F_, N, _ = ray.shape

        vis_gate = vis.unsqueeze(-1)  # (B,F,N,1)
        depth_patch = self._extract_depth_patch(depth_map, uv) * vis_gate
        dino_feat = self._sample_dino(images, uv) * vis_gate

        zr = (
            z_ref
            if z_ref is not None
            else float(z_raw.detach().flatten().median()) + 1e-6
        )

        feat = torch.cat(
            [
                ray,
                (z_raw / zr).unsqueeze(-1),
                vis.unsqueeze(-1),
                depth_patch,
                dino_feat,
            ],
            dim=-1,
        )  # (B,F,N, input_dim)

        x = self.embed(feat)
        x = x.permute(0, 2, 1, 3).reshape(B * N, F_, self.dim)
        for pre_n, layer, post_n in zip(self.pre_norms, self.layers, self.post_norms):
            xn = pre_n(x)
            x = post_n(x + layer(xn, xn))
        x = self.out_norm(x)
        x = x.reshape(B, N, F_, self.dim).permute(0, 2, 1, 3)  # (B,F,N,D)

        dlog = (
            self.dz_head(x)
            .squeeze(-1)
            .clamp(-self.max_log_correction, self.max_log_correction)
        )
        delta_uv = self.max_delta_uv * torch.tanh(self.duv_head(x))  # (B,F,N,2)
        new_uv = uv + delta_uv

        # Re-sample depth at corrected uv, apply Δlog_z
        new_uv_norm = 2.0 * new_uv / self.image_size - 1.0
        z_pred = F.grid_sample(
            depth_map.reshape(B * F_, 1, depth_map.shape[-2], depth_map.shape[-1]),
            new_uv_norm.reshape(B * F_, 1, N, 2),
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        ).reshape(B, F_, N) * torch.exp(dlog)

        if self.per_frame_scale:
            # One global log-scale per frame from within-frame masked-mean pooling of
            # the per-point features, applied to every point in the frame — removes the
            # per-frame scale drift the per-track SSM cannot see (sec:v43).
            vm = vis.unsqueeze(-1)  # (B,F,N,1)
            pooled = (x * vm).sum(2) / vm.sum(2).clamp_min(1.0)  # (B,F,D)
            ds = self.max_scale_correction * torch.tanh(
                self.scale_head(pooled)
            )  # (B,F,1)
            z_pred = z_pred * torch.exp(ds)

        # Unproject new_uv → camera-frame XYZ
        fx = K[:, 0, 0].view(B, 1, 1)
        fy = K[:, 1, 1].view(B, 1, 1)
        cx_ = K[:, 0, 2].view(B, 1, 1)
        cy_ = K[:, 1, 2].view(B, 1, 1)
        xyz = torch.stack(
            [
                (new_uv[..., 0] - cx_) / fx * z_pred,
                (new_uv[..., 1] - cy_) / fy * z_pred,
                z_pred,
            ],
            dim=-1,
        )

        vis_logits = x.new_zeros(B, F_, N)
        return TrackerOutputs(
            xyz=xyz,
            uv=new_uv,
            vis_logits=vis_logits,
            spawn_logits=vis_logits,
            delta_uv=delta_uv,
        )
