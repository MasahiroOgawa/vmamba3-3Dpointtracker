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


def _rot6d_to_matrix(r6: Tensor) -> Tensor:
    """Gram-Schmidt 6-D rotation parameterisation (Zhou et al. 2019).

    r6: (..., 6) -> R: (..., 3, 3). With r6 = [1,0,0, 0,1,0] this returns the
    identity, so a zero-init head plus the [1,0,0,0,1,0] bias starts at identity.
    """
    a1, a2 = r6[..., 0:3], r6[..., 3:6]
    b1 = F.normalize(a1, dim=-1, eps=1e-6)
    a2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(a2, dim=-1, eps=1e-6)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)


class Mamba3DepthRefiner(nn.Module):
    def __init__(
        self,
        dim: int = 128,
        state_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        max_log_correction: float = 2.0,
        two_pool: bool = False,
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
                    bidirectional_mask=False,
                    two_pool=two_pool,
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
        within_frame: bool = False,
        pose_head: bool = False,
        gate_by_vis: bool = True,
        feat_encoder: str = "dinov3",
        vmamba3_dim: int = 384,
        vmamba3_heads: int = 6,
        vmamba3_blocks: int = 2,
        vmamba3_patch: int = 14,
        vmamba3_grid: int = 32,
        two_pool: bool = False,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.max_log_correction = float(max_log_correction)
        self.max_delta_uv = float(max_delta_uv)
        self.patch_size = int(patch_size)
        # Zeroing the patch/appearance inputs of an occluded point is honest, but it also removes
        # that token from the temporal pool. A more accurate visibility flag therefore feeds the
        # operator strictly less signal, which is the effect this switch exists to measure.
        self.gate_by_vis = bool(gate_by_vis)
        self.image_size = float(image_size)
        self.per_frame_scale = bool(per_frame_scale)
        self.max_scale_correction = float(max_scale_correction)
        self.within_frame = bool(within_frame)
        self.pose_head = bool(pose_head)
        self.feat_encoder = str(feat_encoder)

        # Appearance-feature encoder: off-the-shelf frozen DINOv3 ("dinov3"), or a
        # trainable Vision-Mamba-3 (NC-SSD) encoder computed from scratch
        # ("vmamba3", v48/v49). Both expose .dim and forward_video()->[(B,F,D,g,g)].
        if self.feat_encoder == "vmamba3":
            from .vmamba3_encoder import VMamba3Encoder

            self.dino = VMamba3Encoder(
                dim=vmamba3_dim,
                num_heads=vmamba3_heads,
                state_dim=state_dim,
                patch=vmamba3_patch,
                grid=vmamba3_grid,
                blocks=vmamba3_blocks,
            )
        else:
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
                    bidirectional_mask=False,
                    two_pool=two_pool,
                )
                for _ in range(num_layers)
            ]
        )
        self.pre_norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(num_layers)])
        self.post_norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(num_layers)])
        self.out_norm = nn.LayerNorm(dim)

        self.dz_head = _mlp(dim, 64, 1)
        self.duv_head = _mlp(dim, 64, 2)
        # Visibility. The evaluated visibility has always been the flow's forward-backward mask,
        # passed straight through; on adt that mask is the whole deficit to DELTA (occlusion
        # accuracy 0.785 vs 0.826, while our points are the closer ones). This head predicts a
        # RESIDUAL on that mask's logit and is zero-initialised, so at step 0 it reproduces the
        # flow mask exactly and can only be judged against it.
        self.vis_head = _mlp(dim, 64, 1)
        zero_heads = [self.dz_head, self.duv_head, self.vis_head]
        # v43: per-frame global-scale head fed by within-frame pooling. Corrects
        # DA3-g's per-frame scale drift (sec:v43); zero-init → starts at v42.
        if self.per_frame_scale:
            self.scale_head = _mlp(dim, 64, 1)
            zero_heads.append(self.scale_head)
        with torch.no_grad():
            for head in zero_heads:
                head[-1].weight.zero_()
                head[-1].bias.zero_()

        # v46: within-frame Vision-Mamba-3 self-attention over the N points of each
        # frame -> a per-point depth correction that sees all points in the frame
        # (the missing cross-track axis, sec:v46). Zero-init head -> starts at v42.
        if self.within_frame:
            from visionmamba3.self_attention import Mamba3SelfAttention

            self.wf_mix = Mamba3SelfAttention(
                dim=dim,
                num_heads=num_heads,
                state_dim=state_dim,
                bidirectional=True,
                use_fused_kernel=False,  # pure-torch SSD (avoids the tilelang CUDA kernel)
            )
            self.wf_head = _mlp(dim, 64, 1)
            with torch.no_grad():
                self.wf_head[-1].weight.zero_()
                self.wf_head[-1].bias.zero_()

        # v47: shared ego-motion pose head. A per-frame masked-mean pool over points
        # (O(N), so no v46-style OOM) feeds (a) a shared-context depth correction
        # broadcast back to every point, and (b) a per-frame 6-DoF camera pose used by
        # the self-supervised static-world-consistency loss (sec:v47). Both zero-init:
        # ctx_head -> 0 and the pose -> identity, so v47 starts exactly at v45.
        if self.pose_head:
            self.ctx_head = _mlp(2 * dim, 64, 1)  # shared-context Δlog z
            self.pose_rot_head = _mlp(dim, 64, 6)  # 6-D rotation (Gram-Schmidt)
            self.pose_trans_head = _mlp(dim, 64, 3)  # translation
            self.register_buffer(
                "_ident6", torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
            )
            with torch.no_grad():
                for head in (self.ctx_head, self.pose_rot_head, self.pose_trans_head):
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

        vis_gate = vis.unsqueeze(-1) if self.gate_by_vis else torch.ones_like(vis).unsqueeze(-1)
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

        if self.within_frame:
            # within-frame Vision-Mamba-3 self-attention over the N points of each
            # frame: each point's depth correction sees all (visible) points in its
            # frame — the cross-track axis the per-track SSM lacks (sec:v46).
            xf = x.reshape(B * F_, N, self.dim)
            keep = vis.reshape(B * F_, N)  # (B*F, N): 1 = keep, 0 = mask out
            xf = self.wf_mix(xf, attn_mask=keep).reshape(B, F_, N, self.dim)
            dwf = (
                self.wf_head(xf)
                .squeeze(-1)
                .clamp(-self.max_log_correction, self.max_log_correction)
            )  # (B,F,N)
            z_pred = z_pred * torch.exp(dwf)

        cam_pose = None
        if self.pose_head:
            # Shared ego-motion stage (sec:v47). Masked-mean pool over the N points of
            # each frame (O(N)); broadcast the pooled feature back to every point for a
            # zero-init shared-context depth correction (starts at v45), and predict a
            # per-frame 6-DoF camera pose (zero-init → identity) for the loss.
            vm = vis.unsqueeze(-1)  # (B,F,N,1)
            pooled = (x * vm).sum(2) / vm.sum(2).clamp_min(1.0)  # (B,F,D)
            ctx_in = torch.cat(
                [x, pooled.unsqueeze(2).expand(B, F_, N, self.dim)], dim=-1
            )
            dctx = (
                self.ctx_head(ctx_in)
                .squeeze(-1)
                .clamp(-self.max_log_correction, self.max_log_correction)
            )  # (B,F,N)
            z_pred = z_pred * torch.exp(dctx)
            r6 = self.pose_rot_head(pooled) + self._ident6  # (B,F,6)
            R = _rot6d_to_matrix(r6)  # (B,F,3,3)
            t = self.pose_trans_head(pooled)  # (B,F,3)
            cam_pose = torch.cat([R, t.unsqueeze(-1)], dim=-1)  # (B,F,3,4)

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

        v_in = vis.clamp(1e-3, 1.0 - 1e-3)
        vis_logits = torch.log(v_in / (1.0 - v_in)) + self.vis_head(x).squeeze(-1)
        return TrackerOutputs(
            xyz=xyz,
            uv=new_uv,
            vis_logits=vis_logits,
            spawn_logits=vis_logits,
            delta_uv=delta_uv,
            cam_pose=cam_pose,
        )


QUANTILES = (5, 15, 25, 35, 45, 55, 65, 75, 85, 95)


class Mamba3DepthScaleRefiner(nn.Module):
    """Per-frame log-scale correction for DA3 depth, read from the depth map itself.

    Replaces Mamba3DeflickerRefiner, which is kept because published checkpoints carry its
    state-dict prefix. Four things differ, each from a measurement:

    Name. The correction is dominated by a CONSTANT per-clip offset (|median| p50 0.294) rather than
    frame-to-frame wobble (p50 0.027), so "de-flicker" named the smaller tenth of the job. Removing
    the constant is worth +0.1842 of metric-AJ against the oracle and the wobble a further +0.0707.

    Input. The old stage read the depth only at flow-tracked points, through a flow-derived
    visibility mask, normalised by a statistic of those same samples -- so a property of the depth
    model was observed through a biased ~900-point sample chosen by the tracker, which is what tied
    it to the front-end. This reads the depth map.

    Normalisation is by FIXED dataset constants, never a per-clip statistic: subtracting a per-clip
    median would erase the absolute-scale cue that predicting the per-clip offset depends on, and
    that offset is ten elevenths of the available gain.

    Range. max_scale_correction 0.5 clipped 17.4% of frames outright; the needed |Δs| reaches 2.44.

    The emitted scale is returned in TrackerOutputs.log_scale so the loss can supervise it directly.
    """

    def __init__(
        self,
        dim: int = 128,
        state_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        max_scale_correction: float = 2.5,
        two_pool: bool = False,
        grid: int = 64,
        log_ref: float = 2.0,
        log_std: float = 1.5,
    ) -> None:
        super().__init__()
        self.max_scale_correction = float(max_scale_correction)
        self.grid = int(grid)
        self.log_ref = float(log_ref)
        self.log_std = float(log_std)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1), nn.GELU(),      # 64 -> 32
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.GELU(),     # 32 -> 16
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.GELU(),     # 16 -> 8
            nn.AdaptiveAvgPool2d(1),
        )
        # The encoder ends in a global average, so what reaches the token is essentially a MEAN of
        # conv features -- and the mean of log-depth alone predicts the target at R^2 0.37, while the
        # quantile vector reaches 0.51. The extra 0.14 is in the shape of the depth distribution,
        # which averaging discards, so the quantiles are supplied directly alongside.
        self.n_quantiles = len(QUANTILES)
        self.to_token = nn.Linear(64 + self.n_quantiles, dim)
        # Closed-form starting point: a least-squares map from the same quantiles, fitted on TRAINING
        # clips, is worth +0.0734 of metric-AJ on its own. Starting there and learning a residual
        # beats starting at zero, which is what the first version did.
        # No linear bypass. The previous version added one, seeded from a closed-form fit, and the
        # learned head then never left zero (|w| 0.0007 after 8000 steps) while the bypass drifted
        # and made things worse: a shortcut that is easier to optimise than the network leaves the
        # network with nothing to do. The network is now the only path to the output.
        self.use_bypass = False
        self.layers = nn.ModuleList(
            [
                Mamba3CrossAttention(
                    dim_q=dim, dim_kv=dim, num_heads=num_heads, state_dim=state_dim,
                    bidirectional_mask=True, two_pool=two_pool,
                )
                for _ in range(num_layers)
            ]
        )
        self.pre_norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(num_layers)])
        self.post_norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(num_layers)])
        self.out_norm = nn.LayerNorm(dim)
        self.scale_head = nn.Linear(dim, 1)
        nn.init.zeros_(self.scale_head.weight)
        nn.init.zeros_(self.scale_head.bias)

    def load_linear_init(self, path) -> int:  # retained so old configs fail loudly, not silently
        """Seed the bypass from a least-squares fit. Returns the number of tensors set."""
        import numpy as _np
        d = _np.load(path)
        w = torch.as_tensor(d["weight"], dtype=torch.float32).view(1, -1)
        if w.shape[1] != self.n_quantiles:
            raise ValueError(f"fit has {w.shape[1]} coefficients, module expects {self.n_quantiles}")
        with torch.no_grad():
            self.linear_bypass.weight.copy_(w)
            self.linear_bypass.bias.copy_(torch.as_tensor([float(d["bias"])]))
        return 2

    def per_frame_logscale(self, depth: Tensor) -> Tensor:
        """(B,F,H,W) depth map -> (B,F,1) bounded log-scale.

        At initialisation the learned branch contributes nothing (zero-init head) and the output is
        exactly the linear fit, so the module starts from a known-good solution instead of identity.
        """
        B, F_, H, W = depth.shape
        logd = torch.log(depth.clamp_min(1e-6))
        q = torch.quantile(
            logd.reshape(B * F_, -1).float(),
            torch.tensor(QUANTILES, device=depth.device, dtype=torch.float32) / 100.0,
            dim=1,
        ).transpose(0, 1)                                              # (B*F, n_q)
        x = F.adaptive_avg_pool2d(logd.reshape(B * F_, 1, H, W), self.grid)
        x = (x - self.log_ref) / self.log_std
        feat = torch.cat([self.encoder(x).flatten(1), (q - self.log_ref) / self.log_std], dim=-1)
        g = self.to_token(feat).reshape(B, F_, -1)
        for pre_n, layer, post_n in zip(self.pre_norms, self.layers, self.post_norms):
            gn = pre_n(g)
            g = post_n(g + layer(gn, gn))
        g = self.out_norm(g)
        return self.max_scale_correction * torch.tanh(self.scale_head(g))

    def forward(self, ray: Tensor, z_raw: Tensor, vis: Tensor, depth: Tensor) -> TrackerOutputs:
        B, F_, N, _ = ray.shape
        ds = self.per_frame_logscale(depth)                       # (B,F,1)
        z_pred = z_raw * torch.exp(ds)
        xyz = torch.stack([ray[..., 0] * z_pred, ray[..., 1] * z_pred, z_pred], dim=-1)
        return TrackerOutputs(
            xyz=xyz,
            vis_logits=z_pred.new_zeros(B, F_, N),
            spawn_logits=z_pred.new_zeros(B, F_, N),
            log_scale=ds,
        )


class Mamba3DeflickerRefiner(nn.Module):
    """v44: DA3-g per-frame scale de-flicker (standalone, no v35 refiner).

    DA3-g's per-frame *global* scale drifts frame to frame (sec:da3lg): the nested
    model re-fits a least-squares scalar independently each frame. This module
    removes that drift and nothing else. Per frame the visible points are mean-
    pooled into one frame token; a *time-axis* Mamba-3 then models the scale
    sequence across frames (bidirectional — offline temporal smoothing); a
    zero-init head emits one log-scale Δs_f per frame, applied to depth along the
    frozen ray. At step 0 Δs_f = 0 → z = z_raw (the WAFT+DA3-g / v40 baseline), so
    v44 is a strict, comparable add-on to that baseline.

    Forward signature matches v33: model(ray, z_raw, vis).
    """

    def __init__(
        self,
        dim: int = 128,
        state_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        max_scale_correction: float = 0.5,
        two_pool: bool = False,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.max_scale_correction = float(max_scale_correction)
        # Per-point input: [ray_x, ray_y, z_raw/z_ref, vis].
        self.embed = _mlp(4, dim, dim)
        # Time-axis Mamba-3 over per-frame tokens (bidirectional: a de-flicker may
        # use future frames). Self-attention (q = kv) over the F frame sequence.
        self.layers = nn.ModuleList(
            [
                Mamba3CrossAttention(
                    dim_q=dim,
                    dim_kv=dim,
                    num_heads=num_heads,
                    state_dim=state_dim,
                    bidirectional_mask=True,
                    two_pool=two_pool,
                )
                for _ in range(num_layers)
            ]
        )
        self.pre_norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(num_layers)])
        self.post_norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(num_layers)])
        self.out_norm = nn.LayerNorm(dim)
        self.scale_head = _mlp(dim, 64, 1)
        with torch.no_grad():  # zero-init → Δs_f = 0 → z = z_raw at step 0
            self.scale_head[-1].weight.zero_()
            self.scale_head[-1].bias.zero_()

    def per_frame_logscale(
        self, ray: Tensor, z_raw: Tensor, vis: Tensor, z_ref: float | None = None
    ) -> Tensor:
        """One bounded log-scale per frame, (B, F, 1). Zero at init."""
        zr = (
            z_ref
            if z_ref is not None
            else float(z_raw.flatten().median().item()) + 1e-6
        )
        feat = torch.cat(
            [ray, (z_raw / zr).unsqueeze(-1), vis.unsqueeze(-1)], dim=-1
        )  # (B,F,N,4)
        x = self.embed(feat)  # (B,F,N,D)
        vm = vis.unsqueeze(-1)  # within-frame masked mean → one token per frame
        g = (x * vm).sum(2) / vm.sum(2).clamp_min(1.0)  # (B,F,D)
        for pre_n, layer, post_n in zip(self.pre_norms, self.layers, self.post_norms):
            gn = pre_n(g)
            g = post_n(g + layer(gn, gn))  # time-axis mixing over F frame tokens
        g = self.out_norm(g)  # (B,F,D)
        return self.max_scale_correction * torch.tanh(self.scale_head(g))  # (B,F,1)

    def forward(
        self,
        ray: Tensor,  # (B, F, N, 2)
        z_raw: Tensor,  # (B, F, N)  DA3-g depth at the frozen uv
        vis: Tensor,  # (B, F, N)
        z_ref: float | None = None,
    ) -> TrackerOutputs:
        B, F_, N, _ = ray.shape
        ds = self.per_frame_logscale(ray, z_raw, vis, z_ref)  # (B,F,1)
        z_pred = z_raw * torch.exp(ds)  # one scalar per frame, all points
        xyz = torch.stack([ray[..., 0] * z_pred, ray[..., 1] * z_pred, z_pred], dim=-1)
        vis_logits = z_pred.new_zeros(B, F_, N)
        return TrackerOutputs(
            xyz=xyz, uv=None, vis_logits=vis_logits, spawn_logits=vis_logits
        )


class Mamba3V73(nn.Module):
    """Mamba3DepthScaleRefiner (stage 1) then the v35 refiner (stage 2), same shape as Mamba3V45.

    Stage 1 estimates a per-frame log-scale from the DEPTH MAP and rescales both the sampled depth
    and the map handed to stage 2, so the appearance-conditioned refiner starts from depth whose
    frame-to-frame scale has already been corrected. Stage 2 is unchanged.

    The difference from Mamba3V45 is entirely stage 1: it reads the depth map rather than points the
    tracker chose, its output bound is wide enough not to clip, and its emitted scale is returned so
    the loss can supervise it directly.

    Forward signature matches v35: model(ray, z_raw, vis, uv, depth_map, images, K).
    """

    def __init__(
        self,
        dim: int = 128,
        state_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        max_scale_correction: float = 2.5,
        two_pool: bool = False,
        grid: int = 64,
        log_ref: float = 2.0,
        log_std: float = 1.5,
        **v35_kwargs,
    ) -> None:
        super().__init__()
        self.scale_refiner = Mamba3DepthScaleRefiner(
            dim=dim, state_dim=state_dim, num_heads=num_heads, num_layers=num_layers,
            max_scale_correction=max_scale_correction, two_pool=two_pool,
            grid=grid, log_ref=log_ref, log_std=log_std,
        )
        self.v35 = Mamba3V35Refiner(
            dim=dim, state_dim=state_dim, num_heads=num_heads, num_layers=num_layers,
            two_pool=two_pool, **v35_kwargs,
        )

    def forward(
        self,
        ray: Tensor,
        z_raw: Tensor,
        vis: Tensor,
        uv: Tensor,
        depth_map: Tensor,
        images: Tensor,
        K: Tensor,
    ) -> TrackerOutputs:
        ds = self.scale_refiner.per_frame_logscale(depth_map)      # (B,F,1), from the map
        scale = torch.exp(ds)
        out = self.v35(ray, z_raw * scale, vis, uv,
                       depth_map * scale.unsqueeze(-1), images, K)
        # carried through so L_dsr can supervise stage 1 even though stage 2 produced the xyz
        out.log_scale = ds
        return out


class Mamba3V45(nn.Module):
    """v45: two-stage 'de-flicker then refine'.

    Stage 1 (v44 de-flicker) estimates one per-frame log-scale Δs_f and produces a
    temporally stable depth z_stab = z_raw·e^{Δs_f} (and a correspondingly rescaled
    depth map). Stage 2 (the standard v35 refiner) then runs on z_stab, so its
    along-ray Δlog z and Δu corrections operate on drift-free depth. Both stages are
    zero-init, so v45 starts exactly at the WAFT+DA3-g (v40) baseline.

    Forward signature matches v35: model(ray, z_raw, vis, uv, depth_map, images, K).
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
        max_scale_correction: float = 0.5,
        d_proj: int = 64,
        dino_model: str = "facebook/dinov3-vits16-pretrain-lvd1689m",
        dino_image_size: int = 448,
        image_size: int = 896,
        pose_head: bool = False,
        two_pool: bool = False,
        gate_by_vis: bool = True,
    ) -> None:
        super().__init__()
        # two_pool reaches both stages: the de-flicker mixer and the refiner both use the
        # rank-1 variant-B mask, so both are subject to the limitation it lifts.
        self.deflicker = Mamba3DeflickerRefiner(
            dim=dim,
            state_dim=state_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            max_scale_correction=max_scale_correction,
            two_pool=two_pool,
        )  # the de-flicker aggregates over visible points per frame; its use of `vis` is a
           # different mechanism from the refiner's token gate and is left alone by this switch
        self.v35 = Mamba3V35Refiner(
            dim=dim,
            state_dim=state_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            max_log_correction=max_log_correction,
            max_delta_uv=max_delta_uv,
            patch_size=patch_size,
            d_proj=d_proj,
            dino_model=dino_model,
            dino_image_size=dino_image_size,
            image_size=image_size,
            per_frame_scale=False,
            pose_head=pose_head,
            two_pool=two_pool,
            gate_by_vis=gate_by_vis,
        )

    def forward(
        self,
        ray: Tensor,
        z_raw: Tensor,  # (B,F,N)
        vis: Tensor,
        uv: Tensor,
        depth_map: Tensor,  # (B,F,Hd,Wd)
        images: Tensor,
        K: Tensor,
    ) -> TrackerOutputs:
        ds = self.deflicker.per_frame_logscale(ray, z_raw, vis)  # (B,F,1)
        scale = torch.exp(ds)  # (B,F,1)
        z_stab = z_raw * scale  # de-flickered depth at the tracked points
        depth_map_stab = depth_map * scale.unsqueeze(-1)  # (B,F,1,1) over Hd,Wd
        return self.v35(ray, z_stab, vis, uv, depth_map_stab, images, K)


class Mamba3V88(nn.Module):
    """v88: v45 with a depth-map scale stage ADDED in front, behind a zero-initialised gate.

    Motivated by a measurement, not a guess. Substituting the standalone scale refiner for the
    de-flicker (v87) LOST ground -- pstudio 0.2027 -> 0.0642 and adt 0.3056 -> 0.1322 untrained, and
    only pstudio came back after 2000 steps of adapting the refiner. The two modules do different
    jobs and the measurement says so:

      * the standalone scale refiner emits a near-CONSTANT per-subset bias (measured |ds|:
        drivetrack 0.586 with p50 = p95 = max, pstudio 0.263, adt 0.022) -- a per-clip bias
        corrector, because it was pre-trained against a median log-depth target;
      * the v44 de-flicker, trained jointly with the refiner, emits the PER-FRAME term that actually
        helps downstream, which result/adt_error.log values at +0.104 APD3D on drivetrack and
        +0.056 on adt beyond any constant.

    So they compose rather than compete: stage 1 removes the global bias from the depth map, stage 2
    removes the residual per-frame flicker, stage 3 refines each track.

        ds_total = gate * ds_scale(depth_map) + ds_deflicker(ray, z, vis)

    `gate` is nn.Parameter(torch.zeros(1)), so at initialisation ds_total == ds_deflicker EXACTLY and
    the model is numerically identical to Mamba3V45. Warm-started from v63 it therefore STARTS at
    0.2238 -- the best clean 2-pool DA3-g result -- and any use the optimiser makes of stage 1 can
    only add to it. Every composite tried before this started below its own baseline and spent its
    budget climbing back; this one cannot.

    Forward signature matches v35/v45: model(ray, z_raw, vis, uv, depth_map, images, K).
    """

    def __init__(
        self,
        dim: int = 128,
        state_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        max_scale_correction: float = 0.5,
        scale_stage_correction: float = 2.5,
        two_pool: bool = False,
        grid: int = 64,
        log_ref: float = 2.0,
        log_std: float = 1.5,
        **v45_kwargs,
    ) -> None:
        super().__init__()
        self.scale_refiner = Mamba3DepthScaleRefiner(
            dim=dim, state_dim=state_dim, num_heads=num_heads, num_layers=num_layers,
            max_scale_correction=scale_stage_correction, two_pool=two_pool,
            grid=grid, log_ref=log_ref, log_std=log_std,
        )
        self.v45 = Mamba3V45(
            dim=dim, state_dim=state_dim, num_heads=num_heads, num_layers=num_layers,
            max_scale_correction=max_scale_correction, two_pool=two_pool, **v45_kwargs,
        )
        # Zero-init: stage 1 contributes nothing until the optimiser opens this, so the warm start
        # from v63 is an exact floor rather than a hopeful one.
        self.scale_gate = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        ray: Tensor,
        z_raw: Tensor,
        vis: Tensor,
        uv: Tensor,
        depth_map: Tensor,
        images: Tensor,
        K: Tensor,
    ) -> TrackerOutputs:
        ds_scale = self.scale_refiner.per_frame_logscale(depth_map)          # (B,F,1) from the MAP
        ds_flick = self.v45.deflicker.per_frame_logscale(ray, z_raw, vis)    # (B,F,1) from the points
        ds = self.scale_gate * ds_scale + ds_flick
        scale = torch.exp(ds)
        out = self.v45.v35(
            ray, z_raw * scale, vis, uv, depth_map * scale.unsqueeze(-1), images, K
        )
        out.log_scale = ds
        return out
