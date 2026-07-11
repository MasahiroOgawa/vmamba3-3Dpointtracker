"""Flow-Conditioned Mamba-3 Tracker (v32).

Instead of an image encoder, this model takes SEA-RAFT optical-flow vectors
and DA3 metric depth as direct per-point conditioning inputs. A small stack of
Mamba-3 temporal layers refines the 2D positions produced by naive flow-chaining
over time.

Input per (batch, frame, track):
  uv_fwd     (B, F, N, 2)  — 2D pixel position from SEA-RAFT flow-chaining
  flow_vec   (B, F, N, 2)  — SEA-RAFT forward flow at the tracked position
  depth_at   (B, F, N)     — DA3 metric depth sampled at uv_fwd
  vis_fwd    (B, F, N)     — forward-backward consistency flag (float {0, 1})

Output (TrackerOutputs):
  uv         (B, F, N, 2)  — refined 2D position (anchor + delta_uv)
  vis_logits (B, F, N)     — visibility logits

The model has ~480k trainable parameters (D=128, L=2 Mamba-3 layers).
Notation follows doc/vmamba3_3dpointtrack/vmamba3_3dpointtrack.tex.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from visionmamba3.cross_attention import Mamba3CrossAttention
from .heads import TrackerOutputs


def _mlp(in_dim: int, hidden: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, out_dim)
    )


class FlowConditionedTracker(nn.Module):
    def __init__(
        self,
        dim: int = 128,
        state_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        self.dim = dim
        # Input: [uv/S (2), flow/S (2), depth/z_ref (1), vis_flag (1)] = 6
        self.embed = _mlp(6, dim, dim)
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

        self.uv_head = _mlp(dim, 64, 2)
        self.vis_head = _mlp(dim, 64, 1)
        # Zero-init uv_head final layer: delta_uv = 0 at step 0 → pred = SEA-RAFT baseline.
        with torch.no_grad():
            self.uv_head[-1].weight.zero_()
            self.uv_head[-1].bias.zero_()

    def forward(
        self,
        uv_fwd: Tensor,  # (B, F, N, 2)
        flow_vec: Tensor,  # (B, F, N, 2)
        depth_at: Tensor,  # (B, F, N)
        vis_fwd: Tensor,  # (B, F, N)
        image_size: float = 896.0,
        depth_ref: float | None = None,
    ) -> TrackerOutputs:
        B, F_, N, _ = uv_fwd.shape
        z_ref = (
            depth_ref
            if depth_ref is not None
            else float(depth_at.flatten().median().item()) + 1e-6
        )
        feat = torch.cat(
            [
                uv_fwd / image_size,  # (B,F,N,2)
                flow_vec / image_size,  # (B,F,N,2)
                depth_at.unsqueeze(-1) / z_ref,  # (B,F,N,1)
                vis_fwd.unsqueeze(-1),  # (B,F,N,1)
            ],
            dim=-1,
        )  # (B,F,N,6)

        x = self.embed(feat)  # (B,F,N,D)

        x = x.permute(0, 2, 1, 3).reshape(B * N, F_, self.dim)  # (B*N, F, D)
        for pre_n, layer, post_n in zip(self.pre_norms, self.layers, self.post_norms):
            xn = pre_n(x)
            x = post_n(x + layer(xn, xn))
        x = self.out_norm(x)

        x = x.reshape(B, N, F_, self.dim).permute(0, 2, 1, 3)  # (B,F,N,D)

        delta_uv = self.uv_head(x)  # (B,F,N,2)
        uv_pred = uv_fwd + delta_uv  # (B,F,N,2) — zero-init → SEA-RAFT baseline
        vis_logits = self.vis_head(x).squeeze(-1)  # (B,F,N)

        zeros_xyz = x.new_zeros(B, F_, N, 3)
        return TrackerOutputs(
            xyz=zeros_xyz,
            uv=uv_pred,
            vis_logits=vis_logits,
            spawn_logits=vis_logits,  # unused; reuse vis for structural compat
        )
