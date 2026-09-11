"""v94: per-frame visibility predicted from forward and backward optical flow alone.

The v92 head (`Mamba3V35Refiner.vis_head`) read the refiner's trunk feature, which carries
the forward-backward mask as its 4th input channel -- so echoing that channel was available
to it as a near-zero-loss solution, and head-on and head-off scored identically. This head
sees neither the mask nor the refiner: only the two flow vectors the mask is derived from,
so the shortcut does not exist and the head is unaffected by retraining the refiner.

It also drops the mask's structural defect. `flow_tracker._consistent` thresholds one
cycle residual per hop and `track_clip` latches the result
(`vis[t+1] = vis[t] & ok`), so a point marked invisible can never be marked visible
again -- yet 52-66% of TAPVid-3D points re-appear in ground truth. This head predicts each
frame independently from its own flow evidence and has no such latch.
"""

import torch
import torch.nn as nn
from torch import Tensor

from visionmamba3.cross_attention import Mamba3CrossAttention

from .depth_refined_tracker import _mlp

# Pixel scale the raw flow vectors are divided by before the MLP. 32 px is a large
# inter-frame displacement at image_size 896, so typical inputs land near unit norm.
FLOW_SCALE = 32.0

# Number of scalars derived per (point, frame); see `flow_features`.
FEAT_DIM = 8


def flow_features(flow_fwd: Tensor, flow_bwd: Tensor) -> Tensor:
    """(B,F,N,2), (B,F,N,2) -> (B,F,N,8), every channel a function of the two flow fields.

    flow_fwd[t] maps frame t to t+1, flow_bwd[t] maps t to t-1, both sampled at the point's
    tracked position. The two residual channels are the quantity `_consistent` thresholds:
    `flow_fwd[t] + flow_bwd[t+1]` is the cycle error of the hop leaving t, and
    `flow_fwd[t-1] + flow_bwd[t]` that of the hop arriving at t. They are handed over
    explicitly rather than left to be discovered, since a product of two inputs is awkward
    for an MLP and carries no information the two vectors do not already hold.
    """
    f, b = flow_fwd, flow_bwd
    # roll along the frame axis, then zero the wrapped end so no clip boundary leaks
    b_next = torch.roll(b, shifts=-1, dims=1)
    b_next[:, -1] = 0.0
    f_prev = torch.roll(f, shifts=1, dims=1)
    f_prev[:, 0] = 0.0
    r_out = (f + b_next).norm(dim=-1)
    r_in = (f_prev + b).norm(dim=-1)
    return torch.cat(
        [
            f / FLOW_SCALE,
            b / FLOW_SCALE,
            torch.log1p(f.norm(dim=-1)).unsqueeze(-1),
            torch.log1p(b.norm(dim=-1)).unsqueeze(-1),
            torch.log1p(r_out).unsqueeze(-1),
            torch.log1p(r_in).unsqueeze(-1),
        ],
        dim=-1,
    )


class FlowVisHead(nn.Module):
    """Flow -> per-frame visibility logit, mixed over frames per point.

    bidirectional: an occlusion is bracketed in time -- a point is known to have been
    occluded partly because it comes back -- so the non-causal mask is the accurate choice
    for offline scoring. It costs the streaming property the refiner's causal stack keeps,
    which is why it is a setting rather than a constant.
    """

    def __init__(
        self,
        dim: int = 64,
        state_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        bidirectional: bool = True,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.embed = _mlp(FEAT_DIM, dim, dim)
        self.layers = nn.ModuleList(
            [
                Mamba3CrossAttention(
                    dim_q=dim,
                    dim_kv=dim,
                    num_heads=num_heads,
                    state_dim=state_dim,
                    bidirectional_mask=bidirectional,
                )
                for _ in range(num_layers)
            ]
        )
        self.pre_norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(num_layers)])
        self.post_norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(num_layers)])
        self.out_norm = nn.LayerNorm(dim)
        self.head = _mlp(dim, dim, 1)

    def forward(self, flow_fwd: Tensor, flow_bwd: Tensor) -> Tensor:
        """(B,F,N,2), (B,F,N,2) -> vis_logits (B,F,N)."""
        B, F_, N, _ = flow_fwd.shape
        x = self.embed(flow_features(flow_fwd, flow_bwd))
        # fold to one sequence per point: the operator mixes along frames only, never
        # across points, matching the refiner's own temporal stack
        x = x.permute(0, 2, 1, 3).reshape(B * N, F_, self.dim)
        for pre_n, layer, post_n in zip(self.pre_norms, self.layers, self.post_norms):
            xn = pre_n(x)
            x = post_n(x + layer(xn, xn))
        x = self.out_norm(x).reshape(B, N, F_, self.dim).permute(0, 2, 1, 3)
        return self.head(x).squeeze(-1)
