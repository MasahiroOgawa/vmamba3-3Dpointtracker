"""Causal cross-frame propagator (v7 — iterative refinement + correlation volumes).

Each track slot `n` is bound at construction time to a user-supplied
query `(x_n^q, y_n^q, t_n^q)`: a pixel location and an anchor frame
index. The initial bank state `q_n^(0)` is the encoder's finest
pyramid feature bilinear-sampled at that pixel and frame
(`doc/attention/mamba3_attention.tex §8.3, eq. eq:track-bank-init`).

v7 additions vs v6:

  * **Correlation-volume cross-attention** (D): at every pyramid level
    of every frame, we compute a `(N, T_kv)` cosine-similarity map
    `corr[n, j] = cos(Q[n], kv_token[j])`, softmax it over the kv
    axis with a learnable temperature, and weighted-sum the kv tokens.
    This gives each query direct access to the "places in this frame
    that look like me" signal, instead of having the SSD cross-attention
    discover it through gradient descent over inner-product weights.
    The correlation branch's output is summed with the existing
    Mamba-3 variant-B cross-attention output (residual structure).

  * **Iterative refinement** (C): within each frame, the full coarse-
    to-fine pyramid pass is run `N_ITER` times. The current bank `Q`
    after the previous iteration is the input to the next iteration,
    so the model can refine its estimate using the updated `Q` to
    re-attend to the same kv features. Standard RAFT-style recipe.

Per-frame loop carries temporal causality implicitly.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from visionmamba3.cross_attention import Mamba3CrossAttention


def _pyramid_to_flat_kv(level: Tensor) -> Tensor:
    """(B, D, h, w) → (B, h*w, D)."""
    return level.flatten(2).transpose(1, 2)


def _sample_query_features(
    pyramid_finest: Tensor,    # (B, F, D, h, w) — the finest pyramid level
    queries_xyt: Tensor,       # (B, N, 3) — (x, y, t) in input-image coords
    query_mask: Tensor,        # (B, N) bool
    image_size: int,
) -> Tensor:
    """Bilinear-sample the finest pyramid level at each query's (x, y, t).

    Returns (B, N, D). Masked slots are zeros. Out-of-bound x/y are clamped
    to grid edges.
    """
    B, F_, D, h, w = pyramid_finest.shape
    N = queries_xyt.shape[1]
    device = pyramid_finest.device

    px = queries_xyt[..., 0].clamp(0, image_size - 1) / max(image_size - 1, 1)
    py = queries_xyt[..., 1].clamp(0, image_size - 1) / max(image_size - 1, 1)
    grid_xy = torch.stack([px * 2.0 - 1.0, py * 2.0 - 1.0], dim=-1)        # (B, N, 2)

    out = torch.zeros(B, N, D, device=device, dtype=pyramid_finest.dtype)
    for b in range(B):
        n_keep = query_mask[b].nonzero(as_tuple=False).flatten()
        if n_keep.numel() == 0:
            continue
        t_idx = queries_xyt[b, n_keep, 2].long().clamp(0, F_ - 1)
        for t_val in torch.unique(t_idx).tolist():
            sel = n_keep[t_idx == t_val]
            if sel.numel() == 0:
                continue
            feat = pyramid_finest[b:b + 1, t_val]
            grid = grid_xy[b, sel].view(1, -1, 1, 2)
            # `grid_sample` may upcast to fp32 under autocast in certain
            # gradient-tracked contexts (v16 fusion triggers this where
            # v14/v15 didn't, because v14/v15 wrapped the whole encoder in
            # @torch.no_grad() while v16 needs gradient through fuse_proj).
            # Cast grid to feat's dtype, then cast sampled back to out's
            # dtype before the index_put so dtypes always line up.
            sampled = F.grid_sample(
                feat, grid.to(feat.dtype), mode="bilinear",
                padding_mode="border", align_corners=True,
            )
            out[b, sel] = sampled.squeeze(-1).squeeze(0).transpose(0, 1).to(out.dtype)
    return out


class CorrelationCrossAttention(nn.Module):
    """RAFT-style correlation cross-attention.

    Computes per-(query, kv) cosine-similarity as the attention score,
    softmax-normalises over the kv axis, and weighted-sums the kv tokens.
    The learnable temperature controls how peaky the attention can become.
    """

    def __init__(self, dim: int, temperature_init: float = 0.1) -> None:
        super().__init__()
        # Log-temperature so the parameter is unconstrained but temperature
        # stays positive. Initialised so exp(log_tau) = temperature_init.
        self.log_tau = nn.Parameter(torch.log(torch.tensor(float(temperature_init))))
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, q_tokens: Tensor, kv_tokens: Tensor) -> Tensor:
        """
        Args:
            q_tokens:  (B, N, D)
            kv_tokens: (B, T, D)
        Returns:
            (B, N, D)
        """
        q_n = F.normalize(q_tokens, dim=-1)
        k_n = F.normalize(kv_tokens, dim=-1)
        tau = self.log_tau.exp().clamp_min(1e-3)
        corr = torch.einsum("bnd,btd->bnt", q_n, k_n) / tau
        attn = corr.softmax(dim=-1)
        out = torch.einsum("bnt,btd->bnd", attn, kv_tokens)
        return self.out_proj(out)


class CausalCrossPropagator(nn.Module):
    """Query-conditioned bank refined coarse-to-fine + iteratively per frame.

    For each frame `t` and each iteration `i ∈ [0, N_ITER)`:
        for each pyramid level l (coarse → fine):
            Q ← LayerNorm_l( Q + Mamba3CrossAttn_l(LN(Q), LN(Y^(t)_l))
                                + CorrelationCrossAttn_l(LN(Q), LN(Y^(t)_l)) )
    After all iterations at frame t, Q is stored as Q^(t).

    The post-update LayerNorm is the v10 fix: without it, `Q`'s magnitude
    accumulated across 192 additive updates (num_iters × levels × frames)
    until each new residual was invisible against the running sum, and the
    head saw a near-constant direction every frame → static tracks. With
    LayerNorm bounding `Q` after every addition, each new residual has
    visible influence on the direction, so per-frame predictions can
    actually vary — and the position/velocity losses can do their job.

    The two cross-attention branches (Mamba-3 SSD + correlation) sum at
    each level — both contribute gradient. The correlation branch carries
    the explicit feature-matching signal that the rank-1-mask SSD branch
    cannot produce on its own.
    """

    def __init__(
        self,
        dim: int = 384,
        num_pyramid_levels: int = 3,
        num_heads: int = 6,
        state_dim: int = 64,
        num_iters: int = 1,
        use_correlation: bool = False,
        corr_temperature: float = 0.1,
    ) -> None:
        super().__init__()
        if num_pyramid_levels < 1:
            raise ValueError("need at least 1 pyramid level")
        self.dim = dim
        self.num_pyramid_levels = num_pyramid_levels
        self.num_iters = num_iters
        self.use_correlation = use_correlation

        self.ssd_levels = nn.ModuleList(
            [
                Mamba3CrossAttention(
                    dim_q=dim, dim_kv=dim, num_heads=num_heads,
                    state_dim=state_dim,
                )
                for _ in range(num_pyramid_levels)
            ]
        )
        # `CorrelationCrossAttention` (RAFT-style) is kept available in this
        # module but instantiated only when explicitly requested via
        # `use_correlation=True`. v11 default omits it — the diagnostic data
        # showed it didn't move the metric vs cost in gradient conflict with
        # the SSD branch and per-step variance.
        if use_correlation:
            self.corr_levels = nn.ModuleList(
                [
                    CorrelationCrossAttention(dim=dim, temperature_init=corr_temperature)
                    for _ in range(num_pyramid_levels)
                ]
            )
        else:
            self.corr_levels = None
        self.q_norms = nn.ModuleList(
            [nn.LayerNorm(dim) for _ in range(num_pyramid_levels)]
        )
        self.kv_norms = nn.ModuleList(
            [nn.LayerNorm(dim) for _ in range(num_pyramid_levels)]
        )
        # v10 post-update norm: bounds ‖Q‖ after each residual addition so
        # the optimiser can't escape the loss by inflating cross-attention
        # output magnitudes (the v7/v8/v9 failure mode).
        self.out_norms = nn.ModuleList(
            [nn.LayerNorm(dim) for _ in range(num_pyramid_levels)]
        )

    def forward(
        self,
        pyramid: list[Tensor],
        queries_xyt: Tensor,
        query_mask: Tensor,
        image_size: int,
    ) -> Tensor:
        """
        Args:
            pyramid: list of `num_pyramid_levels` (B, F, D, h_l, w_l), coarse → fine.
            queries_xyt: (B, N, 3) — (x, y, t) per track in pixel coords.
            query_mask: (B, N) bool — True at real query slots.
            image_size: input image side length (for the bilinear-sample step).

        Returns:
            Q_history: (B, F, N, D).
        """
        if len(pyramid) != self.num_pyramid_levels:
            raise ValueError(
                f"pyramid has {len(pyramid)} levels, propagator built for "
                f"{self.num_pyramid_levels}"
            )

        F_ = pyramid[0].shape[1]
        Q = _sample_query_features(
            pyramid[-1], queries_xyt, query_mask, image_size=image_size,
        )

        history: list[Tensor] = []
        for t in range(F_):
            for _ in range(self.num_iters):
                for l in range(self.num_pyramid_levels):
                    kv_grid = pyramid[l][:, t]
                    kv_raw = _pyramid_to_flat_kv(kv_grid)
                    kv_tokens = self.kv_norms[l](kv_raw)
                    q_tokens = self.q_norms[l](Q)
                    delta_ssd = self.ssd_levels[l](q_tokens, kv_tokens)
                    if self.corr_levels is not None:
                        delta_corr = self.corr_levels[l](q_tokens, kv_tokens)
                        Q = self.out_norms[l](Q + delta_ssd + delta_corr)
                    else:
                        Q = self.out_norms[l](Q + delta_ssd)
            history.append(Q)

        return torch.stack(history, dim=1)
