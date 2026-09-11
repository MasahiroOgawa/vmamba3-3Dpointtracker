"""Training-free point tracking by chaining SEA-RAFT optical flow.

Each query point is propagated from its anchor frame outward by bilinearly
sampling per-frame dense flow: forward to later frames, backward to earlier
ones. Visibility comes from forward-backward (cycle) consistency — a point
whose forward flow does not invert back onto itself is treated as occluded,
and once occluded it stays occluded along the chain.

All frames share the same flow fields, so every query is propagated in lockstep
with a per-frame "is this point placed yet" mask, rather than looping per query.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def _sample(field: Tensor, pts: Tensor, image_size: int) -> Tensor:
    """Bilinearly sample a (1, 2, S, S) field at pts (N, 2) pixel coords -> (N, 2)."""
    N = pts.shape[0]
    grid = (2.0 * pts / (image_size - 1) - 1.0).view(1, N, 1, 2)
    s = F.grid_sample(
        field, grid, mode="bilinear", padding_mode="border", align_corners=True
    )
    return s.view(2, N).transpose(0, 1)


def _consistent(
    p_from: Tensor,
    disp: Tensor,
    back: Tensor,
    alpha: float,
    beta: float,
) -> Tensor:
    """Cycle-consistency test for one hop p_from -> p_from+disp -> +back ≈ p_from."""
    err = (disp + back).norm(dim=-1)
    tol = alpha * (disp.norm(dim=-1) + back.norm(dim=-1)) + beta
    return err <= tol


@torch.no_grad()
def track_clip(
    flow_model,
    images: Tensor,  # (F, 3, S, S) float RGB in [0, 255] on flow_model.device
    queries_xy: Tensor,  # (N, 2) pixel coords in the S×S image space
    anchor_t: Tensor,  # (N,) long frame index of each query
    image_size: int,
    fb_alpha: float = 0.05,
    fb_beta: float = 1.0,
    bidirectional: bool = False,
) -> tuple[Tensor, Tensor]:
    """Return (uv (F, N, 2), vis (F, N) float in {0,1}) on CPU.

    bidirectional=True runs SEA-RAFT on the reversed video to obtain backward
    flows.  For backward tracking (t < t_query) the model then sees frames in
    its natural forward order, which can give more accurate flow than asking it
    to estimate backward flow directly.  Costs one extra full flow pass.
    Only useful for offline evaluation; real-time inference should leave this False.
    """
    device = flow_model.device
    F_ = int(images.shape[0])
    N = int(queries_xy.shape[0])
    images = images.to(device)
    uv = torch.zeros(F_, N, 2, device=device)
    vis = torch.zeros(F_, N, dtype=torch.bool, device=device)
    idx = torch.arange(N, device=device)
    anchor_t = anchor_t.to(device).clamp(0, F_ - 1)
    uv[anchor_t, idx] = queries_xy.to(device)
    vis[anchor_t, idx] = True

    if F_ == 1:
        return uv.cpu(), vis.float().cpu()

    # Precompute consecutive forward and backward flows on CPU to save GPU VRAM.
    # For long clips (e.g. ADT: 300 frames) keeping all flow tensors on GPU would
    # use ~1.65 GB (fwd+bwd+rev_fwd).  Store on CPU; move to device per frame.
    fwd = [
        flow_model.flow(images[t : t + 1], images[t + 1 : t + 2])[0].cpu().unsqueeze(0)
        for t in range(F_ - 1)
    ]  # each (1, 2, S, S) on CPU
    bwd = [
        flow_model.flow(images[t + 1 : t + 2], images[t : t + 1])[0].cpu().unsqueeze(0)
        for t in range(F_ - 1)
    ]

    # Bidirectional mode: run SEA-RAFT on reversed video so backward tracking
    # uses the model in its natural forward direction.
    # rev_fwd[t] = flow(images[F-1-t] -> images[F-2-t]), i.e. original frame
    # F-1-t displaced toward F-2-t.  To go from original frame k to k-1 use
    # rev_fwd[F-1-k].
    if bidirectional:
        images_rev = images.flip(0)
        rev_fwd = [
            flow_model.flow(images_rev[t : t + 1], images_rev[t + 1 : t + 2])[0]
            .cpu()
            .unsqueeze(0)
            for t in range(F_ - 1)
        ]
    else:
        rev_fwd = None

    # Forward sweep: fill frames after each query's anchor.
    for t in range(F_ - 1):
        m = anchor_t <= t
        if not m.any():
            continue
        d = _sample(fwd[t].to(device), uv[t], image_size)  # t -> t+1
        cand = uv[t] + d
        b = _sample(bwd[t].to(device), cand, image_size)  # t+1 -> t (cycle check)
        ok = _consistent(uv[t], d, b, fb_alpha, fb_beta)
        uv[t + 1, m] = cand[m]
        vis[t + 1, m] = vis[t, m] & ok[m]

    # Backward sweep: fill frames before each query's anchor.
    for t in range(F_ - 1, 0, -1):
        m = anchor_t >= t
        if not m.any():
            continue
        # Use reversed-video forward flow when available; fall back to direct bwd.
        back_field = (rev_fwd[F_ - 1 - t] if rev_fwd is not None else bwd[t - 1]).to(
            device
        )
        d = _sample(back_field, uv[t], image_size)  # t -> t-1
        cand = uv[t] + d
        f = _sample(fwd[t - 1].to(device), cand, image_size)  # t-1 -> t (cycle check)
        ok = _consistent(uv[t], d, f, fb_alpha, fb_beta)
        uv[t - 1, m] = cand[m]
        vis[t - 1, m] = vis[t, m] & ok[m]

    return uv.cpu(), vis.float().cpu()


@torch.no_grad()
def track_clip_with_flow(
    flow_model,
    images: Tensor,  # (F, 3, S, S) float RGB in [0, 255] on flow_model.device
    queries_xy: Tensor,  # (N, 2) pixel coords in the S×S image space
    anchor_t: Tensor,  # (N,) long frame index of each query
    image_size: int,
    fb_alpha: float = 0.05,
    fb_beta: float = 1.0,
    bidirectional: bool = False,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Return (uv (F,N,2), vis (F,N), flow_fwd (F,N,2), flow_bwd (F,N,2)) on CPU.

    flow_fwd[t] = flow t -> t+1 sampled at uv[t]; zeros at t = F-1.
    flow_bwd[t] = flow t -> t-1 sampled at uv[t]; zeros at t = 0.
    Their sum over a hop, flow_fwd[t] + flow_bwd[t+1], is the cycle residual that
    _consistent() thresholds -- the v94 head reads both vectors instead, so it can
    learn a magnitude-dependent tolerance and, unlike the latching mask, re-mark a
    point visible after an occlusion.
    bidirectional: see track_clip.
    """
    device = flow_model.device
    F_ = int(images.shape[0])
    N = int(queries_xy.shape[0])
    images = images.to(device)
    uv = torch.zeros(F_, N, 2, device=device)
    vis = torch.zeros(F_, N, dtype=torch.bool, device=device)
    flow_at = torch.zeros(F_, N, 2, device=device)
    back_at = torch.zeros(F_, N, 2, device=device)
    idx = torch.arange(N, device=device)
    anchor_t = anchor_t.to(device).clamp(0, F_ - 1)
    uv[anchor_t, idx] = queries_xy.to(device)
    vis[anchor_t, idx] = True

    if F_ == 1:
        return uv.cpu(), vis.float().cpu(), flow_at.cpu(), back_at.cpu()

    fwd = [
        flow_model.flow(images[t : t + 1], images[t + 1 : t + 2])[0].cpu().unsqueeze(0)
        for t in range(F_ - 1)
    ]
    bwd = [
        flow_model.flow(images[t + 1 : t + 2], images[t : t + 1])[0].cpu().unsqueeze(0)
        for t in range(F_ - 1)
    ]

    if bidirectional:
        images_rev = images.flip(0)
        rev_fwd = [
            flow_model.flow(images_rev[t : t + 1], images_rev[t + 1 : t + 2])[0]
            .cpu()
            .unsqueeze(0)
            for t in range(F_ - 1)
        ]
    else:
        rev_fwd = None

    for t in range(F_ - 1):
        m = anchor_t <= t
        if not m.any():
            continue
        d = _sample(fwd[t].to(device), uv[t], image_size)
        cand = uv[t] + d
        b = _sample(bwd[t].to(device), cand, image_size)
        ok = _consistent(uv[t], d, b, fb_alpha, fb_beta)
        uv[t + 1, m] = cand[m]
        vis[t + 1, m] = vis[t, m] & ok[m]

    for t in range(F_ - 1, 0, -1):
        m = anchor_t >= t
        if not m.any():
            continue
        back_field = (rev_fwd[F_ - 1 - t] if rev_fwd is not None else bwd[t - 1]).to(
            device
        )
        d = _sample(back_field, uv[t], image_size)
        cand = uv[t] + d
        f = _sample(fwd[t - 1].to(device), cand, image_size)
        ok = _consistent(uv[t], d, f, fb_alpha, fb_beta)
        uv[t - 1, m] = cand[m]
        vis[t - 1, m] = vis[t, m] & ok[m]

    # Sampled after both sweeps so every uv[t] is final. bwd[t-1] maps frame t back to
    # t-1, so the backward flow *at* frame t reads from bwd[t-1], not bwd[t].
    for t in range(F_ - 1):
        flow_at[t] = _sample(fwd[t].to(device), uv[t], image_size)
    for t in range(1, F_):
        back_at[t] = _sample(bwd[t - 1].to(device), uv[t], image_size)

    return uv.cpu(), vis.float().cpu(), flow_at.cpu(), back_at.cpu()


@torch.no_grad()
def track_clip_fuse(
    flow_model,
    images: Tensor,  # (F, 3, S, S) float RGB in [0, 255]
    queries_xy: Tensor,  # (N, 2) pixel coords in the S×S image space
    anchor_t: Tensor,  # (N,) long frame index of each query
    image_size: int,
    fb_alpha: float = 0.05,
    fb_beta: float = 1.0,
) -> tuple[Tensor, Tensor]:
    """Real bidirectional tracking: forward+backward flow FUSION per hop.

    Each propagation step averages the forward flow and the negated backward flow
    sampled at the forward-predicted point,
        d = 0.5 * (d_fwd - d_bwd(p + d_fwd)),
    genuinely coupling both flow directions *within* each step. This is the only
    flow-chaining bidirectional that differs from the forward tracker: the
    reversed-video flag in `track_clip`, and a two-pass whole-track average, both
    reproduce the forward track exactly (the reversed pass re-runs the identical
    flow chains). Visibility uses the forward-backward cycle-consistency test — a
    point whose two flows disagree (|d_fwd + d_bwd| > tol) is marked occluded.
    """
    device = flow_model.device
    F_, N = int(images.shape[0]), int(queries_xy.shape[0])
    images = images.to(device)
    uv = torch.zeros(F_, N, 2, device=device)
    vis = torch.zeros(F_, N, dtype=torch.bool, device=device)
    idx = torch.arange(N, device=device)
    anchor_t = anchor_t.to(device).clamp(0, F_ - 1)
    uv[anchor_t, idx] = queries_xy.to(device)
    vis[anchor_t, idx] = True
    if F_ == 1:
        return uv.cpu(), vis.float().cpu()

    fwd = [
        flow_model.flow(images[t : t + 1], images[t + 1 : t + 2])[0].cpu().unsqueeze(0)
        for t in range(F_ - 1)
    ]
    bwd = [
        flow_model.flow(images[t + 1 : t + 2], images[t : t + 1])[0].cpu().unsqueeze(0)
        for t in range(F_ - 1)
    ]

    def _fuse_step(primary, check, uv_t):
        d_p = _sample(primary, uv_t, image_size)
        cand = uv_t + d_p
        d_c = _sample(check, cand, image_size)
        d = 0.5 * (d_p - d_c)  # d_c ≈ -d_p for a consistent pair
        return uv_t + d, _consistent(uv_t, d_p, d_c, fb_alpha, fb_beta)

    for t in range(F_ - 1):  # forward sweep t -> t+1 (primary=fwd[t], check=bwd[t])
        m = anchor_t <= t
        if not m.any():
            continue
        nxt, ok = _fuse_step(fwd[t].to(device), bwd[t].to(device), uv[t])
        uv[t + 1, m] = nxt[m]
        vis[t + 1, m] = vis[t, m] & ok[m]

    for t in range(
        F_ - 1, 0, -1
    ):  # backward sweep t -> t-1 (primary=bwd[t-1], check=fwd[t-1])
        m = anchor_t >= t
        if not m.any():
            continue
        prv, ok = _fuse_step(bwd[t - 1].to(device), fwd[t - 1].to(device), uv[t])
        uv[t - 1, m] = prv[m]
        vis[t - 1, m] = vis[t, m] & ok[m]

    return uv.cpu(), vis.float().cpu()
