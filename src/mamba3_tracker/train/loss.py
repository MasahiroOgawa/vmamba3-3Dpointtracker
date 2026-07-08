"""Tracking loss v11 — cumulative-sum trajectory + scale-normalised squared L2.

User-driven design after v6–v10's static-tracks failure mode was traced to two
sources: (1) the propagator track-memory accumulator blowing up across many
residual adds without LayerNorm (fixed in v10), and (2) the per-(t, n) Huber
loss scale producing a non-stationary cost surface that made the loss spiky
and non-monotone in training-step space.

v11 keeps the loss's *shape in residual space* simple and smooth — a parabola
in the prediction variable — and replaces the offset-from-anchor prediction
with a cumulative integration of per-frame motion.

Reconstruction:

    Δp̂(t, n) = pred.xyz[..., t, n, :]    model output per frame
    Δp̂(0, n) is IGNORED (the first frame's emitted value is discarded).
    p̂(0, n) = p*(0, n)                    initial position from GT
    p̂(t, n) = p̂(0, n) + Σ_{s=1..t} Δp̂(s, n)    for t ≥ 1

Per-clip scale (training only — eval uses TAPVid-3D's median scaling):

    s_3D = median over visible (t, n) of  ‖p*(t, n)‖₂          [m]
    s_2D = image_size                                            [px]

Three loss terms — all squared L2 in scale-normalised space:

    L_3D(t, n) = ‖( p̂(t, n) − p*(t, n) ) / s_3D‖²
    L_2D(t, n) = ‖( π(p̂(t, n), K) − π(p*(t, n), K) ) / s_2D‖²
    L_vis      = BCEWithLogits(pred.vis_logits, gt_visibility)

Means over visible (t, n) and over the batch. Weighted sum with
normalised weights `Σ λ_i = 1` from `configs/v11.yaml`.

No velocity term. No smoothness term. No threshold. No Huber per-(t, n)
clamp. Three terms, three numbers, smooth U-shape in the prediction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..model.heads import TrackerOutputs


_TERMS = ("pos_3D", "pos_2D", "vis")


@dataclass
class TrackingLossOutput:
    total: Tensor
    pos_3D: Tensor
    pos_2D: Tensor
    vis: Tensor
    scale: Tensor | None = None  # v20: log-scale supervision term; None for v11–v19


def _project(xyz: Tensor, K: Tensor) -> Tensor:
    """Pinhole projection. xyz: (B, ..., 3); K: (B, 3, 3); returns (B, ..., 2)."""
    extra_dims = xyz.dim() - 2
    view = (-1,) + (1,) * extra_dims
    fx = K[:, 0, 0].view(view)
    fy = K[:, 1, 1].view(view)
    cx = K[:, 0, 2].view(view)
    cy = K[:, 1, 2].view(view)
    Z = xyz[..., 2].clamp_min(1e-6)
    u = (xyz[..., 0] / Z) * fx + cx
    v = (xyz[..., 1] / Z) * fy + cy
    return torch.stack([u, v], dim=-1)


def reconstruct_trajectory(
    delta_pred: Tensor,
    init_xyz: Tensor,
    anchor_idx: Tensor,
) -> Tensor:
    """Bidirectional cumsum reconstruction anchored at each track's anchor frame.

    For each track n with anchor frame `a_n = anchor_idx[b, n]`:
        p̂(a_n, n) = init_xyz[b, n]
        p̂(t, n)   = init_xyz[b, n] + Σ_{s = a_n+1 .. t} Δp̂(s, n)       for t > a_n
        p̂(t, n)   = init_xyz[b, n] − Σ_{s = t+1   .. a_n} Δp̂(s, n)     for t < a_n

    Implemented as:
        cs[t] = Σ_{s = 0..t} Δp̂(s)               (cumsum with Δp̂(0) := 0)
        p̂(t) = init_xyz + (cs[t] − cs[a_n])      gather + subtract

    Args:
        delta_pred: (B, F, N, 3) — model's per-frame `Δp̂(t, n)`. The
            entry at t=0 is ignored (Δp̂(0) := 0 inside the cumsum).
        init_xyz: (B, N, 3) — initial 3-D position at each track's
            anchor frame `a_n` (v12 uses GT at `tracks_XYZ[a_n, n]`;
            future = feature-detector output).
        anchor_idx: (B, N) long — each track's anchor frame index in
            window coords. Clamped to [0, F-1].
    Returns:
        p_hat: (B, F, N, 3) with `p̂(a_n) = init_xyz` exactly.
    """
    B, F_, N, _ = delta_pred.shape
    if F_ == 0:
        return delta_pred
    if F_ == 1:
        return init_xyz.unsqueeze(1)
    # Zero out the t=0 entry so cumsum starts effectively at t=1.
    delta_zero_init = delta_pred.clone()
    delta_zero_init[:, 0] = 0.0
    cs = delta_zero_init.cumsum(dim=1)  # (B, F, N, 3)
    a = anchor_idx.clamp(min=0, max=F_ - 1)  # (B, N)
    cs_at_anchor = cs.gather(  # (B, N, 3)
        dim=1,
        index=a.view(B, 1, N, 1).expand(B, 1, N, 3),
    ).squeeze(1)
    p_hat = init_xyz.unsqueeze(1) + cs - cs_at_anchor.unsqueeze(1)
    return p_hat


def _per_clip_median_scale(
    gt_tracks_XYZ: Tensor,  # (B, F, N, 3)
    visible: Tensor,  # (B, F, N) bool/float
    eps: float = 1e-3,
) -> Tensor:
    """`s_3D = median(‖p*‖)` over visible (t, n), one scalar per clip."""
    B = gt_tracks_XYZ.shape[0]
    norms = gt_tracks_XYZ.norm(dim=-1)  # (B, F, N)
    s = torch.full((B,), eps, device=gt_tracks_XYZ.device, dtype=gt_tracks_XYZ.dtype)
    for b in range(B):
        v = visible[b].bool()
        if v.any():
            s[b] = norms[b][v].median()
    return s.clamp_min(eps)


def reconstruct_trajectory_v18(
    delta_tilde: Tensor,
    scale: Tensor,
    init_xyz: Tensor,
    anchor_idx: Tensor,
) -> Tensor:
    """v18 reconstruction (anchor-based with learnable per-clip scale).

    Same shape as v11–v17 — each track anchored at its own query
    frame and integrated bidirectionally — but each per-frame delta
    is multiplied by the model-predicted per-clip scalar `s` before
    accumulating:

        Δp̂(t, n) = s · Δp̃(t, n)
        p̂(a_n, n) = init_xyz[b, n]   (= GT at the anchor frame)
        p̂(t, n)   = init_xyz + (cs[t] − cs[a_n])     for t ≠ a_n
        cs[t]     = Σ_{τ=0..t}  Δp̂(τ, n)   with Δp̂(0) := 0

    Anchoring at GT keeps the predicted Z in a sensible (positive,
    scene-depth) range from step 0 — no projection blow-ups in the
    2D loss during early training, unlike pure-cumsum-from-origin.

    Args:
        delta_tilde: (B, F, N, 3) — raw model output, pre-scale.
        scale:       (B,)        — per-clip positive scalar from ScaleHead.
        init_xyz:    (B, N, 3)   — GT position at each track's anchor frame.
        anchor_idx:  (B, N) long — each track's anchor frame index.
    Returns:
        p_hat: (B, F, N, 3) absolute trajectory in world units.
    """
    s = scale.to(delta_tilde.dtype).view(-1, 1, 1, 1)
    return reconstruct_trajectory(s * delta_tilde, init_xyz, anchor_idx)


def _gt_path_length_from_anchor(
    gt_tracks_XYZ: Tensor,  # (B, F, N, 3)
    anchor_idx: Tensor,  # (B, N) long
    eps: float = 1e-3,
) -> Tensor:
    """Per-(t, n) cumulative 3D path length of GT between anchor a_n and t.

        cs[t]      = Σ_{τ=1..t} ‖p*(τ) − p*(τ−1)‖
        s_gt(t,n)  = |cs[t]  −  cs[a_n]|                       (path between)

    Symmetric with the anchor-based reconstruction: at t = a_n the
    GT motion budget is 0 (and the predicted residual is exactly 0
    by construction); growing in either temporal direction matches
    how far GT has actually moved. Floored at `eps` for safety;
    detached so no gradient flows through GT magnitudes.

    Returns: (B, F, N) positive scalars.
    """
    B, F_, N, _ = gt_tracks_XYZ.shape
    delta_gt = torch.zeros_like(gt_tracks_XYZ)
    if F_ > 1:
        delta_gt[:, 1:] = gt_tracks_XYZ[:, 1:] - gt_tracks_XYZ[:, :-1]
    step_norms = delta_gt.norm(dim=-1)  # (B, F, N), [:, 0] = 0
    cs = step_norms.cumsum(dim=1)  # (B, F, N)
    a = anchor_idx.clamp(min=0, max=F_ - 1).long()  # (B, N)
    cs_at_anchor = cs.gather(
        dim=1,
        index=a.unsqueeze(1).expand(B, 1, N),
    ).squeeze(1)  # (B, N)
    s_gt = (cs - cs_at_anchor.unsqueeze(1)).abs()  # (B, F, N)
    return s_gt.clamp_min(eps).detach()


class TrackingLossV18(nn.Module):
    """v18 loss: anchor-based reconstruction with a learnable per-clip
    scale, squared L² in 3D normalised by GT path length from anchor,
    plus v11–v17 style 2D term and visibility BCE.

    Reconstruction (`reconstruct_trajectory_v18`):
        p̂(a_n, n) = p*_anchor(n)                                   (GT injected at anchor)
        p̂(t, n)   = p*_anchor + s · (cs[t] − cs[a_n])               (per-anchor cumsum)

    3D term (the v18 change vs v11–v17):

        L_3D(t,n) = ((p̂(t,n) − p*(t,n)) / s_gt(t,n))²   summed over xyz

    where `s_gt(t,n)` is the cumulative GT 3D path length BETWEEN
    the anchor and t (see `_gt_path_length_from_anchor`). Removes
    per-clip scale dependence (drivetrack ~ 30 m vs pstudio ~ 1 m)
    and structurally penalises the "predict static at anchor"
    collapse: `Δp̂ ≈ 0` gives residual = GT motion from anchor,
    s_gt = GT path length from anchor → normalised ≈ 1 per (t,n).

    2D term (unchanged from v11–v17):

        L_2D(t,n) = ((π(p̂) − π(p*)) / image_size)²   summed over uv

    `image_size` is the encoder input side length (architectural
    constant). No startup projection blow-ups because the GT anchor
    keeps `p̂_z ≈ scene_depth` from step 0.
    """

    def __init__(
        self,
        weights: Mapping[str, float],
        image_size: int = 224,
    ) -> None:
        super().__init__()
        missing = set(_TERMS) - set(weights)
        if missing:
            raise ValueError(f"TrackingLossV18: missing weights for {sorted(missing)}")
        self.w = {k: float(weights[k]) for k in _TERMS}
        self.image_size = float(image_size)

    def forward(
        self,
        pred: TrackerOutputs,
        gt_tracks_XYZ: Tensor,  # (B, F, N, 3)
        gt_visibility: Tensor,  # (B, F, N) bool
        gt_query_mask: Tensor,  # (B, N) bool
        gt_anchor_frame: Tensor,  # (B, N) long
        K: Tensor,  # (B, 3, 3)
    ) -> TrackingLossOutput:
        if pred.scale is None:
            raise RuntimeError(
                "TrackingLossV18 requires pred.scale; set model.predict_scale=True"
            )

        B, F_, N, _ = gt_tracks_XYZ.shape
        a = gt_anchor_frame.clamp(min=0, max=F_ - 1).long()  # (B, N)
        init_xyz = gt_tracks_XYZ.gather(
            dim=1,
            index=a.view(B, 1, N, 1).expand(B, 1, N, 3),
        ).squeeze(1)  # (B, N, 3)
        p_hat = reconstruct_trajectory_v18(
            pred.xyz, pred.scale, init_xyz, a
        )  # (B, F, N, 3)
        s_gt = _gt_path_length_from_anchor(gt_tracks_XYZ, a)  # (B, F, N)

        vis_f = gt_visibility.float()
        qm = gt_query_mask.unsqueeze(1).expand(B, F_, N).float()
        w_pos = vis_f * qm
        denom = w_pos.sum().clamp_min(1.0)

        r_3D = (p_hat - gt_tracks_XYZ) / s_gt.unsqueeze(-1)  # (B, F, N, 3)
        pos_3D = ((r_3D.pow(2).sum(dim=-1)) * w_pos).sum() / denom

        uv_hat = _project(p_hat, K)
        uv_gt = _project(gt_tracks_XYZ, K)
        r_2D = (uv_hat - uv_gt) / self.image_size
        finite_2D = torch.isfinite(r_2D).all(dim=-1).float()
        w_2D = w_pos * finite_2D
        denom_2D = w_2D.sum().clamp_min(1.0)
        pos_2D = (
            torch.nan_to_num(r_2D.pow(2).sum(dim=-1), nan=0.0, posinf=0.0, neginf=0.0)
            * w_2D
        ).sum() / denom_2D

        vis_loss = F.binary_cross_entropy_with_logits(
            pred.vis_logits,
            vis_f,
            weight=qm,
            reduction="sum",
        ) / qm.sum().clamp_min(1.0)

        total = (
            self.w["pos_3D"] * pos_3D
            + self.w["pos_2D"] * pos_2D
            + self.w["vis"] * vis_loss
        )
        return TrackingLossOutput(
            total=total,
            pos_3D=pos_3D,
            pos_2D=pos_2D,
            vis=vis_loss,
        )


_TERMS_V20 = ("pos_3D", "scale", "pos_2D", "vis")


def _per_clip_motion_scale(
    gt_tracks_XYZ: Tensor,  # (B, F, N, 3)
    init_xyz: Tensor,  # (B, N, 3)  GT at each track's anchor
    w_pos: Tensor,  # (B, F, N)  visible ∧ query mask (float)
    anchor_idx: Tensor | None = None,  # (B, N) long — required to mask anchor frames
    eps: float = 1e-4,
) -> Tensor:
    """Per-clip metric motion scale: RMS over visible NON-ANCHOR (t,n) of the
    displacement from each track's anchor, ‖p*(t,n) − p*_anchor(n)‖.

    Two fixes vs the v20 version:
      * Mask out anchor frames. At t = anchor_idx[b,n], ‖m*‖ ≡ 0 by
        construction (m* is displacement *from* the anchor). Including
        these zeros in the aggregation drags the median toward 0, which
        in v20 produced occasional `scale_gt ≈ floor` → m*/scale_gt
        blowing up by 10³× → pos_3D spikes of 10⁶+ on a few clips.
      * RMS instead of median. RMS is dominated by large values, so it
        reflects the true motion magnitude of the clip rather than being
        depressed by the many small-displacement (t,n) pairs near the
        anchor frame.

    One positive scalar per batch item. This is the target the ScaleHead
    learns to predict from the scene (log-space). Floored at `eps` only as
    a numerical guard for the log on genuinely-static clips.
    """
    B, F_, N = w_pos.shape
    disp = (gt_tracks_XYZ - init_xyz.unsqueeze(1)).norm(dim=-1)  # (B, F, N)
    if anchor_idx is not None:
        t_idx = torch.arange(F_, device=disp.device).view(1, F_, 1).expand(B, F_, N)
        a_idx = anchor_idx.view(B, 1, N).expand(B, F_, N)
        non_anchor = (t_idx != a_idx).float()
    else:
        non_anchor = torch.ones_like(w_pos)
    w = w_pos * non_anchor  # (B, F, N)
    sq = (disp * disp) * w
    n = w.sum(dim=(1, 2)).clamp_min(1.0)
    rms = (sq.sum(dim=(1, 2)) / n).clamp_min(eps * eps).sqrt()
    return rms.detach()


def _per_clip_anchor_depth_scale(
    init_xyz: Tensor,  # (B, N, 3)  GT XYZ at each track's anchor frame
    qmask: Tensor,  # (B, N)     query mask (bool or float)
    eps: float = 1e-2,
) -> Tensor:
    """Per-clip scene-depth scale: median anchor-frame Z over query tracks.

    Why depth, not motion-RMS (v20/v21/v22):
      * Image motion ≈ 3D motion / depth, so depth is the projective scale a
        feature tracker can actually observe from a monocular video.
      * Bounded below by physics (no visible point sits at the optical centre),
        so log(scale_gt) is well-defined regardless of how much the scene moves
        — eliminates the near-singular `(log s − log scale_gt)²` spike that
        v20/v21/v22 suffer on near-static clips.
      * Median aligns with TAPVid-3D's official `scaling="median"` evaluator
        normalisation: train- and eval-time scale conventions match exactly.

    Returns one positive scalar per batch item.
    """
    z = init_xyz[..., 2]  # (B, N)
    valid = qmask.float() * torch.isfinite(z).float() * (z > 0).float()  # (B, N)
    # Sentinel-fill invalid entries with +inf so they sort to the end; pick the
    # floor((n_valid - 1) / 2)-th sorted value per row.
    z_sentinel = torch.where(valid > 0, z, torch.full_like(z, float("inf")))
    sorted_z, _ = torch.sort(z_sentinel, dim=-1)
    n_valid = valid.sum(dim=-1).long().clamp_min(1)  # (B,)
    mid = ((n_valid - 1) // 2).clamp_min(0)  # (B,)
    med = sorted_z.gather(1, mid.unsqueeze(-1)).squeeze(-1)  # (B,)
    med = torch.where(torch.isfinite(med), med, torch.full_like(med, eps))
    return med.clamp_min(eps).detach()


class TrackingLossV20(nn.Module):
    """v20 loss: scale-invariant decomposition — shape and scale supervised
    SEPARATELY so neither starves the other.

    The v18/v19 failure was the multiplicative product `p̂ = p_anchor +
    s · cumsum(Δp̃)`: the shape gradient ∂L/∂Δp̃ ∝ s, so when the ScaleHead
    drove `s → 0` (optimal whenever the shape was still noisy), the shape
    head stopped learning — a deadlock that collapsed pstudio/adt.

    v20 breaks the coupling. With `scale*` = the clip's GT motion magnitude
    (`_per_clip_motion_scale`) and `p̃* = (p* − p*_anchor) / scale*` the
    unitless GT trajectory:

      shape (pos_3D): ‖ cumsum(Δp̃) − p̃* ‖²
                      gradient → xyz_head, NO `s` factor → never starved.
      scale:          ( log s − log scale* )²
                      gradient → ScaleHead, independent of shape quality.
                      `s = exp(z)` (ScaleHead param="exp"), so log s = z;
                      this directly teaches the scene→scale mapping.
      pos_2D:         ‖ (π(p̂) − π(p*)) / image_size ‖²  with the metric
                      reconstruction `p̂ = p*_anchor + scale* · cumsum(Δp̃)`
                      (teacher-forced GT scale, so 2D supervises shape in
                      pixel space without re-coupling to the scale head).
      vis:            BCEWithLogits.

    Inference (eval/render) uses the PREDICTED scale:
        p̂ = p*_anchor + s_pred · cumsum(Δp̃),
    which is exactly `reconstruct_trajectory_v18(pred.xyz, pred.scale, ...)`
    — so the eval/render reconstruction path is unchanged from v18/v19.
    """

    def __init__(
        self,
        weights: Mapping[str, float],
        image_size: int = 224,
        mask_z_negative: bool = False,
        scale_source: str = "motion_rms",
        loss_form: str = "L2",
        metric_2d_uses_pred_scale: bool = False,
        amp_3d: float = 1.0,
        amp_2d: float = 1.0,
    ) -> None:
        super().__init__()
        missing = set(_TERMS_V20) - set(weights)
        if missing:
            raise ValueError(f"TrackingLossV20: missing weights for {sorted(missing)}")
        self.w = {k: float(weights[k]) for k in _TERMS_V20}
        self.image_size = float(image_size)
        # v22+: geometrically reject Z<=0 in 2D loss and tie visibility to Z>0
        # (a point with predicted Z<=0 is behind the camera, hence not on-image;
        # if GT says visible at that frame, that mismatch must show up as vis
        # loss). Replaces adding a duplicate metric-3D term that would just
        # make the total loss fluctuate (per user 2026-05-28).
        self.mask_z_negative = bool(mask_z_negative)
        # v23: scale_gt source. "motion_rms" (v20/v21/v22) uses RMS of GT motion
        # — degenerate on near-static clips → log-MSE spikes to ~80. "anchor_depth_median"
        # uses median anchor-frame Z (scene depth) — bounded by physics, matches
        # TAPVid-3D `scaling="median"` evaluator normalisation, and is the natural
        # projective invariant (image motion ≈ 3D motion / depth).
        if scale_source not in ("motion_rms", "anchor_depth_median"):
            raise ValueError(
                f"TrackingLossV20: scale_source must be 'motion_rms' or "
                f"'anchor_depth_median', got {scale_source!r}"
            )
        self.scale_source = scale_source
        # loss_form for pos_3D and pos_2D terms (v24+).
        #   "L2"    (v20-v23): r² — gradient 2r vanishes as r→0. With depth-
        #            normalised residuals (~0.02), L2 gradient is ~0.04 and the
        #            shape head never learns past random init.
        #   "log1p" (v24): log(|r|+1) element-wise summed. Gradient
        #            sign(r)/(|r|+1) → ±1 at small r (fixes starvation),
        #            saturates at large r (outlier-robust but no outlier push).
        #            Empirically: bucket-median p3D flat across 11k steps —
        #            the m=0 trap is escaped but optimisation doesn't converge.
        #   "L1"    (v25): |r| element-wise summed. Gradient sign(r) constant 1
        #            at all r — fixes the v23 m=0 trap (where L2's 2·(m*/depth)
        #            ≈ 0.04 push left m=0 as a quasi-stable solution). Loss
        #            VALUE grows linearly with r (vs log1p's saturation), so
        #            large-residual clips contribute proportionally more to the
        #            batch sum.
        if loss_form not in ("L2", "log1p", "L1"):
            raise ValueError(
                f"TrackingLossV20: loss_form must be 'L2', 'log1p', or 'L1', got {loss_form!r}"
            )
        self.loss_form = loss_form
        # v26: by default (False) the 2D loss reconstructs with scale_star (the
        # GT-derived median depth) — same as inference uses s_pred, so training
        # never sees the cost of mispredicting scale, leading to a train/inference
        # mismatch (small pos_2D in training but bad tracking on test videos).
        # When True (v26+), reconstruct with `pred.scale` instead — gradient
        # flows back through scale_head from the 2D loss, forcing s_pred to be
        # consistent with what the 2D projection requires. Per user 2026-05-29.
        self.metric_2d_uses_pred_scale = bool(metric_2d_uses_pred_scale)
        # v27: amplification multipliers applied to pos_3D and pos_2D BEFORE the
        # weighted sum. Default 1.0 preserves v20-v26 behaviour. Used in v27 to
        # un-normalise the position losses to their physical scales: pos_2D
        # ≈ pixels (× image_size ≈ 1000) and pos_3D ≈ "100m scale" (× 100) so
        # both dominate the weighted total. The scale_head still gets gradient
        # via the 2D path (v26 fix) so dropping scale_loss to near-zero is safe.
        self.amp_3d = float(amp_3d)
        self.amp_2d = float(amp_2d)

    def forward(
        self,
        pred: TrackerOutputs,
        gt_tracks_XYZ: Tensor,  # (B, F, N, 3)
        gt_visibility: Tensor,  # (B, F, N) bool
        gt_query_mask: Tensor,  # (B, N) bool
        gt_anchor_frame: Tensor,  # (B, N) long
        K: Tensor,  # (B, 3, 3)
    ) -> TrackingLossOutput:
        if pred.scale is None:
            raise RuntimeError(
                "TrackingLossV20 requires pred.scale; set model.predict_scale=True"
            )

        B, F_, N, _ = gt_tracks_XYZ.shape
        a = gt_anchor_frame.clamp(min=0, max=F_ - 1).long()
        init_xyz = gt_tracks_XYZ.gather(
            dim=1,
            index=a.view(B, 1, N, 1).expand(B, 1, N, 3),
        ).squeeze(1)  # (B, N, 3)

        vis_f = gt_visibility.float()
        qm = gt_query_mask.unsqueeze(1).expand(B, F_, N).float()
        w_pos = vis_f * qm
        denom = w_pos.sum().clamp_min(1.0)

        # Unitless shapes: predicted cumsum (no scale, no anchor offset) and GT.
        zeros = init_xyz.new_zeros(B, N, 3)
        p_tilde = reconstruct_trajectory(pred.xyz, zeros, a)  # (B, F, N, 3)
        if self.scale_source == "anchor_depth_median":
            scale_star = _per_clip_anchor_depth_scale(init_xyz, gt_query_mask)  # (B,)
        else:
            scale_star = _per_clip_motion_scale(
                gt_tracks_XYZ,
                init_xyz,
                w_pos,
                anchor_idx=a,
            )  # (B,)
        gt_tilde = (gt_tracks_XYZ - init_xyz.unsqueeze(1)) / scale_star.view(B, 1, 1, 1)

        # Shape term — clean gradient to the trajectory head.
        r_shape = p_tilde - gt_tilde
        if self.loss_form == "log1p":
            # log(|r|+1) summed over xyz; gradient sign(r)/(|r|+1) is ~1 at the
            # small (~0.02) depth-normalised residuals where L2's 2r vanishes.
            pos_3D = (torch.log1p(r_shape.abs()).sum(dim=-1) * w_pos).sum() / denom
        elif self.loss_form == "L1":
            # |r| summed over xyz; gradient sign(r), constant 1 at all r.
            # Loss VALUE = |r| grows linearly with residual — large-residual
            # clips contribute proportionally more to the aggregated batch loss
            # than under log1p (which saturates).
            pos_3D = (r_shape.abs().sum(dim=-1) * w_pos).sum() / denom
        else:
            pos_3D = ((r_shape.pow(2).sum(dim=-1)) * w_pos).sum() / denom

        # Scale term — direct supervision of log-scale (z) toward log scale*.
        log_s = torch.log(pred.scale.clamp_min(1e-6))  # (B,)
        scale_loss = (log_s - torch.log(scale_star)).pow(2).mean()

        # 2D term — project the metric reconstruction. v26: optionally use
        # s_pred instead of scale_star so the 2D loss matches the inference
        # reconstruction; training then sees and corrects mispredicted scale.
        if self.metric_2d_uses_pred_scale:
            scale_for_2d = pred.scale.view(B, 1, 1, 1)
        else:
            scale_for_2d = scale_star.view(B, 1, 1, 1)
        p_hat_metric = init_xyz.unsqueeze(1) + scale_for_2d * p_tilde
        uv_hat = _project(p_hat_metric, K)
        uv_gt = _project(gt_tracks_XYZ, K)
        r_2D = (uv_hat - uv_gt) / self.image_size
        finite_2D = torch.isfinite(r_2D).all(dim=-1).float()
        z_pred = p_hat_metric[..., 2]
        if self.mask_z_negative:
            # A predicted point at Z<=0 is behind the camera and cannot appear
            # on the image; its 2D projection is geometrically meaningless. Drop
            # those entries from the 2D loss entirely (not just clamped via the
            # _project floor) so they neither contribute residual nor blow up.
            z_ok = (z_pred > 0).float()
        else:
            z_ok = torch.ones_like(z_pred)
        w_2D = w_pos * finite_2D * z_ok
        denom_2D = w_2D.sum().clamp_min(1.0)
        if self.loss_form == "log1p":
            # r_2D is already image-width normalised — log(|r_2D|+1) per-axis
            # summed gives the same non-vanishing gradient behaviour as p3D.
            pos_2D_per = torch.log1p(r_2D.abs()).sum(dim=-1)
            pos_2D = (
                torch.nan_to_num(pos_2D_per, nan=0.0, posinf=0.0, neginf=0.0) * w_2D
            ).sum() / denom_2D
        elif self.loss_form == "L1":
            pos_2D_per = r_2D.abs().sum(dim=-1)
            pos_2D = (
                torch.nan_to_num(pos_2D_per, nan=0.0, posinf=0.0, neginf=0.0) * w_2D
            ).sum() / denom_2D
        else:
            pos_2D = (
                torch.nan_to_num(
                    r_2D.pow(2).sum(dim=-1), nan=0.0, posinf=0.0, neginf=0.0
                )
                * w_2D
            ).sum() / denom_2D

        if self.mask_z_negative:
            # Visibility couples to projectability. Effective predicted
            # "visible" probability is sigmoid(vis_logits) AND Z>0 — predicting
            # Z<0 is implicitly predicting "invisible" at that frame, so a GT
            # visible point with predicted Z<0 produces a high BCE on its own,
            # giving Z a gradient toward >0 through the visibility term.
            # `sigmoid(z / 0.05)` is a smooth Z>0 indicator with a ~5cm
            # transition width (so the gradient is well-defined everywhere).
            z_safe = torch.sigmoid(z_pred / 0.05)
            p_vis_eff = (torch.sigmoid(pred.vis_logits) * z_safe).clamp(1e-7, 1 - 1e-7)
            vis_bce = -(vis_f * p_vis_eff.log() + (1 - vis_f) * (1 - p_vis_eff).log())
            vis_loss = (vis_bce * qm).sum() / qm.sum().clamp_min(1.0)
        else:
            vis_loss = F.binary_cross_entropy_with_logits(
                pred.vis_logits,
                vis_f,
                weight=qm,
                reduction="sum",
            ) / qm.sum().clamp_min(1.0)

        # v27: amplify pos_3D and pos_2D BEFORE the weighted sum so they
        # dominate the total. amp_*=1.0 (default) preserves v20-v26 behaviour.
        pos_3D_amp = self.amp_3d * pos_3D
        pos_2D_amp = self.amp_2d * pos_2D
        total = (
            self.w["pos_3D"] * pos_3D_amp
            + self.w["scale"] * scale_loss
            + self.w["pos_2D"] * pos_2D_amp
            + self.w["vis"] * vis_loss
        )
        # Report the AMPLIFIED per-term values in the loss output so the
        # training log and plots reflect what the optimizer is actually seeing.
        return TrackingLossOutput(
            total=total,
            pos_3D=pos_3D_amp,
            pos_2D=pos_2D_amp,
            vis=vis_loss,
            scale=scale_loss,
        )


class TrackingLoss(nn.Module):
    """v11 loss. Construct with normalised weights from `train/config.py`."""

    def __init__(
        self,
        weights: Mapping[str, float],
        image_size: int = 224,
    ) -> None:
        super().__init__()
        missing = set(_TERMS) - set(weights)
        if missing:
            raise ValueError(f"TrackingLoss: missing weights for {sorted(missing)}")
        self.w = {k: float(weights[k]) for k in _TERMS}
        self.image_size = float(image_size)

    def forward(
        self,
        pred: TrackerOutputs,
        gt_tracks_XYZ: Tensor,  # (B, F, N, 3)
        gt_visibility: Tensor,  # (B, F, N) bool
        gt_query_mask: Tensor,  # (B, N) bool
        gt_anchor_frame: Tensor,  # (B, N) long — kept in signature for backwards compat (unused in v11)
        K: Tensor,  # (B, 3, 3)
    ) -> TrackingLossOutput:
        B, F_, N, _ = gt_tracks_XYZ.shape

        # v12: each track is anchored at its own query frame `gt_anchor_frame[b, n]`,
        # not at clip-frame-0. p̂(a_n, n) = GT[a_n, n]; trajectory integrated
        # bidirectionally from there. This matches the TAPVid-3D evaluation
        # convention where queries can be at any frame and all visible frames
        # are scored. See `reconstruct_trajectory` for the cumsum-gather math.
        a = gt_anchor_frame.clamp(min=0, max=F_ - 1).long()  # (B, N)
        init_xyz = gt_tracks_XYZ.gather(
            dim=1,
            index=a.view(B, 1, N, 1).expand(B, 1, N, 3),
        ).squeeze(1)  # (B, N, 3)
        p_hat = reconstruct_trajectory(pred.xyz, init_xyz, a)  # (B, F, N, 3)

        # Mask: visible (t, n) AND real query slot.
        vis_f = gt_visibility.float()
        qm = gt_query_mask.unsqueeze(1).expand(B, F_, N).float()
        w_pos = vis_f * qm  # (B, F, N)
        denom = w_pos.sum().clamp_min(1.0)

        # Per-clip 3-D scale (detached so no gradient flows back through GT magnitudes).
        s_3D = _per_clip_median_scale(gt_tracks_XYZ, w_pos).detach()  # (B,)
        s_3D_b = s_3D.view(B, 1, 1, 1)
        s_2D_b = self.image_size

        # L_3D — scale-normalised squared L2 in 3D.
        r_3D = (p_hat - gt_tracks_XYZ) / s_3D_b  # (B, F, N, 3)
        pos_3D = ((r_3D.pow(2).sum(dim=-1)) * w_pos).sum() / denom

        # L_2D — pinhole-project both predicted and GT, then scale-normalised
        # squared L2 in pixel space.
        uv_hat = _project(p_hat, K)  # (B, F, N, 2)
        uv_gt = _project(gt_tracks_XYZ, K)  # (B, F, N, 2)
        r_2D = (uv_hat - uv_gt) / s_2D_b  # (B, F, N, 2)
        # Mask out non-finite projections (rare: Z near zero — clamp handles it,
        # but be defensive).
        finite_2D = torch.isfinite(r_2D).all(dim=-1).float()
        w_2D = w_pos * finite_2D
        denom_2D = w_2D.sum().clamp_min(1.0)
        pos_2D = (
            (torch.nan_to_num(r_2D.pow(2).sum(dim=-1), nan=0.0, posinf=0.0, neginf=0.0))
            * w_2D
        ).sum() / denom_2D

        # Visibility BCE — masked by qm (per-frame target weighted by query slot).
        vis_loss = F.binary_cross_entropy_with_logits(
            pred.vis_logits,
            vis_f,
            weight=qm,
            reduction="sum",
        ) / qm.sum().clamp_min(1.0)

        total = (
            self.w["pos_3D"] * pos_3D
            + self.w["pos_2D"] * pos_2D
            + self.w["vis"] * vis_loss
        )
        return TrackingLossOutput(
            total=total,
            pos_3D=pos_3D,
            pos_2D=pos_2D,
            vis=vis_loss,
        )


def _unproject_with_depth(
    uv: Tensor,  # (B, F, N, 2) pixel coords on the encoder-resolution image
    depth_map: Tensor,  # (B, F, Hd, Wd) per-frame metric depth (DA3-Large, frozen)
    K: Tensor,  # (B, 3, 3) intrinsics matching the encoder-resolution image
    image_size: float,  # encoder square resolution (depth_map covers same FOV)
) -> Tensor:
    """Differentiably sample `depth_map` at `uv` and unproject to camera-frame XYZ.

    The DA3 depth map is at the model's own working resolution (e.g. 504), while
    `uv` lives in the encoder's pixel-coord grid (e.g. 896). `F.grid_sample`
    handles the resolution mismatch — the gradient flows back to `uv` via
    bilinear interpolation weights, and `depth_map` itself is the frozen DA3
    output (no gradient needed through it).
    """
    B, F_, N, _ = uv.shape
    Hd, Wd = depth_map.shape[-2:]
    grid = 2.0 * (uv / image_size) - 1.0  # (B, F, N, 2)
    grid = grid.view(B * F_, 1, N, 2)
    z = F.grid_sample(
        depth_map.view(B * F_, 1, Hd, Wd),
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    ).view(B, F_, N)
    fx = K[:, 0, 0].view(B, 1, 1)
    fy = K[:, 1, 1].view(B, 1, 1)
    cx = K[:, 0, 2].view(B, 1, 1)
    cy = K[:, 1, 2].view(B, 1, 1)
    u = uv[..., 0]
    v = uv[..., 1]
    x = (u - cx) / fx * z
    y = (v - cy) / fy * z
    return torch.stack([x, y, z], dim=-1)  # (B, F, N, 3)


class TrackingLossV31(nn.Module):
    """v31 loss: 2D-track loss + 3D-unproject loss (DA3 metric depth) + visibility.

    The 2D pixel-tracker emits `uv_pred = anchor_uv + delta_uv` per (t, n).
    A frozen pretrained metric-depth model (DA3-Large) supplies the absolute
    scale. We sample that depth differentiably at `uv_pred` and unproject with
    per-clip intrinsics K to get `xyz_pred`. Gradients flow to the uv_head from
    BOTH 2D and 3D loss terms (via F.grid_sample for the 3D branch).

    Three terms — all L1 (v25's empirical finding: constant gradient is what
    this regime needs):

      L_pos_3D = mean over visible (t, n)   |xyz_pred − xyz*| / scale_gt
      L_pos_2D = mean over visible (t, n)   |uv_pred − uv*|  / image_size
      L_vis    = BCE(vis_logits, gt_visibility)

    `scale_gt` is per-clip median anchor-frame Z (`_per_clip_anchor_depth_scale`),
    matching scale_est v8 and the v23+ depth-normalised loss — puts pstudio
    (~3 m), drivetrack (~20 m), and adt (~1 m) on the same gradient footing.
    """

    def __init__(
        self,
        weights: Mapping[str, float],
        image_size: int = 224,
    ) -> None:
        super().__init__()
        missing = set(_TERMS) - set(weights)
        if missing:
            raise ValueError(f"TrackingLossV31: missing weights for {sorted(missing)}")
        self.w = {k: float(weights[k]) for k in _TERMS}
        self.image_size = float(image_size)

    def forward(
        self,
        pred: TrackerOutputs,
        gt_tracks_XYZ: Tensor,  # (B, F, N, 3)
        gt_visibility: Tensor,  # (B, F, N) bool
        gt_query_mask: Tensor,  # (B, N) bool
        gt_anchor_frame: Tensor,  # (B, N) long
        K: Tensor,  # (B, 3, 3)
        depth: Tensor,  # (B, F, Hd, Wd) cached DA3-Large metric depth
    ) -> TrackingLossOutput:
        if pred.uv is None:
            raise RuntimeError(
                "TrackingLossV31 requires pred.uv; set model.head_mode='uv'"
            )

        B, F_, N, _ = gt_tracks_XYZ.shape
        a = gt_anchor_frame.clamp(min=0, max=F_ - 1).long()
        init_xyz = gt_tracks_XYZ.gather(
            dim=1,
            index=a.view(B, 1, N, 1).expand(B, 1, N, 3),
        ).squeeze(1)  # (B, N, 3)

        vis_f = gt_visibility.float()
        qm = gt_query_mask.unsqueeze(1).expand(B, F_, N).float()
        w_pos = vis_f * qm

        scale_gt = _per_clip_anchor_depth_scale(init_xyz, gt_query_mask)  # (B,)

        uv_gt = _project(gt_tracks_XYZ, K)  # (B, F, N, 2)
        r_2D = (pred.uv - uv_gt) / self.image_size  # (B, F, N, 2)
        finite_2D = torch.isfinite(r_2D).all(dim=-1).float()
        w_2D = w_pos * finite_2D
        denom_2D = w_2D.sum().clamp_min(1.0)
        pos_2D = (
            torch.nan_to_num(r_2D.abs().sum(dim=-1), nan=0.0, posinf=0.0, neginf=0.0)
            * w_2D
        ).sum() / denom_2D

        xyz_pred = _unproject_with_depth(
            pred.uv, depth, K, self.image_size
        )  # (B, F, N, 3)
        r_3D = (xyz_pred - gt_tracks_XYZ) / scale_gt.view(B, 1, 1, 1)
        finite_3D = torch.isfinite(r_3D).all(dim=-1).float()
        w_3D = w_pos * finite_3D
        denom_3D = w_3D.sum().clamp_min(1.0)
        pos_3D = (
            torch.nan_to_num(r_3D.abs().sum(dim=-1), nan=0.0, posinf=0.0, neginf=0.0)
            * w_3D
        ).sum() / denom_3D

        vis_loss = F.binary_cross_entropy_with_logits(
            pred.vis_logits,
            vis_f,
            weight=qm,
            reduction="sum",
        ) / qm.sum().clamp_min(1.0)

        total = (
            self.w["pos_3D"] * pos_3D
            + self.w["pos_2D"] * pos_2D
            + self.w["vis"] * vis_loss
        )
        return TrackingLossOutput(
            total=total,
            pos_3D=pos_3D,
            pos_2D=pos_2D,
            vis=vis_loss,
        )


class TrackingLossV33(nn.Module):
    """v33 loss: 3D-position-only (depth-along-ray refiner).

    The model emits a refined 3D track `pred.xyz` directly — the SEA-RAFT 2D
    position and FB-consistency visibility are frozen and NOT model outputs, so
    there is nothing to supervise for 2D or visibility (their gradients would be
    zero anyway). The only learnable signal is the per-track depth correction,
    supervised by the 3D position error:

        L_pos_3D = mean over visible (t, n)   |xyz_pred - xyz*| / scale_gt   (L1)

    `scale_gt` is the per-clip median anchor-frame Z (matches V31 / the official
    median-scaling evaluator). pos_2D and vis are reported as zero for logging
    parity with TrackingLossV31.
    """

    def __init__(self, weights: Mapping[str, float], image_size: int = 896) -> None:
        super().__init__()
        if "pos_3D" not in weights:
            raise ValueError("TrackingLossV33: missing weight for 'pos_3D'")
        self.w_pos_3D = float(weights["pos_3D"])
        self.image_size = float(image_size)

    def forward(
        self,
        pred: TrackerOutputs,
        gt_tracks_XYZ: Tensor,  # (B, F, N, 3)
        gt_visibility: Tensor,  # (B, F, N) bool
        gt_query_mask: Tensor,  # (B, N) bool
        gt_anchor_frame: Tensor,  # (B, N) long
        K: Tensor,  # (B, 3, 3) — unused (xyz already in camera frame)
    ) -> TrackingLossOutput:
        if pred.xyz is None:
            raise RuntimeError("TrackingLossV33 requires pred.xyz")

        B, F_, N, _ = gt_tracks_XYZ.shape
        a = gt_anchor_frame.clamp(min=0, max=F_ - 1).long()
        init_xyz = gt_tracks_XYZ.gather(
            dim=1,
            index=a.view(B, 1, N, 1).expand(B, 1, N, 3),
        ).squeeze(1)  # (B, N, 3)

        vis_f = gt_visibility.float()
        qm = gt_query_mask.unsqueeze(1).expand(B, F_, N).float()
        w_pos = vis_f * qm

        scale_gt = _per_clip_anchor_depth_scale(init_xyz, gt_query_mask)  # (B,)
        r_3D = (pred.xyz - gt_tracks_XYZ) / scale_gt.view(B, 1, 1, 1)
        finite_3D = torch.isfinite(r_3D).all(dim=-1).float()
        w_3D = w_pos * finite_3D
        denom_3D = w_3D.sum().clamp_min(1.0)
        pos_3D = (
            torch.nan_to_num(r_3D.abs().sum(dim=-1), nan=0.0, posinf=0.0, neginf=0.0)
            * w_3D
        ).sum() / denom_3D

        zero = pos_3D.new_zeros(())
        return TrackingLossOutput(
            total=self.w_pos_3D * pos_3D,
            pos_3D=pos_3D,
            pos_2D=zero,
            vis=zero,
        )


class TrackingLossV35(nn.Module):
    """v35 loss: vis-masked L1 3D + Δuv L2 regularization.

    L = vis * |xyz_pred - xyz_gt|_1 / scale_gt     (L_3D, same as v33)
      + λ_uv * vis * |delta_uv|²                    (Δuv regularization)

    Invisible points (vis=0) contribute zero gradient. L_2D field in the
    returned TrackingLossOutput carries the Δuv regularization for logging.
    """

    def __init__(self, weights: Mapping[str, float], image_size: int = 896) -> None:
        super().__init__()
        if "pos_3D" not in weights:
            raise ValueError("TrackingLossV35: missing weight for 'pos_3D'")
        self.w_pos_3D = float(weights["pos_3D"])
        self.w_reg_uv = float(weights.get("reg_uv", 0.0))
        self.image_size = float(image_size)

    def forward(
        self,
        pred: TrackerOutputs,
        gt_tracks_XYZ: Tensor,  # (B, F, N, 3)
        gt_visibility: Tensor,  # (B, F, N)
        gt_query_mask: Tensor,  # (B, N)
        gt_anchor_frame: Tensor,  # (B, N) long
        K: Tensor,  # (B, 3, 3) — unused (xyz already camera-frame)
    ) -> TrackingLossOutput:
        B, F_, N, _ = gt_tracks_XYZ.shape
        a = gt_anchor_frame.clamp(min=0, max=F_ - 1).long()
        init_xyz = gt_tracks_XYZ.gather(
            dim=1,
            index=a.view(B, 1, N, 1).expand(B, 1, N, 3),
        ).squeeze(1)

        vis_f = gt_visibility.float()
        qm = gt_query_mask.unsqueeze(1).expand(B, F_, N).float()
        w_pos = vis_f * qm

        scale_gt = _per_clip_anchor_depth_scale(init_xyz, gt_query_mask)
        r_3D = (pred.xyz - gt_tracks_XYZ) / scale_gt.view(B, 1, 1, 1)
        finite_3D = torch.isfinite(r_3D).all(dim=-1).float()
        w_3D = w_pos * finite_3D
        denom = w_3D.sum().clamp_min(1.0)
        pos_3D = (torch.nan_to_num(r_3D.abs().sum(dim=-1)) * w_3D).sum() / denom

        if pred.delta_uv is not None and self.w_reg_uv > 0.0:
            reg_uv = (pred.delta_uv.pow(2).sum(dim=-1) * w_pos).sum() / denom
        else:
            reg_uv = pos_3D.new_zeros(())

        zero = pos_3D.new_zeros(())
        total = self.w_pos_3D * pos_3D + self.w_reg_uv * reg_uv
        return TrackingLossOutput(total=total, pos_3D=pos_3D, pos_2D=reg_uv, vis=zero)
