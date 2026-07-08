"""Sanity tests for the v11 tracking loss (cumsum trajectory + scale-normalised L2)."""

from __future__ import annotations

from pathlib import Path

import torch

from mamba3_tracker.model.heads import TrackerOutputs
from mamba3_tracker.train.config import load_config
from mamba3_tracker.train.loss import TrackingLoss, reconstruct_trajectory


REPO = Path(__file__).resolve().parents[2]


def _toy_gt(B: int = 1, F: int = 4, N: int = 3, drift: float = 0.1):
    """GT with linear motion. Anchor at frame 0."""
    torch.manual_seed(0)
    anchor = torch.tensor([[1.0, 0.5, 5.0]]).expand(B, N, 3).contiguous()
    t = torch.arange(F, dtype=torch.float32).view(1, F, 1, 1)
    motion = torch.tensor([drift, drift, drift]).view(1, 1, 1, 3) * t
    tracks = anchor.unsqueeze(1) + motion          # (B, F, N, 3)
    vis = torch.ones(B, F, N, dtype=torch.bool)
    qmask = torch.ones(B, N, dtype=torch.bool)
    anchor_idx = torch.zeros(B, N, dtype=torch.long)
    K = torch.tensor([[500.0, 0.0, 224.0],
                      [0.0, 500.0, 224.0],
                      [0.0, 0.0, 1.0]]).view(1, 3, 3).expand(B, 3, 3).contiguous()
    return tracks, vis, qmask, anchor_idx, K


def _equal_weights() -> dict[str, float]:
    return {"pos_3D": 1.0 / 3, "pos_2D": 1.0 / 3, "vis": 1.0 / 3}


def test_reconstruct_trajectory_at_anchor_equals_init():
    """p̂(a_n) must equal init_xyz exactly, regardless of delta_pred values."""
    B, F, N = 2, 8, 3
    delta = torch.randn(B, F, N, 3) * 5.0
    init = torch.tensor([[1.0, 2.0, 3.0]]).expand(B, N, 3).contiguous()
    anchor = torch.randint(low=0, high=F, size=(B, N))
    p_hat = reconstruct_trajectory(delta, init, anchor)
    assert p_hat.shape == (B, F, N, 3)
    for b in range(B):
        for n in range(N):
            a = anchor[b, n].item()
            assert torch.allclose(p_hat[b, a, n], init[b, n], atol=1e-5), \
                f"p̂(a_n={a}) should equal init_xyz at (b={b}, n={n})"


def test_reconstruct_trajectory_bidirectional_cumsum():
    """For anchor at t=2 (middle), forward t>2 adds Δp̂ from anchor+1; backward t<2 subtracts."""
    B, N = 1, 1
    delta = torch.tensor([[[0.0, 0.0, 0.0],          # t=0
                           [1.0, 0.0, 0.0],          # t=1: motion (1,0,0)
                           [0.0, 2.0, 0.0],          # t=2: motion (0,2,0)  ← anchor here
                           [0.0, 0.0, 3.0],          # t=3: motion (0,0,3)
                           [10.0, 0.0, 0.0]]]).view(B, 5, N, 3)  # t=4: motion (10,0,0)
    init = torch.tensor([[100.0, 200.0, 300.0]]).view(B, N, 3)
    anchor = torch.tensor([[2]])
    p_hat = reconstruct_trajectory(delta, init, anchor)
    # Forward from anchor=2:
    #   p̂(3) = init + Δp̂(3) = (100, 200, 303)
    #   p̂(4) = init + Δp̂(3) + Δp̂(4) = (110, 200, 303)
    # Backward from anchor=2:
    #   p̂(1) = init - Δp̂(2) = (100, 200 - 2, 300) = (100, 198, 300)
    #   p̂(0) = init - Δp̂(2) - Δp̂(1) = (99, 198, 300)
    expected = torch.tensor([[[[99.0, 198.0, 300.0]],     # t=0
                              [[100.0, 198.0, 300.0]],    # t=1
                              [[100.0, 200.0, 300.0]],    # t=2 = init
                              [[100.0, 200.0, 303.0]],    # t=3
                              [[110.0, 200.0, 303.0]]]])  # t=4
    assert torch.allclose(p_hat, expected, atol=1e-5), \
        f"got {p_hat}, expected {expected}"


def test_loss_zero_when_predicted_motion_equals_gt_motion():
    """If Δp̂(t) for t ≥ 1 exactly matches the GT per-frame motion, loss is ~0."""
    tracks, vis, qmask, anchor_idx, K = _toy_gt(drift=0.05)
    # GT motion per frame = tracks[t] - tracks[t-1] for t ≥ 1; 0 at t=0 (will be ignored).
    delta = torch.zeros_like(tracks)
    delta[:, 1:] = tracks[:, 1:] - tracks[:, :-1]
    pred = TrackerOutputs(
        xyz=delta.clone(),
        vis_logits=torch.full(vis.shape, 10.0),   # sigmoid ≈ 1, BCE ≈ 0
        spawn_logits=torch.zeros(vis.shape),
    )
    loss = TrackingLoss(weights=_equal_weights(), image_size=448)
    out = loss(pred, tracks, vis, qmask, anchor_idx, K)
    assert out.pos_3D.item() < 1e-9, f"pos_3D={out.pos_3D.item()} should be ~0"
    assert out.pos_2D.item() < 1e-9, f"pos_2D={out.pos_2D.item()} should be ~0"


def test_static_zero_predictor_is_penalised_on_moving_gt():
    """Δp̂ = 0 for all t ≥ 1 means p̂(t) = p̂(0) = GT[0] for all t. With GT moving
    away from frame 0, this is a position error that grows linearly in t. The loss
    must be non-trivial (the v6/v7/v8/v9 collapse must NOT pass this test)."""
    tracks, vis, qmask, anchor_idx, K = _toy_gt(drift=0.5)
    pred = TrackerOutputs(
        xyz=torch.zeros_like(tracks),
        vis_logits=torch.full(vis.shape, 10.0),
        spawn_logits=torch.zeros(vis.shape),
    )
    loss = TrackingLoss(weights=_equal_weights(), image_size=448)
    out = loss(pred, tracks, vis, qmask, anchor_idx, K)
    assert out.pos_3D.item() > 0.01, \
        f"pos_3D={out.pos_3D.item()} — static predictor should produce real loss"
    assert out.pos_2D.item() > 1e-4, \
        f"pos_2D={out.pos_2D.item()} — static predictor should miss in pixel space too"


def test_backward_populates_xyz_grad_only_for_t_ge_1():
    """Gradient on Δp̂(0) must be zero (it's ignored). Gradient on Δp̂(t ≥ 1) non-zero."""
    tracks, vis, qmask, anchor_idx, K = _toy_gt(drift=0.3)
    delta_pred = torch.zeros_like(tracks, requires_grad=True)
    pred = TrackerOutputs(
        xyz=delta_pred,
        vis_logits=torch.full(vis.shape, 10.0, requires_grad=True),
        spawn_logits=torch.zeros(vis.shape, requires_grad=True),
    )
    loss = TrackingLoss(weights=_equal_weights(), image_size=448)
    out = loss(pred, tracks, vis, qmask, anchor_idx, K)
    out.total.backward()
    assert delta_pred.grad is not None
    grad_t0 = delta_pred.grad[:, 0].abs().sum().item()
    grad_t_ge_1 = delta_pred.grad[:, 1:].abs().sum().item()
    assert grad_t0 < 1e-9, f"Δp̂(0) should have ZERO gradient (it's ignored), got {grad_t0}"
    assert grad_t_ge_1 > 0.0, "Δp̂(t≥1) should have non-zero gradient"


def test_config_loader_accepts_v11():
    cfg = load_config(REPO / "configs" / "v11.yaml")
    assert cfg["version"] == "v11"
    s = sum(cfg["loss"]["weights"].values())
    assert abs(s - 1.0) < 1e-9
    assert cfg["model"]["level_sizes"] == [16, 32, 64, 128]
    assert cfg["model"]["num_iters"] == 1
    assert cfg["model"]["use_correlation"] is False
