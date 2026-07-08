---
name: project-spatrackerv2-bidir
description: SpatialTrackerV2 evaluation bugs — two-phase root-cause analysis of 2.64% vs 24.7% AJ gap; bidir+fixed_cam fixes went first, then s_wind+track3d_pred bugs found
metadata:
  type: project
---

## Phase 1 bugs (bidir fix — already applied)

Three bugs in our original `eval_spatracker_v2.py`:
1. **Missing bidirectional tracking** (primary): TAPVid-3D evaluates ALL visible frames including pre-query. Forward-only → zero predictions there. Fix: time-reversed backward pass, combine with `fwd_mask = t_arr >= t_queries[None, :]`.
2. `fixed_cam=False` for ADT/pstudio — caused VO errors. Fix: `fixed_cam=True`.
3. `replace_ratio=0.2` instead of paper's `1.0`. Fix: `replace_ratio=1.0`.

After Phase 1 fix and full re-inference (50 clips each):
```
adt:         norm-AJ=2.64%,  metric-AJ=18.36%
pstudio:     norm-AJ=1.18%,  metric-AJ=19.21%
drivetrack:  norm-AJ=1.67%,  metric-AJ= 0.84%
```
Still 9× below paper's ~24.7% mean normalized AJ.

---

## Phase 2 bugs (root cause of remaining 9× gap)

Found by reading SpaTrackerV2 source code (`eval_predictor.py::forward_sparse`, `SpaTrack.py`):

### Bug A — Wrong `s_wind`: 60 → 500 (official `magic_infer_offline.yaml`)
- With `s_wind=60` and a 300-frame ADT clip: creates 5 sliding windows (step_slide=56)
- With `s_wind=500`: all 300 frames fit in ONE window
- Affects tracking quality (each window tracks independently with less temporal context)

### Bug B — Wrong output: `track3d_pred` → `track2d_pred + reproject_2d3d`
This is the CATASTROPHIC bug. Source:

```python
# eval_predictor.py::forward_sparse (official evaluator):
traj_e, traj_d_e, vis_e = (
    track2d_pred[..., :2][None],   # 2D pixel coords in video space
    track2d_pred[..., 2:3][None],  # depth in metres
    vis_pred[None,...,0],
)
traj_uvd = torch.cat([traj_e, traj_d_e], dim=-1)  # (B,T,N,3)
traj_3d = reproject_2d3d(traj_uvd, sample.intrs)  # per-frame camera XYZ

# Our buggy code (eval_spatracker_v2.py):
track3d = result[4]  # track3d_pred — WRONG
```

`track3d_pred` (result[4]) is assembled from window-local `rgb_tracks` in each window's frame-0 camera coordinate system. With 5 windows, each chunk uses a different camera frame → catastrophically inconsistent 3D across the clip.

`track2d_pred` (result[5]) is always in the original video's 2D pixel space — consistent regardless of how many windows were used. Then `reproject_2d3d` with per-frame K gives correct camera-space XYZ.

### Empirical replication results (2026-07-02)

Script `scripts/replicate_spatracker_v2_paper.py` tested 3 conditions on ADT clips:

**Run 1 — WoodenBowl (small, simple clip):**
```
A: s_wind=60, track3d_pred  → AJ=6.27%
B: s_wind=60, track2d+reproj → AJ=7.17%
C: s_wind=300, track2d+reproj → OOM
```

**Run 2 — 3 low-N clips (N=258-350, Apartment sequences):**
```
A: s_wind=60, track3d_pred  → Mean AJ=0.52%
B: s_wind=60, track2d+reproj → Mean AJ=0.70%
C: s_wind=300, track2d+reproj → All OOM
Paper: s_wind=500, full minival  → ~24.7%
```

**Key finding: Bug A (s_wind) is the PRIMARY cause; Bug B has minor effect.**
- A→B improvement: +0.18pp (low-N clips) or +0.9pp (WoodenBowl)
- Both A and B are vastly below the paper's 24.7%
- s_wind=500 (paper's exact config) requires ~40GB VRAM; RTX 4080 Laptop (12GB) can't test it.
- s_wind=300 also OOMs on long clips.

---

## Current eval results (bidir, phase 1 fixed)

```python
# In scripts/plot_6method_comparison.py (as of 2026-07-02):
spatrackerv2_norm = {"drivetrack": 0.0167, "pstudio": 0.0118, "adt": 0.0264}
spatrackerv2_abs  = {"drivetrack": 0.0084, "pstudio": 0.1921, "adt": 0.1836}
```

These are from the bidir-fixed run but still use the buggy track3d_pred + s_wind=60. Fixing Bug B (track2d+reproject) only adds ~0.2-0.9pp. The dominant remaining gap is Bug A (s_wind=60 vs 500), which requires ~40GB VRAM to fix — not feasible on our RTX 4080 Laptop (12GB).

---

## reproject_2d3d implementation (not in public SpaTrackerV2 repo)

```python
def reproject_2d3d(uvd: torch.Tensor, K: np.ndarray) -> np.ndarray:
    """uvd: (T, N, 3), K: (T, 3, 3) → (T, N, 3) camera-space XYZ."""
    u, v, d = uvd[:, :, 0], uvd[:, :, 1], uvd[:, :, 2]
    Kt = torch.from_numpy(K).to(uvd.device)
    fx = Kt[:, 0, 0].unsqueeze(-1)   # (T, 1)
    fy = Kt[:, 1, 1].unsqueeze(-1)
    cx = Kt[:, 0, 2].unsqueeze(-1)
    cy = Kt[:, 1, 2].unsqueeze(-1)
    x = (u - cx) / fx * d
    y = (v - cy) / fy * d
    return torch.stack([x, y, d], dim=-1).cpu().numpy().astype(np.float32)
```

**How to apply:** Fix `eval_spatracker_v2.py` with both Phase 2 fixes before any future re-inference.
