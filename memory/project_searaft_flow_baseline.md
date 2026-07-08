---
name: project-searaft-flow-baseline
description: SEA-RAFT optical-flow point tracker — training-free baseline that replaces v31's collapsed Mamba-3 propagator; fixes zero-motion collapse, mean minival 3D-AJ 5.2%.
metadata:
  type: project
---

On 2026-06-15, after v31 (and every learned variant v19–v31) collapsed to the
zero-motion baseline (mean minival 3D-AJ ≈ 0.5%, motion ratio 0.0%), the user
asked to swap the learned Mamba-3 propagator for the **latest 2D optical flow**.
Chosen: **SEA-RAFT** (ECCV 2024, `MemorySlices/Tartan-C-T-TSKH-spring540x960-M`),
added as a git submodule at `third_party/SEA-RAFT` (read-only upstream).

**Pipeline (training-free):** frozen SEA-RAFT dense flow chained frame-to-frame
to propagate each query's `uv` from its anchor (forward + backward sweeps);
forward-backward consistency gives visibility; then the SAME v31 path —
unproject through the frozen DA3-Large depth cache + TAPVid-3D metrics. Code:
`src/searaft_flow/{model.py,flow_tracker.py}`, `scripts/eval_searaft_tracker.py`,
`configs/searaft.yaml`. Reuses `loss._unproject_with_depth` and
`tapvid3d_eval.{compute_clip_metrics,aggregate}` verbatim for apples-to-apples.

**Result (official minival, 3D-AJ %, 12.9 frames/s on one GPU):**

| method | pstudio | drivetrack | aria/adt | mean |
|---|---|---|---|---|
| SEA-RAFT+DA3 (ours) | 7.5 | 1.6 | 6.6 | **5.2** |
| v31 Mamba3+DA3 | 0.5 | 0.2 | 0.7 | 0.5 |
| BootsTAPIR+ZoeDepth* | 10.2 | 5.1 | 8.6 | 8.0 |
| SpatialTracker* | 9.8 | 5.8 | 9.2 | 8.3 |

Zero-motion collapse is **fixed** (motion ratio ~100–118%, not 0%) and AJ is
~11× v31. Still below published TAP+depth baselines; **drivetrack is the weak
spot** (1.6 vs 5–9) — large fast driving motion makes frame-to-frame flow
chaining drift. This is the new strong baseline / ablation floor; see
[[feedback-tracker-ablation-v6v7v8]] and [[feedback-efficiency-and-accuracy-together]].
