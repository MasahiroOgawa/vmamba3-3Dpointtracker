---
name: project_v32_flow_conditioned_tracker
description: v32 FlowConditionedTracker result — learned SSM on top of SEA-RAFT flow HURTS vs the training-free baseline
metadata:
  type: project
---

v32 (`FlowConditionedTracker`, `configs/v32.yaml`) is a tiny (0.406M param) causal
Mamba-3 SSM that takes frozen SEA-RAFT flow vectors + DA3 metric depth as direct
per-point features (no image encoder) and refines the 2D positions from naive
flow-chaining via a zero-init `delta_uv` head, then unprojects through DA3 depth.

**Result (TAPVid-3D minival, 3D-AJ, trained 20k steps ~9.3h):**

| subset | SEA-RAFT (no train) | v32 (trained) | Δ |
|---|---|---|---|
| pstudio | 7.5% | 5.7% | −1.8 |
| drivetrack | 1.6% | 0.4% | −1.2 |
| adt | 6.6% | 4.2% | −2.4 |
| **mean** | **5.2%** | **3.4%** | **−1.8** |

**Negative result: the learned SSM degrades 3D accuracy vs the training-free
SEA-RAFT+DA3 baseline on every subset.** It IS far above the collapsed v31
(0.5%) — `uv_head` gradients stayed live, no zero-motion collapse, drivetrack
motion ratio reached ~92% in training and ~107% at eval.

**Why:** Motion ratios at eval are near-identical to SEA-RAFT (100.6/106.9/117.6%
vs 100.6/106.9/117.5%), so `delta_uv` is tiny in aggregate magnitude yet
consistently *adds* small systematic errors. Likely the nudges push `uv` across
DA3 depth discontinuities (object boundaries), flipping points out of the AJ
within-threshold band — SEA-RAFT chaining lands more reliably on the tracked
surface. The 2D loss term has almost nothing to correct (SEA-RAFT uv is already
near GT 2D), so training mostly perturbs a near-optimal input.

**How to apply:** The training-free SEA-RAFT+DA3 baseline (5.2%) is the best of
our tracking approaches so far. Do NOT layer a learned 2D-refinement SSM on top
without changing the formulation — a residual `delta_uv` on an already-good flow
estimate is the wrong lever. Better next directions: (a) learn depth/uv
*consistency gating* rather than position deltas, (b) train the refinement only
where FB-consistency is low (occlusion/fast-motion), (c) supervise against a
depth-aware 3D loss that penalises crossing depth edges. Artifacts:
`outputs/v32_20260616-0001` (ckpt), `outputs/v32_eval_20260617-0925` (eval +
comparison). See [[project_searaft_flow_baseline]].
