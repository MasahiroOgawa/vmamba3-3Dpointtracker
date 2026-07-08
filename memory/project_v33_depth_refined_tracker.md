---
name: project_v33_depth_refined_tracker
description: v33 Mamba3DepthRefiner result — learned depth-only refinement on top of SEA-RAFT+DA3 also HURTS vs training-free baseline
metadata:
  type: project
---

v33 (`Mamba3DepthRefiner`, `configs/v33.yaml`) is the follow-up to [[project_v32_flow_conditioned_tracker]].
After v32's `delta_uv` hurt, the constraint was "the SSM only treats 3D positions,
never touches the SEA-RAFT 2D track". With `uv` frozen, the only 3D DOF that
preserves the 2D projection is depth along the pixel ray, so a tiny (0.398M param)
causal Mamba-3 SSM refines per-track depth: `z = z_raw * exp(Δlog z)` (Δlog-z head
zero-init → starts at the SEA-RAFT+DA3 baseline). Loss is 3D-position-only
(`TrackingLossV33`, L1 normalised by per-clip anchor depth).

**Result (TAPVid-3D minival, 3D-AJ, trained 20k steps ~9.5h):**

| subset | SEA-RAFT (no train) | v33 (trained) | Δ |
|---|---|---|---|
| pstudio | 7.5% | 2.7% | −4.8 |
| drivetrack | 1.6% | 0.8% | −0.8 |
| adt | 6.6% | 6.1% | −0.5 |
| **mean** | **5.2%** | **3.2%** | **−2.0** |

**Negative result again** — depth-only refinement degrades 3D-AJ, worst on
pstudio (7.5→2.7). The design constraints held PERFECTLY: eval motion ratios
(100.58/106.88/117.53%) and occlusion accuracy (0.689/0.767/0.727, mean 0.728)
are byte-identical to SEA-RAFT, confirming the SSM never touched the 2D track or
the frozen FB-consistency visibility.

**Key diagnosis — train/eval gap + loss/metric misalignment.** The val 3D L1 loss
dropped cleanly (0.21 @ step500 → 0.088 @ step19500), yet minival 3D-AJ went DOWN.
Two compounding causes: (1) 3D-AJ applies global **median scaling** before
thresholding, so it measures *relative* depth structure — the refiner lowered
absolute normalised-L1 on train while distorting the relative structure AJ scores;
(2) the per-point depth corrections overfit the train depth distribution (pstudio
~3m, drivetrack ~20m, adt ~1m) and transfer poorly. Raw DA3 depth at the
SEA-RAFT-tracked `uv` is already a strong signal that small learned corrections
mostly damage.

**How to apply — STOP refining on top of SEA-RAFT+DA3 with an L1-depth objective.**
BOTH learned refinements tried (v32 2D `delta_uv`, v33 depth-only) underperform the
training-free baseline (5.2%), which remains the best tracking approach in this
repo. If pursuing a learned refiner further: (a) train against a **median-scaled /
scale-invariant 3D loss** that matches the AJ metric rather than raw L1; (b) gate
corrections to low-FB-consistency points only (leave confident points at baseline);
(c) hold out a proper val set by subset to catch the depth-distribution overfit.
Otherwise pivot away from refinement entirely. Artifacts:
`outputs/v33_20260617-0001` (ckpt), `outputs/v33_eval_20260618-0818` (eval +
comparison). See [[project_searaft_flow_baseline]], [[project_v32_flow_conditioned_tracker]].
