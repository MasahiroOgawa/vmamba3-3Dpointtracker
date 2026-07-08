---
name: tracker-ablation-v6v7v8
description: Mamba-3 tracker ablation pipeline. Three numbered runs (v6/v7/v8); each adds one set of fixes on top of the previous. Implement → train → eval → commit per run; results are for the next paper's ablation table.
metadata:
  type: feedback
---

User decided (2026-05-19) to address the v5 "trivial zero-motion baseline"
collapse via a staged ablation, one variant at a time, with results
recorded per run for the paper.

**Ordering:** v6 → v7 → v8. Do **not** combine multiple fixes per run —
each must be evaluable independently.

| Variant | Builds on | Adds | Where |
|---|---|---|---|
| **v6 = BFA** | v5 (broken-zero-baseline) | (B) Smooth-L1 / Huber + direction-cosine term in position loss; (F) 2D reprojection loss `L_2D = SmoothL1(uv̂ − uv*) / image_size`; (A) scaled magnitude penalty `((‖Δp̂‖ − ‖Δp*‖) / s)²` | `src/mamba3_tracker/train/loss.py`, threading K through to loss in train script |
| **v7 = CD** | v6 | (C) iterative refinement: N refinement passes through the propagator per frame, each producing a Δ-update to Δp; (D) correlation-volume features fed to the propagator (`corr[n, h, w] = cos(f_query_n, frame_t_feature[h, w])`) | `src/mamba3_tracker/model/propagator.py`, `src/mamba3_tracker/model/tracker.py` |
| **v8 = E** | v7 | (E) freeze a pretrained backbone (DINOv2-S/14 or DA3-SMALL DINOv2) instead of the from-scratch `PyramidEncoder` | `src/mamba3_tracker/model/encoder.py`, plus weight loader |

**Per-run workflow (mandatory):**

1. Implement the new pieces in code.
2. Smoke-test (≤200 steps) to confirm forward + loss + backward stay finite.
3. Launch `scripts/train_mamba3_tracker.py --out-dir outputs/runs/mamba3_tracker_v{N}` 30k steps fp32. Keep `--ckpt-every 500` for mid-train render capability.
4. After training completes, render qualitative MP4s (3 clips, one per subset, latest ckpt).
5. Run `scripts/eval_mamba3_tracker.py` on all 3 subsets → 3D-AJ / APD3D / OA JSONs + summary.md.
6. **Use the `/cleanup-commit-push` skill** to land the implementation + the
   eval `summary.md` + the training-curve PNG. Commit subject (the skill's
   first commit-message line):
   `feat(mamba3_tracker): v{N} — <one-line description> (vs v{N-1}: <delta>)`
   Do NOT plain `git commit` for ablation variants — the skill enforces the
   security scan + final diff review.
7. Only then start the next variant.

**Why one-fix-per-run:** for the paper's ablation table to be meaningful,
each variant's contribution must be independently measurable. Combining
all six fixes into one v6 means we can't attribute the eventual win.

**Reference for the v5 failure mode this is correcting:**
`memory/feedback_no_auto_render.md` and the v5 30k-step training-curve
PNG at `outputs/runs/mamba3_tracker_v5/training_curve.png` (best val pos
0.0097 looked clean but the actual predicted Δp magnitude was 4mm vs
GT 41cm — trivial zero-motion baseline that the scale-normalized L1
loss made look like convergence).
