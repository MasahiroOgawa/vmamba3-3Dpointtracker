---
name: every step must report efficiency AND accuracy together
description: When comparing variants vs DA3 at any pipeline stage, always measure BOTH compute-efficiency (mem/latency/FLOPs) AND task-accuracy (depth |rel_err|, F-score@5cm, pose AUC@30°)
type: feedback
---

For this project's evaluation framing, **never** report
efficiency-only or accuracy-only when comparing a variant against
DA3-SMALL. Each step's PLAN entry must produce both:

- **Efficiency**: peak GPU memory, wall-clock latency, FLOPs at
  multiple input sizes (`scripts/bench_efficiency.py`).
- **Accuracy**: depth `|rel_err|`, F-score@5cm (back-proj and TSDF),
  pose AUC@30° on ETH3D `terrains` (`scripts/eval_ckpt_sweep.py`,
  `scripts/eval_recon_metrics.py`, `scripts/eval_ray_metrics.py`).

**Why:** the project goal is "DA3-quality on mobile via SSD attention."
Efficiency without accuracy doesn't show the swap is functional;
accuracy without efficiency doesn't show the swap is worth doing.
Both numbers go in the same paper claim.

**How to apply:**
- When a PLAN step says "benchmark X" or "evaluate X", treat that as
  a shorthand for "produce the efficiency table AND the accuracy
  table for X".
- Existing eval scripts already implement the accuracy half — reuse
  them, don't write new ones. They take a ckpt path; you may need to
  build a ckpt of the new architecture (e.g., DA3-SMALL warm-started
  into the new backbone) when no trained ckpt exists yet.
- Architecture-only changes (e.g., enabling `alt_start=4` with no
  training) still need an accuracy probe — at minimum via warm-start
  from DA3-SMALL — so the reader knows whether the architecture is
  functionally sane before training.
