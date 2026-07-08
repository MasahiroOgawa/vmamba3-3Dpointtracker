---
name: never re-run a training that already produced its outputs
description: If a training has already produced its checkpoints/eval logs under a given recipe + seed, do NOT re-run it. Reuse the existing artifacts — even if the orchestrator's default would re-do them.
type: feedback
---

Trainings are expensive (minutes to hours of GPU time). Once a training has produced its checkpoints and eval logs for a given (recipe, seed, scene), those artifacts are deterministic modulo small CUDA non-determinism — re-running them produces near-identical numbers and burns redundant compute.

**Why:** Re-running consumes the user's time and GPU cycles. The user has flagged this twice this session (re-running V1 zero-shot evaluation; re-running V2 under the same recipe). They want this to be a permanent rule.

**How to apply:**

- Before launching an orchestrator that runs many variants, check which ckpts and eval logs already exist on disk. Skip variants that are already done.
- The `train_scene_overfit.py` orchestrator runs `--variants all` by default. Always pass `--variants <comma-list>` to skip variants whose ckpts already exist under the same `(recipe, seed, scene)` in some other output dir.
- For evals, prefer running the `phase4_evaluator` directly on the existing ckpt rather than triggering the orchestrator's eval-of-everything default.
- If you do need to re-run for a clean comparison.md aggregation, copy the eval.log files from the old run dir into the new one rather than re-evaluating.
- Self-contained "all rows in one comparison.md" is a *nice-to-have*, not a reason to burn compute. Aggregate post-hoc from existing eval logs if needed.
- Different recipe ⇒ new training is justified. Same recipe + same seed + same scene ⇒ never.
