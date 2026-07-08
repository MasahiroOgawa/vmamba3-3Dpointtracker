---
name: cleanup-commit-push between each step in a planned sequence
description: When executing a multi-step plan autonomously, run /cleanup-commit-push (skill) between each step
type: feedback
---

When executing through a pre-approved multi-step plan (Step 0 → 1 →
2 → ...), invoke the **`/cleanup-commit-push` skill between each
step**, not just at the end. Each step's work should land as its own
self-contained commit, pushed to remote.

**Why:**
- Each step is a checkpoint that needs to land cleanly so prior
  reasoning is captured before the next step's work entangles with
  it.
- If a later step reveals a bug in an earlier step, having clean
  per-step commits makes bisection / partial-revert easy.
- Visible progress on the remote between steps.

**How to apply:**
- After completing the work of Step N (code + tests + PLAN.md
  update for that step), invoke the `cleanup-commit-push` skill.
- After cleanup-commit-push lands, immediately start Step N+1
  without re-asking permission (per the
  `feedback_dont_gate_planned_steps` memory).
- Combine: plan → code Step N → tests → PLAN.md update for Step N
  → /cleanup-commit-push → start Step N+1.
