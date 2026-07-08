---
name: don't gate between pre-approved sequential steps
description: When the user has agreed to a multi-step plan (Step 0 → 1 → 2 → ...), proceed through the steps autonomously without asking before each one
type: feedback
---

When the user has agreed on a multi-step plan (e.g., "Steps 0, 1, 2, 3
in this order"), execute through the steps **without asking
permission between each one**. The agreement up front is the green
light for the whole sequence.

**Why:**
- Asking "OK to proceed to Step N?" between every planned step
  duplicates the original yes/no decision and slows progress.
- The user has already committed to the sequence; mid-sequence
  confirmation is friction, not safety.

**How to apply:**
- Before starting a multi-step run, ensure the plan is committed to
  PLAN.md (or another shared doc) so each step's intent is visible.
- During execution, end each step with a brief result + immediately
  start the next step. Don't ask "ready for Step N+1?".
- Exceptions where it IS appropriate to pause:
  - Step result revealed something that invalidates the plan.
  - A truly destructive / non-recoverable action sits in the next
    step (e.g., dropping a database, force-pushing).
  - A genuine unexpected obstacle (kernel doesn't compile, OOM at
    unexpected size, GT data missing).
- Otherwise, just go.
