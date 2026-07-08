---
name: Draw architecture diagrams before implementation
description: When proposing changes across multiple unfamiliar files, sketch an ASCII/flowchart diagram of the existing vs. proposed pipeline before writing code
type: feedback
---

Before implementing cross-file changes (plans that touch a model, its backbone,
a head, and an eval driver together), produce a visual flow/process diagram of
(a) the baseline system, (b) the modified system, and (c) a side-by-side
comparison with the changed component highlighted.

**Why:** On 2026-04-20, while approving the SSM-3D vs. DA3 evaluation plan the
user rejected `ExitPlanMode` with: "before implementing, I couldn't understand
the implemented structure. so draw the visually easy to understand process
diagram or flow chart for our mamba 3 attention based 3D reconstruction. and
draw the same diagram for Depth anything 3, and draw comparison of those with
enhance for the changed part." A text-only file-and-function list was not enough
to let them verify the plan.

**How to apply:** For any plan where the comparison or architectural difference
matters (swapping a module, reusing pretrained weights across a changed
backbone, multi-stage pipelines), add ASCII-box diagrams to the plan file —
one per system, plus a side-by-side that visually calls out the change. Place
them early in the plan (before the file list), not as an appendix.
