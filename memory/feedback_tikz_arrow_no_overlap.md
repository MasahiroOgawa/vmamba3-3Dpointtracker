---
name: feedback-tikz-arrow-no-overlap
description: TikZ arrows must never overlap a box interior or run tangential to a box edge; single-point contact only
metadata:
  type: feedback
---

Arrows must contact each box at exactly ONE point (the anchor). Two failure modes to avoid:

1. **Overlap**: the path passes through a box interior. Common cause: `(A.east) |- (B.south)` where A is wide enough that A.east.x > B.center.x — the horizontal goes leftward back through A.

2. **Tangential**: the path runs parallel to and along a box edge (touching along a line, not a point).

**Why:** Both make the figure ugly and hard to read. User rule: "if arrow is overlapped with box, it is very ugly and hard to see." Single-point contact is mandatory.

**How to apply:**
- Before writing any arrow path, check that no segment can enter a box interior.
- For bypass arrows (e.g. skipping a row of boxes), ALWAYS exit from the source box's `.south` or `.north`, route in empty space *below* or *above* all boxes, then approach the target from the clear side. Pattern:
  ```latex
  \coordinate (wa) at ($(src.south)+(0,-3mm)$);
  \draw[arr,rounded corners=2pt] (src.south) -- (wa) -- (wa -| tgt.south) -- (tgt.south);
  ```
- Never use `(A.east) |- (B.south)` without first verifying A.east.x < B.center.x and no box lies in the horizontal/vertical segments' paths.
- If the route must cross the y-level of other boxes, offset horizontally into a clear gap (e.g. 3mm inside a 6mm gap between two boxes).
