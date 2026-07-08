---
name: stay close to the DA3 paper setup
description: Don't deviate from the DA3 paper's training setup (loss form, hyperparameters) without empirical evidence the deviation pays. Default to original DA3, not "improved" variants.
type: feedback
---

When something in the DA3 training pipeline misbehaves (collapse, OOM, non-convergence), do not preemptively swap in a different loss / parameterization / regularizer (e.g., the §15.59.2 Kendall-Gal log-scale Laplace pivot) to "fix" it. Instead:

1. Find and fix the proximate root cause (the §15.59.1 OOM was actually solvable by the eval-side TSDF guard alone — the loss change was unnecessary).
2. Keep the original DA3 setup as the comparison baseline.
3. Only deviate if the deviation has been *measured* to give better held-out test metrics on the same data, and document the win.

**Why:** Deviations that "should help" but don't measurably help (a) muddy the experimental record (variant vs DA3 comparison rows are no longer apples-to-apples), (b) make the result harder to reproduce against the published paper's number, and (c) leave dead code paths that bit-rot.

**How to apply:** When the `da3_*` training pipeline misbehaves, look for *external* fixes (eval-side guards, recipe tweaks like step count / LR / weight decay, regularization knobs that the DA3 paper itself uses) before changing the loss form or any other paper-defined component. If you do propose a deviation, lead with held-out evidence that the alternative wins, not just theoretical reasoning. The §15.59.2 Kendall-Gal pivot was reverted because it never measurably improved test metrics — the OOM that motivated it was solved by an unrelated change.
