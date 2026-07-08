---
name: No paper without competitive accuracy
description: User refuses to publish efficiency-only results; the paper must report accuracy ≥ DA3-SMALL (or competitive) with the Mamba-3 swap. Efficiency alone is not enough.
type: feedback
---

The user will not publish a paper that reports only efficiency wins (latency / memory / FLOPs) when accuracy is materially below the baseline. Even if Mamba-3 attention is N× faster than DA3-SMALL, the result is not publishable unless accuracy is **compatible with or better than DA3-SMALL** on the headline metrics (depth `|relative_depth_error|`, F-score@5cm, pose AUC@30°).

**Why:** Stated explicitly on 2026-05-03 in response to a plan that listed "pure efficiency contribution (CIFAR-10 architectural-parity + §15.55 Phase 4b efficiency)" as a fallback when accuracy gates miss: *"I never publish paper without compatible accuracy or better accuracy. otherwise, only efficiency is meaningless."* This explicitly closes the §15.58 "Path 1" fallback path the prior plan considered.

**How to apply:**
- Never propose an "efficiency-only" framing as a final fallback. If accuracy gates miss, propose more accuracy work (longer schedule, different state_dim, different num_heads, hybrid swap, MIMO, etc.) until accuracy is reached or the experiment is honestly abandoned.
- When suggesting countermeasure ladders that include "give up on accuracy and ship efficiency," **remove that branch**.
- When listing acceptance gates, treat accuracy gates as hard, not soft. Efficiency is the *secondary* claim that rides on top of an accuracy result, never the primary claim.
- The acceptable end-states are: (i) accuracy compatible-or-better at materially better compute → publish; (ii) honestly abandon the experiment / pivot to a different problem statement. There is no third option.
