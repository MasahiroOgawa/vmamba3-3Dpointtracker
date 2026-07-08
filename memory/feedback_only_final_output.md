---
name: For long-running tasks, only report the final result
description: Don't narrate every intermediate step (per-step loss prints, ckpt saves, per-eval MEAN lines) during a long training/eval run. Wait silently and report once when the run completes or a real failure occurs.
type: feedback
---

When a training or evaluation runs for hours, do not respond on every monitor event with a one-line "step N test=X, continuing". The user sees each response as a notification, so per-step play-by-play becomes spam.

**Why:** The user said it directly: "you don't need to notify every step intermediate result. instead, only notify me the final output comes." (during the §15.59.9 lr=1e-5 run, after ~20 per-step acknowledgements between step 100 and step 1900).

**How to apply:**
- Configure monitors so only *final/actionable* events fire: `DONE`, `summary.md written`, `Traceback`, `MemoryError`, `Killed`, hard failures. Don't fire on every `step XXX/YYYY` or per-eval `MEAN` line.
- When a per-step event does land, do not echo it back as a chat message. Stay silent.
- Respond when: (a) the whole run completes and final results exist, (b) a real failure happens and the user needs to decide, (c) the user asks for a status update. Otherwise silent.
- For multi-stage runs (train → eval → summary), still only respond at the end of each user-visible deliverable, not at each ckpt save.
- This applies to /loop dynamic mode too: schedule the next wake far enough out that the natural next response is a real deliverable, not "still running".
