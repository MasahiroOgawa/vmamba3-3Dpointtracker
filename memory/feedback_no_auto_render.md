---
name: no-auto-render-tracking-videos
description: During tracker training, do not auto-render videos on every ckpt save or val event. Only render when the user explicitly asks. If no ckpt exists yet, wait for one.
metadata:
  type: feedback
---

For the Mamba-3 tracker training runs (`outputs/runs/mamba3_tracker_v*/`),
do not automatically render qualitative tracking videos on every checkpoint
save or every validation step. The user finds the constant per-500-step
event stream noisy.

**Why:** During the v5 fp32 training I auto-monitored ckpt saves and offered
renders on each one. The user clarified: "you don't need to create videos
for every 500 steps. but whenever I asked, create video using latest saved
checkpoint."

**How to apply:**
- Default Monitor filter for tracker runs: step *milestones* (e.g. 2000,
  4000, 8000, …) + failure signatures (Traceback, OOM, non-finite). Skip
  the `saved` and frequent `VAL` lines unless the user has asked for them.
- When the user asks "render val tracking video" (or similar): pick the
  newest `ckpt_*.pt` under the active run dir by step number, run
  `scripts/render_tracker_video.py` against it, and report the resulting
  MP4 paths. If no ckpt yet exists, tell the user and wait — do not save
  a one-off ckpt or rerun training to make one appear.
- `--ckpt-every` is the user's choice; don't reduce it just to make
  renders available sooner.
