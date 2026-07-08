---
name: tracker-train-in-systemd-scope
description: Wrap long Mamba-3 tracker train runs in `systemd-run --user --scope -p MemoryMax=18G`. systemd-oomd kills tmux + gnome-shell otherwise when RAM pressure tips. Defence-in-depth — the actual root cause is the dataloader; see [[tapvid-dataloader-decode-window-only]].
metadata:
  type: feedback
---

When kicking off any `scripts/train_mamba3_tracker.py` run that will live
more than a few minutes, wrap it in:

```bash
systemd-run --user --scope --quiet \
    -p MemoryMax=18G -p MemorySwapMax=8G -p OOMPolicy=continue \
    bash -c '<train command>'
```

**Why:** `systemd-oomd` uses Pressure Stall Information (PSI), not the
per-scope `MemoryMax` hard limit, to decide when to kill. At its
default 60 % memory-pressure threshold it will pick the heaviest user
cgroup and kill it — together with siblings (tmux, gnome-shell, dbus,
ibus, gnome-terminal) when the system as a whole is under pressure.

v7's first two launches (2026-05-19, before the dataloader fix in
[[tapvid-dataloader-decode-window-only]]) died this way around step
~200, even with the scope at `MemoryMax=22G`. The scope didn't help
because the trigger was global PSI, not the scope's hard limit.

Once the dataloader leak is fixed, the scope is still useful as
defence-in-depth: with `MemoryMax=18G` on a 32 GB box, system overhead
(~4 GB) + buff/cache (~6 GB) + train (≤18 GB) ≈ 28 GB → stays below
the PSI tripping threshold. `OOMPolicy=continue` prevents the scope
failure from cascading to its parent.

**How to apply:**
- Default for new tracker training launches: wrap in the scope at
  `MemoryMax=18G`, not 22G.
- Pair with `--num-workers 1` (or 0 if you want zero worker process).
- Same conceptual pattern as
  `memory/feedback_da3_bench_num_fusion_workers.md`.
- `loginctl enable-linger $USER` is already on (per earlier session);
  no action needed there.
