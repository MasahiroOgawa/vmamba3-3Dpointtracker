---
name: da3-bench-num-fusion-workers-1
description: Always pass `inference.num_fusion_workers=1` to DA3's evaluator; default parallel fusion OOMs and cascades to tmux death
metadata:
  type: feedback
---

When running `python -m depth_anything_3.bench.evaluator` (directly or via
`scripts/run_da3_bench_eval.py`) on this 32 GB RAM machine, always:

1. Pass `inference.num_fusion_workers=1`, AND
2. Wrap the invocation in a memory-capped systemd scope:
   `systemd-run --user --scope -p MemoryMax=22G -p MemorySwapMax=8G -p OOMPolicy=continue -- <cmd>`

**Why:** Even with `num_fusion_workers=1`, a single TSDF fusion on a
100-image ETH3D scene reached 28.7 GB anon-RSS / 72 GB VM (process 1547695,
2026-05-16 15:12). The kernel global-OOM fired, killing the bench and
marking `user@1000.service` as having lost a process; systemd then timed
out the user scope's stop and SIGKILLed everything in it — including the
tmux server. Confirmed twice (13:18 and 15:12 events in journalctl).

`num_fusion_workers=1` is necessary but not sufficient. The systemd scope
makes OOM cgroup-local so it kills only the bench, doesn't escalate to
`user@1000.service`, and tmux survives. `loginctl enable-linger $USER` is
also enabled so the user service persists across logouts.

**How to apply:** Every invocation of `run_da3_bench_eval.py` /
`depth_anything_3.bench.evaluator` includes the flag. The shell wrapper
`scripts/eval_vssd_full_pipeline.sh` should also force it. When iterating
on the bench in a fresh shell, default to per-dataset runs
(`eval.datasets=[one]`) with `eval.eval_only=true` if inference is already
cached — cheaper and avoids re-OOMing.
