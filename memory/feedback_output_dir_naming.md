---
name: output-dir-naming-flat-track-version-datetime
description: Output dirs use datetime-first naming — `result/<YYYYMMDD-HHMM>_<name>/` — so ls and shell history show newest runs first.
metadata:
  type: feedback
---

## Convention

Datetime goes **first** in every result directory name so that `ls` and shell
history sort chronologically at a glance:

```
result/<YYYYMMDD-HHMM>_<name>/      ← datetime FIRST, always
```

e.g. `result/20260702-1739_metric3d_v36_bidir/`

Each run gets **one** top-level directory. **Child directories inside it do
NOT repeat the datetime** — the parent already disambiguates the run.

```
result/<YYYYMMDD-HHMM>_track_v<version>/    ← datetime ONLY here
    ├─ cfg.json                         resolved config + launch args snapshot
    ├─ train.log                        stdout of the training process
    ├─ loss_history.json                per-step + per-val loss rows
    ├─ motion_history.json              v13+: [{step, pstudio_ratio, drivetrack_ratio, ...}, ...]
    ├─ ckpt_500.pt, ckpt_1000.pt, ...   training checkpoints
    ├─ ckpt_<final>.pt
    ├─ eval/                            TAPVid-3D eval results (per-subset JSONs)
    │     ├─ pstudio.json
    │     └─ drivetrack.json
    ├─ viz_ckpt<step>/                  rendered MP4s, named by source ckpt step
    │     ├─ pstudio_basketball_1.mp4
    │     └─ drivetrack_<clip_id>.mp4
    └─ plots/                           training-curve, motion-ratio plots
          ├─ training_curve.png
          └─ motion_ratio.png
```

`<version>` = `v13`, `v14`, etc. `<YYYYMMDD-HHMM>` is set at first launch and
appears EXACTLY ONCE in the path. Child dirs use descriptive single-purpose
names (`viz_ckpt30k`, `viz_step12500`, `eval`, `plots`). If a render is
re-run on the same checkpoint, manually remove or rename the existing
`viz_ckpt<step>/` first; we do not bake datetimes into child paths to avoid
the collision.

## Why one dir

- Chronological order: `ls result/` shows newest runs last; trivial to find the latest.
- Resume: pass the same `--out-dir result/<datetime>_track_v13` and the train
  script auto-resumes from the latest `ckpt_*.pt` inside.
- Co-located artifacts: easy diff across runs (`diff -r outputs/track_v13_… outputs/track_v14_…`).
- One commit message references one path.
- The user explicitly requested this on 2026-05-22 over the previous split
  (`outputs/runs/...` vs `outputs/eval_tracker/...`).

## Anti-patterns (do NOT do)

- `outputs/eval_tracker/v13_<…>/` — split eval/render away from training.
- `outputs/runs/mamba3_tracker_v13/` — buried under a fixed `mamba3_tracker_*`
  prefix that we used for v6–v12. New runs go straight under `outputs/`.
- `result/track_v13_<datetime>/` (old style) — datetime last, ruins chronological `ls`.
- `result/track_v13/` without a datetime — risks silent overwrite on re-launches.
- `outputs/track_v13_<datetime>/runs/`, `outputs/track_v13_<datetime>/eval/v2/`,
  or any deeper version nesting. One dir per run, period.
- **`outputs/track_v16_<datetime>/viz_step12500_<datetime>/`** — datetime
  appearing twice in the path. Child dirs use descriptive names only
  (`viz_step12500`, `viz_ckpt30k`, `eval`, `plots`); never a second timestamp.
  Rule added 2026-05-23 after the redundant timestamp pattern made
  paths unreadable in shell history.

## Per-checkpoint outputs (viz + single-model eval)

Scripts that take `--ckpt` write their outputs **inside the checkpoint's own
result directory**, not as a new top-level entry under `result/`. The `--out-dir`
argument is optional and auto-derived when omitted:

| Script | Default sub-directory |
|--------|----------------------|
| `render_tracks.py` | `viz_ckpt<step>_<style>/` |
| `render_3d_tracks.py` | `viz3d_ckpt<step>/` |
| `render_space_time_tracks.py` | `viz_st_ckpt<step>/` |
| `render_tracker_video.py` | `viz_ckpt<step>/` |
| `eval_mamba3_tracker.py` | `eval_ckpt<step>/` |
| `eval_flow_conditioned_tracker.py` | `eval_ckpt<step>/` |

Example after running with `--ckpt result/v35_20260701/ckpt_20000.pt`:
```
result/v35_20260701/
    ckpt_20000.pt
    viz_ckpt20000_tapvid/    ← render_tracks --style tapvid
    viz_ckpt20000_d4rt/      ← render_tracks --style d4rt
    viz3d_ckpt20000/         ← render_3d_tracks
    eval_ckpt20000/          ← eval_mamba3_tracker
```

## Multi-method / cross-run evaluation

Scripts that compare multiple training results (e.g. `eval_metric3d.py`,
shell pipelines) write to `result/eval/YYYYMMDD-HHMM_<name>/`:

```bash
--out-dir result/eval/$(date +%Y%m%d-%H%M)_metric3d_v35
LOG_DIR="$REPO/result/eval/${TS}_megasam_eval"
```

## Scope

- Applies to **v13 and later**.
- v6–v12 stay where they are: `outputs/runs/mamba3_tracker_v<6..12>/` and
  `outputs/eval_tracker/...`. No retroactive move.
- The launch command must include the new path explicitly:
  ```bash
  systemd-run --user --scope --quiet -p MemoryMax=18G ... \
      uv run python scripts/train_mamba3_tracker.py \
      --config configs/v13.yaml \
      --out-dir result/$(date +%Y%m%d-%H%M)_track_v13
  ```
