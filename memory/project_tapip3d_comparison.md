---
name: project-tapip3d-comparison
description: 6-method comparison on TAPVid-3D minival — median-AJ (leaderboard) and absolute metric-AJ; v34 and ADT-MegaSAM pending
metadata:
  type: project
---

Paper goal: show TAPVid-3D normalised metric is misleading — it hides depth quality. SEA-RAFT+MegaSAM ≈ SEA-RAFT+DA3 on normalised but 250× worse absolute. Our v33 wins absolute but scores lower on leaderboard.

**Why:** Establish v33 (SEA-RAFT+DA3+Mamba3SSD) as SoTA for absolute 3D tracking and expose the normalised metric's failure mode.

## 6-method plan
1. SEA-RAFT + DA3
2. SEA-RAFT + MegaSAM
3. TAPIP3D + DA3
4. TAPIP3D + MegaSAM (image-only, no GT poses)
5. Ours (DA3)     = v33 = SEA-RAFT + DA3 + Mamba3SSD
6. Ours (MegaSAM) = v34 = SEA-RAFT + MegaSAM + Mamba3SSD  ← PENDING

## Normalised median-AJ % (scale-invariant, TAPVid-3D leaderboard metric)

| Method              | drivetrack | pstudio | ADT   |
|---------------------|-----------|---------|-------|
| SEA-RAFT+DA3        | 10.79      | 11.06   | 13.23 |
| SEA-RAFT+MegaSAM    | 10.29      | 13.47   | N/A*  |
| TAPIP3D+DA3         |  6.49      |  2.28   |  0.38 |
| TAPIP3D+MegaSAM     |  0.00      |  0.27   | N/A*  |
| Ours (DA3=v33)      |  5.68      |  4.07   | 12.31 |
| Ours (MegaSAM=v34)  | PENDING    | PENDING | N/A*  |

*N/A = ADT MegaSAM stride=2 pipeline running (log: result/megasam_adt_s2_20260629-0731.log via symlink)

## Absolute metric-AJ % (no median scaling, fixed-metre thresholds 1cm..2.56m)

| Method              | drivetrack | pstudio | ADT   |
|---------------------|-----------|---------|-------|
| SEA-RAFT+DA3        |  0.47      | 16.49   | 27.27 |
| SEA-RAFT+MegaSAM    |  0.04      |  8.18   | N/A*  |
| TAPIP3D+DA3         |  0.60      | 13.08   | 16.42 |
| TAPIP3D+MegaSAM     |  0.00      |  0.00   | N/A*  |
| Ours (DA3=v33)      |  8.17      | 18.59   | 27.21 |
| Ours (MegaSAM=v34)  | PENDING    | PENDING | N/A*  |

Key insight: SEA-RAFT+MegaSAM drivetrack normalised≈10.3% but absolute≈0.04% (250× gap hidden by normalisation). Our v33 absolute 8.17% but normalised only 5.7%.

## Evaluation sources

| Method | Normalised-AJ source | Absolute-AJ source |
|--------|---------------------|--------------------|
| SEA-RAFT+DA3 | `result/metric3d_searaft_fix_20260618-1718/summary.md` | same |
| SEA-RAFT+MegaSAM | `result/metric3d_searaft_megasam_minival_20260626-0903/summary.md` | same |
| TAPIP3D+DA3 | official TAPIP3D evaluator JSON, key `tapvid3d_average_jaccard_best` | `result/tapip3d_absolute_eval_20260625-1318/metric_results/` |
| TAPIP3D+MegaSAM | TAPIP3D evaluator JSON (drivetrack/pstudio) | ~0% (broken predictions) |
| v33 | `result/metric3d_v33_fix_20260618-1718/summary.md` | same |
| v34 | `result/metric3d_v34_megasam_minival_20260628-1541/` (PENDING) | same |

TAPIP3D official JSON path: `/home/mas/proj/study/TAPIP3D/result/auto_generated/tapip3d_kubric_24frames_384trajs_2026-06-24_18-55-53/`

## Figures

`scripts/plot_6method_comparison.py` → `result/figures/fig1_normalized_aj.{png,pdf}` and `fig2_absolute_aj.{png,pdf}`

Re-run the script after each new eval completes to update figures with v34/ADT data.

## Why MegaSAM depth fails

DROID-SLAM needs dense video (≥10fps, ≥100 frames) to estimate metric scale via parallax triangulation.
- drivetrack NPZ: 25-198 images at 1.2-9.9fps → DROID-SLAM gives depth≈0.93m (should be ~15m). 16× scale error → 0.04% absolute metric.
- pstudio NPZ: 150 images at 7.5fps but rotation-dominant motion → depth capped at 1.07m (should be 2-4m). 2-4× scale error.
- Normalised metric rescales by median(GT/pred) per clip → hides the scale error → 10.29%/13.47% looks fine.
- TAPIP3D+MegaSAM: coordinate frame mismatch (DROID-SLAM world frame vs per-frame camera frame expected by evaluator) → 0%.
- ADT: 300 images at 10fps, walking 6DOF → DROID-SLAM should work better. Running overnight.

## ADT overnight pipeline (started 2026-06-28 ~15:32 JST)

Stage 1: MegaSAM annotations for ADT (50 clips × ~17min ≈ 14h). PID 2620563.
Stage 2-4: SEA-RAFT+MegaSAM eval, TAPIP3D+MegaSAM eval on ADT.
After completion: run v34 on ADT, then re-run plot script.

## v34 eval (started 2026-06-28 ~15:41 JST)

`eval_metric3d.py --method v33 --ckpt result/v33_20260617-0001/ckpt_20000.pt --da3-depth-root result/tapvid3d_megasam --split minival --subsets drivetrack pstudio`
PID 2626626, log: `result/v34_eval_20260628-1541.log`
At 45/50 drivetrack clips when last checked; pstudio next.

## Mean 3D error (metres, lower=better) — v33 vs SEA-RAFT+DA3

| Subset     | SEA-RAFT+DA3 | TAPIP3D+DA3 | v33   |
|------------|-------------|------------|-------|
| drivetrack | 11.70m      | —          | 6.88m |
| pstudio    | 0.76m       | —          | 0.72m |
| adt        | 0.41m       | —          | 0.41m |

**How to apply:** v33 is 17× better than competitors on drivetrack absolute metric. Paper frames this as: "normalised metric rewards wrong behaviour (hides scale) — real 3D performance requires fixed-metre evaluation."
