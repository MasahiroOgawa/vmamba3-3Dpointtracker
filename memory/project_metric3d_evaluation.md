---
name: project_metric3d_evaluation
description: Absolute-metric TAPVid-3D eval reverses the v33 verdict — v33 beats SEA-RAFT+DA3 in real metres; leaderboard median scaling hid it
metadata:
  type: project
---

The TAPVid-3D headline metric is scale-invariant twice over: per clip it (1)
median-rescales predicted depth to GT and (2) uses depth-relative pixel
thresholds. For metric 3D reconstruction (this project's goal) both discard the
quantity we care about. We added an **absolute-metric** evaluation path:
`compute_clip_metrics_absolute` in `src/mamba3_tracker/eval/tapvid3d_eval.py`
calls the official vendored metric with `scaling="none"` +
`use_fixed_metric_threshold=True` (fixed-metre thresholds 1cm..2.56m) →
metric-AJ / metric-APD3D, plus mean/median real 3D error in metres. Runner:
`scripts/eval_metric3d.py` (full minival, `--method searaft|v33`); roll-up:
`scripts/compare_metric3d.py`.

**Full 150-clip minival result.** NOTE: median-AJ below is the CORRECTED value
after the 256-px intrinsics fix (commit a706255). The earlier 0.052/0.032 were an
artifact of passing full-resolution intrinsics (depth-relative thresholds too
tight); absolute metric-AJ and metres are unaffected by the fix.

| | median-AJ (corrected) | metric-AJ | metric-APD3D | err mean/med (m) |
|---|---|---|---|---|
| SEA-RAFT+DA3 | **0.117** | 0.147 | 0.254 | 4.29 / 4.01 |
| v33 | 0.074 | **0.180** | **0.306** | **2.67 / 2.13** |

Corrected median-AJ vs 2024 published baselines (SpatialTracker 0.083,
BootsTAPIR+ZoeDepth 0.080): our training-free SEA-RAFT+DA3 (0.117) EXCEEDS them
(we use 2025 DA3-Metric-Large depth vs their 2024 ZoeDepth), but is far below
current SOTA TAPIP3D (~0.30, 2025) and SpatialTrackerV2 (2025). v33 (0.074) <
SEA-RAFT on median-AJ (depth refiner trades relative structure for absolute scale).

**The verdict reverses under the metric that matters.** v33 loses on the
median-scaled leaderboard metric (3.2% vs 5.2%) but WINS every absolute-metric
measure overall: metric-AJ +22%, metric-APD3D +20%, mean error −38% (median
−47%). The gain is concentrated where DA3's metric scale is most wrong:
drivetrack (far-field ~20m) metric-AJ 0.005→0.082 (16×), error 11.70→6.89 m;
adt (indoor ~1m, DA3 already accurate) is a tie — the learned depth corrector
helps when needed, no harm otherwise. Both run ~12.5 fps, 0 failures.

**Why:** median scaling grants every method a free per-clip global-scale fix, so
it cannot see that DA3's raw metric depth is ~40% scale-biased and that v33
corrects it. v33's raw error ≈ its median-scaled error → its output is already
near the correct absolute scale.

**SOTA comparison — SpatialTracker, drivetrack (provisional).** Google releases
precomputed baseline predictions at
`https://storage.googleapis.com/dm-tapnet/tapvid3d/release_predictions_files/spatracker/drivetrack/<clip>.npz`
(keys `tracks_XYZ (F,N,3)`, `visibility (F,N)`) — **only the spatracker method, only
drivetrack**. Download via the public bucket (object GET works; listing is 401).
Scored with the SAME pipeline (`eval_metric3d.py --method external --pred-dir
~/data/tapvid3d_baseline_preds/spatracker`), all 50 drivetrack minival clips, absolute metric:

| method (drivetrack) | metric-AJ | metric-APD3D | err mean/med (m) |
|---|---|---|---|
| SpatialTracker (SOTA) | 0.068 | 0.106 | 6.84 / 6.03 |
| SEA-RAFT+DA3 (ours) | 0.005 | 0.012 | 11.70 / 11.07 |
| **v33 (ours)** | **0.082** | **0.134** | 6.89 / **5.50** |

Under identical scoring, **v33 beats SpatialTracker on drivetrack** in metric-AJ
(+20%), metric-APD3D (+26%), median error (5.50 vs 6.03 m); ties mean error
(~6.85 m). **VALIDATED** (`scripts/validate_official_metric.py`): the released preds ARE the
real paper-grade SpatialTracker. Reproducing the official `evaluate_model.py`
invocation exactly (raw GT npz, `query_points=queries_xyt[...,::-1]`, `order='t n'`)
on our vendored metric gives drivetrack median 3D-AJ = **0.0584**, matching the
paper's 0.058 — but ONLY after the official intrinsics resize: TAPVid-3D defines
pixel thresholds relative to 256-px images, so intrinsics must be scaled by
`256/min(H,W)` (drivetrack 1280×1920 → ×0.2). Without it we got 0.0076 (~7× low).

Key consequences:
- Our metric code is correct; absolute metric-AJ (fixed-metre thresholds) is
  **independent of the intrinsics resize** (0.0685 either way), so the
  absolute-metric comparison is sound and convention-free.
- **v33 (0.082) genuinely beats the real SpatialTracker (0.0685) in absolute
  metric-AJ on drivetrack** (+20%; metric-APD3D 0.134 vs 0.106).
- FIXED (commit a706255): `eval_metric3d` now resizes intrinsics by
  `256/min(orig_H,orig_W)` for the median metric call (load_clip K/resolution
  verified identical to raw npz). Corrected median-AJ above. Absolute unaffected.

Baseline-prediction availability (probed GCS bucket, listing 401 but object GET
works): **only `spatracker/drivetrack` is downloadable** — no pstudio/adt for any
method, no BootsTAPIR. So old-baseline comparison on other subsets needs running
the models. Current SOTA is **TAPIP3D (~0.30, Apr 2025)** and SpatialTrackerV2
(Jul 2025), NOT SpatialTracker V1 (2024). Plan: user ran **TAPIP3D inference on
TAPVid-3D minival** on another PC; ingest its per-clip preds (`<subset>/<clip>.npz`,
keys tracks_XYZ (F,N,3) cam-frame metric, visibility (F,N), GT query order) via
`eval_metric3d.py --method external`. Validity gates: (1) TAPIP3D must use
MONOCULAR depth not GT/sensor depth, else absolute comparison is unfair; (2)
cross-check our scoring reproduces TAPIP3D's published per-subset minival 3D-AJ
before trusting absolute numbers. Open: final official-invocation cross-check on
our OWN saved predictions (the SEA-RAFT 0.117 > 2024-baselines result is
surprising and should be re-confirmed by saving preds + validate_official_metric).

**How to apply / paper decision gate.** The "v33 is a failure" conclusion in
[[project_v33_depth_refined_tracker]] is metric-dependent and must be qualified:
v33 is the better model for absolute-metric 3D tracking. A paper framing is
viable ("median scaling hides metric-depth quality; a tiny learned depth
corrector gives SoTA-grade absolute-metric tracking") BUT requires a fair
baseline: **no published absolute-metric SOTA numbers exist** (the whole
leaderboard is median-scaled) and most SOTA trackers aren't metric-depth, so we
must run a metric-capable baseline ourselves (e.g. BootsTAPIR+ZoeDepth) in
absolute metric. That run is the agreed next decision gate, now justified since
the v33 metric win is solid at full scale. See
[[project_searaft_flow_baseline]], [[project_v33_depth_refined_tracker]].
