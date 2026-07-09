---
name: project_external_methods_not_lying
description: The 3 external SOTA methods' near-zero minival scores are GENUINE (hardware/depth limits), not an eval-harness bug — verified
metadata:
  type: project
---

Investigated (2026-07-08) whether SpatialTrackerV2 / TAPIP3D / TrackCraft3R scoring
~1-2% norm 3D-AJ on minival (vs published 18-25%) is a harness bug. **It is not — the
scorer is correct and the methods genuinely underperform on this PC.** Do not "fix" the
scorer for these.

Evidence the scorer/GT are correct:
- TAPVid-3D GT `tracks_XYZ` is **per-frame camera coordinates** (not first-frame): projects
  to the query pixel at 0.00 px on all 150 clips. The old docstring in `data/tapvid3d.py`
  claiming "first-frame camera / fixed pose" was wrong and was corrected.
- Per-frame Umeyama best-fit (upper bound for ANY frame/axis error) does NOT recover
  SpatialTrackerV2's AJ (~0.002) → not a coordinate-frame bug.
- Our own methods (SEA-RAFT+DA3 = 11.7%) and DELTA+DA3 (14.1%, [[project_delta_eval]])
  score fine through the same scorer with the same DA3 depth.

Why each genuinely fails on this hardware:
- **SpatialTrackerV2**: temporal windowing drift; forced to `s_wind=60` by 12 GB VRAM
  (paper uses ~500, needs ~40 GB). Error grows 0.09→0.56 m with distance from query.
- **TAPIP3D+DA3**: aggregates into a world frame that cancels camera motion → needs real
  poses (MegaSAM, slow). Pose-free with DA3 depth it fails on rotation-heavy adt/pstudio.
  **Confirmed by TAPIP3D's OWN official evaluator** on `da3_minival`: adt 0.0038,
  pstudio 0.0228, drivetrack 0.0649 — matches our wrapper (0.004/0.023/0.065) to rounding.
- **TrackCraft3R**: Wan2.1 video-diffusion backbone, OOD on outdoor drivetrack; also far
  too slow (~32 min/clip). [[project_trackcraft3r_eval]]

Diagnostic tool encoding this: `scripts/diagnose_external_pred.py` (query-reproj +
per-frame-Umeyama upper bound + drift-vs-query-distance) — run it before assuming a new
baseline's low score is a bug.
