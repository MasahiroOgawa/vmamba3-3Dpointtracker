# Plan: re-evaluate SOTA baselines with DA3-g (nested-giant depth)

## Naming (precise)

- **DA3-l** = `da3metric-large` (0.35 B), the OLD depth cache `~/data/tapvid3d_da3`.
  Scale-biased on drivetrack far-field (~0.55×).
- **DA3-g** = `DA3NESTED-GIANT-LARGE` (1.4 B), the NEW depth cache `~/data/tapvid3d_da3nested`.
  ~1.0× metric. (CC BY-NC — research only.)

Rename every ambiguous "old DA3"/"DA3" that means the large model to **DA3-l**, and the
nested-giant to **DA3-g**, across **code, docs, and README**.

## What to re-evaluate

Every method whose 3D lift samples a DA3 depth cache was run with **DA3-l**. Re-run each
with **DA3-g** and present a **separate DA3-g table + graph** beside the DA3-l one. The
DA3-g minival cache is already complete, so this runs now (no wait for the full precompute).

**In scope**
- **External SOTA baselines** (each in its own venv; DA3 depth fed in via `DA3_ROOT`):
  - **SpaTrackerV2** — `scripts/eval_spatracker_v2.py` (SpaTrackerV2 venv)
  - **TrackCraft3R** — `scripts/eval_trackcraft3r.py` (TrackCraft3R venv). **SKIPPED for DA3-g**
    (31 min/clip ≈ 3 GPU-days for 150 clips; lowest-performing baseline, DA3-l 0.020).
    Revisit AFTER v41 is fully done — ask the user then whether to re-run (resumes via
    skip-existing; 1 pstudio clip already cached).
  - **DELTA** — `scripts/eval_delta.py` (DELTA venv)
  - **TAPIP3D** — `scripts/eval_tapip3d_absolute.py` (TAPIP3D venv; depth via `eval/*_da3_minival`
    dataset configs — swap mechanism under investigation)
- **Training-free (no-refiner) baselines** (our harness, `eval_metric3d.py`):
  - **SEA-RAFT + DA3-g** — `--method searaft` (running now)
  - **WAFT + DA3-g** — = **v40**, already done (abs-AJ 0.208)

**Out of scope** (explicitly excluded per user): the **DA3-l-trained refiners** v33/v35/v39.
Feeding them DA3-g zero-shot only shows the over-correction collapse; the meaningful
DA3-g refiner is the retrained **v41** (separate work, after the full precompute).

## The swap mechanism

- `eval_spatracker_v2.py`, `eval_trackcraft3r.py`, `eval_delta.py`: `DA3_ROOT` is now
  env-overridable — `DA3_ROOT=~/data/tapvid3d_da3nested <venv-python> scripts/eval_<m>.py …`.
  Default stays DA3-l for backward compat. Output to `<method>_da3g/`.
- `eval_metric3d.py` (SEA-RAFT baseline): `--da3-depth-root ~/data/tapvid3d_da3nested`.
- TAPIP3D: TBD from the config investigation (env / config override, no third_party edits).
- Score each external run with `eval_metric3d.py --method external --pred-dir <…_da3g>`.

## GPU strategy

Single 12 GB GPU, ~4.5 GB free while the precompute runs. The no-refiner baselines are
light and run alongside the precompute. The external SOTA models (SpaTrackerV2, TrackCraft3R)
are heavy and likely need the GPU to themselves — measure one first; if it OOMs, either
briefly pause the (resumable) precompute to run the external sweep with the full card, or
queue them after the precompute. Decide with the user based on the measured footprint.

## Deliverables

- Separate **DA3-g results table** (mirrors DA3-l `tab:median`/`tab:abs`) with per-subset
  breakdown, in the memo `doc/vmamba3_3dpointtrack/vmamba3_3dpointtrack.tex`.
- Separate **DA3-l vs DA3-g comparison graph** (same methods) via the existing plot tool.
- Keep the DA3-l tables untouched for side-by-side comparison.
