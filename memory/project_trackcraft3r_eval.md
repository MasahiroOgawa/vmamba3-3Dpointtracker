---
name: project-trackcraft3r-eval
description: TrackCraft3R TAPVid-3D eval setup — smoke test passed, timing, known fixes applied
metadata:
  type: project
---

TrackCraft3R (CVPR 2025 video diffusion 3D tracker) smoke test completed successfully on 2026-07-03.

**Why:** Paper reports Sim(3)-aligned AJ (not comparable to standard median-scaled normalized AJ). We need to run it through our standard eval pipeline for a fair row in tab:median / tab:abs.

**Setup location:** `~/proj/study/TrackCraft3R` (feature/uv branch), symlinked at `third_party/TrackCraft3R`. Eval script: `scripts/eval_trackcraft3r.py`.

**Checkpoints:**
- Base model: `checkpoints/wan_models/Wan-AI/Wan2.1-T2V-1.3B/` (17 GB total: DiT 5.3 GB, T5 11 GB, VAE 485 MB)
- Fine-tuned: `checkpoints/trackcraft3r/model.safetensors` (17 GB)
- Null context cache: `checkpoints/null_context.pt` (created once, skips T5 thereafter)

**Fixes applied (feature/uv branch):**
1. `_load_checkpoint` streams safetensors in-place (avoids 17 GB dict spike; peak drops from ~24 GB to ~8 GB)
2. `mmap=True` in `load_state_dict_from_bin` (reduces PSI burst from T5 pth loading)
3. T5 null-context encoding always on CPU (avoids GPU VRAM conflict with Ollama)
4. `set_num_interop_threads` guarded with try/except (called twice → RuntimeError)
5. systemd-oomd threshold raised: `/etc/systemd/oomd.conf.d/ml-training.conf` → 90% / 60s

**Run command:**
```
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
uv run python -u scripts/eval_trackcraft3r.py \
  --subsets pstudio adt drivetrack \
  --out-dir /home/mas/data/tapvid3d_baseline_preds/trackcraft3r
```
(run from `~/proj/study/TrackCraft3R`)

**Timing:** ~32 min/clip at 480×832 on RTX 4080 (90 windows × 21 s/window). Full 150-clip eval ≈ 79 hours.

**Why:** Expandable segments (`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`) is required — model uses 9.6 GB of 12 GB VRAM; without it, allocator fragmentation causes CUDA OOM on every forward pass.

**How to apply:** Always set `expandable_segments:True` when running TrackCraft3R inference.
