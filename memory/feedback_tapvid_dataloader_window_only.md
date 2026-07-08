---
name: tapvid-dataloader-decode-window-only
description: In TAPVid-3D dataset loaders, decode only the windowed JPEG frames — never the full clip. Plus keep `persistent_workers=False` so the worker's caching allocator is freed each epoch.
metadata:
  type: feedback
---

When loading TAPVid-3D clips for training, the loader must:

1. **Peek `F` cheaply** with `peek_clip_F(path)` (reads only the
   `images_jpeg_bytes` object-array shape, no JPEG decode).
2. **Decode only the windowed frames** via `load_clip(path, frames=(s, e))`
   — never call `load_clip(path)` without `frames=` in the training hot path.
3. **Set `persistent_workers=False`** in the `DataLoader`.

**Why:** A drivetrack clip at 1280×1920 with 24 frames decodes to
~566 MB in float32. The previous code decoded the whole clip per
`__getitem__`, sliced an 8-frame window, and discarded the rest. With
`num_workers=1, persistent_workers=True, prefetch_factor=2`, PyTorch's
caching allocator inside the worker held the residual buffers across
the entire 30k-step run. RAM climbed slowly until `systemd-oomd`'s
default 60 % memory-pressure threshold tripped at ~step 200; it then
killed our scope plus tmux, gnome-shell, dbus, ibus, gnome-terminal —
exactly the cascade [[tracker-train-in-systemd-scope]] tries to
mitigate. v7's first two launches died this way.

After the fix, an RSS smoke test (40 `__getitem__` calls across
pstudio + drivetrack + adt) stabilises at ~619 MB and stops growing.
The systemd-run scope is still recommended as defence-in-depth, but
the dataloader is the actual root cause.

**How to apply:**
- In any new dataset/__getitem__ on `tapvid3d.py`, route through the
  windowed code path. Do not introduce a "load whole clip then slice"
  shortcut.
- Eval code is allowed to call `load_clip(path)` without `frames=`
  because eval runs one clip at a time, no persistent worker.
- If you bump `num_workers` above 1, profile RSS for 200 iterations
  before committing — multi-worker scenarios were never measured.
