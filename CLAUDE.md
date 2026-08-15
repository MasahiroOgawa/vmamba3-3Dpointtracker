# Project Rules — vmamba3-3Dpointtracker

These rules are repo-local. They override any global `~/.claude/CLAUDE.md` behavior for this project.

## Python / packaging

- **Never invoke `pip` directly.** Use `uv` for everything.
  - Add third-party packages: `uv add <pkg>`
  - Add a local editable source: configure `[tool.uv.sources]` in `pyproject.toml` with `path = "..."` + `editable = true`, then `uv add <name>`
  - Run anything in the venv: `uv run python ...`, `uv run pytest`
  - Sync after manual edits to `pyproject.toml`: `uv sync`
- `visionmamba3` is provided by `third_party/visionMamba3` (editable install via `[tool.uv.sources]`).

## third_party layout

All external repos live at `~/proj/study/` (single shared clone) and are symlinked here:

| Symlink | Points to |
|---------|-----------|
| `third_party/depth-anything-3` | `/home/mas/proj/study/depth-anything-3` |
| `third_party/mamba-ssm` | `/home/mas/proj/study/mamba-ssm` |
| `third_party/SEA-RAFT` | `/home/mas/proj/study/SEA-RAFT` |
| `third_party/SpaTrackerV2` | `/home/mas/proj/study/SpaTrackerV2` |
| `third_party/TrackCraft3R` | `/home/mas/proj/study/TrackCraft3R` |
| `third_party/DELTA_densetrack3d` | `/home/mas/proj/study/DELTA_densetrack3d` |
| `third_party/WAFT` | `/home/mas/proj/study/WAFT` |
| `third_party/visionMamba3` | proper git submodule (version-pinned) |

Never edit files inside any `third_party/` entry.

### `third_party/visionMamba3` is a mirror, never a place to work

`~/proj/study/visionMamba3` is the **single, canonical repository** for all Vision-Mamba-3 code.
`third_party/visionMamba3` is a submodule pointer at a commit of it and nothing else. Therefore,
in `third_party/visionMamba3`:

- **Never commit.**
- **Never create a branch.**
- **Never push.**

To change Vision-Mamba-3 code, edit `~/proj/study/visionMamba3`, commit and push there, then in
this repository run `git -C third_party/visionMamba3 fetch && git -C third_party/visionMamba3
checkout <sha>` and commit the moved pointer. The pointer must always name a commit reachable
from `origin/main` of the canonical repo.

If anything original is ever found committed inside `third_party/visionMamba3`, move it to
`~/proj/study/visionMamba3`, push it there, and re-pin.

**Why this is spelled out, when "never edit third_party" already appears above.** On 2026-08-15
the two-pool (VSSD-beta,gamma) operator was implemented correctly in the canonical repo, then
implemented a *second* time as a local branch inside `third_party/visionMamba3` and committed
there. The reasoning was superficially good: the canonical `main` had moved ahead of the pinned
commit, bumping the pin might change the operator numerically, and that would invalidate the
baselines the new arm was being compared against — so patching the pinned commit "avoided the
risk". Two things were wrong with it.

1. It produced a submodule pointer to a commit that existed only on one machine. `git ls-remote`
   matched zero refs; `git clone --recursive` would have failed for anyone else, which is exactly
   the reproducibility property the submodule exists to provide.
2. The risk it was avoiding was never measured. When finally checked, the divergence was additive
   and gated off by default (`rope_angles=False` keeps the projection width and the outputs
   unchanged), and the canonical operator was **bit-identical** to the patched pinned one: the
   same weights gave `max|diff| = 0` and the trained checkpoint loaded with zero missing or
   unexpected keys.

The rule to take from it: if bumping the pin looks risky, **measure the divergence** — build the
layer both ways with the same weights and compare — rather than forking to avoid finding out.
Forking is not the cautious option; it is the one that silently breaks the clone.

After cloning on a new machine:
```bash
git submodule update --init
for name in depth-anything-3 mamba-ssm SEA-RAFT SpaTrackerV2 TrackCraft3R DELTA_densetrack3d WAFT; do
  ln -s /home/mas/proj/study/$name third_party/$name
done
```

`SpaTrackerV2`, `TrackCraft3R`, `DELTA_densetrack3d`, and `WAFT` have their own uv venvs; run their eval scripts from within those directories.

## Sudo

- Never run `sudo` directly. If a command needs sudo, print the exact command and ask the user to run it.

## 3D track figures

- Every plot of world XYZ tracks (`make_teaser_3d.py`, `render_qual_3d.py`, `render_3d_tracks.py`)
  must call `mamba3_tracker.viz.track3d_axes.apply_image_like_view(ax)` so all of them share one
  image-like camera: +X right, +Y down, +Z (depth) 45° up-right. Plot world X/Y/Z on the plot's
  x/y/z directly — never permute axes to fake a viewpoint, or the figures disagree with each other
  and with the video frame. The camera is fixed on purpose; do not expose elev/azim as CLI flags.
- Frame the box with `apply_equal_cube(ax, lims)` from the same module: equal span on all three
  axes plus one shared tick interval. Don't hand-tune a per-figure box aspect or pad a thin axis
  by hand — that made the teaser's height axis a different on-page length than Fig. 13's.
- Keep the inference caches (`--replot`) working: a viewpoint or styling change must be
  re-renderable without a checkpoint, the depth cache, or a GPU.

## Code style

- Prefer concise expressions (comprehensions over for+append loops).
- Do not add comments that restate the code; only add comments when the *why* is non-obvious.

## Memory

- Memory files for this project live at `memory/*.md` under this repo, indexed by `MEMORY.md` at the repo root.
