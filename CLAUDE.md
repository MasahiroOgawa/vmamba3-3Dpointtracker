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

## Code style

- Prefer concise expressions (comprehensions over for+append loops).
- Do not add comments that restate the code; only add comments when the *why* is non-obvious.

## Memory

- Memory files for this project live at `memory/*.md` under this repo, indexed by `MEMORY.md` at the repo root.
