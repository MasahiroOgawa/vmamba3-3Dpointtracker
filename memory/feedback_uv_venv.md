---
name: feedback-uv-venv
description: Always use uv for virtual environments in third-party repos, not python3 -m venv
metadata:
  type: feedback
---

For third-party repos that need isolated environments, set up `uv` instead of `python3 -m venv`.

**Why:** User prefers consistent uv tooling across all projects, including forks of upstream repos.

**How to apply:**
- When setting up a third-party tool (e.g. SpaTrackerV2, TrackCraft3R), use `uv init` + `uv add <deps>` instead of `python3 -m venv .venv` + `.venv/bin/pip install`.
- Fork the upstream repo to `MasahiroOgawa/` on GitHub, create a `feature/uv` branch, add `pyproject.toml` for uv, and push.
- Run inference via `uv run python <script>` instead of `.venv/bin/python <script>`.

See [[feedback-no-pip]] — never invoke pip directly; uv is the universal package manager for this user.
