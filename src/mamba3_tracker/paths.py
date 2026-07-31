"""Filesystem conventions shared by the scripts."""

from __future__ import annotations

import os
from pathlib import Path


def cache_dir() -> Path:
    """Per-user directory for caches that must survive between runs.

    Figure scripts cache their inference here so `--replot` can re-render without
    a checkpoint or a GPU. That rules out `mktemp`, and a fixed `/tmp/<name>` would
    collide between users on a shared machine -- hence XDG_CACHE_HOME.
    """
    root = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
    d = root / "vmamba3-3Dpointtracker"
    d.mkdir(parents=True, exist_ok=True)
    return d
