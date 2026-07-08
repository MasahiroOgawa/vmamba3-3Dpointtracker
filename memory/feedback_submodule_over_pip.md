---
name: prefer git submodule over pip for hackable upstream deps
description: For deps where we want to inspect / potentially patch the source (Mamba, DA3, etc.), use git submodule under third_party/ instead of pip install
type: feedback
---

When adding a dependency where transparency or local inspection matters
(e.g., kernel implementations, model architectures we may swap or fork),
prefer **git submodule under `third_party/<name>/`** with sys.path
injection (see `src/mamba3_attn/__init__.py` for the DA3 pattern), rather than
`uv add <pkg>` from PyPI.

**Why:**
- Dependency is visible in the repo (vs. opaque `uv.lock` entry).
- Specific commit pinned and inspectable.
- Source is locally editable for debugging or patching without escaping
  to a fork-and-pip-install dance.
- Matches the existing pattern with `third_party/depth-anything-3`.

**How to apply:**
- For research/architecture deps where we will read or modify the
  source: submodule.
- For utility/standard deps (numpy, torch, open3d, etc.): pip via
  `uv add` is fine.
- Pip-installed pure-utility packages already in `pyproject.toml` need
  not be migrated; this rule applies to new additions.
