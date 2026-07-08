---
name: avoid-persistent-cd
description: never use `cd <dir> && cmd` in bash — it persists across calls; use `make -C` or absolute paths instead
metadata:
  type: feedback
---

In Claude Code bash invocations, working directory persists between calls. Using `cd <dir> && cmd` for a one-shot command silently leaves the cwd in `<dir>` for every subsequent command, breaking later invocations that assume the project root.

**Why:** Concrete incident, 2026-05-14 — ran `cd doc/attention && make` to build a LaTeX PDF, then later launched a CIFAR-10 training script with `uv run python scripts/cifar10_compare.py ...` from "the project root." The training crashed with `can't open file '.../doc/attention/scripts/cifar10_compare.py'` because cwd was still `doc/attention/`.

**How to apply:**
- For `make` in a subdir: use `make -C <subdir> [target]`.
- For other tools: pass absolute paths, or specify the subdir as part of the command's own path arguments.
- Only use `cd` if the *user* explicitly requested it for the rest of the session.
- Project's CLAUDE.md already says "maintain your current working directory throughout the session by using absolute paths and avoiding usage of `cd`" — follow that.
