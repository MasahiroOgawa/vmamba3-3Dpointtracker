---
name: one-unified-yaml-per-ablation
description: Each ablation run (mamba3_tracker v8, v9, …) keeps every hyperparameter in **one** YAML file at `configs/<run>.yaml`, sectioned by part (`model:`, `data:`, `train:`, `loss:`). Not split across multiple files.
metadata:
  type: feedback
---

When introducing a new ablation run, put every knob — model dims, data
selection, training schedule, loss weights, anything else — into a
single sectioned YAML at `configs/<run>.yaml` (e.g. `configs/v8.yaml`).
**Do not** split into `loss_v8.yaml` / `train_v8.yaml` / etc.

**Why:** Two-fold:

1. The user explicitly asked for it on 2026-05-20 while approving the
   v8 = F plan: *"all the parameter should be in 1 config; yaml file.
   and sectioned by each part, e.g. loss."*
2. One file per run keeps reproducibility tight — the cfg.json snapshot
   written by the train script is a 1-to-1 mirror of the input YAML
   plus any CLI overrides, so a single path / commit hash uniquely
   identifies an ablation row.

**How to apply:**

- Schema: top-level `version:` string + nested sections. For trackers,
  the standard sections are `model:`, `data:`, `train:`, `loss:`.
  Loss weights nest under `loss.weights:`.
- The train script takes one `--config configs/<run>.yaml` and a few
  trivial CLI overrides (`--out-dir`, `--steps`, `--seed`, etc.). All
  other knobs come from the YAML.
- The loader (`src/mamba3_tracker/train/config.py`) asserts the
  `version:` string matches the code version and refuses to start
  otherwise — silent v6→v8 schema drift produces meaningless metrics
  in `loss_history.json`.
- Loss weights are read raw, normalised so `Σ λ_i = 1` after load,
  and BOTH raw + normalised get written into the run's `cfg.json`
  for reproducibility.
- Existing v6/v7 runs used argparse-only configuration. That pattern
  is grandfathered, not extended. New runs use the unified YAML.
