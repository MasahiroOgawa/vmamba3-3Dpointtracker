"""Unified YAML config loader for mamba3_tracker ablation runs.

See `memory/feedback_one_unified_yaml_per_ablation.md` for the rule:
every knob lives in one sectioned YAML at `configs/<run>.yaml`.
Sections: `model`, `data`, `train`, `loss`. Top-level `version:` string
is asserted against the supported set so silent v6→v8 schema drift
can't quietly produce meaningless metrics.

Loss-weight semantics:
  * Read raw weights from `loss.weights`.
  * Assert `Σ raw_λ_i > 0`.
  * Normalise so `Σ λ_i = 1`. Both raw and normalised are kept on the
    returned cfg so they end up in `cfg.json` for reproducibility.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

SUPPORTED_VERSIONS = ("v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31", "v32", "v33", "v35")


def _deep_merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    out = {**base}
    for k, v in overrides.items():
        if v is None:
            continue
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _normalise_loss_weights(raw: dict[str, float]) -> dict[str, float]:
    total = sum(float(v) for v in raw.values())
    if total <= 0:
        raise ValueError(
            f"loss.weights must have a positive sum, got {total} from {raw}"
        )
    return {k: float(v) / total for k, v in raw.items()}


def load_config(
    path: str | Path,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Load a unified ablation YAML, merge CLI overrides, normalise weights.

    Args:
        path: path to e.g. `configs/v8.yaml`.
        overrides: nested dict of CLI overrides (e.g.
            `{"train": {"steps": 50, "batch": 1}, "data": {"subsets": ...}}`)
            that take precedence over the YAML.

    Returns the resolved cfg with three guaranteed keys:
        - `version` (str, asserted ∈ SUPPORTED_VERSIONS)
        - `loss.weights_raw`   (dict[str, float]) — what the user wrote
        - `loss.weights`       (dict[str, float]) — same keys, sum to 1
    plus the original `model`, `data`, `train`, `loss` sections.
    """
    path = Path(path).expanduser().resolve()
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: top-level must be a mapping")

    cfg = _deep_merge(raw, overrides or {})

    version = cfg.get("version")
    if version not in SUPPORTED_VERSIONS:
        raise ValueError(
            f"{path}: version={version!r} not in supported set {SUPPORTED_VERSIONS}"
        )

    loss = cfg.setdefault("loss", {})
    weights_raw = dict(loss.get("weights", {}))
    if not weights_raw:
        raise ValueError(f"{path}: loss.weights is empty")
    loss["weights_raw"] = weights_raw
    loss["weights"] = _normalise_loss_weights(weights_raw)
    return cfg


def dump_resolved(cfg: dict[str, Any], out_path: str | Path) -> None:
    """Write resolved cfg as JSON snapshot next to the checkpoints."""
    import json
    out_path = Path(out_path)
    out_path.write_text(json.dumps(cfg, indent=2, sort_keys=True))
