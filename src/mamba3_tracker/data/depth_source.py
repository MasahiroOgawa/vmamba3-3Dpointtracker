"""One place that decides which DA3 depth is being used, and one loader that reads it.

WHY THIS EXISTS. The DA3-l / DA3-g choice was made in four different ways -- a DA3_ROOT env var, a
--da3-depth-root CLI flag, a data.da3_depth_root config key, and hardcoded paths -- with
load_da3_depth copy-pasted into six scripts and fourteen scripts silently defaulting to DA3-l. That
produced the same class of bug three times:

  1. v73-v85 loaded a DA3-l-trained scale refiner and fed it DA3-g depth (already ruled out in
     doc/plan_da3g_reeval.md for v33/v35/v39, and reintroduced anyway).
  2. eval_waft.py hardcoded DA3-l, so every WAFT track set carries DA3-l depth with nothing saying so.
  3. train_scale_standalone.py read DA3-g maps but computed its target from a WAFT prediction whose
     z came from DA3-l, training the model to emit the DA3-l correction. Measured on minival: it
     predicted +0.586 where the truth was -0.927, and applying it scored 0.1435 against 0.208 for
     no correction at all.

Each was invisible because nothing records which depth a derived artifact was built from. So this
module also provides stamp()/verify() for prediction directories: the source is written next to the
data and checked when it is consumed.

USE:
    from mamba3_tracker.data.depth_source import resolve
    src = resolve("da3g")            # or "da3l", or a path (identified against the registry)
    depth = src.load(subset, clip_name, n_frames)
    print(src)                       # DA3-g (DA3NESTED-GIANT-LARGE) at ~/data/tapvid3d_da3nested
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# The registry. Names are what configs and CLIs should say; paths are an implementation detail.
_REGISTRY: dict[str, dict] = {
    "da3l": {
        "root": Path("~/data/tapvid3d_da3").expanduser(),
        "model": "da3metric-large (0.35B)",
        "note": "scale-biased on drivetrack far-field (~0.55x)",
    },
    "da3g": {
        "root": Path("~/data/tapvid3d_da3nested").expanduser(),
        "model": "DA3NESTED-GIANT-LARGE (1.4B)",
        "note": "~1.0x metric; CC BY-NC, research only",
    },
}
ALIASES = {
    "da3-l": "da3l",
    "da3_l": "da3l",
    "large": "da3l",
    "tapvid3d_da3": "da3l",
    "da3-g": "da3g",
    "da3_g": "da3g",
    "nested": "da3g",
    "tapvid3d_da3nested": "da3g",
}


@dataclass(frozen=True)
class DepthSource:
    name: str  # canonical: "da3l" or "da3g"
    root: Path
    model: str
    note: str

    @property
    def label(self) -> str:
        return "DA3-l" if self.name == "da3l" else "DA3-g"

    def __str__(self) -> str:
        return f"{self.label} ({self.model}) at {self.root}"

    def path(self, subset: str, clip_name: str) -> Path:
        stem = clip_name if clip_name.endswith(".npz") else clip_name + ".npz"
        return self.root / subset / stem

    def load(
        self, subset: str, clip_name: str, n_frames: int | None = None
    ) -> np.ndarray:
        """(F, Hd, Wd) metric depth. n_frames=None means all frames."""
        with np.load(self.path(subset, clip_name)) as d:
            q = d["depth_q"].astype(np.float32)
            if n_frames is not None:
                q = q[:n_frames]
            lo, hi = float(d["d_min"]), float(d["d_max"])
        return lo + q * ((hi - lo) / 65535.0)

    def stamp(self, out_dir: Path, **extra) -> None:
        """Record which depth produced the artifacts in out_dir, beside the artifacts."""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        p = out_dir / "depth_source.json"
        payload = {
            "depth_source": self.name,
            "label": self.label,
            "model": self.model,
            "root": str(self.root),
            **extra,
        }
        p.write_text(json.dumps(payload, indent=1, sort_keys=True))

    def verify(self, artifact_dir: Path, *, strict: bool = True) -> str:
        """Check an artifact directory was built with THIS source. Returns a status string."""
        p = Path(artifact_dir) / "depth_source.json"
        if not p.exists():
            msg = (
                f"{artifact_dir} has no depth_source.json, so which DA3 depth built it is "
                f"UNKNOWN. This is how the scale-refiner target bug hid: waft_full_eval was "
                f"DA3-l and nothing said so. Re-stamp it or regenerate it."
            )
            if strict:
                raise SystemExit(f"[depth_source] {msg}")
            return f"UNVERIFIED: {msg}"
        got = json.loads(p.read_text()).get("depth_source")
        if got != self.name:
            msg = (
                f"{artifact_dir} was built with {got!r} but this run uses {self.name!r}. Mixing "
                f"them is the bug that invalidated v73-v85 and the scale refiner's training target."
            )
            # strict=False means "report, do not block", and that must cover the MISMATCH case too,
            # not only a missing stamp: a track set's uv is invariant to the depth used to build it
            # (verified, max|d uv| = 2.3e-13), so a da3l track set is a legitimate uv source for a
            # da3g run. Only a consumer that uses the artifact's z should refuse.
            if strict:
                raise SystemExit(f"[depth_source] {msg} Use matching artifacts or regenerate.")
            return f"MISMATCH {got} vs {self.name} -- ok if only uv is used"
        return f"verified {self.name}"


def resolve(spec: str | Path | None, *, default: str | None = None) -> DepthSource:
    """Accept a canonical name, an alias, or a path. Paths are identified against the registry."""
    if spec is None:
        if default is None:
            raise SystemExit(
                "[depth_source] no depth source given and no default allowed here. Pass 'da3l' or "
                "'da3g' explicitly -- 14 scripts used to default to DA3-l silently, which is how "
                "DA3-l artifacts kept leaking into DA3-g runs."
            )
        spec = default
    s = str(spec).strip()
    key = ALIASES.get(s.lower(), s.lower())
    if key in _REGISTRY:
        e = _REGISTRY[key]
        return DepthSource(key, e["root"], e["model"], e["note"])
    # a path: identify it rather than trusting it
    p = Path(s).expanduser().resolve()
    for k, e in _REGISTRY.items():
        if p == e["root"].resolve():
            return DepthSource(k, e["root"], e["model"], e["note"])
    raise SystemExit(
        f"[depth_source] {spec!r} is not a known depth source. Known: {sorted(_REGISTRY)} "
        f"({', '.join(str(e['root']) for e in _REGISTRY.values())}). Add it to the registry in "
        f"src/mamba3_tracker/data/depth_source.py rather than passing a bare path."
    )


def known() -> list[str]:
    return sorted(_REGISTRY)
