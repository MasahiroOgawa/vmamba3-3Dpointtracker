"""Plot all v1-v33 + SEA-RAFT+DA3 on TAPVid-3D 3D-AJ (median scaling).

Colour coding:
  - blue/orange/green : paper baselines (solid, full-eval set for v5-v18 group)
  - dark red, hatched : our Mamba-3 runs on FULL EVAL set (v5–v18)
  - red, solid        : our Mamba-3 runs on MINIVAL (v19–v30)
  - teal              : SEA-RAFT+DA3 training-free (minival)
  - purple            : Mamba-3 learned refiner on SEA-RAFT (v32, v33, minival)

Full-eval and minival use DIFFERENT test clips, so they are NOT directly comparable.
The chart separates them visually.

Writes:
  result/comparison_all_v1_v33/comparison.png
  result/comparison_all_v1_v33/comparison.md
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

BASE = Path(__file__).parent.parent
OUTPUTS = BASE / "result"
EVAL_TRACKER = OUTPUTS / "eval_tracker"

SUBSETS = ("pstudio", "drivetrack", "aria")
SUBSET_ON_DISK = {"pstudio": "pstudio", "drivetrack": "drivetrack", "aria": "adt"}


def _load(metric_results_dir: Path) -> dict[str, float | None]:
    out: dict[str, float | None] = {s: None for s in SUBSETS}
    for sub in SUBSETS:
        p = metric_results_dir / f"{SUBSET_ON_DISK[sub]}.json"
        if not p.exists():
            continue
        clips = json.loads(p.read_text())
        ajs = [
            c["average_jaccard"]
            for c in clips
            if isinstance(c.get("average_jaccard"), (int, float))
        ]
        out[sub] = sum(ajs) / len(ajs) if ajs else None
    return out


def _mean(d: dict[str, float | None]) -> float | None:
    vals = [v for v in d.values() if isinstance(v, (int, float))]
    return sum(vals) / len(vals) if vals else None


def _fmt(v: float | None) -> str:
    return "—" if v is None else f"{v * 100:.1f}"


def main() -> None:
    # ── full-eval-set runs (v5-v18, hatched) ──────────────────────────────────
    full_eval_runs: list[tuple[str, dict]] = []
    for label, mr_dir in [
        ("v5", EVAL_TRACKER / "v5_step30k" / "metric_results"),
        ("v6", EVAL_TRACKER / "v6_step4k" / "metric_results"),
        ("v14", EVAL_TRACKER / "v14_step30000" / "metric_results"),
        ("v15", EVAL_TRACKER / "v15_step30000" / "metric_results"),
        ("v16", EVAL_TRACKER / "v16_step30000" / "metric_results"),
        ("v17", EVAL_TRACKER / "v17_step30000" / "metric_results"),
        ("v18", EVAL_TRACKER / "v18_step30000" / "metric_results"),
    ]:
        if mr_dir.exists():
            full_eval_runs.append((label, _load(mr_dir)))

    # ── minival runs (v19-v30, solid red) ────────────────────────────────────
    minival_mamba3: list[tuple[str, dict]] = []
    for label, mr_dir in [
        ("v19", EVAL_TRACKER / "v19_20260526-1610_official" / "metric_results"),
        ("v20", EVAL_TRACKER / "v20_20260527-0913_official" / "metric_results"),
        ("v21", OUTPUTS / "track_v21_20260528-1202" / "eval" / "metric_results"),
        ("v22", OUTPUTS / "track_v22_20260528-1721" / "eval" / "metric_results"),
        ("v23", OUTPUTS / "track_v23_20260528-1922" / "eval" / "metric_results"),
        ("v25", OUTPUTS / "track_v25_20260529-1214" / "eval" / "metric_results"),
        ("v26", OUTPUTS / "track_v26_20260529-1808" / "eval" / "metric_results"),
        ("v28", OUTPUTS / "track_v28_20260531-1802" / "eval" / "metric_results"),
        ("v29b", OUTPUTS / "track_v29b_20260602-1149" / "eval" / "metric_results"),
        ("v30", OUTPUTS / "track_v30_20260602-1900" / "eval" / "metric_results"),
    ]:
        if mr_dir.exists():
            minival_mamba3.append((label, _load(mr_dir)))

    # ── SEA-RAFT+DA3 training-free (minival, teal) ────────────────────────────
    searaft_dir = OUTPUTS / "searaft_20260615-1908" / "metric_results"
    searaft = _load(searaft_dir) if searaft_dir.exists() else None

    # ── v32/v33 learned refiners (minival, purple) ────────────────────────────
    refiner_runs: list[tuple[str, dict]] = []
    for label, mr_dir in [
        ("v32", OUTPUTS / "v32_eval_20260617-0925" / "metric_results"),
        ("v33", OUTPUTS / "v33_eval_20260618-0818" / "metric_results"),
    ]:
        if mr_dir.exists():
            refiner_runs.append((label, _load(mr_dir)))

    # ── Minival paper baselines ────────────────────────────────────────────────
    baselines_cfg = BASE / "configs" / "tapvid3d_baselines.yaml"
    cfg = yaml.safe_load(baselines_cfg.read_text())
    minival_baselines = cfg.get("baselines", [])  # BootsTAPIR, SpatialTracker
    fulleval_sota = cfg.get("sota_full_eval", [])  # CoTracker3, DELTA (full-eval)

    # ── assemble methods list ─────────────────────────────────────────────────
    # Each entry: (label, {subset: float|None}, kind)
    # kind → controls colour + hatch
    methods: list[tuple[str, dict, str]] = []

    # Minival baselines
    for b in minival_baselines:
        methods.append((b["name"], {s: b.get(s) for s in SUBSETS}, "baseline_minival"))

    # Full-eval sota reference (hatched grey, same as existing chart)
    for s in fulleval_sota:
        methods.append((s["name"], {k: s.get(k) for k in SUBSETS}, "sota_fulleval"))

    # Separator: full-eval versions
    for label, m in full_eval_runs:
        methods.append((label, m, "mamba3_fulleval"))

    # Minival Mamba-3 runs
    for label, m in minival_mamba3:
        methods.append((label, m, "mamba3_minival"))

    # SEA-RAFT+DA3
    if searaft:
        methods.append(("SEA-RAFT\n+DA3", searaft, "searaft"))

    # v32/v33 refiners
    for label, m in refiner_runs:
        methods.append((label, m, "refiner"))

    # ── colour map ───────────────────────────────────────────────────────────
    KIND_STYLE = {
        "baseline_minival": dict(
            color=None, alpha=0.85, hatch=None, edge="none", lw=0.6
        ),
        "sota_fulleval": dict(
            color="#9e9e9e", alpha=0.55, hatch="///", edge="#444", lw=0.6
        ),
        "mamba3_fulleval": dict(
            color="#d62728", alpha=0.50, hatch="\\\\", edge="black", lw=0.6
        ),
        "mamba3_minival": dict(
            color="#d62728", alpha=1.0, hatch=None, edge="black", lw=0.6
        ),
        "searaft": dict(color="#17becf", alpha=1.0, hatch=None, edge="black", lw=0.8),
        "refiner": dict(color="#9467bd", alpha=1.0, hatch=None, edge="black", lw=0.8),
    }

    x_labels = ["pstudio", "drivetrack", "aria", "mean"]
    n_groups = len(x_labels)
    n_methods = len(methods)
    width = 0.8 / max(n_methods, 1)
    fig_w = max(18, 0.55 * n_methods + 6)
    fig, ax = plt.subplots(figsize=(fig_w, 6))
    x = np.arange(n_groups)

    for i, (name, m, kind) in enumerate(methods):
        vals = [m.get(s) for s in SUBSETS] + [_mean(m)]
        ys = [0 if v is None else v * 100 for v in vals]
        st = KIND_STYLE[kind]
        bars = ax.bar(
            x + i * width - 0.4 + width / 2,
            ys,
            width,
            label=name,
            color=st["color"],
            alpha=st["alpha"],
            hatch=st["hatch"],
            edgecolor=st["edge"],
            linewidth=st["lw"],
        )
        for bar, v in zip(bars, vals):
            if v is not None and v * 100 >= 0.5:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.15,
                    f"{v * 100:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=6,
                    rotation=90 if n_methods > 20 else 0,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=11)
    ax.set_ylabel("3D-AJ (%)", fontsize=11)
    ax.set_title(
        "TAPVid-3D 3D-AJ — all versions v5–v33 + SEA-RAFT+DA3 (median scaling)\n"
        "solid red = Mamba-3 on minival  |  hatched red = Mamba-3 on full-eval (diff. test set)  |"
        "  teal = SEA-RAFT+DA3 (training-free)  |  purple = Mamba-3 refiner on SEA-RAFT",
        fontsize=9,
    )
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=7, ncol=3, loc="upper right")
    fig.tight_layout()

    out_dir = OUTPUTS / "comparison_all_v1_v33"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / "comparison.png"
    fig.savefig(out_png, dpi=96, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {out_png}")

    # ── markdown table ────────────────────────────────────────────────────────
    lines = [
        "# TAPVid-3D 3D-AJ — all versions v5–v33 (median scaling, %)",
        "",
        "| method | split | pstudio | drivetrack | aria | mean |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for b in minival_baselines:
        vals = [b.get(s) for s in SUBSETS]
        lines.append(
            f"| {b['name']} | minival | {_fmt(vals[0])} | {_fmt(vals[1])} | {_fmt(vals[2])} | {_fmt(_mean({s: v for s, v in zip(SUBSETS, vals)}))} |"
        )
    for label, m in full_eval_runs:
        vals = [m.get(s) for s in SUBSETS]
        lines.append(
            f"| **{label}** | full-eval | {_fmt(vals[0])} | {_fmt(vals[1])} | {_fmt(vals[2])} | {_fmt(_mean(m))} |"
        )
    for label, m in minival_mamba3:
        vals = [m.get(s) for s in SUBSETS]
        lines.append(
            f"| **{label}** | minival | {_fmt(vals[0])} | {_fmt(vals[1])} | {_fmt(vals[2])} | {_fmt(_mean(m))} |"
        )
    if searaft:
        vals = [searaft.get(s) for s in SUBSETS]
        lines.append(
            f"| **SEA-RAFT+DA3** | minival | {_fmt(vals[0])} | {_fmt(vals[1])} | {_fmt(vals[2])} | {_fmt(_mean(searaft))} |"
        )
    for label, m in refiner_runs:
        vals = [m.get(s) for s in SUBSETS]
        lines.append(
            f"| **{label}** | minival | {_fmt(vals[0])} | {_fmt(vals[1])} | {_fmt(vals[2])} | {_fmt(_mean(m))} |"
        )

    (out_dir / "comparison.md").write_text("\n".join(lines) + "\n")
    print(f"[plot] wrote {out_dir / 'comparison.md'}")


if __name__ == "__main__":
    main()
