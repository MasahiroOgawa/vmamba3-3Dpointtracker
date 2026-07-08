"""Plot "where we are" — TAPVid-3D 3D-AJ for every minival-comparable run we have,
against paper baselines and (with caveat) current SoTA published on full eval.

Reads:
  - configs/tapvid3d_baselines.yaml
       baselines:       Table 3 minival numbers, directly comparable to our minival eval.
       sota_full_eval:  newer methods (CoTracker3, DELTA) published ONLY on the full
                        eval set — included as reference, drawn with a hatched style
                        and a caveat in the plot so it's visually clear they are
                        not on the same test set.
  - --eval-dir <path>...   each `metric_results/{pstudio,drivetrack,adt}.json`
                           (output of scripts/eval_mamba3_tracker.py --split minival).

  When no --eval-dir is given, auto-discovers every minival-evaluated run on disk:
      result/track_v*_*/eval/metric_results/                   (v21+ official pipeline)
      result/eval_tracker/*_official/metric_results/           (v19/v20 re-evals)

Writes (to --out-dir):
  - comparison.md   markdown table (baselines + ours + SoTA reference)
  - comparison.png  grouped-bar chart: subsets × methods, with our runs in red,
                    baselines in solid colour, SoTA in hatched grey.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml


SUBSETS = ("pstudio", "drivetrack", "aria")
SUBSET_ON_DISK = {"pstudio": "pstudio", "drivetrack": "drivetrack", "aria": "adt"}


def _load_run(eval_dir: Path) -> dict[str, float | None]:
    """{subset: 3D-AJ as fraction in [0, 1]} for one of our runs."""
    out: dict[str, float | None] = {s: None for s in SUBSETS}
    for sub in SUBSETS:
        json_path = eval_dir / "metric_results" / f"{SUBSET_ON_DISK[sub]}.json"
        if not json_path.exists():
            continue
        per_clip = json.loads(json_path.read_text())
        ajs = [
            c["average_jaccard"]
            for c in per_clip
            if isinstance(c.get("average_jaccard"), (int, float))
        ]
        out[sub] = sum(ajs) / len(ajs) if ajs else None
    return out


def _mean(vals: list[float | None]) -> float | None:
    finite = [v for v in vals if isinstance(v, (int, float))]
    return sum(finite) / len(finite) if finite else None


def _fmt(v: float | None) -> str:
    return "—" if v is None else f"{v * 100:.1f}"


def _auto_discover_runs() -> list[tuple[str, Path]]:
    """Find every minival-comparable eval directory on disk.

    Returns sorted list of (label, eval_dir). Sort key extracts the version
    number from the directory name so v22 lands to the right of v19/v20/v21
    in the final plot ("our latest" rightmost).
    """
    found: list[tuple[str, Path]] = []
    outputs = Path("result")
    if outputs.is_dir():
        for d in sorted(outputs.glob("track_v*_*")):
            ed = d / "eval"
            if (ed / "metric_results").is_dir():
                m = re.match(r"track_(v\d+)_", d.name)
                found.append((m.group(1) if m else d.name, ed))
    eval_root = Path("result/eval_tracker")
    if eval_root.is_dir():
        for d in sorted(eval_root.glob("*_official")):
            if (d / "metric_results").is_dir():
                m = re.search(r"(v\d+)", d.name)
                tag = m.group(1) if m else d.name
                if any(t == tag for t, _ in found):
                    continue
                found.append((tag, d))

    def _key(item):
        m = re.match(r"v(\d+)", item[0])
        return int(m.group(1)) if m else 1_000_000

    return sorted(found, key=_key)


def _write_markdown(
    baselines: list[dict],
    ours: list[tuple[str, dict[str, float | None]]],
    sota: list[dict],
    out_path: Path,
) -> None:
    lines = [
        "# TAPVid-3D 3D-AJ — where we are (median scaling, %)",
        "",
        "Per-clip mean 3D Average Jaccard × 100 on the official MINIVAL split.",
        "Baselines (Table 3 minival) are directly comparable. SoTA below is",
        "on the full-eval split (different test set, larger and on-average",
        "easier) — included only for directional reference.",
        "",
        "| method | pstudio | drivetrack | aria (adt) | mean |",
        "|---|---:|---:|---:|---:|",
    ]
    for b in baselines:
        vals = [b.get(s) for s in SUBSETS]
        lines.append(
            f"| {b['name']} | {_fmt(vals[0])} | {_fmt(vals[1])} | "
            f"{_fmt(vals[2])} | {_fmt(_mean(vals))} |"
        )
    for label, m in ours:
        vals = [m.get(s) for s in SUBSETS]
        lines.append(
            f"| **{label}** | **{_fmt(vals[0])}** | **{_fmt(vals[1])}** | "
            f"**{_fmt(vals[2])}** | **{_fmt(_mean(vals))}** |"
        )
    if sota:
        lines.append("")
        lines.append(
            "### Reference (full-eval split — not directly comparable to above)"
        )
        lines.append("")
        lines.append("| method | pstudio | drivetrack | aria (adt) | mean |")
        lines.append("|---|---:|---:|---:|---:|")
        for s in sota:
            vals = [s.get(k) for k in SUBSETS]
            lines.append(
                f"| _{s['name']}_ | {_fmt(vals[0])} | {_fmt(vals[1])} | "
                f"{_fmt(vals[2])} | {_fmt(_mean(vals))} |"
            )
    out_path.write_text("\n".join(lines) + "\n")
    print(f"[compare] wrote {out_path}")


def _plot(
    baselines: list[dict],
    ours: list[tuple[str, dict[str, float | None]]],
    sota: list[dict],
    out_path: Path,
) -> None:
    methods: list[tuple[str, dict[str, float | None], str]] = []
    methods.extend(
        (b["name"], {s: b.get(s) for s in SUBSETS}, "baseline") for b in baselines
    )
    methods.extend((label, m, "ours") for label, m in ours)
    methods.extend((s["name"], {k: s.get(k) for k in SUBSETS}, "sota") for s in sota)

    x_labels = ["pstudio", "drivetrack", "aria", "mean"]
    n_methods = len(methods)
    n_groups = len(x_labels)
    width = 0.8 / max(n_methods, 1)
    fig, ax = plt.subplots(figsize=(max(11, 1.0 * n_methods + 4), 6))
    x = np.arange(n_groups)
    for i, (name, m, kind) in enumerate(methods):
        vals = [m.get(s) for s in SUBSETS] + [_mean([m.get(s) for s in SUBSETS])]
        ys = [0 if v is None else v * 100 for v in vals]
        if kind == "ours":
            color, alpha, hatch, edge = "#d62728", 1.0, None, "black"
        elif kind == "sota":
            color, alpha, hatch, edge = "#9e9e9e", 0.55, "///", "#444"
        else:
            color, alpha, hatch, edge = None, 0.85, None, "none"
        bars = ax.bar(
            x + i * width - 0.4 + width / 2,
            ys,
            width,
            label=name,
            color=color,
            alpha=alpha,
            hatch=hatch,
            edgecolor=edge,
            linewidth=0.6,
        )
        for bar, v in zip(bars, vals):
            if v is not None and v * 100 > 0.2:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.2,
                    f"{v * 100:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_ylabel("3D-AJ (%)")
    ax.set_title(
        "TAPVid-3D 3D-AJ — where we are\n"
        "(solid = paper baselines on minival, red = our runs on minival, "
        "hatched = SoTA on full-eval, NOT same test set)",
        fontsize=10,
    )
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=7, ncol=2, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"[compare] wrote {out_path}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--eval-dir",
        type=Path,
        action="append",
        default=[],
        help="Path to a run's eval dir. Repeat to pass several. "
        "If omitted, auto-discovers every minival-comparable run on disk.",
    )
    ap.add_argument(
        "--baselines",
        type=Path,
        default=Path("configs/tapvid3d_baselines.yaml"),
    )
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument(
        "--label",
        action="append",
        default=None,
        help="Display label per --eval-dir (in order). Default: inferred.",
    )
    ap.add_argument(
        "--no-sota",
        action="store_true",
        help="Hide the full-eval SoTA reference bars from the chart.",
    )
    args = ap.parse_args()

    cfg = yaml.safe_load(args.baselines.read_text())
    baselines = cfg.get("baselines", [])
    sota = [] if args.no_sota else cfg.get("sota_full_eval", [])

    if args.eval_dir:
        if args.label is not None and len(args.label) != len(args.eval_dir):
            ap.error("--label count must match --eval-dir count")
        ours: list[tuple[str, dict[str, float | None]]] = []
        for i, ed in enumerate(args.eval_dir):
            label = args.label[i] if args.label else ed.parent.name
            ours.append((label, _load_run(ed)))
    else:
        discovered = _auto_discover_runs()
        print(f"[compare] auto-discovered {len(discovered)} minival-comparable run(s):")
        for label, ed in discovered:
            print(f"  {label}  ←  {ed}")
        ours = [(label, _load_run(ed)) for label, ed in discovered]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    _write_markdown(baselines, ours, sota, args.out_dir / "comparison.md")
    _plot(baselines, ours, sota, args.out_dir / "comparison.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
