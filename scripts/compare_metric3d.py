"""Roll up several eval_metric3d.py runs into one comparison table.

Reads >=2 `metrics.json` produced by scripts/eval_metric3d.py and emits a
comparison.md with, per subset + overall, both the scale-invariant leaderboard
metric and the absolute-metric scores side by side:

    median-AJ | metric-AJ | metric-APD3D | err mean(m) | err median(m)

Usage:
    uv run python scripts/compare_metric3d.py \
        --run result/<dt>_metric3d_searaft --label "SEA-RAFT+DA3" \
        --run result/<dt>_metric3d_v33     --label "v33 (depth refiner)" \
        --out-dir result/<dt>_metric3d_compare
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_COLS = [
    ("average_jaccard", "median-AJ"),
    ("metric_average_jaccard", "metric-AJ"),
    ("metric_average_pts_within_thresh", "metric-APD3D"),
    ("metric_err_mean_m", "err mean(m)"),
    ("metric_err_median_m", "err median(m)"),
]


def _fmt(v, meters):
    if not isinstance(v, (int, float)) or v != v:
        return "—"
    return f"{v:.3f}" if meters else f"{v:.4f}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, action="append", default=[],
                    help="a metric3d out-dir (containing metrics.json); repeatable")
    ap.add_argument("--label", action="append", default=[],
                    help="display label per --run, in order")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    if len(args.run) < 2:
        ap.error("need >=2 --run dirs to compare")
    if args.label and len(args.label) != len(args.run):
        ap.error("--label count must match --run count")

    runs = []
    for i, rd in enumerate(args.run):
        data = json.loads((rd / "metrics.json").read_text())
        label = args.label[i] if args.label else data.get("method", rd.name)
        runs.append((label, data))

    subsets = list(runs[0][1]["per_subset"].keys())
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows = [
        "# Absolute-metric 3D tracking — method comparison (TAPVid-3D minival)",
        "\nmedian-AJ = scale-invariant leaderboard metric. metric-* = absolute "
        "(no median scaling, fixed-metre thresholds). err = real 3D error in metres.\n",
    ]
    for scope in subsets + ["overall"]:
        rows.append(f"\n## {scope}\n")
        rows.append("| method | " + " | ".join(c[1] for c in _COLS) + " |")
        rows.append("|" + "---|" * (len(_COLS) + 1))
        for label, data in runs:
            m = data["overall"] if scope == "overall" else data["per_subset"][scope]
            cells = [_fmt(m.get(k), "m" in lbl) for k, lbl in _COLS]
            rows.append(f"| {label} | " + " | ".join(cells) + " |")

    out = args.out_dir / "comparison.md"
    out.write_text("\n".join(rows) + "\n")
    print("\n".join(rows))
    print(f"\n[compare] wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
