"""Generate result figures for doc/vmamba3_3dpointtrack/ from metric3d eval JSONs.

Produces two figures into doc/vmamba3_3dpointtrack/figs/:
  fig_metric_reversal.png  — median-AJ vs absolute metric-AJ, SEA-RAFT+DA3 vs v33,
      per subset + overall. Shows v33 loses the (scale-invariant) leaderboard metric
      but wins the absolute-metric one.
  fig_sota_drivetrack.png  — drivetrack absolute metric-AJ / metric-APD3D for
      SpatialTracker (SOTA) vs SEA-RAFT+DA3 vs v33.

Reads the committed-source eval JSONs under result/ (gitignored), so the PNGs are
written into the doc tree to be committed alongside the .tex.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
FIGS = REPO / "doc" / "vmamba3_3dpointtrack" / "figs"


def _load(p: str) -> dict:
    return json.loads((REPO / p / "metrics.json").read_text())


def fig_metric_reversal(searaft: dict, v33: dict) -> None:
    subs = ["pstudio", "drivetrack", "adt", "overall"]
    labels = ["pstudio", "drivetrack", "adt", "mean"]

    def col(d, key):
        return [(d["overall"] if s == "overall" else d["per_subset"][s])[key] for s in subs]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    x = np.arange(len(subs))
    w = 0.38
    for ax, key, title in [
        (axes[0], "average_jaccard", "Leaderboard metric (median-scaled 3D-AJ)\nscale-invariant"),
        (axes[1], "metric_average_jaccard", "Absolute metric-AJ\n(no scaling, fixed-metre thresholds)"),
    ]:
        sr, vv = col(searaft, key), col(v33, key)
        ax.bar(x - w / 2, sr, w, label="SEA-RAFT+DA3 (baseline)", color="#7f7f7f")
        ax.bar(x + w / 2, vv, w, label="v33 (depth refiner)", color="#1f77b4")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("3D-AJ")
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", alpha=0.3)
        for xi, (a, b) in enumerate(zip(sr, vv)):
            ax.text(xi - w / 2, a + 0.004, f"{a:.3f}", ha="center", va="bottom", fontsize=7)
            ax.text(xi + w / 2, b + 0.004, f"{b:.3f}", ha="center", va="bottom", fontsize=7)
    axes[0].legend(loc="upper left", fontsize=8)
    fig.suptitle("Median scaling hides metric-depth quality: v33 loses the leaderboard metric "
                 "but wins the absolute metric", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = FIGS / "fig_metric_reversal.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[fig] wrote {out}")


def fig_sota_drivetrack(searaft: dict, v33: dict, st: dict) -> None:
    methods = ["SpatialTracker\n(SOTA 2024)", "SEA-RAFT+DA3\n(baseline)", "v33\n(ours)"]
    colors = ["#d62728", "#7f7f7f", "#1f77b4"]
    aj = [st["per_subset"]["drivetrack"]["metric_average_jaccard"],
          searaft["per_subset"]["drivetrack"]["metric_average_jaccard"],
          v33["per_subset"]["drivetrack"]["metric_average_jaccard"]]
    apd = [st["per_subset"]["drivetrack"]["metric_average_pts_within_thresh"],
           searaft["per_subset"]["drivetrack"]["metric_average_pts_within_thresh"],
           v33["per_subset"]["drivetrack"]["metric_average_pts_within_thresh"]]

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
    x = np.arange(len(methods))
    for ax, vals, title in [(axes[0], aj, "absolute metric-AJ"),
                            (axes[1], apd, "absolute metric-APD3D")]:
        ax.bar(x, vals, 0.6, color=colors)
        ax.set_xticks(x)
        ax.set_xticklabels(methods, fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", alpha=0.3)
        for xi, v in enumerate(vals):
            ax.text(xi, v + 0.002, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    fig.suptitle("TAPVid-3D drivetrack (50 clips), absolute metric: v33 > SpatialTracker", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = FIGS / "fig_sota_drivetrack.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[fig] wrote {out}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--searaft", default="result/metric3d_searaft_fix_20260618-1718")
    ap.add_argument("--v33", default="result/metric3d_v33_fix_20260618-1718")
    ap.add_argument("--spatracker", default="result/metric3d_spatracker_drivetrack_fixed")
    args = ap.parse_args()
    FIGS.mkdir(parents=True, exist_ok=True)
    searaft, v33, st = _load(args.searaft), _load(args.v33), _load(args.spatracker)
    fig_metric_reversal(searaft, v33)
    fig_sota_drivetrack(searaft, v33, st)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
