"""Plot TAPIP3D vs SEA-RAFT+DA3 vs v33 comparison bar charts.

Generates:
  result/plots/comparison_median_aj.png  — median-scaled 3D-AJ per subset
  result/plots/comparison_metric_aj.png  — absolute metric-AJ per subset
  result/plots/comparison_err_mean.png   — absolute mean 3D error per subset

Usage:
    uv run python scripts/plot_tapip3d_vs_v33.py \\
        --tapip3d-abs-dir result/YYYYMMDD-HHMM_tapip3d_absolute_eval
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np

matplotlib.rcParams.update(
    {
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.dpi": 150,
    }
)

REPO = Path(__file__).resolve().parent.parent
SUBSETS = ["drivetrack", "pstudio", "adt"]

# Per-subset results from saved eval files (result/metric3d_*/metric_results/*.json)
V33_MEDIAN_AJ = {"drivetrack": 5.68, "pstudio": 4.07, "adt": 12.31}
V33_METRIC_AJ = {"drivetrack": 8.17, "pstudio": 18.59, "adt": 27.21}
V33_ERR_MEAN = {"drivetrack": 6.884, "pstudio": 0.724, "adt": 0.413}

SEARAFT_MEDIAN_AJ = {"drivetrack": 10.79, "pstudio": 11.06, "adt": 13.23}
SEARAFT_METRIC_AJ = {"drivetrack": 0.47, "pstudio": 16.49, "adt": 27.27}
SEARAFT_ERR_MEAN = {"drivetrack": 11.70, "pstudio": 0.759, "adt": 0.410}

# TAPIP3D median-AJ from TAPIP3D's official evaluator output
TAPIP3D_MEDIAN_AJ = {"drivetrack": 6.49, "pstudio": 2.28, "adt": 0.38}


def _load_tapip3d_abs(abs_dir: Path) -> tuple[dict, dict]:
    metric_aj, err_mean = {}, {}
    for sub in SUBSETS:
        p = abs_dir / "metric_results" / f"{sub}.json"
        if not p.exists():
            raise FileNotFoundError(f"Missing {p}")
        clips = json.loads(p.read_text())
        metric_aj[sub] = (
            float(np.mean([c["metric_average_jaccard"] for c in clips])) * 100
        )
        err_mean[sub] = float(np.mean([c["metric_err_mean_m"] for c in clips]))
    return metric_aj, err_mean


def bar_group(ax, groups, series, ylabel, title, pct=True, log_scale=False):
    n_groups = len(groups)
    n_series = len(series)
    w = 0.8 / n_series
    x = np.arange(n_groups)
    colors = ["#4C72B0", "#DD8452", "#55A868"]

    for i, (label, vals) in enumerate(series.items()):
        ys = [vals[g] for g in groups]
        bars = ax.bar(
            x + (i - n_series / 2 + 0.5) * w, ys, w * 0.9, label=label, color=colors[i]
        )
        if not log_scale:
            for bar, y in zip(bars, ys):
                fmt = f"{y:.1f}%" if pct else f"{y:.2f}m"
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    fmt,
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="upper right")
    if log_scale:
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.2f"))
        for container in ax.containers:
            ax.bar_label(container, fmt="%.2fm", padding=3, fontsize=8)
    else:
        ax.set_ylim(0, ax.get_ylim()[1] * 1.2)
    ax.grid(axis="y", alpha=0.35)
    ax.spines[["top", "right"]].set_visible(False)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tapip3d-abs-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=REPO / "result" / "plots")
    args = ap.parse_args()

    tapip3d_mj, tapip3d_em = _load_tapip3d_abs(args.tapip3d_abs_dir)

    print("\n=== TAPIP3D vs SEA-RAFT+DA3 vs v33 ===")
    header = f"{'':30} {'drivetrack':>12} {'pstudio':>12} {'adt':>12} {'mean':>12}"
    print(header)
    for tag, med, mj, em in [
        ("SEA-RAFT+DA3", SEARAFT_MEDIAN_AJ, SEARAFT_METRIC_AJ, SEARAFT_ERR_MEAN),
        ("TAPIP3D", TAPIP3D_MEDIAN_AJ, tapip3d_mj, tapip3d_em),
        ("v33", V33_MEDIAN_AJ, V33_METRIC_AJ, V33_ERR_MEAN),
    ]:
        print(
            f"  {tag} median-AJ:   {med['drivetrack']:>11.2f}% {med['pstudio']:>11.2f}% {med['adt']:>11.2f}% {np.mean(list(med.values())):>11.2f}%"
        )
        print(
            f"  {tag} metric-AJ:   {mj['drivetrack']:>11.2f}% {mj['pstudio']:>11.2f}% {mj['adt']:>11.2f}% {np.mean(list(mj.values())):>11.2f}%"
        )
        print(
            f"  {tag} err_mean:    {em['drivetrack']:>11.3f}m {em['pstudio']:>11.3f}m {em['adt']:>11.3f}m {np.mean(list(em.values())):>11.3f}m"
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Plot 1: Median-scaled 3D-AJ
    fig, ax = plt.subplots(figsize=(8, 5))
    bar_group(
        ax,
        SUBSETS,
        {
            "SEA-RAFT + DA3": SEARAFT_MEDIAN_AJ,
            "TAPIP3D (DA3)": TAPIP3D_MEDIAN_AJ,
            "v33 (RAFT+DA3+Mamba3)": V33_MEDIAN_AJ,
        },
        "Median-scaled 3D-AJ (%)",
        "Median-scaled 3D-AJ (TAPVid-3D minival, 150 clips)",
    )
    fig.tight_layout()
    out1 = args.out_dir / "comparison_median_aj.png"
    fig.savefig(out1)
    plt.close(fig)
    print(f"\n[plot] {out1}")

    # Plot 2: Absolute metric-AJ
    fig, ax = plt.subplots(figsize=(8, 5))
    bar_group(
        ax,
        SUBSETS,
        {
            "SEA-RAFT + DA3": SEARAFT_METRIC_AJ,
            "TAPIP3D (DA3)": tapip3d_mj,
            "v33 (RAFT+DA3+Mamba3)": V33_METRIC_AJ,
        },
        "Absolute Metric-AJ (%)",
        "Absolute Metric-AJ (no median scaling, fixed-metre thresholds)",
    )
    fig.tight_layout()
    out2 = args.out_dir / "comparison_metric_aj.png"
    fig.savefig(out2)
    plt.close(fig)
    print(f"[plot] {out2}")

    # Plot 3: Mean 3D error (log scale for drivetrack outlier)
    fig, ax = plt.subplots(figsize=(8, 5))
    bar_group(
        ax,
        SUBSETS,
        {
            "SEA-RAFT + DA3": SEARAFT_ERR_MEAN,
            "TAPIP3D (DA3)": tapip3d_em,
            "v33 (RAFT+DA3+Mamba3)": V33_ERR_MEAN,
        },
        "Mean 3D Error (m)",
        "Mean Absolute 3D Error over visible GT points",
        pct=False,
        log_scale=True,
    )
    fig.tight_layout()
    out3 = args.out_dir / "comparison_err_mean.png"
    fig.savefig(out3)
    plt.close(fig)
    print(f"[plot] {out3}")


if __name__ == "__main__":
    main()
