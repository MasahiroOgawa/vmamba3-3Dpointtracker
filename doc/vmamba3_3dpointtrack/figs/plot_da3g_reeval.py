"""External-SOTA re-eval on TAPVid-3D minival, drawn as INDEPENDENT DA3-l and DA3-g
figures, each with the normalized (leaderboard) 3D-AJ and the absolute metric-AJ.

Writes da3l_reeval.pdf and da3g_reeval.pdf. Numbers: normalized from tab:median /
the external eval summaries; metric from tab:da3l-sota / tab:da3g-sota.

Run: uv run python doc/vmamba3_3dpointtrack/figs/plot_da3g_reeval.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

methods = ["SEA-RAFT", "WAFT", "DELTA", "SpatialTr.\nV2", "TAPIP3D"]
subsets = ["drivetrack", "pstudio", "adt"]
colors = {"drivetrack": "#4c78a8", "pstudio": "#f58518", "adt": "#54a24b"}
NAN = np.nan  # TAPIP3D+DA3-g adt is OOM-limited

# ---- normalized 3D-AJ (leaderboard, scale-invariant) ----
norm_l = {
    "drivetrack": [0.108, 0.115, 0.130, 0.017, 0.065],
    "pstudio":    [0.111, 0.116, 0.141, 0.012, 0.023],
    "adt":        [0.132, 0.137, 0.152, 0.025, 0.004],
}
norm_g = {
    "drivetrack": [0.112, 0.118, 0.137, 0.018, 0.057],
    "pstudio":    [0.036, 0.039, 0.049, 0.008, 0.011],
    "adt":        [0.120, 0.123, 0.141, 0.027, NAN],
}
# ---- absolute metric-AJ (fixed-metre thresholds) ----
metric_l = {
    "drivetrack": [0.005, 0.005, 0.006, 0.008, 0.006],
    "pstudio":    [0.165, 0.181, 0.199, 0.192, 0.131],
    "adt":        [0.273, 0.289, 0.344, 0.179, 0.164],
}
metric_g = {
    "drivetrack": [0.136, 0.146, 0.164, 0.043, 0.110],
    "pstudio":    [0.175, 0.192, 0.194, 0.184, 0.148],
    "adt":        [0.269, 0.285, 0.324, 0.170, NAN],
}

x = np.arange(len(methods))
w = 0.26


def panel(ax, data, title, ymax):
    for i, s in enumerate(subsets):
        vals = np.array(data[s], dtype=float)
        ax.bar(x + (i - 1) * w, np.nan_to_num(vals), w, label=s, color=colors[s])
        for xi, v in zip(x + (i - 1) * w, vals):
            if np.isnan(v):
                ax.text(xi, 0.004, "OOM", ha="center", va="bottom", fontsize=6,
                        rotation=90, color="0.4")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=8)
    ax.set_title(title, fontsize=10)
    ax.set_ylim(0, ymax)
    ax.grid(axis="y", ls=":", alpha=0.5)


def figure(norm, metric, tag, out):
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4))
    panel(axes[0], norm, f"Normalized 3D-AJ ({tag})", 0.17)
    panel(axes[1], metric, f"Absolute metric-AJ ({tag})", 0.37)
    axes[0].set_ylabel("AJ (higher better)")
    axes[0].legend(fontsize=8, loc="upper right", ncol=1)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")


here = Path(__file__).parent
figure(norm_l, metric_l, "DA3-l", here / "da3l_reeval.pdf")
figure(norm_g, metric_g, "DA3-g", here / "da3g_reeval.pdf")
