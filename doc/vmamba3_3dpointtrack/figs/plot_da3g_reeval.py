"""Grouped-bar comparison of external-SOTA absolute metric-AJ under DA3-l vs DA3-g.

Two panels: far-field drivetrack (where the DA3-l scale bias bites) and the mean.
Numbers are from the DA3-l tables and the DA3-g re-eval (\S DA3-g re-evaluation).

Run: uv run python doc/vmamba3_3dpointtrack/figs/plot_da3g_reeval.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

methods = ["SEA-RAFT", "WAFT\n(v40)", "DELTA", "SpatialTr.\nV2", "TAPIP3D"]

# absolute metric-AJ
drive_l = [0.005, 0.005, 0.006, 0.008, 0.006]
drive_g = [0.136, 0.146, 0.164, 0.043, 0.110]
mean_l = [0.147, 0.159, 0.183, 0.126, 0.100]
mean_g = [0.193, 0.208, 0.227, 0.132, 0.129]  # TAPIP3D DA3-g mean = drivetrack+pstudio only

x = np.arange(len(methods))
w = 0.38
c_l, c_g = "#9e9e9e", "#2f6fb0"

fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4))
for ax, (yl, yg, title) in zip(
    axes, [(drive_l, drive_g, "drivetrack (far-field)"), (mean_l, mean_g, "mean")]
):
    ax.bar(x - w / 2, yl, w, label="DA3-l", color=c_l)
    ax.bar(x + w / 2, yg, w, label="DA3-g", color=c_g)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=8)
    ax.set_title(f"absolute metric-AJ — {title}", fontsize=10)
    ax.set_ylabel("metric-AJ")
    ax.grid(axis="y", ls=":", alpha=0.5)
    for xi, (a, b) in enumerate(zip(yl, yg)):
        ax.text(xi - w / 2, a + 0.004, f"{a:.3f}", ha="center", va="bottom", fontsize=6)
        ax.text(xi + w / 2, b + 0.004, f"{b:.3f}", ha="center", va="bottom", fontsize=6)
axes[0].legend(fontsize=9, loc="upper right")
fig.tight_layout()
out = Path(__file__).with_name("da3g_reeval.pdf")
fig.savefig(out, bbox_inches="tight")
print(f"wrote {out}")
