"""Generate two grouped-bar figures comparing methods on TAPVid-3D minival.

Figure 1: TAPVid-3D normalised median-AJ (leaderboard metric, scale-invariant)
Figure 2: Absolute metric-AJ (fixed-metre thresholds, real 3D accuracy)

Methods:
  SOTA: SpatialTrackerV2, TAPIP3D+DA3, TAPIP3D+MegaSAM, TrackCraft3R
  Ours: SEA-RAFT+DA3, SEA-RAFT+MegaSAM, bidir, v33, v34, v35, v36

Run after each new eval completes to update the figures.
NaN values are shown as hatched empty bars labelled "N/A".
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

REPO = Path(__file__).parent.parent
OUT_DIR = REPO / "result" / "figures"

SUBSETS = ["drivetrack", "pstudio", "adt"]
SUBSET_LABELS = {"drivetrack": "Drivetrack", "pstudio": "Panoptic Studio", "adt": "ADT"}

METHODS = [
    # ── SOTA (not ours) ──────────────────────────────────────────────────────
    "SpatialTracker\nV2",
    "TAPIP3D\n+DA3",
    "TAPIP3D\n+MegaSAM",
    "TrackCraft3R",
    # ── Ours ─────────────────────────────────────────────────────────────────
    "SEA-RAFT+DA3\n(Ours)",
    "SEA-RAFT+MegaSAM\n(Ours)",
    "SEA-RAFT+DA3\nbidir (Ours)",
    "DA3+SSM\n(Ours)",
    "MegaSAM+SSM\n(Ours)",
    "DINOv3+SSM\n(Ours)",
    "DINOv3+SSM\nbidir (Ours)",
]

# Number of SOTA methods listed first; used to draw the group separator.
SOTA_COUNT = 4

SUBSET_COLORS = {
    "drivetrack": "#4878CF",  # steel blue
    "pstudio": "#E06C2B",  # burnt orange
    "adt": "#3A9E5C",  # forest green
}

HATCH_PENDING = "///"



def build_data_tables() -> tuple[dict, dict]:
    """Return (norm_aj, abs_aj) tables: {method_idx: {subset: float_or_nan}}."""

    # All values are hardcoded from the paper tables (tab:median / tab:abs) so that
    # figures can be regenerated without needing the gitignored result directories.
    # Source columns order: drivetrack, pstudio, adt  (matches METHODS order above).

    # fmt: off
    # (3D-AJ norm, metric-AJ abs) per method × subset
    data_norm = {
        # SOTA ──────────────────────────────────────────────────────────────────
        "SpatialTrackerV2":   {"drivetrack": 0.017,  "pstudio": 0.012,  "adt": 0.025},
        "TAPIP3D+DA3":        {"drivetrack": 0.065,  "pstudio": 0.023,  "adt": 0.004},
        "TAPIP3D+MegaSAM":    {"drivetrack": 0.000,  "pstudio": 0.003,  "adt": 0.004},
        "TrackCraft3R":       {"drivetrack": 0.003,  "pstudio": 0.013,  "adt": 0.010},
        # Ours ──────────────────────────────────────────────────────────────────
        "SEA-RAFT+DA3":       {"drivetrack": 0.108,  "pstudio": 0.111,  "adt": 0.132},
        "SEA-RAFT+MegaSAM":   {"drivetrack": 0.103,  "pstudio": 0.135,  "adt": 0.163},
        "SEA-RAFT+DA3 bidir": {"drivetrack": 0.112,  "pstudio": 0.111,  "adt": 0.132},
        "v33":                {"drivetrack": 0.057,  "pstudio": 0.041,  "adt": 0.123},
        "v34":                {"drivetrack": 0.059,  "pstudio": 0.048,  "adt": 0.119},
        "v35":                {"drivetrack": 0.093,  "pstudio": 0.054,  "adt": 0.141},
        "v36":                {"drivetrack": 0.093,  "pstudio": 0.057,  "adt": 0.141},
    }
    data_abs = {
        # SOTA ──────────────────────────────────────────────────────────────────
        "SpatialTrackerV2":   {"drivetrack": 0.008,  "pstudio": 0.192,  "adt": 0.179},
        "TAPIP3D+DA3":        {"drivetrack": 0.006,  "pstudio": 0.131,  "adt": 0.164},
        "TAPIP3D+MegaSAM":    {"drivetrack": 0.000,  "pstudio": 0.000,  "adt": 0.000},
        "TrackCraft3R":       {"drivetrack": 0.002,  "pstudio": 0.025,  "adt": 0.032},
        # Ours ──────────────────────────────────────────────────────────────────
        "SEA-RAFT+DA3":       {"drivetrack": 0.005,  "pstudio": 0.165,  "adt": 0.273},
        "SEA-RAFT+MegaSAM":   {"drivetrack": 0.000,  "pstudio": 0.082,  "adt": 0.209},
        "SEA-RAFT+DA3 bidir": {"drivetrack": 0.005,  "pstudio": 0.165,  "adt": 0.273},
        "v33":                {"drivetrack": 0.082,  "pstudio": 0.186,  "adt": 0.272},
        "v34":                {"drivetrack": 0.001,  "pstudio": 0.092,  "adt": 0.208},
        "v35":                {"drivetrack": 0.130,  "pstudio": 0.274,  "adt": 0.298},
        "v36":                {"drivetrack": 0.130,  "pstudio": 0.249,  "adt": 0.298},
    }
    # fmt: on

    # Map METHODS list → data dicts (same order as METHODS)
    _keys = [
        "SpatialTrackerV2", "TAPIP3D+DA3", "TAPIP3D+MegaSAM", "TrackCraft3R",
        "SEA-RAFT+DA3", "SEA-RAFT+MegaSAM", "SEA-RAFT+DA3 bidir",
        "v33", "v34", "v35", "v36",
    ]
    NaN = float("nan")
    norm_aj: dict[int, dict[str, float]] = {}
    abs_aj: dict[int, dict[str, float]] = {}
    for m_idx, key in enumerate(_keys):
        norm_aj[m_idx] = {s: data_norm[key].get(s, NaN) for s in SUBSETS}
        abs_aj[m_idx] = {s: data_abs[key].get(s, NaN) for s in SUBSETS}

    return norm_aj, abs_aj


def _percent(v: float) -> float:
    return v * 100.0


def make_figure(
    data: dict[int, dict[str, float]], title: str, ylabel: str, fname: str
) -> None:
    n_methods = len(METHODS)
    n_subsets = len(SUBSETS)
    bar_w = 0.22
    group_gap = 0.2
    group_w = n_subsets * bar_w + group_gap
    x_centers = np.arange(n_methods) * group_w

    fig, ax = plt.subplots(figsize=(14, 5))

    offsets = np.linspace(-(n_subsets - 1) / 2, (n_subsets - 1) / 2, n_subsets) * bar_w
    legend_handles = []

    for s_idx, subset in enumerate(SUBSETS):
        color = SUBSET_COLORS[subset]
        xs = x_centers + offsets[s_idx]
        heights = [
            _percent(data[m].get(subset, float("nan"))) for m in range(n_methods)
        ]

        for xi, hi in zip(xs, heights):
            if np.isnan(hi):
                # N/A bar
                ax.bar(
                    xi,
                    0.0,
                    width=bar_w,
                    color="none",
                    edgecolor="#888",
                    linewidth=0.8,
                    hatch=HATCH_PENDING,
                )
                ax.text(
                    xi,
                    0.5,
                    "N/A",
                    ha="center",
                    va="bottom",
                    fontsize=6.5,
                    color="#888",
                    rotation=90,
                )
            else:
                ax.bar(
                    xi,
                    hi,
                    width=bar_w,
                    color=color,
                    alpha=0.85,
                    edgecolor="white",
                    linewidth=0.5,
                )
                label = f"{hi:.2f}" if hi < 1.0 else f"{hi:.1f}"
                if hi < 0.8:  # small bar: rotate label upward to avoid crowding
                    ax.text(
                        xi,
                        hi + 0.15,
                        label,
                        ha="center",
                        va="bottom",
                        fontsize=5.5,
                        color="#444",
                        rotation=90,
                    )
                else:
                    ax.text(
                        xi,
                        hi + 0.3,
                        label,
                        ha="center",
                        va="bottom",
                        fontsize=6.0,
                        color="#222",
                    )

        patch = mpatches.Patch(color=color, alpha=0.85, label=SUBSET_LABELS[subset])
        legend_handles.append(patch)

    ax.set_xticks(x_centers)
    ax.set_xticklabels(METHODS, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11, pad=10)
    ax.legend(handles=legend_handles, fontsize=9, framealpha=0.85)
    ax.set_xlim(x_centers[0] - group_w / 2, x_centers[-1] + group_w / 2)
    ax.set_ylim(0, None)
    ax.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0f}%")
    )
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
    ax.spines[["top", "right"]].set_visible(False)

    # Light separators between all methods
    boundaries = (x_centers[:-1] + x_centers[1:]) / 2
    for xi in boundaries:
        ax.axvline(xi, color="#ddd", linewidth=0.8, zorder=0)

    # Prominent separator + group labels between SOTA and Ours
    sota_ours_x = boundaries[SOTA_COUNT - 1]
    ax.axvline(sota_ours_x, color="#777", linewidth=1.5, linestyle="--", zorder=1)
    y_top = ax.get_ylim()[1]
    ax.text(
        (x_centers[0] + x_centers[SOTA_COUNT - 1]) / 2,
        y_top * 0.97,
        "SOTA",
        ha="center",
        va="top",
        fontsize=9,
        color="#555",
        style="italic",
    )
    ax.text(
        (x_centers[SOTA_COUNT] + x_centers[-1]) / 2,
        y_top * 0.97,
        "Ours",
        ha="center",
        va="top",
        fontsize=9,
        color="#555",
        style="italic",
    )

    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        out = OUT_DIR / f"{fname}.{ext}"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved: {out}")
    plt.close(fig)


def print_table(norm_aj, abs_aj):
    header = f"{'Method':<24}" + "".join(f"{s:>12}" for s in SUBSETS)
    print("\n=== Normalised median-AJ (%) ===")
    print(header)
    for m_idx, name in enumerate(METHODS):
        row = name.replace("\n", " ").ljust(24)
        for s in SUBSETS:
            v = _percent(norm_aj[m_idx].get(s, float("nan")))
            row += f"{'N/A':>12}" if np.isnan(v) else f"{v:>11.2f}%"
        print(row)

    print("\n=== Absolute metric-AJ (%) ===")
    print(header)
    for m_idx, name in enumerate(METHODS):
        row = name.replace("\n", " ").ljust(24)
        for s in SUBSETS:
            v = _percent(abs_aj[m_idx].get(s, float("nan")))
            row += f"{'N/A':>12}" if np.isnan(v) else f"{v:>11.2f}%"
        print(row)


def main():
    norm_aj, abs_aj = build_data_tables()
    print_table(norm_aj, abs_aj)

    make_figure(
        norm_aj,
        title="TAPVid-3D Normalised Median-AJ (leaderboard metric, higher = better)",
        ylabel="3D-AJ [%]  (normalised, scale-invariant)",
        fname="fig1_normalized_aj",
    )
    make_figure(
        abs_aj,
        title="Absolute Metric-AJ (fixed-metre thresholds, real 3D accuracy)",
        ylabel="Metric-AJ [%]  (absolute, 1cm–2.56m thresholds)",
        fname="fig2_absolute_aj",
    )
    print("\nDone.")


if __name__ == "__main__":
    main()
