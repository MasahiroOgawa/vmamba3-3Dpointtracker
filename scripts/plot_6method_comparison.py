"""Generate two grouped-bar figures comparing methods on TAPVid-3D minival.

Figure 1: TAPVid-3D normalised median-AJ (leaderboard metric, scale-invariant)
Figure 2: Absolute metric-AJ (fixed-metre thresholds, real 3D accuracy)

Methods:
  SOTA: SpatialTrackerV2, TAPIP3D+DA3, TAPIP3D+MegaSAM, TrackCraft3R
  Ours: SEA-RAFT+DA3, SEA-RAFT+MegaSAM, bidir, v33–v39

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
    "DELTA\n+DA3",
    # ── Ours ─────────────────────────────────────────────────────────────────
    # The "Ours" group header disambiguates authorship, so labels omit "(Ours)".
    # DA3 is the common depth backbone across v33–v39, so it is dropped; only the
    # distinguishing components (flow front-end, non-DA3 depth, refiner) are shown.
    "SEA-RAFT\n(baseline)",
    "SEA-RAFT\n+MegaSAM",
    "SEA-RAFTbidir\n(baseline)",
    "v33\n(SEARAFT\n+mamba3)",
    "v34\n(SEARAFT\n+MegaSAM\n+mamba3)",
    "v35\n(SEARAFT\n+vmamba3)",
    "v36\n(SEARAFTbidir\n+vmamba3)",
    "v37\n(WAFT)",
    "v38\n(WAFTbidir)",
    "v39\n(WAFT\n+vmamba3)",
]

# Number of SOTA methods listed first; used to draw the group separator.
SOTA_COUNT = 5

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
        "DELTA+DA3":          {"drivetrack": 0.130,  "pstudio": 0.141,  "adt": 0.152},
        # Ours ──────────────────────────────────────────────────────────────────
        "SEA-RAFT+DA3":       {"drivetrack": 0.108,  "pstudio": 0.111,  "adt": 0.132},
        "SEA-RAFT+MegaSAM":   {"drivetrack": 0.103,  "pstudio": 0.135,  "adt": 0.163},
        "SEA-RAFT+DA3 bidir": {"drivetrack": 0.112,  "pstudio": 0.111,  "adt": 0.132},
        "v33":                {"drivetrack": 0.057,  "pstudio": 0.041,  "adt": 0.123},
        "v34":                {"drivetrack": 0.059,  "pstudio": 0.048,  "adt": 0.119},
        "v35":                {"drivetrack": 0.093,  "pstudio": 0.054,  "adt": 0.141},
        "v36":                {"drivetrack": 0.093,  "pstudio": 0.057,  "adt": 0.141},
        "WAFT+DA3":           {"drivetrack": 0.115,  "pstudio": 0.116,  "adt": 0.137},
        "WAFT+DA3 bidir":     {"drivetrack": 0.116,  "pstudio": 0.120,  "adt": 0.141},
        "WAFT+v35":           {"drivetrack": 0.098,  "pstudio": 0.073,  "adt": 0.151},
    }
    data_abs = {
        # SOTA ──────────────────────────────────────────────────────────────────
        "SpatialTrackerV2":   {"drivetrack": 0.008,  "pstudio": 0.192,  "adt": 0.179},
        "TAPIP3D+DA3":        {"drivetrack": 0.006,  "pstudio": 0.131,  "adt": 0.164},
        "TAPIP3D+MegaSAM":    {"drivetrack": 0.000,  "pstudio": 0.000,  "adt": 0.000},
        "TrackCraft3R":       {"drivetrack": 0.002,  "pstudio": 0.025,  "adt": 0.032},
        "DELTA+DA3":          {"drivetrack": 0.006,  "pstudio": 0.199,  "adt": 0.344},
        # Ours ──────────────────────────────────────────────────────────────────
        "SEA-RAFT+DA3":       {"drivetrack": 0.005,  "pstudio": 0.165,  "adt": 0.273},
        "SEA-RAFT+MegaSAM":   {"drivetrack": 0.000,  "pstudio": 0.082,  "adt": 0.209},
        "SEA-RAFT+DA3 bidir": {"drivetrack": 0.005,  "pstudio": 0.165,  "adt": 0.273},
        "v33":                {"drivetrack": 0.082,  "pstudio": 0.186,  "adt": 0.272},
        "v34":                {"drivetrack": 0.001,  "pstudio": 0.092,  "adt": 0.208},
        "v35":                {"drivetrack": 0.130,  "pstudio": 0.274,  "adt": 0.298},
        "v36":                {"drivetrack": 0.130,  "pstudio": 0.249,  "adt": 0.298},
        "WAFT+DA3":           {"drivetrack": 0.005,  "pstudio": 0.181,  "adt": 0.289},
        "WAFT+DA3 bidir":     {"drivetrack": 0.005,  "pstudio": 0.181,  "adt": 0.292},
        "WAFT+v35":           {"drivetrack": 0.141,  "pstudio": 0.279,  "adt": 0.318},
    }
    # fmt: on

    # Map METHODS list → data dicts (same order as METHODS)
    _keys = [
        "SpatialTrackerV2",
        "TAPIP3D+DA3",
        "TAPIP3D+MegaSAM",
        "TrackCraft3R",
        "DELTA+DA3",
        "SEA-RAFT+DA3",
        "SEA-RAFT+MegaSAM",
        "SEA-RAFT+DA3 bidir",
        "v33",
        "v34",
        "v35",
        "v36",
        "WAFT+DA3",
        "WAFT+DA3 bidir",
        "WAFT+v35",
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
    data: dict[int, dict[str, float]],
    title: str,
    ylabel: str,
    fname: str,
    methods: list[str] = METHODS,
    sota_count: int = SOTA_COUNT,
) -> None:
    n_methods = len(methods)
    n_subsets = len(SUBSETS)
    bar_w = 0.22
    group_gap = 0.2
    group_w = n_subsets * bar_w + group_gap
    x_centers = np.arange(n_methods) * group_w

    fig, ax = plt.subplots(figsize=(17, 5.5))

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
    ax.set_xticklabels(methods, fontsize=7.5)
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
    sota_ours_x = boundaries[sota_count - 1]
    ax.axvline(sota_ours_x, color="#777", linewidth=1.5, linestyle="--", zorder=1)
    y_top = ax.get_ylim()[1]
    ax.text(
        (x_centers[0] + x_centers[sota_count - 1]) / 2,
        y_top * 0.97,
        "SOTA",
        ha="center",
        va="top",
        fontsize=9,
        color="#555",
        style="italic",
    )
    ax.text(
        (x_centers[sota_count] + x_centers[-1]) / 2,
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


# ── DA3-g whole-method comparison (methods that consume DA3-g depth) ──────────
METHODS_DA3G = [
    "SpatialTracker\nV2",
    "TAPIP3D\n+DA3-g",
    "DELTA\n+DA3-g",
    "SEA-RAFT\n+DA3-g\n(baseline)",
    "v40\n(WAFT\n+DA3-g)",
    "v41\n(WAFT+DA3-g\n+vmamba3)",
    "v42\n(v41 +pstudio\n×20)",
    "v43\n(v42 +scale\nhead)",
]
SOTA_COUNT_DA3G = 3


def build_da3g_tables() -> tuple[dict, dict]:
    """DA3-g norm/abs AJ, per subset, in METHODS_DA3G order. NaN = TAPIP3D adt OOM."""
    NaN = float("nan")
    # fmt: off
    keyed_norm = [
        {"drivetrack": 0.018, "pstudio": 0.008, "adt": 0.027},  # SpatialTrackerV2
        {"drivetrack": 0.057, "pstudio": 0.011, "adt": NaN},    # TAPIP3D+DA3-g
        {"drivetrack": 0.137, "pstudio": 0.049, "adt": 0.141},  # DELTA+DA3-g
        {"drivetrack": 0.112, "pstudio": 0.036, "adt": 0.120},  # SEA-RAFT+DA3-g
        {"drivetrack": 0.118, "pstudio": 0.039, "adt": 0.123},  # v40 WAFT+DA3-g
        {"drivetrack": 0.110, "pstudio": 0.022, "adt": 0.108},  # v41
        {"drivetrack": 0.118, "pstudio": 0.026, "adt": 0.112},  # v42 (v41 + pstudio x20)
        {"drivetrack": 0.119, "pstudio": 0.035, "adt": 0.123},  # v43 (v42 + scale head)
    ]
    keyed_abs = [
        {"drivetrack": 0.043, "pstudio": 0.184, "adt": 0.170},  # SpatialTrackerV2
        {"drivetrack": 0.110, "pstudio": 0.148, "adt": NaN},    # TAPIP3D+DA3-g
        {"drivetrack": 0.164, "pstudio": 0.194, "adt": 0.324},  # DELTA+DA3-g
        {"drivetrack": 0.136, "pstudio": 0.175, "adt": 0.269},  # SEA-RAFT+DA3-g
        {"drivetrack": 0.146, "pstudio": 0.192, "adt": 0.285},  # v40 WAFT+DA3-g
        {"drivetrack": 0.166, "pstudio": 0.198, "adt": 0.285},  # v41
        {"drivetrack": 0.169, "pstudio": 0.213, "adt": 0.292},  # v42 (v41 + pstudio x20)
        {"drivetrack": 0.171, "pstudio": 0.220, "adt": 0.297},  # v43 (v42 + scale head)
    ]
    # fmt: on
    norm = {i: {s: d.get(s, NaN) for s in SUBSETS} for i, d in enumerate(keyed_norm)}
    ab = {i: {s: d.get(s, NaN) for s in SUBSETS} for i, d in enumerate(keyed_abs)}
    return norm, ab


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

    # DA3-g counterparts (same style, DA3-g methods only)
    norm_g, abs_g = build_da3g_tables()
    make_figure(
        norm_g,
        title="TAPVid-3D Normalised Median-AJ — DA3-g depth (higher = better)",
        ylabel="3D-AJ [%]  (normalised, scale-invariant)",
        fname="fig1_normalized_aj_da3g",
        methods=METHODS_DA3G,
        sota_count=SOTA_COUNT_DA3G,
    )
    make_figure(
        abs_g,
        title="Absolute Metric-AJ — DA3-g depth (fixed-metre thresholds, real 3D accuracy)",
        ylabel="Metric-AJ [%]  (absolute, 1cm–2.56m thresholds)",
        fname="fig2_absolute_aj_da3g",
        methods=METHODS_DA3G,
        sota_count=SOTA_COUNT_DA3G,
    )
    print("\nDone.")


if __name__ == "__main__":
    main()
