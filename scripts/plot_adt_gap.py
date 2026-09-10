"""Where the adt gap to DELTA actually is, per clip.

adt is the subset with the largest deficit (-0.0182 metric-AJ). This splits that number into its
two parts -- how close the predicted points are, and whether they are called visible -- because
the two point in opposite directions and the headline metric hides it.
"""

import json
import pathlib
import statistics as st

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OURS = "result/v91/eval_400/metric_results/adt.json"
DELTA = "result/20260715-1443_da3g_delta/metric_results/adt.json"
OUT = pathlib.Path("result/analysis/adt_gap.png")


def load(p: str) -> dict:
    return {c["clip_id"]: c for c in json.loads(pathlib.Path(p).read_text())}


def main() -> int:
    V, D = load(OURS), load(DELTA)
    ids = sorted(set(V) & set(D))
    aj_v = [V[i]["metric_average_jaccard"] for i in ids]
    aj_d = [D[i]["metric_average_jaccard"] for i in ids]
    pt_v = [V[i]["metric_average_pts_within_thresh"] for i in ids]
    pt_d = [D[i]["metric_average_pts_within_thresh"] for i in ids]
    oc_v = [V[i]["occlusion_accuracy"] for i in ids]
    oc_d = [D[i]["occlusion_accuracy"] for i in ids]
    er_v = [V[i]["metric_err_median_m"] for i in ids]
    er_d = [D[i]["metric_err_median_m"] for i in ids]

    fig, ax = plt.subplots(2, 2, figsize=(12, 9))

    def paired(a, d, v, title, xlabel, lower_better=False):
        lo = min(min(d), min(v)) * 0.95
        hi = max(max(d), max(v)) * 1.05
        a.plot([lo, hi], [lo, hi], color="black", lw=1, ls="--")
        a.scatter(d, v, s=34, color="#1b6ca8", edgecolor="black", linewidth=0.5, alpha=0.85)
        wins = sum(1 for x, y in zip(d, v) if (y < x if lower_better else y > x))
        a.set_xlim(lo, hi)
        a.set_ylim(lo, hi)
        a.set_xlabel(f"DELTA  {xlabel}")
        a.set_ylabel(f"v91  {xlabel}")
        side = "below" if lower_better else "above"
        a.set_title(f"{title}\nv91 better on {wins}/{len(d)} clips ({side} the line)", loc="left",
                    fontsize=11)
        a.grid(alpha=0.3)

    paired(ax[0][0], aj_d, aj_v, "metric-AJ  — the headline, and we lose it", "metric-AJ")
    paired(ax[0][1], pt_d, pt_v, "Points within threshold — we WIN this", "pts within thresh")
    paired(ax[1][0], oc_d, oc_v, "Occlusion accuracy — we LOSE this", "occlusion acc.")
    paired(ax[1][1], er_d, er_v, "Median 3D error (m) — we WIN this", "median err (m)",
           lower_better=True)

    for a in ax.flat:
        a.set_aspect("equal", adjustable="box")
    fig.suptitle(
        "adt, 50 clips: our 3D positions are closer than DELTA's, and metric-AJ is still worse.\n"
        f"pts-within-threshold {st.mean(pt_d):.4f} -> {st.mean(pt_v):.4f} (+{st.mean(pt_v)-st.mean(pt_d):.4f}), "
        f"occlusion acc {st.mean(oc_d):.4f} -> {st.mean(oc_v):.4f} ({st.mean(oc_v)-st.mean(oc_d):+.4f}), "
        f"metric-AJ {st.mean(aj_d):.4f} -> {st.mean(aj_v):.4f} ({st.mean(aj_v)-st.mean(aj_d):+.4f})",
        fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT, dpi=130)
    print(f"  wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
