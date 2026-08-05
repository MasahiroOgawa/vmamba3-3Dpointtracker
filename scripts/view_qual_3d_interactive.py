#!/usr/bin/env python3
"""Rotatable 3D view of the qualitative tracks, as a self-contained HTML page.

`render_qual_3d.py` produces the paper's static panels, which fix one camera. This
renders the same cached tracks into a browser view you can orbit, zoom and toggle --
useful for judging whether a prediction actually follows the ground truth in depth,
which a single projection can hide.

Reads only `render_qual_3d.py`'s inference cache, so it needs no checkpoint, no depth
cache and **no GPU**. That matters: the GPU may be busy with a training run, and
`eval.efficiency`'s exclusivity guard means a stray CUDA process can abort someone
else's measurement.

Axis convention matches the static figures and paper Fig. 1(b): +X right, +Y down,
+Z depth away from the camera, with one equal-span cube so a metre reads the same on
every axis (`viz/track3d_axes.py` explains why equal span rather than per-axis
autoscale).

  uv run python scripts/view_qual_3d_interactive.py --subset drivetrack
  uv run python scripts/view_qual_3d_interactive.py --subset drivetrack --with-delta
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mamba3_tracker.paths import cache_dir  # noqa: E402

CACHE_DIR = cache_dir() / "qual_3d"
# Predicted solid, ground truth dashed -- the same pairing the static panels use.
STYLE = {"v39": ("solid", "v39 (ours)"), "delta": ("solid", "DELTA+DA3-l"),
         "gt": ("dash", "ground truth")}


def equal_cube(pts: np.ndarray) -> list[tuple[float, float]]:
    """Ranges giving all three axes the same span, centred on their own midpoints.

    Equal span, not per-axis autoscale: it is what lets an extent along X be compared
    against one along Z by eye. A clip whose extents are anisotropic then fills its
    short axes only partly, which is honest rather than a defect -- drivetrack spans
    ~1.2 m across against ~5.4 m in depth.
    """
    lo, hi = np.nanmin(pts, axis=0), np.nanmax(pts, axis=0)
    half = float(np.max(hi - lo)) / 2.0
    mid = (lo + hi) / 2.0
    return [(float(m - half), float(m + half)) for m in mid]


def add_tracks(fig, xyz, vis, key, colours, width):
    dash, name = STYLE[key]
    for t in range(xyz.shape[0]):
        m = vis[t] > 0
        if m.sum() < 2:
            continue
        p = xyz[t][m]
        fig.add_trace(go.Scatter3d(
            x=p[:, 0], y=p[:, 1], z=p[:, 2], mode="lines",
            line=dict(color=colours[t % len(colours)], width=width, dash=dash),
            name=name, legendgroup=key, showlegend=(t == 0),
            hovertemplate=f"{name} track {t}<br>X %{{x:.2f}} Y %{{y:.2f}} Z %{{z:.2f}} m<extra></extra>",
        ))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subset", default="drivetrack", help="drivetrack | pstudio | adt")
    ap.add_argument("--max-tracks", type=int, default=40,
                    help="Subsample tracks; every track at once is unreadable and slow.")
    ap.add_argument("--with-delta", action="store_true", help="overlay the DELTA baseline too")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    cache = CACHE_DIR / f"{args.subset}.npz"
    if not cache.exists():
        print(f"no cache for '{args.subset}' at {cache}\n"
              f"Run render_qual_3d.py once (with checkpoint and data) to populate it.",
              file=sys.stderr)
        return 1

    # No allow_pickle: this cache holds only arrays and one unicode scalar, and
    # render_qual_3d.py writes+reads it the same way. Pickle would let a tampered
    # cache file execute code.
    d = np.load(cache)
    gt, v39, vis = d["gt"], d["v39"], d["vis"]
    delta = d["delta"] if args.with_delta else None
    n = min(args.max_tracks, gt.shape[0])
    # Prefer the longest-visible tracks: the short ones carry little shape and just
    # add clutter to a view whose purpose is judging trajectory agreement.
    order = np.argsort(-(vis > 0).sum(axis=1))[:n]
    gt, v39, vis = gt[order], v39[order], vis[order]
    if delta is not None:
        delta = delta[order]

    colours = ["#4878CF", "#E06C2B", "#3A9E5C", "#8E44AD", "#C0392B",
               "#16A085", "#D4AC0D", "#7F8C8D"]
    fig = go.Figure()
    add_tracks(fig, gt, vis, "gt", ["#888888"] * 8, 3)
    add_tracks(fig, v39, vis, "v39", colours, 4)
    if delta is not None:
        add_tracks(fig, delta, vis, "delta", ["#B0B0B0"] * 8, 3)

    anchors = np.stack([v39[t, int(d["anchor"][order][t])] for t in range(n)])
    fig.add_trace(go.Scatter3d(
        x=anchors[:, 0], y=anchors[:, 1], z=anchors[:, 2], mode="markers",
        marker=dict(size=4, color="black"), name="anchor frame"))

    finite = np.concatenate([v39.reshape(-1, 3), gt.reshape(-1, 3)])
    finite = finite[np.isfinite(finite).all(axis=1)]
    (xr, yr, zr) = equal_cube(finite)
    fig.update_layout(
        title=f"{args.subset}: v39 vs ground truth &mdash; {str(d['clip_id'])}",
        scene=dict(
            xaxis=dict(title="X (m)", range=list(xr)),
            # +Y points down in the TAPVid-3D camera frame, so reverse it to read
            # like the video rather than flipping the data.
            yaxis=dict(title="Y (m)", range=list(yr)[::-1]),
            zaxis=dict(title="Z depth (m)", range=list(zr)),
            aspectmode="cube",
        ),
        legend=dict(itemsizing="constant"), margin=dict(l=0, r=0, t=40, b=0),
    )

    out = args.out or (Path("result") / f"qual_{args.subset}_v39_3d_interactive.html")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out, include_plotlyjs="cdn")
    print(f"wrote {out}  ({n} longest-visible tracks of {d['gt'].shape[0]})")
    print(f"view with:  xdg-open {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
