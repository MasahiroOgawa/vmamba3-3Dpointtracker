"""Render point-tracking videos and interactive visualizations.

**TAPVid style** (`render_tracking_video`):
  Filled/hollow circles (visible/occluded) with a fading line tail.  Output: MP4.

**D4RT style** (`render_tracking_d4rt_html`):
  Interactive animated 3D visualization: vivid rainbow colors, alpha-fading 3D dot
  trails, play/pause controls and a frame slider, fully rotatable/zoomable 3D scene.
  Open the output .html in any browser.  Matches the D4RT paper aesthetic.
  https://deepmind.google/blog/d4rt-teaching-ai-to-see-the-world-in-four-dimensions/
  Output: self-contained HTML via Plotly.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def _colormap(n: int) -> np.ndarray:
    """`n` BGR uint8 colors from a hash-like distribution."""
    rng = np.random.default_rng(0)
    hsv = np.zeros((n, 1, 3), dtype=np.uint8)
    hsv[:, 0, 0] = (rng.integers(0, 180, size=n)).astype(np.uint8)
    hsv[:, 0, 1] = 200
    hsv[:, 0, 2] = 230
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[:, 0]


def _colormap_vivid(n: int) -> np.ndarray:
    """`n` evenly-spaced vivid rainbow colors (max saturation + value) → BGR."""
    hue = np.linspace(0, 179, n, endpoint=False).astype(np.uint8)
    hsv = np.zeros((n, 1, 3), dtype=np.uint8)
    hsv[:, 0, 0] = hue
    hsv[:, 0, 1] = 255
    hsv[:, 0, 2] = 255
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[:, 0]


def project_to_uv(xyz: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Pinhole projection: (..., 3) world XYZ → (..., 2) pixel uv.

    Tracks are in *camera coordinates of the first frame* (TAPVid-3D
    convention §3.1), so a static camera at the canonical pose is assumed.
    """
    Z = np.clip(xyz[..., 2:3], 1e-6, None)
    uv = (xyz[..., :2] / Z) * np.array([K[0, 0], K[1, 1]]) + np.array(
        [K[0, 2], K[1, 2]]
    )
    return uv


def render_tracking_video(
    frames_rgb_uint8: np.ndarray,  # (F, H, W, 3)
    pred_tracks_NT3: np.ndarray,  # (N, F, 3)
    pred_visibility_NT: np.ndarray,  # (N, F) in [0, 1]
    intrinsics_K: np.ndarray,  # (3, 3)
    out_path: str | Path,
    fps: int = 30,
    tail_len: int = 8,
    radius: int = 4,
    vis_thresh: float = 0.5,
) -> Path:
    """Write an MP4 video with overlaid predicted point tracks."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    F, H, W, _ = frames_rgb_uint8.shape
    N = pred_tracks_NT3.shape[0]
    uv = project_to_uv(pred_tracks_NT3, intrinsics_K)  # (N, F, 2)

    colors = _colormap(N)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(str(out_path), fourcc, fps, (W, H))
    if not vw.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter at {out_path}")

    for t in range(F):
        frame_bgr = cv2.cvtColor(frames_rgb_uint8[t], cv2.COLOR_RGB2BGR).copy()

        # Trajectory tails (oldest faded, newest opaque)
        for n in range(N):
            for k in range(1, min(tail_len, t) + 1):
                p0 = uv[n, t - k]
                p1 = uv[n, t - k + 1]
                if not (np.isfinite(p0).all() and np.isfinite(p1).all()):
                    continue
                p0 = tuple(int(round(v)) for v in p0)
                p1 = tuple(int(round(v)) for v in p1)
                if (
                    min(p0 + p1) < -50
                    or max(p0[0], p1[0]) > W + 50
                    or max(p0[1], p1[1]) > H + 50
                ):
                    continue
                alpha = 1.0 - (k - 1) / max(1, tail_len)
                bgr = tuple(int(c * alpha) for c in colors[n].tolist())
                cv2.line(frame_bgr, p0, p1, bgr, thickness=1, lineType=cv2.LINE_AA)

        # Current-frame markers
        for n in range(N):
            p = uv[n, t]
            if not np.isfinite(p).all():
                continue
            cx, cy = (int(round(p[0])), int(round(p[1])))
            if not (0 <= cx < W and 0 <= cy < H):
                continue
            color = tuple(int(c) for c in colors[n].tolist())
            if pred_visibility_NT[n, t] >= vis_thresh:
                cv2.circle(
                    frame_bgr,
                    (cx, cy),
                    radius,
                    color,
                    thickness=-1,
                    lineType=cv2.LINE_AA,
                )
            else:
                cv2.circle(
                    frame_bgr,
                    (cx, cy),
                    radius,
                    color,
                    thickness=1,
                    lineType=cv2.LINE_AA,
                )

        vw.write(frame_bgr)
    vw.release()
    return out_path


def render_tracking_video_d4rt(
    frames_rgb_uint8: np.ndarray,  # (F, H, W, 3)
    pred_tracks_NT3: np.ndarray,  # (N, F, 3)
    pred_visibility_NT: np.ndarray,  # (N, F) in [0, 1]
    intrinsics_K: np.ndarray,  # (3, 3)
    out_path: str | Path,
    fps: int = 15,
    tail_len: int = 16,
    head_radius: int = 5,
    vis_thresh: float = 0.5,
) -> Path:
    """D4RT-style: vivid rainbow colors, alpha-fading dot trails, no connecting lines.

    Each track has a unique high-saturation hue. The current frame shows a
    bright filled dot with a white sparkle; the past `tail_len` positions are
    dots with decreasing opacity and radius. Occluded points are omitted.
    Alpha-blending per tail step uses cv2.addWeighted.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    F, H, W, _ = frames_rgb_uint8.shape
    N = pred_tracks_NT3.shape[0]
    uv = project_to_uv(pred_tracks_NT3, intrinsics_K)  # (N, F, 2)
    colors = _colormap_vivid(N)  # (N, 3) BGR

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(str(out_path), fourcc, fps, (W, H))
    if not vw.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter at {out_path}")

    for t in range(F):
        canvas = cv2.cvtColor(frames_rgb_uint8[t], cv2.COLOR_RGB2BGR).copy()

        # Tail: draw oldest steps first so newer dots paint over older ones
        for k in range(tail_len, 0, -1):
            t_k = t - k
            if t_k < 0:
                continue
            alpha = (1.0 - k / (tail_len + 1)) ** 1.5
            r = max(1, int(round(head_radius * (1.0 - 0.6 * k / tail_len))))
            overlay = canvas.copy()
            drew = False
            for n in range(N):
                if pred_visibility_NT[n, t_k] < vis_thresh:
                    continue
                p = uv[n, t_k]
                if not np.isfinite(p).all():
                    continue
                cx, cy = int(round(p[0])), int(round(p[1]))
                if not (0 <= cx < W and 0 <= cy < H):
                    continue
                cv2.circle(
                    overlay,
                    (cx, cy),
                    r,
                    tuple(int(v) for v in colors[n]),
                    -1,
                    cv2.LINE_AA,
                )
                drew = True
            if drew:
                cv2.addWeighted(overlay, alpha, canvas, 1.0 - alpha, 0, canvas)

        # Head: full-opacity dot + white sparkle at center for visible points
        for n in range(N):
            if pred_visibility_NT[n, t] < vis_thresh:
                continue
            p = uv[n, t]
            if not np.isfinite(p).all():
                continue
            cx, cy = int(round(p[0])), int(round(p[1]))
            if not (0 <= cx < W and 0 <= cy < H):
                continue
            c = tuple(int(v) for v in colors[n])
            cv2.circle(canvas, (cx, cy), head_radius, c, -1, cv2.LINE_AA)
            cv2.circle(
                canvas,
                (cx, cy),
                max(1, head_radius // 3),
                (255, 255, 255),
                -1,
                cv2.LINE_AA,
            )

        vw.write(canvas)

    vw.release()
    return out_path


def render_tracking_d4rt_html(
    pred_tracks_NT3: np.ndarray,  # (N, F, 3)  camera-space XYZ metres
    pred_visibility_NT: np.ndarray,  # (N, F)     float 0/1
    out_path: str | Path,
    fps: int = 15,
    tail_seconds: float = 1.0,
    head_size: float = 5.0,
    vis_thresh: float = 0.5,
    max_tracks: int = 64,
    images_rgb: np.ndarray | None = None,  # (F, H, W, 3) uint8
    depth_hw: np.ndarray | None = None,  # (F, H, W) float32 metres
    K_mat: np.ndarray | None = None,  # (3, 3) intrinsics for H × W
    pcloud_stride: int = 16,
) -> Path:
    """Interactive animated 3D D4RT-style visualization — self-contained HTML.

    Produces a Plotly animated Scatter3d: vivid rainbow-colored tracks in 3D space
    with alpha-fading tails, plus an image-colored scene point cloud when
    `images_rgb`, `depth_hw`, and `K_mat` are supplied (two Scatter3d traces).
    Open the file in any browser to:
      - ▶ Play / ⏸ Pause the animation with the buttons
      - Drag the frame slider to jump to any frame
      - Rotate / zoom / pan the 3D scene freely

    Plotly.js is loaded from CDN (requires internet on first open).
    To cap HTML size, at most `max_tracks` tracks are shown (longest-visible first).
    `pcloud_stride` controls point cloud density (higher = sparser, smaller HTML).
    """
    import plotly.graph_objects as go  # lazy import — not needed for TAPVid style

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tail_len = max(1, round(tail_seconds * fps))

    N_full, F, _ = pred_tracks_NT3.shape

    # Select longest-visible tracks up to max_tracks
    vis_count = pred_visibility_NT.sum(axis=1)
    sel = np.argsort(-vis_count)[:max_tracks]
    tracks = pred_tracks_NT3[sel]  # (N, F, 3)
    vis = pred_visibility_NT[sel]  # (N, F)
    N = len(sel)

    # Per-track vivid RGB colors (same hue logic as _colormap_vivid, but as (R,G,B) tuples)
    hue = np.linspace(0, 179, N, endpoint=False).astype(np.uint8)
    hsv = np.zeros((N, 1, 3), dtype=np.uint8)
    hsv[:, 0, 0] = hue
    hsv[:, 0, 1] = 255
    hsv[:, 0, 2] = 255
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[:, 0, ::-1]  # (N, 3) RGB

    def _rgba(n: int, alpha: float) -> str:
        r, g, b = int(rgb[n, 0]), int(rgb[n, 1]), int(rgb[n, 2])
        return f"rgba({r},{g},{b},{alpha:.3f})"

    # === Precompute per-frame scene point clouds from depth + images ===
    pcloud_frames: list | None = None
    if images_rgb is not None and depth_hw is not None and K_mat is not None:
        Hp, Wp = int(depth_hw.shape[1]), int(depth_hw.shape[2])
        fx, fy = float(K_mat[0, 0]), float(K_mat[1, 1])
        cx, cy = float(K_mat[0, 2]), float(K_mat[1, 2])

        # Fixed pixel grid — same positions every frame for a stable point cloud
        vs_arr = np.arange(0, Hp, pcloud_stride)
        us_arr = np.arange(0, Wp, pcloud_stride)
        ug, vg = np.meshgrid(us_arr, vs_arr, indexing="xy")
        ug = ug.flatten().astype(np.int32)
        vg = vg.flatten().astype(np.int32)

        # Nearest-neighbour remap if images and depth are different resolutions
        Hi, Wi = int(images_rgb.shape[1]), int(images_rgb.shape[2])
        if Hi != Hp or Wi != Wp:
            yi = np.round(np.arange(Hp) * Hi / Hp).astype(int).clip(0, Hi - 1)
            xi = np.round(np.arange(Wp) * Wi / Wp).astype(int).clip(0, Wi - 1)
            img_src = images_rgb[:, yi[:, None], xi[None, :], :]  # (F, Hp, Wp, 3)
        else:
            img_src = images_rgb

        pcloud_frames = []
        for t in range(F):
            d = depth_hw[t, vg, ug]
            valid = (d > 0.05) & (d < 500.0) & np.isfinite(d)
            if not valid.any():
                pcloud_frames.append(([], [], [], []))
                continue
            uv, vv, dv = ug[valid], vg[valid], d[valid]
            X = ((uv - cx) * dv / fx).tolist()
            Y = ((vv - cy) * dv / fy).tolist()
            Z = dv.tolist()
            rgb_pts = img_src[t, vv, uv]  # (M, 3) uint8
            colors = ["rgb(%d,%d,%d)" % (int(r), int(g), int(b)) for r, g, b in rgb_pts]
            pcloud_frames.append((X, Y, Z, colors))

    # Build animation frames — two Scatter3d traces when point cloud is available:
    #   trace 0: scene point cloud (image-colored background)
    #   trace 1: track tails + heads (vivid hue, RGBA opacity)
    plotly_frames = []
    for t in range(F):
        frame_traces = []

        # Trace 0: scene point cloud
        if pcloud_frames is not None:
            px, py, pz, pc = pcloud_frames[t]
            frame_traces.append(
                go.Scatter3d(
                    x=px,
                    y=py,
                    z=pz,
                    mode="markers",
                    marker=dict(color=pc, size=1.5, line=dict(width=0), opacity=0.8),
                    hoverinfo="skip",
                )
            )

        # Trace 1 (or 0 without pcloud): track tails + heads
        xs, ys, zs, clrs, szs = [], [], [], [], []

        # Tail: oldest step first; 3 px dots at 0.15–0.9 opacity fade into a visible trail
        for k in range(tail_len, 0, -1):
            t_k = t - k
            if t_k < 0:
                continue
            # alpha: 0.15 (oldest) → 0.90 (newest) so the full tail stays readable
            alpha = 0.15 + 0.75 * (1.0 - k / tail_len)
            for n in range(N):
                if vis[n, t_k] < vis_thresh:
                    continue
                p = tracks[n, t_k]
                if not np.isfinite(p).all():
                    continue
                xs.append(float(p[0]))
                ys.append(float(p[1]))
                zs.append(float(p[2]))
                clrs.append(_rgba(n, alpha))
                szs.append(3.0)

        # Head: current-frame dot, full opacity
        for n in range(N):
            if vis[n, t] < vis_thresh:
                continue
            p = tracks[n, t]
            if not np.isfinite(p).all():
                continue
            xs.append(float(p[0]))
            ys.append(float(p[1]))
            zs.append(float(p[2]))
            clrs.append(_rgba(n, 1.0))
            szs.append(head_size)

        frame_traces.append(
            go.Scatter3d(
                x=xs,
                y=ys,
                z=zs,
                mode="markers",
                marker=dict(color=clrs, size=szs, line=dict(width=0)),
                hoverinfo="skip",
            )
        )

        plotly_frames.append(go.Frame(data=frame_traces, name=str(t)))

    frame_ms = int(1000 / max(1, fps))
    fig = go.Figure(
        data=plotly_frames[0].data if plotly_frames else [],
        frames=plotly_frames,
        layout=go.Layout(
            paper_bgcolor="#0d0d0d",
            scene=dict(
                bgcolor="#0d0d0d",
                xaxis=dict(visible=False, showgrid=False, zeroline=False),
                yaxis=dict(visible=False, showgrid=False, zeroline=False),
                zaxis=dict(visible=False, showgrid=False, zeroline=False),
                aspectmode="data",
            ),
            title=dict(
                text="3D point tracks  (rotate: drag | zoom: scroll)",
                font=dict(color="#cccccc", size=13),
            ),
            updatemenus=[
                dict(
                    type="buttons",
                    showactive=False,
                    x=0.05,
                    y=0.02,
                    xanchor="left",
                    yanchor="bottom",
                    bgcolor="#1a1a1a",
                    bordercolor="#444",
                    font=dict(color="#eeeeee"),
                    buttons=[
                        dict(
                            label="▶ Play",
                            method="animate",
                            args=[
                                None,
                                dict(
                                    frame=dict(duration=frame_ms, redraw=True),
                                    fromcurrent=True,
                                    transition=dict(duration=0),
                                ),
                            ],
                        ),
                        dict(
                            label="⏸ Pause",
                            method="animate",
                            args=[
                                [None],
                                dict(
                                    frame=dict(duration=0, redraw=False),
                                    mode="immediate",
                                    transition=dict(duration=0),
                                ),
                            ],
                        ),
                    ],
                )
            ],
            sliders=[
                dict(
                    active=0,
                    x=0.1,
                    len=0.85,
                    y=0.02,
                    yanchor="bottom",
                    bgcolor="#1a1a1a",
                    bordercolor="#333",
                    tickcolor="#555",
                    font=dict(color="#cccccc", size=9),
                    currentvalue=dict(
                        prefix="Frame ", font=dict(color="#cccccc", size=11)
                    ),
                    transition=dict(duration=0),
                    steps=[
                        dict(
                            args=[
                                [str(t)],
                                dict(
                                    frame=dict(duration=0, redraw=True),
                                    mode="immediate",
                                    transition=dict(duration=0),
                                ),
                            ],
                            label=str(t) if t % max(1, F // 20) == 0 else "",
                            method="animate",
                        )
                        for t in range(F)
                    ],
                )
            ],
            margin=dict(l=0, r=0, t=50, b=100),
            height=750,
            showlegend=False,
        ),
    )

    fig.write_html(str(out_path), include_plotlyjs="cdn", full_html=True)
    return out_path
