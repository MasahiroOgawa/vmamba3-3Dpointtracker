"""Run v33 (SEA-RAFT + DA3 + Mamba-3 depth refiner) on arbitrary video files.

Query points: a user-specified grid (default 20×20) placed at frame 0.
Intrinsics: estimated from image dimensions (f ≈ max(H, W); principal point = centre).
            Pass --fx / --fy / --cx / --cy to override.

Outputs per video (in --out-dir/<stem>/):
    tracks.mp4          2D track overlay
    tracks_3d.html      interactive 3D trajectory (plotly)

Usage:
    uv run python scripts/track_custom_video.py \
        ~/data/study/*.MOV \
        --ckpt result/v33_20260617-0001/ckpt_20000.pt \
        --out-dir result/custom_video_tracks
"""
from __future__ import annotations

import argparse
import sys
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# ── DA3 needs this stub before any mamba3_tracker import ─────────────────────
sys.modules.setdefault("moviepy.editor", types.ModuleType("moviepy.editor"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "third_party" / "depth-anything-3" / "src"))

from depth_anything_3.api import DepthAnything3          # noqa: E402
from mamba3_tracker.model.depth_refined_tracker import Mamba3DepthRefiner  # noqa: E402
from mamba3_tracker.viz.track_video import render_tracking_video           # noqa: E402
from searaft_flow import FlowModel, track_clip                              # noqa: E402


# ── video I/O ─────────────────────────────────────────────────────────────────
def _load_video_frames(path: Path, max_frames: int = 0) -> np.ndarray:
    """Return (F, H, W, 3) uint8 RGB array.  Raises if estimated RAM > 16 GB."""
    import cv2
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    limit = min(total, max_frames) if max_frames else total
    ram_gb = limit * h * w * 3 / 1e9
    if ram_gb > 16:
        raise RuntimeError(
            f"{path.name}: {total} frames ({h}×{w}) would need {limit*h*w*3/1e9:.1f} GB RAM. "
            f"Use --max-frames (e.g. --max-frames 300) to cap the video."
        )
    frames = []
    for _ in range(limit):
        ok, bgr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise RuntimeError(f"No frames read from {path}")
    return np.stack(frames)


def _estimate_K(H: int, W: int) -> np.ndarray:
    """Pinhole intrinsics heuristic: f = max(H,W), principal point = centre."""
    f = float(max(H, W))
    return np.array([[f, 0, W / 2.0], [0, f, H / 2.0], [0, 0, 1]], dtype=np.float32)


# ── grid queries ──────────────────────────────────────────────────────────────
def _grid_queries(H: int, W: int, nx: int = 20, ny: int = 20) -> torch.Tensor:
    """Return (N, 3) tensor [u, v, anchor_t=0] for an nx×ny grid at frame 0."""
    xs = torch.linspace(W / (2 * nx), W - W / (2 * nx), nx)
    ys = torch.linspace(H / (2 * ny), H - H / (2 * ny), ny)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")   # (ny, nx)
    u = gx.reshape(-1)
    v = gy.reshape(-1)
    t = torch.zeros(u.shape[0])
    return torch.stack([u, v, t], dim=-1)   # (N, 3): [u, v, anchor_t]


# ── DA3 depth ─────────────────────────────────────────────────────────────────
def _run_da3(da3_model, frames_uint8: np.ndarray, process_res: int = 504) -> np.ndarray:
    """Return (F, Hd, Wd) float32 metric depth in metres."""
    from PIL import Image
    rgbs = [Image.fromarray(f) for f in frames_uint8]
    chunk = 8   # run 8 frames at a time to stay within VRAM
    chunks = []
    for i in range(0, len(rgbs), chunk):
        pred = da3_model.inference(rgbs[i:i + chunk], process_res=process_res, export_format="mini_npz")
        chunks.append(np.asarray(pred.depth, dtype=np.float32))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return np.concatenate(chunks, axis=0)   # (F, Hd, Wd)


# ── ray helper ────────────────────────────────────────────────────────────────
def _ray_from_uv(uv: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    """uv: (..., 2); K: (3,3) → (..., 2) normalised ray."""
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    return torch.stack([(uv[..., 0] - cx) / fx, (uv[..., 1] - cy) / fy], dim=-1)


# ── main inference ────────────────────────────────────────────────────────────
@torch.no_grad()
def _infer(flow_model, da3_model, v33_model, frames_uint8, K_np, image_size, nx, ny, device):
    H, W = frames_uint8.shape[1], frames_uint8.shape[2]
    F_ = frames_uint8.shape[0]
    sx, sy = image_size / float(W), image_size / float(H)

    # Resize to working resolution for SEA-RAFT
    frames_t = torch.from_numpy(frames_uint8).float().permute(0, 3, 1, 2) / 255.0   # (F,3,H,W)
    frames_r = F.interpolate(frames_t, size=(image_size, image_size), mode="bilinear", align_corners=False)
    images_255 = frames_r * 255.0   # (F,3,S,S)

    queries = _grid_queries(H, W, nx, ny)   # (N,3) in original pixel coords
    queries_xy = torch.stack([queries[:, 0] * sx, queries[:, 1] * sy], dim=-1)
    anchor_t = queries[:, 2].long().clamp(0, F_ - 1)

    # SEA-RAFT 2D tracking
    uv, vis = track_clip(flow_model, images_255.to(device), queries_xy, anchor_t, image_size)
    # uv: (F, N, 2) in working-resolution pixels; vis: (F, N)

    # Intrinsics scaled to working resolution
    K = torch.from_numpy(K_np.copy())
    K[0] *= sx
    K[1] *= sy

    # DA3 depth for all frames
    print(f"  [da3] running on {F_} frames …", flush=True)
    depth_np = _run_da3(da3_model, frames_uint8)   # (F, Hd, Wd)
    depth_t = torch.from_numpy(depth_np).to(device).unsqueeze(1)   # (F, 1, Hd, Wd)

    # Sample depth at tracked positions
    uv_d = uv.unsqueeze(0).to(device)           # (1, F, N, 2)
    vis_d = vis.unsqueeze(0).to(device)          # (1, F, N)
    grid = (2.0 * uv_d / image_size - 1.0).view(F_, 1, -1, 2)   # (F, 1, N, 2)
    z_raw = F.grid_sample(depth_t, grid, mode="bilinear",
                          padding_mode="border", align_corners=False).view(1, F_, -1)  # (1,F,N)

    # Ray directions — uv_d is (1,F,N,2); _ray_from_uv broadcasts → (1,F,N,2)
    K_dev = K.to(device)
    ray = _ray_from_uv(uv_d, K_dev)   # (1, F, N, 2)

    # v33 depth refinement
    pred = v33_model(ray, z_raw, vis_d)          # TrackerOutputs
    xyz = pred.xyz[0].cpu().numpy()              # (F, N, 3) camera-frame metric XYZ
    pred_vis = vis.cpu().numpy()                 # (F, N)

    # Bring back to (N, F, 3) convention expected by render_tracking_video
    tracks_NT3 = xyz.transpose(1, 0, 2)         # (N, F, 3)
    vis_NT = pred_vis.T                          # (N, F)
    return tracks_NT3, vis_NT


# ── 3D HTML plot ──────────────────────────────────────────────────────────────
def _save_3d_html(tracks_NT3, vis_NT, out_path: Path):
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("  [3d] plotly not installed, skipping HTML plot")
        return
    fig = go.Figure()
    N = tracks_NT3.shape[0]
    colors = [f"hsl({int(360*i/N)},70%,50%)" for i in range(N)]
    for n in range(N):
        vis_mask = vis_NT[n] > 0.5
        x, y, z = tracks_NT3[n, vis_mask].T
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z, mode="lines+markers",
            marker=dict(size=2, color=colors[n]),
            line=dict(width=1, color=colors[n]),
            showlegend=False,
        ))
    fig.update_layout(
        scene=dict(xaxis_title="X (m)", yaxis_title="Y (m)", zaxis_title="Z (m)"),
        title="v33 3D tracks",
    )
    fig.write_html(str(out_path))
    print(f"  [3d] wrote {out_path}")


# ── entry point ───────────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("videos", nargs="+", type=Path, help="Input .MOV/.mp4/... files")
    ap.add_argument("--ckpt", type=Path,
                    default=Path("result/v33_20260617-0001/ckpt_20000.pt"),
                    help="v33 checkpoint")
    ap.add_argument("--out-dir", type=Path, default=Path("result/custom_video_tracks"))
    ap.add_argument("--image-size", type=int, default=896, help="SEA-RAFT working resolution")
    ap.add_argument("--da3-res", type=int, default=504, help="DA3 processing resolution")
    ap.add_argument("--da3-model", default="da3metric-large")
    ap.add_argument("--nx", type=int, default=20, help="Grid columns")
    ap.add_argument("--ny", type=int, default=20, help="Grid rows")
    ap.add_argument("--max-frames", type=int, default=0, help="Truncate to first N frames (0=all)")
    ap.add_argument("--url", default="MemorySlices/Tartan-C-T-TSKH-spring540x960-M",
                    help="SEA-RAFT model variant URL")
    ap.add_argument("--iters", type=int, default=None, help="SEA-RAFT RAFT iterations")
    # Intrinsics override
    ap.add_argument("--fx", type=float, default=None)
    ap.add_argument("--fy", type=float, default=None)
    ap.add_argument("--cx", type=float, default=None)
    ap.add_argument("--cy", type=float, default=None)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[v33-video] device={device}  grid={args.nx}×{args.ny}={args.nx*args.ny} points")

    # ── load models ───────────────────────────────────────────────────────────
    print("[v33-video] loading SEA-RAFT …", flush=True)
    flow_model = FlowModel(device, url=args.url, iters=args.iters)

    print("[v33-video] loading DA3 …", flush=True)
    da3_model = DepthAnything3.from_pretrained(f"depth-anything/{args.da3_model}").to(device)
    da3_model.eval()

    print("[v33-video] loading v33 …", flush=True)
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    v33_model = Mamba3DepthRefiner()
    v33_model.load_state_dict(ckpt["model"])
    v33_model.to(device).eval()

    # ── process each video ────────────────────────────────────────────────────
    for video_path in args.videos:
        video_path = video_path.expanduser()
        print(f"\n[v33-video] {video_path.name}", flush=True)

        frames = _load_video_frames(video_path, max_frames=args.max_frames)
        F_, H, W, _ = frames.shape
        print(f"  frames={F_}  size={H}×{W}", flush=True)

        # Intrinsics
        K_np = _estimate_K(H, W)
        for val, (i, j) in [(args.fx, (0, 0)), (args.fy, (1, 1)),
                            (args.cx, (0, 2)), (args.cy, (1, 2))]:
            if val is not None:
                K_np[i, j] = val

        tracks_NT3, vis_NT = _infer(
            flow_model, da3_model, v33_model,
            frames, K_np, args.image_size, args.nx, args.ny, device,
        )

        out_dir = args.out_dir / video_path.stem
        out_dir.mkdir(parents=True, exist_ok=True)

        # 2D video
        mp4_path = out_dir / "tracks.mp4"
        render_tracking_video(frames, tracks_NT3, vis_NT, K_np, mp4_path)
        print(f"  [2d] wrote {mp4_path}")

        # 3D HTML
        _save_3d_html(tracks_NT3, vis_NT, out_dir / "tracks_3d.html")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
