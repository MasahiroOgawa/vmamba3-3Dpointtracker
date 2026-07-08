"""TAPVid-3D clip loader.

Each clip is one .npz with the following keys (confirmed against
~/data/tapvid3d/pstudio/boxes_12.npz):

    images_jpeg_bytes : (F,)  object array of JPEG-encoded bytes
    queries_xyt       : (N_q, 3) float32  — (x, y, t) of each query in pixels + frame
    tracks_XYZ        : (F, N_q, 3) float32  — GT 3D position per (frame, query)
                                                  in the camera coordinate frame
    visibility        : (F, N_q) bool
    fx_fy_cx_cy       : (4,) float64 — pinhole intrinsics

Coordinate frame: TAPVid-3D releases tracks in *camera coordinates of the
first frame* (per the TAPVid-3D paper §3.1). For training we keep them as-is;
the camera is treated as fixed at the canonical pose.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image

SUBSETS = ("pstudio", "drivetrack", "adt")


@dataclass
class TAPVidClip:
    """One TAPVid-3D clip, decoded into tensors."""
    images: torch.Tensor          # (F, 3, H, W) float32 in [0, 1]
    queries_xyt: torch.Tensor     # (N_q, 3) float32
    tracks_XYZ: torch.Tensor      # (F, N_q, 3) float32
    visibility: torch.Tensor      # (F, N_q) bool
    K: torch.Tensor               # (3, 3) float32
    clip_id: str                  # filename stem (e.g. "boxes_12")
    subset: str                   # "pstudio" | "drivetrack" | "adt"

    @property
    def F(self) -> int:
        return int(self.images.shape[0])

    @property
    def H(self) -> int:
        return int(self.images.shape[-2])

    @property
    def W(self) -> int:
        return int(self.images.shape[-1])

    @property
    def N_q(self) -> int:
        return int(self.queries_xyt.shape[0])


def _decode_jpeg_frames(jpeg_bytes_arr: np.ndarray) -> torch.Tensor:
    """Decode a (F,) object array of JPEG bytes → (F, 3, H, W) float32 in [0, 1]."""
    frames: list[torch.Tensor] = []
    for jb in jpeg_bytes_arr:
        img = Image.open(io.BytesIO(bytes(jb))).convert("RGB")
        # numpy (H, W, 3) uint8 → torch (3, H, W) float32 in [0, 1]
        arr = np.asarray(img, dtype=np.uint8)
        t = torch.from_numpy(arr).permute(2, 0, 1).float().div_(255.0)
        frames.append(t)
    return torch.stack(frames, dim=0)


def _build_K(fx_fy_cx_cy: np.ndarray) -> torch.Tensor:
    fx, fy, cx, cy = (float(v) for v in fx_fy_cx_cy)
    K = torch.tensor(
        [[fx, 0.0, cx],
         [0.0, fy, cy],
         [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )
    return K


def _infer_subset(path: Path) -> str:
    for sub in SUBSETS:
        if sub in path.parts:
            return sub
    return "unknown"


def peek_clip_F(path: str | Path) -> int:
    """Return F (number of frames) without decoding any JPEG."""
    path = Path(path).expanduser().resolve()
    with np.load(path, allow_pickle=True) as d:
        return int(d["images_jpeg_bytes"].shape[0])


def has_images(path: str | Path) -> bool:
    """True if the .npz contains image bytes (False for labels-only clips).

    Reads only the zip directory (cheap, no decompression). Many TAPVid-3D
    ADT clips ship labels-only on the HF mirror — they can be scored against
    GT but not rendered/encoded, so renderers should skip them.
    """
    try:
        with np.load(path, allow_pickle=True) as d:
            return "images_jpeg_bytes" in d.files
    except Exception:
        return False


def load_clip(
    path: str | Path,
    frames: tuple[int, int] | None = None,
) -> TAPVidClip:
    """Load one TAPVid-3D clip from a `.npz` file.

    If `frames=(start, end)` is given, only those frames are JPEG-decoded
    and `tracks_XYZ` / `visibility` are sliced to match. This avoids
    materialising the full clip in RAM — drivetrack clips are 1280×1920
    float32 (~28 MB per frame), so a 24-frame clip is ~670 MB even when
    we only need 8 frames. The query list (`queries_xyt`) is NOT sliced;
    its `t` axis still refers to original-clip frame indices and the
    caller shifts it.
    """
    path = Path(path).expanduser().resolve()
    d = np.load(path, allow_pickle=True)
    jpeg_arr = d["images_jpeg_bytes"]
    if frames is not None:
        s, e = frames
        jpeg_arr = jpeg_arr[s:e]
        tracks = torch.from_numpy(np.asarray(d["tracks_XYZ"][s:e])).float()
        vis = torch.from_numpy(np.asarray(d["visibility"][s:e])).bool()
    else:
        tracks = torch.from_numpy(d["tracks_XYZ"]).float()
        vis = torch.from_numpy(d["visibility"]).bool()
    images = _decode_jpeg_frames(jpeg_arr)
    queries = torch.from_numpy(d["queries_xyt"]).float()
    K = _build_K(d["fx_fy_cx_cy"])
    return TAPVidClip(
        images=images,
        queries_xyt=queries,
        tracks_XYZ=tracks,
        visibility=vis,
        K=K,
        clip_id=path.stem,
        subset=_infer_subset(path),
    )


def list_clips(
    data_root: str | Path,
    subsets: Iterable[str] = SUBSETS,
) -> list[Path]:
    """List every .npz path under the named subsets of `data_root/tapvid3d/`."""
    root = Path(data_root).expanduser()
    if root.name != "tapvid3d":
        root = root / "tapvid3d"
    out: list[Path] = []
    for sub in subsets:
        out += sorted((root / sub).glob("*.npz"))
    return out
