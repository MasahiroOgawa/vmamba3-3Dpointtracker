"""One-shot data downloader: ETH3D terrains (primary) + COCO-mini + NeRF-lego (fallback).

Usage:
    uv run python scripts/download_data.py [--data-root DATA_ROOT]

All downloads check free space (2 GB minimum) before starting.
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

from mamba3_attn.data import (
    download_coco_mini,
    download_eth3d_terrains,
    download_nerf_lego,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=Path("data"))
    ap.add_argument("--skip-eth3d", action="store_true")
    ap.add_argument("--skip-nerf", action="store_true")
    ap.add_argument("--skip-coco", action="store_true")
    args = ap.parse_args()

    root = args.data_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    print(f"Data root: {root}")

    ok = True

    if not args.skip_eth3d:
        try:
            p = download_eth3d_terrains(root)
            print(f"[ok] ETH3D terrains at {p}")
        except Exception as e:
            print(f"[fail] ETH3D: {e}")
            traceback.print_exc()
            ok = False

    if not args.skip_nerf:
        try:
            p = download_nerf_lego(root)
            print(f"[ok] NeRF lego at {p}")
        except Exception as e:
            print(f"[fail] NeRF lego: {e}")
            traceback.print_exc()

    if not args.skip_coco:
        try:
            ann, ids = download_coco_mini(root, num_images=20)
            print(f"[ok] COCO-mini annotations at {ann}, {len(ids)} image ids")
        except Exception as e:
            print(f"[fail] COCO-mini: {e}")
            traceback.print_exc()
            ok = False

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
