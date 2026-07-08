"""Download DINOv2-small pretrained weights.

Fetches `dinov2_vits14_pretrain.pth` (~84 MB) into
`data/weights/dinov2_vits14_pretrain.pth`.

Usage:
    uv run python scripts/download_dinov2.py [--dest PATH]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from mamba3_attn.data._util import download

URL = "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth"
DEFAULT_DEST = Path("data/weights/dinov2_vits14_pretrain.pth")


def ensure_dinov2_vits14(dest: Path = DEFAULT_DEST) -> Path:
    """Download the checkpoint if missing. Returns the local path."""
    return download(URL, dest, min_bytes=50_000_000)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    args = ap.parse_args()
    p = ensure_dinov2_vits14(args.dest)
    print(f"[ok] {p}  ({p.stat().st_size / 2**20:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
