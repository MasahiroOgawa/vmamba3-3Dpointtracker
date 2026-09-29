#!/usr/bin/env python3
"""Download the WAFT checkpoints that are not in its git repository into third_party/WAFT/.

  uv run python scripts/download_weights.py

- ckpts/waft_a1_recommended.pth (~245 MB): the a1 checkpoint WAFT's README recommends for
  downstream use, from the authors' Google Drive.
- depth-anything-ckpts/depth_anything_v2_vits.pth (~95 MB): Depth-Anything-V2-Small, which WAFT's
  backbone loads from that relative path, from the official Hugging Face repository.

Files already present are skipped. DA3 (depth-anything/da3metric-large) and DINOv3 are not
handled here: they are fetched from the Hugging Face Hub on first use.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

WAFT = Path(__file__).resolve().parents[1] / "third_party" / "WAFT"
WEIGHTS = {
    WAFT / "ckpts" / "waft_a1_recommended.pth":
        "https://drive.usercontent.google.com/download?id=1CxzBQx0iSg6AyIgt6MF0ROlF_cAeZLPC"
        "&export=download&confirm=t",
    WAFT / "depth-anything-ckpts" / "depth_anything_v2_vits.pth":
        "https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/"
        "depth_anything_v2_vits.pth",
}


def fetch(url: str, dest: Path) -> None:
    """Download to `<dest>.part` and rename on success, so a killed run never leaves a
    truncated checkpoint that looks complete."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    print(f"[download] {dest.relative_to(WAFT.parents[1])}", flush=True)
    with urllib.request.urlopen(url, timeout=60) as r, open(part, "wb") as f:
        total = int(r.headers.get("Content-Length", 0))
        done = 0
        while chunk := r.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {done / 2**20:8.0f} / {total / 2**20:.0f} MB", end="", flush=True)
    print()
    part.rename(dest)


def main() -> None:
    if not (WAFT / "model").exists():
        raise SystemExit("third_party/WAFT is empty; run `git submodule update --init` first.")
    for dest, url in WEIGHTS.items():
        if dest.exists():
            print(f"[skip] {dest.relative_to(WAFT.parents[1])} already present")
        else:
            fetch(url, dest)


if __name__ == "__main__":
    main()
