"""Download the ADT (Aria Digital Twin) subset of TAPVid-3D.

The official Google release does not redistribute the ADT .npz files (Aria
license requires generating them locally via aria_dataset_downloader after
accepting the licence). The HF mirror at `ZhengGuangze/TAPVid-3D` re-hosts
them pre-generated as 10 tarball batches of ~17 GB each.

This script pulls one or more batches and extracts the .npz files directly
under `~/data/tapvid3d/adt/`. After extraction, file names match the
`tapvid3d_*.npz` pattern used by the official MINIVAL_FILES split.

Usage:
    uv run python scripts/download_tapvid3d_adt.py            # batch 0 only
    uv run python scripts/download_tapvid3d_adt.py --batches 0 1 2
    uv run python scripts/download_tapvid3d_adt.py --batches all   # all 10
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tarfile
from pathlib import Path

import requests
from tqdm import tqdm


HF_BASE = "https://huggingface.co/datasets/ZhengGuangze/TAPVid-3D/resolve/main"
NUM_BATCHES = 10  # adt_batch_0 .. adt_batch_9


def _streamed_download(url: str, dest: Path, max_attempts: int = 8) -> None:
    """Download `url` to `dest` with Range-based resume + retry on connection
    drops. HuggingFace's CDN was dropping the connection mid-stream on these
    17 GB tarballs; this resumes from however many bytes are already on disk."""
    dest.parent.mkdir(parents=True, exist_ok=True)

    def _expected_total() -> int:
        with requests.head(url, allow_redirects=True, timeout=60) as r:
            r.raise_for_status()
            return int(r.headers.get("content-length", 0))

    expected = None
    for attempt in range(1, max_attempts + 1):
        have = dest.stat().st_size if dest.exists() else 0
        if expected is None:
            try:
                expected = _expected_total()
            except Exception as e:
                print(f"[adt] attempt {attempt}: HEAD failed ({e}); will infer size from GET")
        if expected and have >= expected:
            print(f"[adt] {dest.name} fully present ({have / 1e9:.1f} GB)")
            return
        if have:
            print(f"[adt] resuming {dest.name} from {have / 1e9:.2f} GB "
                  f"(attempt {attempt}/{max_attempts})")
        headers = {"Range": f"bytes={have}-"} if have else {}
        try:
            with requests.get(url, headers=headers, stream=True, timeout=60) as r:
                r.raise_for_status()
                if expected is None:
                    expected = have + int(r.headers.get("content-length", 0))
                with open(dest, "ab") as f, tqdm(
                    total=expected, initial=have, unit="B", unit_scale=True,
                    desc=dest.name, leave=False,
                ) as pbar:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if not chunk:
                            continue
                        f.write(chunk)
                        pbar.update(len(chunk))
            # Success — verify size, return.
            if not expected or dest.stat().st_size >= expected:
                return
            print(f"[adt] {dest.name} short read ({dest.stat().st_size}/{expected}); retrying")
        except (requests.exceptions.ChunkedEncodingError,
                requests.exceptions.ConnectionError,
                requests.exceptions.Timeout) as e:
            print(f"[adt] attempt {attempt}/{max_attempts} failed: {type(e).__name__}: {e}")
            if attempt == max_attempts:
                raise
    raise RuntimeError(f"[adt] {dest.name}: exhausted {max_attempts} attempts")


def _is_labels_only_npz(path: Path) -> bool:
    """True if the npz is missing `images_jpeg_bytes` (the official Google
    release ships labels-only files for some clips; the HF mirror tarballs
    re-host them with images included)."""
    try:
        import numpy as np
        with np.load(path, allow_pickle=True) as d:
            return "images_jpeg_bytes" not in d.files
    except Exception:
        return False


def _extract_to(archive: Path, dest: Path) -> tuple[int, int]:
    """Extract .npz files from archive into dest. Returns (n_new, n_replaced).

    Skips files already on disk that already have images. Overwrites files
    on disk that are labels-only (no `images_jpeg_bytes`) so re-running the
    download fixes minival ADT clips that originally landed as labels-only.
    """
    dest.mkdir(parents=True, exist_ok=True)
    n_new = 0
    n_replaced = 0
    with tarfile.open(archive, "r:gz") as tf:
        for m in tf:
            if not m.isfile() or not m.name.endswith(".npz"):
                continue
            target = dest / Path(m.name).name
            if target.exists() and target.stat().st_size > 0:
                if not _is_labels_only_npz(target):
                    continue
                # labels-only on disk → overwrite with the with-images version
                target.unlink()
                kind = "replace"
            else:
                kind = "new"
            extracted = tf.extractfile(m)
            if extracted is None:
                continue
            with open(target, "wb") as out:
                shutil.copyfileobj(extracted, out)
            if kind == "new":
                n_new += 1
            else:
                n_replaced += 1
    return n_new, n_replaced


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path,
                    default=Path.home() / "data" / "tapvid3d")
    ap.add_argument("--batches", nargs="+", default=["0"],
                    help="Batch indices to download (or 'all').")
    ap.add_argument("--keep-tarballs", action="store_true",
                    help="Keep .tar.gz files after extraction.")
    args = ap.parse_args()

    if args.batches == ["all"]:
        batches = list(range(NUM_BATCHES))
    else:
        batches = [int(b) for b in args.batches]

    adt_dir = args.data_root / "adt"
    cache_dir = adt_dir / "_tarballs"
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"[adt] target: {adt_dir}")
    print(f"[adt] batches: {batches}")

    total_new = 0
    total_replaced = 0
    for n in batches:
        url = f"{HF_BASE}/adt_batch_{n}.tar.gz"
        tar_path = cache_dir / f"adt_batch_{n}.tar.gz"
        print(f"\n[adt] batch {n}: {url}")
        _streamed_download(url, tar_path)
        print(f"[adt] extracting {tar_path.name} ...")
        n_new, n_replaced = _extract_to(tar_path, adt_dir)
        total_new += n_new
        total_replaced += n_replaced
        print(f"[adt] +{n_new} new, ~{n_replaced} replaced (labels-only → with-images)")
        if not args.keep_tarballs:
            tar_path.unlink()
            print(f"[adt] removed {tar_path.name} (use --keep-tarballs to retain)")

    if cache_dir.exists() and not any(cache_dir.iterdir()):
        cache_dir.rmdir()

    print(f"\n[adt] done. +{total_new} new, ~{total_replaced} labels-only replaced under {adt_dir}")
    print(f"[adt] total .npz now: {len(list(adt_dir.glob('*.npz')))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
