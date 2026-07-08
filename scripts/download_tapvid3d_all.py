"""Download all three TAPVid-3D subsets (pstudio + adt + drivetrack).

The HF mirror at `ZhengGuangze/TAPVid-3D` re-hosts the entire eval pool
of ~4569 clips (= 4419 FULL_EVAL + 150 MINIVAL) as 24 tarballs:

    pstudio.tar.gz             3.64 GB    (1 file, includes both FULL_EVAL and MINIVAL pstudio)
    adt_batch_0..9.tar.gz    ~161 GB    (10 batches, ~17 GB each, includes all 1906 FULL_EVAL + 50 MINIVAL adt)
    drivetrack_batch_0..12  ~309 GB    (13 batches, 7-28 GB each, includes 2407 FULL_EVAL + 50 MINIVAL drivetrack)
                            ~474 GB    compressed total
                            ~525 GB    extracted

Default behaviour: sequentially download each tarball, extract its
.npz files to `~/data/tapvid3d/<subset>/`, then delete the tarball
(peak disk = one tarball + extracted-so-far). Use `--keep-tarballs`
to retain the .tar.gz files (uses ~470 GB more disk).

For the official train / test split used by v19+, see
`src/mamba3_tracker/data/dataset.py:official_train_test_split`:
    train = FULL_EVAL_FILES (4419 clips)
    test  = MINIVAL_FILES   (150 clips)

Usage:
    uv run python scripts/download_tapvid3d_all.py                  # all subsets
    uv run python scripts/download_tapvid3d_all.py --subsets pstudio   # one subset
    uv run python scripts/download_tapvid3d_all.py --keep-tarballs     # don't auto-delete
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tarfile
import time
from pathlib import Path

import requests
from tqdm import tqdm


HF_BASE = "https://huggingface.co/datasets/ZhengGuangze/TAPVid-3D/resolve/main"

# {subset: [tarball_name, ...]}.  Tarballs extract into ~/data/tapvid3d/<subset>/.
SUBSET_TARBALLS: dict[str, list[str]] = {
    "pstudio":    ["pstudio.tar.gz"],
    "adt":        [f"adt_batch_{i}.tar.gz" for i in range(10)],
    "drivetrack": [f"drivetrack_batch_{i}.tar.gz" for i in range(13)],
}


def _streamed_download(url: str, dest: Path) -> bool:
    """Returns True if file was downloaded, False if already present."""
    if dest.exists() and dest.stat().st_size > 1024:
        print(f"  {dest.name} already present ({dest.stat().st_size / 1e9:.2f} GB), skipping download")
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(dest, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc=dest.name, leave=False
        ) as pbar:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                f.write(chunk)
                pbar.update(len(chunk))
    return True


def _extract_to(archive: Path, dest: Path) -> int:
    """Extract all .npz members of `archive` into `dest` (flat, no subdirs).
    Returns count of newly-written files (existing same-name files are kept)."""
    dest.mkdir(parents=True, exist_ok=True)
    n = 0
    with tarfile.open(archive, "r:gz") as tf:
        for m in tf:
            if not m.isfile() or not m.name.endswith(".npz"):
                continue
            target = dest / Path(m.name).name
            if target.exists() and target.stat().st_size > 1024:
                continue
            extracted = tf.extractfile(m)
            if extracted is None:
                continue
            with open(target, "wb") as out:
                shutil.copyfileobj(extracted, out)
            n += 1
    return n


def _free_gb(path: Path) -> float:
    s = shutil.disk_usage(path)
    return s.free / 1e9


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", type=Path, default=Path.home() / "data" / "tapvid3d",
                    help="Root for extracted .npz files. Created if missing.")
    ap.add_argument("--subsets", nargs="+", default=list(SUBSET_TARBALLS.keys()),
                    choices=list(SUBSET_TARBALLS.keys()))
    ap.add_argument("--keep-tarballs", action="store_true",
                    help="Keep .tar.gz files after extraction (otherwise deleted).")
    args = ap.parse_args()

    args.data_root.mkdir(parents=True, exist_ok=True)
    print(f"[tapvid3d] target: {args.data_root}")
    print(f"[tapvid3d] subsets: {args.subsets}")
    print(f"[tapvid3d] keep tarballs: {args.keep_tarballs}")
    print(f"[tapvid3d] disk free at target: {_free_gb(args.data_root):.1f} GB")
    print()

    total_files_extracted = 0
    t0 = time.perf_counter()
    for sub in args.subsets:
        sub_dir = args.data_root / sub
        cache_dir = sub_dir / "_tarballs"
        cache_dir.mkdir(parents=True, exist_ok=True)
        tarballs = SUBSET_TARBALLS[sub]
        print(f"=== {sub} ({len(tarballs)} tarballs) ===")
        for i, name in enumerate(tarballs):
            url = f"{HF_BASE}/{name}"
            tar_path = cache_dir / name
            print(f"[{sub}] {i+1}/{len(tarballs)}: {name}  (free: {_free_gb(args.data_root):.1f} GB)")
            try:
                _streamed_download(url, tar_path)
                n_new = _extract_to(tar_path, sub_dir)
                total_files_extracted += n_new
                print(f"  +{n_new} .npz extracted into {sub_dir}/")
            except Exception as e:
                print(f"  FAIL {type(e).__name__}: {e}")
                if tar_path.exists():
                    tar_path.unlink()
                continue
            if not args.keep_tarballs:
                tar_path.unlink()
                print(f"  removed {tar_path.name}")
        if cache_dir.exists() and not any(cache_dir.iterdir()):
            cache_dir.rmdir()
        n_npz_total = len(list(sub_dir.glob("*.npz")))
        print(f"[{sub}] DONE — {n_npz_total} .npz files on disk\n")

    elapsed = time.perf_counter() - t0
    print(f"=== ALL DONE in {elapsed/3600:.2f} h ===")
    print(f"[tapvid3d] new .npz files this run: {total_files_extracted}")
    print(f"[tapvid3d] disk free now: {_free_gb(args.data_root):.1f} GB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
