"""Recover a TAPVid-3D clip that is listed in full_eval but absent locally.

The HuggingFace mirror publishes only 13 drivetrack tarballs (~309 GB total) and no individual
.npz files, so a single clip cannot be requested directly. This streams each tarball through gzip
without ever storing it, stops at the first matching member, writes just that file, and aborts the
transfer. Disk cost is one clip; network cost is however much of the archives is read before the
match, which is why the batch that succeeded is printed -- record it if this is ever needed again.

Usage:
    uv run python scripts/fetch_missing_tapvid3d_clip.py                 # find and fetch all missing
    uv run python scripts/fetch_missing_tapvid3d_clip.py --dry-run       # just report what is missing
"""

from __future__ import annotations

import argparse
import sys
import tarfile
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).parent))
from eval_waft import TAPVID3D_ROOT, get_full_eval_files  # noqa: E402

HF_BASE = "https://huggingface.co/datasets/ZhengGuangze/TAPVid-3D/resolve/main"
BATCHES = {"drivetrack": 13, "adt": 8, "pstudio": 1}


def missing_clips(subset: str) -> list[str]:
    return [c for c in get_full_eval_files(subset) if not (TAPVID3D_ROOT / subset / c).exists()]


def fetch_from_tarballs(subset: str, wanted: set[str], out_dir: Path, reverse: bool = True,
                        batches: list[int] | None = None, attempts: int = 3) -> set[str]:
    """Stream each tarball, extracting any wanted member. Returns the names still missing.

    Batches are searched last-first by default. The local copy was populated by downloading
    batch 0..N in order, so an interrupted tail leaves its gap in the final archives, not the
    early ones -- the extraction-time histogram of the surviving clips shows the last hour
    producing 195 files against 391-647 in the hours before it. Searching forward means
    streaming almost the whole 309 GB before reaching the batch that actually holds the clip.
    """
    order = batches if batches else (
        range(BATCHES[subset] - 1, -1, -1) if reverse else range(BATCHES[subset]))
    for i in order:
        if not wanted:
            break
        url = f"{HF_BASE}/{subset}_batch_{i}.tar.gz"
        for attempt in range(1, attempts + 1):
            if not wanted:
                break
            print(f"  [batch {i}] streaming {url} (attempt {attempt}/{attempts})", flush=True)
            found_before = len(wanted)
            try:
                with requests.get(url, stream=True, timeout=120) as r:
                    r.raise_for_status()
                    # "r|gz" is streaming mode: sequential, no seeking, so the transfer can be
                    # abandoned the moment the member is found rather than read to the end.
                    with tarfile.open(fileobj=r.raw, mode="r|gz") as tar:
                        for member in tar:
                            name = Path(member.name).name
                            if name not in wanted:
                                continue
                            src = tar.extractfile(member)
                            if src is None:
                                continue
                            out_dir.mkdir(parents=True, exist_ok=True)
                            dest = out_dir / name
                            tmp = dest.with_suffix(dest.suffix + ".part")
                            with tmp.open("wb") as fh:
                                while chunk := src.read(1 << 20):
                                    fh.write(chunk)
                            tmp.rename(dest)
                            wanted.discard(name)
                            print(f"  [batch {i}] recovered {name} "
                                  f"({dest.stat().st_size/2**20:.1f} MB)", flush=True)
                            if not wanted:
                                break
                break  # archive read to the end without error; no retry needed
            except Exception as e:
                # A truncated transfer means this batch was NOT fully searched, so "absent" cannot
                # be concluded from it. Batch 8 died at 17.4 GB of 25.5 GB on the first sweep and
                # was silently treated as searched, which is why this retries instead of moving on.
                print(f"  [batch {i}] attempt {attempt} aborted: {type(e).__name__}: "
                      f"{str(e)[:80]}", flush=True)
                if len(wanted) < found_before:
                    break
    return wanted


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subsets", nargs="+", default=["drivetrack", "pstudio", "adt"])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--forward", action="store_true",
                    help="search batches 0..N instead of the default last-first order")
    ap.add_argument("--batches", type=int, nargs="+",
                    help="search only these batch numbers, in the order given")
    ap.add_argument("--attempts", type=int, default=3,
                    help="retries per batch; a truncated transfer leaves it unsearched")
    args = ap.parse_args()

    rc = 0
    for subset in args.subsets:
        miss = missing_clips(subset)
        print(f"{subset}: {len(miss)} missing")
        for c in miss:
            print(f"    {c}")
        if not miss or args.dry_run:
            continue
        left = fetch_from_tarballs(subset, set(miss), TAPVID3D_ROOT / subset,
                                   reverse=not args.forward, batches=args.batches,
                                   attempts=args.attempts)
        if left:
            print(f"{subset}: STILL MISSING {len(left)}: {sorted(left)}")
            rc = 1
        else:
            print(f"{subset}: complete")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
