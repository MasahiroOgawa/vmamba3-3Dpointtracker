"""Cache forward/backward flow at each tracked point -- the v94 visibility head's only input.

Usage: uv run python scripts/cache_flow_vis.py configs/v94_cache.yaml

The head is deliberately independent of the depth refiner, so this cache holds no trunk
feature, no depth and no forward-backward mask: only the two flow vectors and the
ground-truth visibility to train against. The flow mask is stored too, but as the
baseline to beat, never as an input.
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from mamba3_tracker.data.tapvid3d_splits import MINIVAL_FILES  # noqa: E402
from searaft_flow import track_clip_with_flow  # noqa: E402


def decode_images(jpeg_bytes) -> np.ndarray:
    import io

    from PIL import Image

    return np.stack(
        [np.asarray(Image.open(io.BytesIO(b)).convert("RGB")) for b in jpeg_bytes]
    ).transpose(0, 3, 1, 2).astype(np.float32) / 255.0


def build_waft(device, scale, iters):
    """WAFT exactly as the v93 trainer builds it; eval_waft chdirs, so restore the cwd."""
    import importlib

    cwd = os.getcwd()
    try:
        ew = importlib.import_module("eval_waft")
        return ew.build_flow(
            ew.WAFT_ROOT / "config" / "a1" / "tar-c-t.json",
            ew.WAFT_ROOT / "ckpts" / "waft_a1_recommended.pth",
            device,
            scale=scale,
            iters=iters,
        )
    finally:
        os.chdir(cwd)


def interleave(per_subset):
    """Round-robin over subsets. The trainer samples uniformly over subsets, so a cache that
    fills one subset before starting the next is unusable until the last one lands."""
    its = {s: iter(names) for s, names in per_subset.items()}
    while its:
        for s in list(its):
            name = next(its[s], None)
            if name is None:
                del its[s]
            else:
                yield s, name


def clip_lists(cfg):
    """minival is fixed; train/heldout are drawn from everything else, deterministically."""
    root = Path(str(cfg["data"]["root"])).expanduser()
    c = cfg["cache"]
    rng = random.Random(int(c["seed"]))
    out = {}
    for subset in c["subsets"]:
        mini = set(MINIVAL_FILES[subset])
        rest = sorted(p.name for p in (root / subset).glob("*.npz") if p.name not in mini)
        rng.shuffle(rest)
        n_tr, n_ho = int(c["n_train_per_subset"]), int(c["n_heldout_per_subset"])
        if bool(c.get("heldout_from_end", False)):
            # Taking held-out from the end makes it independent of how many clips training
            # asks for. With the default (held-out immediately after train) a large n_train
            # walks past the end of the list and pstudio -- 106 clips outside minival, against
            # adt's 1907 -- is left with no held-out set at all.
            out.setdefault("heldout", {})[subset] = rest[len(rest) - n_ho :]
            out.setdefault("train", {})[subset] = rest[: len(rest) - n_ho][:n_tr]
        else:
            out.setdefault("train", {})[subset] = rest[:n_tr]
            out.setdefault("heldout", {})[subset] = rest[n_tr : n_tr + n_ho]
        n_mv = c.get("n_minival_per_subset")
        out.setdefault("minival", {})[subset] = (
            sorted(mini) if n_mv is None else sorted(mini)[: int(n_mv)]
        )
    return out


@torch.no_grad()
def run_clip(flow_model, data, image_size, fb_alpha, fb_beta, device):
    imgs = decode_images(data["images_jpeg_bytes"])
    F_, _, H, W = imgs.shape
    images = torch.from_numpy(imgs)
    if (H, W) != (image_size, image_size):
        images = F.interpolate(
            images, size=(image_size, image_size), mode="bilinear", align_corners=False
        )
    # Scaled in place and freed eagerly. A 300-frame clip at 896^2 is 2.7 GB as float32, and
    # holding the decoded, the resized and the scaled copies at once drove peak RSS to 16 GB.
    # In-place multiply is the same arithmetic, so the cache stays consistent across restarts.
    images_255 = images.mul_(255.0).to(device)
    del imgs, images
    sx, sy = image_size / float(W), image_size / float(H)
    q = data["queries_xyt"].astype(np.float32)
    queries_xy = torch.from_numpy(np.stack([q[:, 0] * sx, q[:, 1] * sy], -1)).float()
    anchor_t = torch.from_numpy(q[:, 2]).long().clamp(0, F_ - 1)
    return track_clip_with_flow(
        flow_model, images_255, queries_xy, anchor_t, image_size, fb_alpha, fb_beta,
        bidirectional=False,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config", type=Path)
    cfg = yaml.safe_load(ap.parse_args().config.read_text())

    c, d, fl = cfg["cache"], cfg["data"], cfg["flow"]
    root = Path(str(d["root"])).expanduser()
    out_root = Path(str(c["out_dir"])).expanduser()
    verify = Path(str(c["verify_against"])).expanduser() if c.get("verify_against") else None
    image_size = int(d["image_size"])
    fb_alpha, fb_beta = float(fl["fb_alpha"]), float(fl["fb_beta"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "source.json").write_text(json.dumps(
        {"flow": fl, "image_size": image_size, "data_root": str(root)}, indent=2))

    print(f"[cache] building WAFT (scale={fl['scale']} iters={fl['iters']})", flush=True)
    flow_model = build_waft(device, fl.get("scale"), fl.get("iters"))

    splits = clip_lists(cfg)
    # The split is defined here and nowhere else. Without this manifest the trainer has to
    # infer it by listing the directory, and a clip left behind by an earlier config -- v94
    # held out five per subset that v95 trains on -- silently becomes a train/test leak.
    (out_root / "splits.json").write_text(json.dumps(splits, indent=2))
    total = sum(len(v) for s in splits.values() for v in s.values())
    print(f"[cache] {total} clips: "
          + "  ".join(f"{k}={sum(len(v) for v in s.values())}" for k, s in splits.items()),
          flush=True)

    budget = float(c["time_budget_s"])
    done = mismatch = 0
    t0 = time.time()
    for split, per_subset in splits.items():
        for subset, name in interleave(per_subset):
            od = out_root / split / subset
            od.mkdir(parents=True, exist_ok=True)
            op = od / name
            if op.exists():
                done += 1
                continue
            # Stopping on a clock, not a clip count: the cache is the long pole (adt clips
            # cost ~177 s each) and the interleaved order means whatever has landed by the
            # deadline is already balanced across subsets, hence trainable as it stands.
            if budget > 0 and time.time() - t0 > budget:
                print(f"[cache] BUDGET {budget:.0f}s reached at {done}/{total}", flush=True)
                break
            with np.load(root / subset / name, allow_pickle=True) as data:
                uv, vis, ff, fb = run_clip(
                    flow_model, data, image_size, fb_alpha, fb_beta, device)
                vis_gt = np.asarray(data["visibility"]).astype(np.uint8)
            # The mask this reproduces must be the one v93b was scored with, or the head
            # would be trained against a front-end the refiner never saw.
            if verify is not None and split == "minival":
                vp = verify / subset / name
                if vp.exists():
                    with np.load(vp) as wd:
                        ref = np.asarray(wd["visibility"])[: vis.shape[0]] > 0.5
                    if not np.array_equal(ref, vis.numpy() > 0.5):
                        mismatch += 1
                        print(f"[cache] MASK MISMATCH {subset}/{name}: "
                              f"{int((ref != (vis.numpy() > 0.5)).sum())} of {ref.size}",
                              flush=True)
            # Written aside and renamed: a kill mid-write leaves a truncated .npz that the
            # resume check would skip forever, since it only tests existence.
            tmp = od / (name + ".partial")
            with open(tmp, "wb") as fh:
                np.savez_compressed(
                    fh,
                    flow_fwd=ff.numpy().astype(np.float32),
                    flow_bwd=fb.numpy().astype(np.float32),
                    uv=uv.numpy().astype(np.float32),
                    vis_flow=(vis.numpy() > 0.5).astype(np.uint8),
                    vis_gt=vis_gt[: vis.shape[0]],
                )
            os.replace(tmp, op)
            done += 1
            el = time.time() - t0
            print(f"[cache] {done}/{total} {split}/{subset}/{name}  "
                  f"{el:.0f}s elapsed, eta {el / max(done, 1) * (total - done) / 60:.0f} min",
                  flush=True)
        if budget > 0 and time.time() - t0 > budget:
            break
    for split, per_subset in splits.items():
        for subset in per_subset:
            d = out_root / split / subset
            print(f"[cache] cached {split}/{subset}: "
                  f"{len(list(d.glob('*.npz'))) if d.exists() else 0}", flush=True)
    print(f"[cache] DONE {done}/{total} clips, mask mismatches: {mismatch}", flush=True)


if __name__ == "__main__":
    main()
