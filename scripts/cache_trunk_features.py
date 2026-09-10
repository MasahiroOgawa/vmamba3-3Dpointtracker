"""Cache the frozen trunk's per-point features, so the visibility head can be fitted offline.

Only `v35.vis_head` is being trained, and it reads the trunk's per-point feature `x`. The trunk
is frozen, so `x` is a constant of the data: computing it once and reusing it turns an 85 s/step
GPU job into an MLP fit over a fixed array, and makes re-fitting the head -- class weights,
capacity, per-subset calibration -- cost minutes instead of a run each.

Two things this buys beyond speed:

  * Whole clips. Training used an 8-frame window, where 95% of observations are visible; the
    evaluator scores whole clips, where 54-71% are. v92 was fitted to the wrong distribution and
    its per-subset damage tracked that gap exactly. Caching at full clip length removes it.
  * A balanced mix. pstudio has 106 training clips against drivetrack's 2406, and adt clips are
    300 frames against drivetrack's 27-199, so sampling clips uniformly would fit the head almost
    entirely to drivetrack and adt. Selection is balanced on OBSERVATIONS, not clips.

The trunk features are captured with a forward hook on `vis_head`, whose input is exactly `x`;
that avoids duplicating the forward pass and cannot drift from it.

Flow comes from the cached WAFT tracks (waft_full_eval), which is what makes this affordable.
Note those tracks are computed from CLEAN images, so the photometric augmentation the live path
applies is absent -- fine for fitting a small head, but these features are not interchangeable
with live-path training.
"""

from __future__ import annotations

import argparse
import importlib.util
import pathlib
import sys

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def _load_trainer():
    """Import the training script as a module, to reuse its data and model paths verbatim."""
    spec = importlib.util.spec_from_file_location("tr", ROOT / "scripts" / "train_depth_refined_tracker.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="config supplying model, depth and flow")
    ap.add_argument("--out", default="result/vis_feature_cache")
    ap.add_argument("--obs-per-subset", type=int, default=4_000_000,
                    help="point-observations to collect per subset (F*N summed over clips)")
    ap.add_argument("--waft-pred-dir",
                    default="~/data/tapvid3d_baseline_preds/waft_full_eval",
                    help="cached WAFT tracks; the training configs run flow live, which costs "
                         "~85 s/clip and is what makes caching worthwhile in the first place")
    args = ap.parse_args()

    tr = _load_trainer()
    cfg = tr.load_config(args.config)
    data_cfg, model_cfg, train_cfg = cfg["data"], cfg["model"], cfg["train"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_size = int(data_cfg["image_size"])

    from mamba3_tracker.data.depth_source import resolve as resolve_depth
    depth_src = resolve_depth(data_cfg.get("depth") or data_cfg["da3_depth_root"])
    print(f"[cache] depth source: {depth_src}", flush=True)

    flow_cfg = cfg.get("flow", {})
    waft_dir = pathlib.Path(flow_cfg.get("waft_pred_dir", args.waft_pred_dir)).expanduser()
    if not waft_dir.is_dir():
        raise SystemExit(f"[cache] no cached WAFT tracks at {waft_dir}")
    print(f"[cache] cached WAFT tracks: {waft_dir}", flush=True)

    train_clips, _ = tr.official_train_test_split(subsets=data_cfg["subsets"])

    # Balance on observations, not clips: pstudio has 106 clips to drivetrack's 2406, and adt
    # clips are 300 frames to drivetrack's 27-199, so equal clip counts would still hand adt
    # several times the supervision.
    by_sub: dict[str, list] = {}
    for c in train_clips:
        by_sub.setdefault(c.parent.name, []).append(c)
    rng = np.random.default_rng(0)
    picked = []
    for sub, cl in sorted(by_sub.items()):
        rng.shuffle(cl)
        got, taken = 0, 0
        for c in cl:
            if got >= args.obs_per_subset:
                break
            picked.append(c)
            taken += 1
            got += _clip_obs(c, int(data_cfg["num_tracks"]))
        print(f"[cache] {sub:<11} {taken:>4} clips -> {got/1e6:.2f}M observations", flush=True)
    print(f"[cache] {len(picked)} clips total", flush=True)

    model = tr.Mamba3V73(
        dim=int(model_cfg["dim"]), state_dim=int(model_cfg["state_dim"]),
        num_heads=int(model_cfg["num_heads"]), num_layers=int(model_cfg["num_layers"]),
        max_scale_correction=float(model_cfg.get("max_scale_correction", 2.5)),
        two_pool=bool(model_cfg.get("two_pool", False)),
        grid=int(model_cfg.get("grid", 64)), log_ref=float(model_cfg.get("log_ref", 2.0)),
        log_std=float(model_cfg.get("log_std", 1.5)),
        max_log_correction=float(model_cfg["max_log_correction"]),
        max_delta_uv=float(model_cfg["max_delta_uv"]),
        patch_size=int(model_cfg["patch_size"]), d_proj=int(model_cfg["d_proj"]),
    ).to(device).eval()
    state = torch.load(pathlib.Path(train_cfg["init_ckpt"]).expanduser(),
                       map_location=device, weights_only=False)
    missing, unexpected = model.load_state_dict(state["model"], strict=False)
    print(f"[cache] loaded {train_cfg['init_ckpt']} (missing={len(missing)} "
          f"unexpected={len(unexpected)})", flush=True)

    # vis_head's input IS the per-point trunk feature, so a hook on it captures exactly what the
    # head will be fitted on -- no second copy of the forward pass to drift out of step.
    grabbed: list[torch.Tensor] = []
    model.v35.vis_head.register_forward_hook(lambda m, inp, out: grabbed.append(inp[0].detach()))

    flow_model = tr._build_waft_flow(device, scale=flow_cfg.get("scale"),
                                     iters=flow_cfg.get("iters"))
    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    ds = tr.TAPVid3DDataset(
        picked, window_size=10**6, augment=False, seed=0,
        max_queries=int(data_cfg["num_tracks"]), image_size=image_size,
        da3_depth_root=depth_src.root, reanchor_window=True)

    kept = 0
    for i in range(len(ds)):
        try:
            batch = tr.collate_tracking([ds[i]])
            grabbed.clear()
            with torch.no_grad():
                ray, z_raw, vis, uv, images, K = tr._run_flow_batch(
                    flow_model, batch, device, image_size,
                    float(flow_cfg.get("fb_alpha", 0.05)), float(flow_cfg.get("fb_beta", 1.0)),
                    waft_pred_dir=waft_dir)
                model(ray, z_raw, vis, uv, batch.depth.to(device), images, K)
            if not grabbed:
                print(f"[cache] {batch.clip_ids[0]}: hook captured nothing; skipped", flush=True)
                continue
            x = grabbed[0][0].to(torch.float16).cpu().numpy()      # (F,N,D)
            np.savez_compressed(
                out_dir / f"{batch.subsets[0]}__{batch.clip_ids[0]}.npz",
                x=x,
                gt_vis=batch.visibility[0].to(torch.bool).numpy(),
                flow_vis=(vis[0] > 0.5).cpu().numpy(),
                qmask=batch.query_mask[0].numpy(),
                subset=batch.subsets[0])
            kept += 1
            if kept % 20 == 0:
                print(f"[cache] {kept}/{len(ds)} clips", flush=True)
        except torch.OutOfMemoryError:
            print(f"[cache] OOM on clip {i}; skipped", flush=True)
            torch.cuda.empty_cache()
    print(f"[cache] wrote {kept} clips to {out_dir}", flush=True)
    return 0


def _clip_obs(path: pathlib.Path, max_queries: int) -> int:
    """Frames x queries actually cached, without decoding the clip's images.

    Counts min(N_q, max_queries): the dataset subsamples to max_queries, so using the clip's full
    query count over-counts. adt has ~824 queries against a cap of 256, so budgeting by the raw
    count gave adt a third of its intended share -- the subset the work is aimed at.
    """
    with np.load(path) as d:
        f, n = d["visibility"].shape
    return int(f) * min(int(n), max_queries)


if __name__ == "__main__":
    raise SystemExit(main())
