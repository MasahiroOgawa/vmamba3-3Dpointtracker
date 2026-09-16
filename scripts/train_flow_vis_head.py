"""Train the v94 visibility head on cached forward/backward flow.

Usage: uv run python scripts/train_flow_vis_head.py configs/v94.yaml

The head never sees the refiner, so this trains standalone in minutes rather than
requiring the 8.5 h refiner run. Validation reports the head's per-frame accuracy beside
the forward-backward mask's on the same held-out clips: the mask is the incumbent, and a
head that does not beat it here cannot help downstream.
"""

import argparse
import json
import random
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mamba3_tracker.model.flow_vis_head import FlowVisHead  # noqa: E402
from mamba3_tracker.train.schedule import wsd  # noqa: E402


class ClipStore:
    """A sequence of clips loaded from disk on use, with a bounded number kept resident.

    Holding the whole pool would need ~12 GB against 13 GB of free RAM: one adt clip is
    300 frames x 838 points x 4 float32 arrays, and there are 1856 of them. Indexing and
    iteration are the list protocol, so random.choice and `for c in store` work unchanged.
    """

    def __init__(self, paths, max_frames: int, resident: int | None):
        self.paths = list(paths)
        self.max_frames = max_frames
        self.resident = len(self.paths) if resident is None else int(resident)
        self._lru: OrderedDict = OrderedDict()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        p = self.paths[i]
        if p in self._lru:
            self._lru.move_to_end(p)
            return self._lru[p]
        with np.load(p) as d:
            F_ = min(int(d["flow_fwd"].shape[0]), self.max_frames)
            clip = {
                "flow_fwd": d["flow_fwd"][:F_],
                "flow_bwd": d["flow_bwd"][:F_],
                "vis_gt": d["vis_gt"][:F_].astype(np.float32),
                "vis_flow": d["vis_flow"][:F_].astype(np.float32),
                "name": p.name,
            }
        self._lru[p] = clip
        while len(self._lru) > self.resident:
            self._lru.popitem(last=False)
        return clip


def load_split(cache_dir: Path, split: str, subsets, max_frames: int, resident: int | None):
    """-> {subset: ClipStore}. `resident` caps how many clips of each subset stay in RAM.

    Membership comes from the cache's own splits.json, never from listing the directory:
    the cache is shared between configs, and a clip one config held out may be another's
    training data. Intersected with what is on disk, since a budgeted cache stops early.
    """
    manifest = json.loads((cache_dir / "splits.json").read_text())[split]
    out = {}
    for ss in subsets:
        d = cache_dir / split / ss
        paths = [d / n for n in sorted(manifest[ss]) if (d / n).exists()]
        out[ss] = ClipStore(paths, max_frames, resident)
    return out


def batch_from(clip, num_points, rng, device):
    """One clip -> (flow_fwd, flow_bwd, vis_gt, vis_flow) as (1,F,n,*) tensors."""
    n_all = clip["flow_fwd"].shape[1]
    idx = (np.arange(n_all) if n_all <= num_points
           else rng.choice(n_all, num_points, replace=False))
    t = lambda k: torch.from_numpy(clip[k][:, idx]).unsqueeze(0).to(device)  # noqa: E731
    return t("flow_fwd"), t("flow_bwd"), t("vis_gt"), t("vis_flow")


@torch.no_grad()
def validate(model, data, device):
    """Head vs the incumbent flow mask, per subset, on whole held-out clips."""
    model.eval()
    rows = {}
    for ss, clips in data.items():
        h_ok = m_ok = tot = 0
        for c in clips:
            ff = torch.from_numpy(c["flow_fwd"]).unsqueeze(0).to(device)
            fb = torch.from_numpy(c["flow_bwd"]).unsqueeze(0).to(device)
            pred = (torch.sigmoid(model(ff, fb))[0] > 0.5).cpu().numpy()
            gt = c["vis_gt"] > 0.5
            h_ok += int((pred == gt).sum())
            m_ok += int(((c["vis_flow"] > 0.5) == gt).sum())
            tot += gt.size
        rows[ss] = (h_ok / tot, m_ok / tot)
    model.train()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config", type=Path)
    cfg_path = ap.parse_args().config
    cfg = yaml.safe_load(cfg_path.read_text())
    mc, dc, tc = cfg["model"], cfg["data"], cfg["train"]

    torch.manual_seed(int(tc["seed"]))
    rng = np.random.default_rng(int(tc["seed"]))
    pick = random.Random(int(tc["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache = Path(str(dc["cache_dir"])).expanduser()
    out_dir = Path(str(tc["out_dir"]))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.yaml").write_text(cfg_path.read_text())

    subsets = list(dc["subsets"])
    max_frames = int(dc["max_frames"])
    # Held-out is small (150 clips) and re-read at every validation, so it stays resident.
    train = load_split(cache, "train", subsets, max_frames, int(dc["clip_cache"]))
    held = load_split(cache, "heldout", subsets, max_frames, None)
    print(f"[v94] train {[len(train[s]) for s in subsets]} clips, "
          f"heldout {[len(held[s]) for s in subsets]} -- subsets {subsets}", flush=True)

    pw = tc["pos_weight"]
    if pw == "auto":
        pos = sum(float(c["vis_gt"].sum()) for s in subsets for c in train[s])
        neg = sum(float(c["vis_gt"].size) for s in subsets for c in train[s]) - pos
        pw = neg / max(pos, 1.0)
    pw = float(pw)
    print(f"[v94] pos_weight resolved to {pw:.4f}", flush=True)

    model = FlowVisHead(
        dim=int(mc["dim"]), state_dim=int(mc["state_dim"]),
        num_heads=int(mc["num_heads"]), num_layers=int(mc["num_layers"]),
        bidirectional=bool(mc["bidirectional"]),
    ).to(device)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"[v94] FlowVisHead {n_par} params, bidirectional={mc['bidirectional']}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=float(tc["lr"]))
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pw, device=device))
    steps = int(tc["steps"])
    best = (-1.0, -1)
    run_loss, run_n = 0.0, 0

    for step in range(1, steps + 1):
        for g in opt.param_groups:
            g["lr"] = float(tc["lr"]) * wsd(step, int(tc["warmup"]), int(tc["decay"]), steps)
        ss = pick.choice(subsets)          # uniform over subsets, not over clips:
        clip = pick.choice(train[ss])      # pstudio has far fewer clips than adt
        ff, fb, vg, _ = batch_from(clip, int(dc["num_points"]), rng, device)
        loss = lossf(model(ff, fb), vg)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), float(tc["grad_clip"]))
        opt.step()
        run_loss += float(loss.detach())
        run_n += 1

        if step % int(tc["log_every"]) == 0:
            print(f"[v94] step {step:5d}  loss {run_loss / run_n:.4f}  "
                  f"lr {opt.param_groups[0]['lr']:.2e}", flush=True)
            run_loss, run_n = 0.0, 0
        if step % int(tc["val_every"]) == 0 or step == steps:
            rows = validate(model, held, device)
            mean_h = sum(h for h, _ in rows.values()) / len(rows)
            mean_m = sum(m for _, m in rows.values()) / len(rows)
            txt = "  ".join(f"{s}: head {h:.4f} mask {m:.4f}" for s, (h, m) in rows.items())
            print(f"[v94] step {step:5d}  VAL  {txt}   mean head {mean_h:.4f} "
                  f"mask {mean_m:.4f}  delta {mean_h - mean_m:+.4f}", flush=True)
            if mean_h > best[0]:
                best = (mean_h, step)
                torch.save({"model": model.state_dict(), "step": step, "cfg": cfg},
                           out_dir / "best.pt")
                (out_dir / "best.json").write_text(json.dumps(
                    {"step": step, "mean_head": mean_h, "mean_mask": mean_m,
                     "per_subset": {s: {"head": h, "mask": m} for s, (h, m) in rows.items()}},
                    indent=2))
        if step % int(tc["ckpt_every"]) == 0:
            torch.save({"model": model.state_dict(), "step": step, "cfg": cfg},
                       out_dir / f"ckpt_{step}.pt")

    print(f"[v94] DONE best mean held-out accuracy {best[0]:.4f} at step {best[1]}", flush=True)


if __name__ == "__main__":
    main()
