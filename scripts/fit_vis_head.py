"""Fit v35.vis_head on cached trunk features, and score it against the flow mask.

The trunk is frozen, so its per-point feature is a constant of the data and the head is just an
MLP over a fixed array. That makes this a minutes-long fit rather than a GPU run, and it makes
the interesting comparisons -- one joint head against three per-subset heads, class weighting,
capacity -- cheap enough to actually run.

The number that matters is not the BCE but whether the head agrees with ground-truth visibility
MORE OFTEN than the flow mask does, on clips it never saw. The flow mask is wrong on 20-23% of
observations, and only a head that beats that is worth putting in front of the evaluator.
"""

from __future__ import annotations

import argparse
import pathlib

import numpy as np
import torch
import torch.nn as nn


def load_cache(d: pathlib.Path, holdout_frac: float, seed: int):
    """-> {subset: (train_x, train_y, hold_x, hold_y, hold_flow)}, split by CLIP not observation."""
    by_sub: dict[str, list] = {}
    for f in sorted(d.glob("*.npz")):
        by_sub.setdefault(f.name.split("__", 1)[0], []).append(f)
    rng = np.random.default_rng(seed)
    out = {}
    for sub, files in sorted(by_sub.items()):
        files = list(files)
        rng.shuffle(files)
        n_hold = max(1, int(len(files) * holdout_frac))
        parts = {"hold": files[:n_hold], "train": files[n_hold:]}
        packed = {}
        for tag, fs in parts.items():
            xs, ys, fv = [], [], []
            for f in fs:
                z = np.load(f, allow_pickle=True)
                m = np.broadcast_to(z["qmask"][None, :], z["gt_vis"].shape)
                xs.append(z["x"][m])
                ys.append(z["gt_vis"][m])
                fv.append(z["flow_vis"][m])
            packed[tag] = (np.concatenate(xs), np.concatenate(ys), np.concatenate(fv))
        out[sub] = packed
        print(f"  {sub:<12} train {len(packed['train'][0]):>9,} obs ({len(parts['train'])} clips)"
              f"   holdout {len(packed['hold'][0]):>9,} obs ({len(parts['hold'])} clips)", flush=True)
    return out


def fit(x, y, dim, steps, lr, pos_weight, device, batch=131072, seed=0):
    torch.manual_seed(seed)
    head = nn.Sequential(nn.Linear(dim, 64), nn.GELU(), nn.Linear(64, 1)).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=0.01)
    xt = torch.from_numpy(x).to(device, torch.float32)
    yt = torch.from_numpy(y).to(device, torch.float32)
    pw = torch.tensor([pos_weight], device=device)
    n = xt.shape[0]
    g = torch.Generator(device="cpu").manual_seed(seed)
    for s in range(steps):
        idx = torch.randint(0, n, (min(batch, n),), generator=g).to(device)
        loss = nn.functional.binary_cross_entropy_with_logits(
            head(xt[idx]).squeeze(-1), yt[idx], pos_weight=pw)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if s % max(1, steps // 5) == 0:
            print(f"    step {s:>5}  bce {loss.item():.4f}", flush=True)
    return head


@torch.no_grad()
def score(head, x, y, flow, device, batch=262144):
    xt = torch.from_numpy(x)
    pred = np.empty(len(y), dtype=bool)
    for i in range(0, len(y), batch):
        chunk = xt[i:i + batch].to(device, torch.float32)
        pred[i:i + batch] = (head(chunk).squeeze(-1) > 0).cpu().numpy()
    return (pred == y).mean(), (flow == y).mean(), pred.mean()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="result/vis_feature_cache")
    ap.add_argument("--out", default="result/vis_head")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--pos-weight", type=float, default=1.0)
    ap.add_argument("--holdout-frac", type=float, default=0.25)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_cache(pathlib.Path(args.cache), args.holdout_frac, seed=0)
    dim = next(iter(data.values()))["train"][0].shape[1]
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print("\n  === joint head, all subsets ===", flush=True)
    jx = np.concatenate([d["train"][0] for d in data.values()])
    jy = np.concatenate([d["train"][1] for d in data.values()])
    joint = fit(jx, jy, dim, args.steps, args.lr, args.pos_weight, device)
    torch.save(joint.state_dict(), out / "joint.pt")

    print(f"\n  {'subset':<12}{'flow acc':>10}{'joint acc':>11}{'per-sub acc':>13}"
          f"{'head says vis':>15}", flush=True)
    rows = []
    for sub, d in data.items():
        hx, hy, hf = d["hold"]
        ja, fa, jm = score(joint, hx, hy, hf, device)
        per = fit(d["train"][0], d["train"][1], dim, args.steps, args.lr,
                  args.pos_weight, device)
        torch.save(per.state_dict(), out / f"{sub}.pt")
        pa, _, _ = score(per, hx, hy, hf, device)
        rows.append((sub, fa, ja, pa, jm))
    print()
    for sub, fa, ja, pa, jm in rows:
        print(f"  {sub:<12}{fa:>10.4f}{ja:>11.4f}{pa:>13.4f}{jm:>15.4f}", flush=True)
    print("\n  a head only helps the tracker if its accuracy beats the flow mask's", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
