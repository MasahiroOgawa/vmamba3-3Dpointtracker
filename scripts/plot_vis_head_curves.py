"""Held-out accuracy against training step for the visibility-head data-scaling probes.

Usage: uv run python scripts/plot_vis_head_curves.py configs/plot_vis_head_curves.yaml
"""

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mamba3_tracker.train.schedule import wsd  # noqa: E402

VAL = re.compile(r"step\s+(\d+)\s+VAL.*?mean head (\d\.\d+) mask (\d\.\d+)")
LOSS = re.compile(r"step\s+(\d+)\s+loss (\d\.\d+)")


def read_curve(log: Path):
    rows = [(int(s), float(h), float(m)) for s, h, m in VAL.findall(log.read_text())]
    return [r[0] for r in rows], [r[1] for r in rows], rows[0][2]


def read_loss(log: Path):
    rows = [(int(s), float(v)) for s, v in LOSS.findall(log.read_text())]
    return [r[0] for r in rows], [r[1] for r in rows]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config", type=Path)
    p = yaml.safe_load(ap.parse_args().config.read_text())["plot"]

    fig, (ax, axt, axl) = plt.subplots(
        3, 1, figsize=(9, 9.0), sharex=True, gridspec_kw={"height_ratios": [3, 2, 1]})

    mask = None
    for r in p["runs"]:
        steps, head, mask = read_curve(Path(r["log"]))
        ax.plot(steps, head, "-o", ms=3.5, lw=1.6, color=r["color"], label=r["label"])
        ls, lv = read_loss(Path(r["log"]))
        axt.plot(ls, lv, lw=1.0, color=r["color"], alpha=0.85, label=r["label"])

    ref = json.loads(Path(p["reference"]).read_text())
    ax.axhline(ref["mean_head"], ls="--", lw=1.6, color="#2ca02c",
               label=f"{p['reference_label']}: {ref['mean_head']:.4f}")
    ax.axhline(mask, ls=":", lw=1.8, color="0.35",
               label=f"flow mask (incumbent): {mask:.4f}")

    ax.set_ylabel("held-out mean accuracy", fontsize=12)
    ax.legend(fontsize=11, loc="lower right", ncol=2)
    ax.grid(alpha=0.3)
    ax.set_title("Visibility head: held-out accuracy, training loss, and the schedule "
                 "that produced them", fontsize=12)

    axt.set_ylabel("training loss", fontsize=12)
    axt.legend(fontsize=11, loc="upper right")
    axt.grid(alpha=0.3)

    for r in p["runs"]:
        sc = r["schedule"]
        total, lr = int(sc["steps"]), float(sc["lr"])
        xs = list(range(1, total + 1, 50))
        axl.plot(xs, [lr * wsd(x, int(sc["warmup"]), int(sc["decay"]), total) for x in xs],
                 lw=1.8, color=r["color"], alpha=0.85)
    axl.set_xlabel("training step", fontsize=12)
    axl.set_ylabel("learning rate", fontsize=12)
    axl.grid(alpha=0.3)
    for a in (ax, axt, axl):
        a.tick_params(labelsize=11)

    out = Path(p["out"])
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
