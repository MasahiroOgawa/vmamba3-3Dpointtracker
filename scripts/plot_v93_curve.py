"""v93 + v93b as one trajectory: loss, validation, and the only signal that has tracked reality.

v93b resumed v93 with its optimiser and scheduler state, so the two logs are one run. The point
of the figure is the contrast: the training loss and the held-out validation are both flat across
8297 steps while metric-AJ rises +0.0086. On this arm validation has mispredicted metric-AJ 7
times out of 7, so the scored checkpoints are plotted separately and are what the extrapolation
uses.
"""

from __future__ import annotations

import pathlib
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

STEP = re.compile(r"^\[train\] step\s+(\d+)/\S+\s+mean\s*loss=([\d.]+)")
VAL = re.compile(r"^\[train\] step\s+(\d+)\s+VAL\s+loss=([\d.]+)")
OUT = pathlib.Path("result/analysis/v93_curve.png")
SCORED = {  # step -> metric-AJ, measured
    0: 0.21886, 3191: 0.22531, 5500: 0.22310, 8297: 0.22744,
}
DELTA = 0.22737


def read(path: str, marker: str) -> tuple[list, list, list, list]:
    text = pathlib.Path(path).read_text(errors="ignore").replace("\r", "\n")
    seg = text[text.rindex(marker):] if marker in text else text
    ts, tl, vs, vl = [], [], [], []
    for ln in seg.split("\n"):
        if m := STEP.match(ln):
            ts.append(int(m[1]))
            tl.append(float(m[2]))
        elif m := VAL.match(ln):
            vs.append(int(m[1]))
            vl.append(float(m[2]))
    return ts, tl, vs, vl


def main() -> int:
    a = read("result/v93.log", "[train] config configs/v93.yaml")
    b = read("result/v93b.log", "[train] config configs/v93b.yaml")
    ts, tl = a[0] + b[0], a[1] + b[1]
    vs, vl = a[2] + b[2], a[3] + b[3]
    print(f"  {len(ts)} loss points, {len(vs)} validations, {len(SCORED)} scored checkpoints")

    fig, ax = plt.subplots(3, 1, figsize=(11, 11), sharex=True)
    ax[0].plot(ts, tl, lw=0.7, alpha=0.4, color="#1b6ca8")
    k = 9
    med = [sorted(tl[max(0, i - k // 2):i + k // 2 + 1])[
        len(tl[max(0, i - k // 2):i + k // 2 + 1]) // 2] for i in range(len(tl))]
    ax[0].plot(ts, med, lw=2, color="#1b6ca8", label="running median (9)")
    ax[0].set_yscale("log")
    ax[0].set_ylabel("training loss")
    ax[0].set_title("Training loss — flat throughout, and it was never the signal", loc="left")
    ax[0].legend(fontsize=9)
    ax[0].grid(alpha=0.3)

    ax[1].plot(vs, vl, "o-", color="#2e8b57", lw=1.8, ms=4)
    ax[1].set_ylabel("held-out validation")
    ax[1].set_title("Held-out validation — also flat; it picked ckpt_5500, the worst of the three",
                    loc="left")
    ax[1].grid(alpha=0.3)

    xs = sorted(SCORED)
    ys = [SCORED[x] for x in xs]
    ax[2].axhline(DELTA, color="crimson", ls="--", lw=1.6)
    ax[2].annotate(f"DELTA {DELTA:.5f}", (100, DELTA), fontsize=9, color="crimson", va="bottom")
    ax[2].plot(xs, ys, "D-", color="darkorange", lw=2, ms=9)
    for x, y in zip(xs, ys):
        ax[2].annotate(f"{y:.5f}", (x, y), textcoords="offset points", xytext=(6, -12),
                       fontsize=9, color="darkorange")
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    den = sum((x - mx) ** 2 for x in xs)
    sl = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den
    res = [y - (my + sl * (x - mx)) for x, y in zip(xs, ys)]
    se = (sum(r * r for r in res) / (n - 2) / den) ** 0.5
    ax[2].plot([0, 12000], [my + sl * (0 - mx), my + sl * (12000 - mx)], ls=":", color="grey",
               label=f"fit {1000*sl:+.5f}/1000 steps  ({abs(sl)/se:.1f}$\\sigma$)")
    ax[2].set_ylabel("metric-AJ")
    ax[2].set_xlabel("optimiser step")
    ax[2].set_title("Metric-AJ on scored checkpoints — the only signal that moved", loc="left")
    ax[2].legend(fontsize=9, loc="lower right")
    ax[2].grid(alpha=0.3)

    fig.tight_layout()
    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT, dpi=130)
    print(f"  metric-AJ trend {1000*sl:+.5f} +/- {1000*se:.5f} per 1000 steps -> {abs(sl)/se:.1f} sigma")
    print(f"  wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
