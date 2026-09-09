"""v91 training and validation curves, with the measured metric-AJ overlaid.

Separate from plot_v90_curves.py because v91 is one run rather than three stitched together, and
because the question it answers is different: whether the loss is still falling at the last step,
i.e. whether more training is worth buying.

Every number plotted is read back from disk -- the scored eval directories and the run log. An
earlier version carried the metric-AJ values inline and was stale within the hour, which is the
same failure the report bars had.
"""

import importlib.util
import json
import pathlib
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RUN = "v91"
LOG = pathlib.Path("result/v73_plan.log")
OUT = pathlib.Path(f"result/analysis/{RUN}_curve.png")
STEP = re.compile(r"\[train\] step\s+(\d+)/\d+\s+loss=([\d.eE+-]+)")
VAL = re.compile(r"\[train\] step\s+(\d+)\s+VAL\s+loss=([\d.eE+-]+)")


def bars() -> tuple[dict, float]:
    """The comparison bars, from the module that owns them rather than retyped here."""
    spec = importlib.util.spec_from_file_location("r", "scripts/report_metric_aj.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.BARS, mod.NOISE


def scored() -> dict[int, float]:
    """{step: metric-AJ} for every eval directory this run has produced.

    eval_<step> is written by the checkpoint watcher; eval_final is the last checkpoint, whose
    step is recovered from the run's configured total rather than assumed.
    """
    out = {}
    for d in sorted(pathlib.Path(f"result/{RUN}").glob("eval_*")):
        f = d / "metrics.json"
        if not f.exists():
            continue
        aj = json.loads(f.read_text())["overall"]["metric_average_jaccard"]
        tag = d.name.removeprefix("eval_")
        if tag.isdigit():
            out[int(tag)] = aj
        else:
            ck = sorted(pathlib.Path(f"result/{RUN}").glob("ckpt_*.pt"),
                        key=lambda p: int(p.stem.split("_")[1]))
            if ck:
                out[int(ck[-1].stem.split("_")[1])] = aj
    return out


def running_median(xs: list[float], k: int = 7) -> list[float]:
    return [sorted(xs[max(0, i - k // 2): i + k // 2 + 1])[
        len(xs[max(0, i - k // 2): i + k // 2 + 1]) // 2] for i in range(len(xs))]


def main() -> int:
    text = LOG.read_text(errors="ignore").replace("\r", "\n")
    # rindex, not index: a run that was started, killed and relaunched writes its banner more
    # than once, and taking the first one splices the abandoned attempt onto the real curve.
    seg = text[text.rindex(f"### {RUN} ("):]
    ts, tl, vs, vl = [], [], [], []
    for ln in seg.split("\n"):
        if m := STEP.search(ln):
            ts.append(int(m[1]))
            tl.append(float(m[2]))
        elif m := VAL.search(ln):
            vs.append(int(m[1]))
            vl.append(float(m[2]))
    aj = scored()
    BARS, _ = bars()
    print(f"  {len(ts)} training points, {len(vs)} validations, {len(aj)} scored checkpoints")

    fig, ax = plt.subplots(2, 1, figsize=(10, 8))
    ax[0].plot(ts, tl, lw=0.7, alpha=0.35, color="#1b6ca8")
    ax[0].plot(ts, running_median(tl), lw=2.2, color="#1b6ca8", label="running median (7)")
    ax[0].set_yscale("log")
    ax[0].set_ylabel("training loss (log)")
    ax[0].set_title(f"{RUN} training loss -- batch=1, one clip per point", loc="left")
    ax[0].legend(fontsize=9)
    ax[0].grid(alpha=0.3)

    ax2 = ax[1].twinx()
    ax[1].plot(vs, vl, "o-", color="#2e8b57", lw=2, ms=6, label="held-out validation")
    for name, v in BARS.items():
        ax2.axhline(v, color="grey", ls="--", lw=1.1)
        ax2.annotate(f"{name} {v:.4f}", (0, v), fontsize=8, color="grey", va="bottom")
    if aj:
        ax2.scatter(list(aj), list(aj.values()), s=110, marker="D", color="darkorange", zorder=5,
                    label="metric-AJ (measured)")
        for k, v in aj.items():
            ax2.annotate(f"{v:.4f}", (k, v), textcoords="offset points", xytext=(8, -3),
                         fontsize=9, color="darkorange")
    ax[1].set_xlabel("step")
    ax[1].set_ylabel("held-out validation loss", color="#2e8b57")
    ax2.set_ylabel("metric-AJ", color="darkorange")
    ax[1].set_title("Validation and metric-AJ -- is anything still falling?", loc="left")
    ax[1].legend(fontsize=9, loc="upper right")
    ax[1].grid(alpha=0.3)

    fig.tight_layout()
    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT, dpi=130)
    print(f"  wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
