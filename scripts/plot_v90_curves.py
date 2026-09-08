"""Plot the v90 line's training and validation curves against metric-AJ.

Written to answer why training lands below its own initial weights: the loss is read from the run
log rather than from a summary, so a flat or noise-dominated optimisation is visible directly.
"""

import re
import pathlib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

LOG = pathlib.Path("result/v73_plan.log")
STEP = re.compile(
    r"\[train\] step\s+(\d+)/(\d+)\s+loss=([\d.eE+-]+).*?\|grad\|=([\d.eE+-]+)"
)
MODS = ("embed", "layers", "dz_head", "duv_head")
MOD = {m: re.compile(rf"{m}=([\d.eE+-]+)") for m in MODS}
VAL = re.compile(r"\[train\] step\s+(\d+)\s+VAL\s+loss=([\d.eE+-]+)")
# (arm, first log line, metric-AJ of the checkpoint that arm produced)
ARMS = [("v90a", 410, 4717), ("v90b", 4717, 4817), ("v90c", 4817, 10**9)]
AJ = {  # step -> (arm, metric-AJ), all measured on minival, DA3-g
    ("v90a", 500): 0.2132, ("v90a", 2000): 0.2154,
    ("v90b", 2000): 0.2164,
    ("v90c", 3000): 0.2151, ("v90c", 3600): 0.2169,
}
WARM_AJ, WARM_VAL = 0.2191, 0.2155

lines = LOG.read_text(errors="ignore").replace("\r", "\n").split("\n")
runs = {}
for arm, lo, hi in ARMS:
    tr_s, tr_l, tr_g, va_s, va_l = [], [], [], [], []
    mods = {m: [] for m in MODS}
    for ln in lines[lo - 1 : min(hi - 1, len(lines))]:
        if m := STEP.search(ln):
            tr_s.append(int(m[1])); tr_l.append(float(m[3])); tr_g.append(float(m[4]))
            for k, rx in MOD.items():
                mm = rx.search(ln); mods[k].append(float(mm[1]) if mm else float("nan"))
        elif m := VAL.search(ln):
            va_s.append(int(m[1])); va_l.append(float(m[2]))
    runs[arm] = (tr_s, tr_l, tr_g, va_s, va_l, mods)
    print(f"  {arm}: {len(tr_s)} step lines, {len(va_s)} validations")

def running_median(xs, k=9):
    return [sorted(xs[max(0, i - k // 2) : i + k // 2 + 1])[len(xs[max(0, i - k // 2) : i + k // 2 + 1]) // 2]
            for i in range(len(xs))]

C = {"v90a": "#1b6ca8", "v90b": "#d1495b", "v90c": "#2e8b57"}
fig, ax = plt.subplots(4, 1, figsize=(11, 15), sharex=False)

# --- training loss, log scale: batch=1 makes this swing by orders of magnitude per clip
off = 0
for arm, _, _ in ARMS:
    s, l, *_ = runs[arm]
    if not s:
        continue
    x = [v + (off if arm == "v90a" else 0) for v in s]
    ax[0].plot(x, l, lw=0.6, alpha=0.30, color=C[arm])
    ax[0].plot(x, running_median(l, 9), lw=2.0, color=C[arm], label=f"{arm} (running median, 9)")
ax[0].set_yscale("log"); ax[0].set_ylabel("training loss (log)")
ax[0].set_title("Training loss — batch=1, so each point is one clip", loc="left")
ax[0].legend(fontsize=9); ax[0].grid(alpha=0.3)

# --- gradient norm against the clip threshold
for arm, _, _ in ARMS:
    s, _, g, *_ = runs[arm]
    if s:
        ax[1].plot(s, g, lw=0.7, alpha=0.55, color=C[arm], label=arm)
ax[1].axhline(1.0, color="k", ls="--", lw=1.5, label="grad_clip = 1.0")
ax[1].set_yscale("log"); ax[1].set_ylabel("|grad| before clipping (log)")
ax[1].set_title("Gradient norm vs the clip threshold", loc="left")
ax[1].legend(fontsize=9, ncol=4); ax[1].grid(alpha=0.3)

# --- per-module gradient: which parameter group consumes the global clip budget
MC = {"embed": "#7570b3", "layers": "#66a61e", "dz_head": "#e6194b", "duv_head": "#999999"}
xoff = 0
for arm, _, _ in ARMS:
    s_, _, _, _, _, mods = runs[arm]
    if not s_:
        continue
    for k in MODS:
        ax[2].plot(s_, mods[k], lw=1.4 if k == "dz_head" else 0.9,
                   alpha=0.95 if k == "dz_head" else 0.55, color=MC[k],
                   label=k if arm == "v90a" else None)
ax[2].axhline(1.0, color="k", ls="--", lw=1.5, label="grad_clip = 1.0 (global)")
ax[2].axvline(2000, color="k", lw=0.8, alpha=0.5)
ax[2].annotate("scale refiner UNFROZEN from here (v90b/c)", (2020, 20), fontsize=9)
ax[2].set_yscale("log"); ax[2].set_ylabel("per-module |grad| (log)")
ax[2].set_title("dz_head dominates the global clip budget once the refiner is unfrozen", loc="left")
ax[2].legend(fontsize=9, ncol=5); ax[2].grid(alpha=0.3)

# --- validation loss and metric-AJ on one axis pair
ax2 = ax[3].twinx()
ax[3].axhline(WARM_VAL, color="grey", ls=":", lw=1.5)
ax[3].annotate("untrained warm start: val 0.2155", (0, WARM_VAL), fontsize=8,
               color="grey", va="bottom")
ax2.axhline(WARM_AJ, color="darkorange", ls=":", lw=2.0)
ax2.annotate("untrained warm start: metric-AJ 0.2191  ← nothing beats this",
             (0, WARM_AJ), fontsize=9, color="darkorange", va="bottom")
for arm, _, _ in ARMS:
    _, _, _, vs, vl, _ = runs[arm]
    if vs:
        ax[3].plot(vs, vl, "o-", color=C[arm], lw=2, ms=5, label=f"{arm} val")
xs = [s for (a, s) in AJ]; ys = [AJ[k] for k in AJ]
ax2.scatter(xs, ys, s=90, marker="D", color="darkorange", zorder=5, label="metric-AJ (measured)")
for (a, s), v in AJ.items():
    ax2.annotate(f"{v:.4f}", (s, v), textcoords="offset points", xytext=(6, -3),
                 fontsize=8, color="darkorange")
ax[3].set_xlabel("step"); ax[3].set_ylabel("held-out validation loss")
ax2.set_ylabel("metric-AJ", color="darkorange")
ax[3].set_title("Validation loss falls; metric-AJ never regains the untrained weights", loc="left")
ax[3].legend(fontsize=9, loc="upper right"); ax[3].grid(alpha=0.3)

fig.tight_layout()
out = pathlib.Path("result/analysis/v90_curves.png")
fig.savefig(out, dpi=130)
print(f"  wrote {out}")
