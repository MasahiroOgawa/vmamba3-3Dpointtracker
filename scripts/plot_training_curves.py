"""Render training/validation loss curves + motion-ratio plot for a tracker run.

Inputs (under `--run-dir`):
  loss_history.json    list of rows; train rows have {step, lr, total, <per-term>...};
                       val rows have {step, val: {total, <per-term>...}}.
  motion_history.json  optional; rows {step, <subset>_ratio, ...}. Target ratio = 1.0.

Outputs (written to `--run-dir/plots/`, per memory/feedback_output_dir_naming.md):
  training_curve.png   subplot per loss term (incl. total). Train + val overlaid.
  motion_ratio.png     per-subset motion ratio vs step, with y=1.0 reference.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _load_history(run_dir: Path) -> tuple[list[dict], list[dict]]:
    rows = json.loads((run_dir / "loss_history.json").read_text())
    train = [r for r in rows if "lr" in r]
    val = [{"step": r["step"], **r["val"]} for r in rows if "val" in r]
    return train, val


def _term_keys(train: list[dict]) -> list[str]:
    """Loss terms only — scalar columns, excluding meta and nested grad logs."""
    skip = {"step", "lr", "grad_norm", "head_grad", "term_grad", "val"}
    keys: set[str] = set()
    for r in train:
        for k, v in r.items():
            if k in skip:
                continue
            if isinstance(v, (int, float)):
                keys.add(k)
    ordered = ["total"] + sorted(k for k in keys if k != "total")
    return ordered


def plot_training_curve(run_dir: Path, out_path: Path) -> None:
    train, val = _load_history(run_dir)
    if not train:
        print(f"[plot] no train rows in {run_dir}/loss_history.json — skipping curve")
        return
    terms = _term_keys(train)
    n = len(terms)
    cols = 2 if n > 1 else 1
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 3 * rows), squeeze=False)
    train_steps = [r["step"] for r in train]
    val_steps = [r["step"] for r in val]
    for i, key in enumerate(terms):
        ax = axes[i // cols][i % cols]
        ty = [r.get(key) for r in train]
        ax.plot(train_steps, ty, label="train", linewidth=1, color="#1f77b4")
        if val:
            vy = [r.get(key) for r in val]
            if any(v is not None for v in vy):
                ax.plot(
                    val_steps,
                    vy,
                    label="val",
                    linewidth=1.4,
                    color="#d62728",
                    marker="o",
                    markersize=3,
                )
        ax.set_title(key)
        ax.set_xlabel("step")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    for j in range(n, rows * cols):
        axes[j // cols][j % cols].axis("off")
    fig.suptitle(f"{run_dir.name} — training curves", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"[plot] wrote {out_path}")


def plot_grad_curves(run_dir: Path, out_path: Path) -> None:
    """Per-head gradient L2 norms over training. Diagnoses which heads are
    actually learning — e.g. v23 had xyz_head ~50× smaller than scale_head,
    v24 brought them to within 2× of each other."""
    rows = json.loads((run_dir / "loss_history.json").read_text())
    train = [r for r in rows if "lr" in r and "head_grad" in r]
    if not train:
        print(
            f"[plot] no head_grad in {run_dir.name} (older run pre-grad-logging) — skipping"
        )
        return
    steps = [r["step"] for r in train]
    head_names = sorted({k for r in train for k in r["head_grad"].keys()})
    has_total = any("grad_norm" in r for r in train)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    if has_total:
        gn = [r.get("grad_norm") for r in train]
        ax.plot(
            steps,
            gn,
            label="|grad| (total, post-clip)",
            linewidth=1.6,
            color="black",
            alpha=0.7,
        )
    for name in head_names:
        ys = [r["head_grad"].get(name) for r in train]
        ax.plot(steps, ys, label=name, linewidth=1)
    ax.set_xlabel("step")
    ax.set_ylabel("L2 gradient norm")
    ax.set_title(f"{run_dir.name} — per-head gradient norms")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"[plot] wrote {out_path}")


def plot_motion_ratio(run_dir: Path, out_path: Path) -> None:
    p = run_dir / "motion_history.json"
    if not p.exists():
        print(f"[plot] no {p.name} — skipping motion plot")
        return
    rows = json.loads(p.read_text())
    if not rows:
        print(f"[plot] {p.name} is empty — skipping")
        return
    subsets = sorted(
        {k.removesuffix("_ratio") for r in rows for k in r if k.endswith("_ratio")}
    )
    fig, ax = plt.subplots(figsize=(8, 4))
    steps = [r["step"] for r in rows]
    for s in subsets:
        ys = [r.get(f"{s}_ratio") for r in rows]
        ax.plot(steps, ys, label=s, linewidth=1)
    ax.axhline(
        1.0,
        color="black",
        linestyle="--",
        linewidth=0.8,
        alpha=0.6,
        label="target (=1.0)",
    )
    ax.set_xlabel("step")
    ax.set_ylabel("Σ pred 2D travel / Σ GT 2D travel")
    ax.set_title(f"{run_dir.name} — motion ratio per subset")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"[plot] wrote {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="path to result/<dt>_track_v*/ (default: most recent)",
    )
    args = ap.parse_args()
    run_dir = args.run_dir
    if run_dir is None:
        candidates = sorted(
            Path("result").glob("track_v*_*"), key=lambda p: p.stat().st_mtime
        )
        if not candidates:
            raise SystemExit("no result/track_v*_*/ run dirs found")
        run_dir = candidates[-1]
    run_dir = run_dir.resolve()
    if not run_dir.is_dir():
        raise SystemExit(f"not a directory: {run_dir}")
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    plot_training_curve(run_dir, plots_dir / "training_curve.png")
    plot_grad_curves(run_dir, plots_dir / "grad_norms.png")
    plot_motion_ratio(run_dir, plots_dir / "motion_ratio.png")


if __name__ == "__main__":
    main()
