#!/usr/bin/env bash
# Does a larger batch fit, and what does it cost per step?
#
# batch=1 is why the gradient direction is noise: one clip per step, and the per-clip loss spans
# 21x. Averaging B clips cuts the gradient's standard error by sqrt(B), which is a bigger lever
# than the learning rate -- with a gradient this noisy no rate is right. This measures whether
# B=2/4/8 fits in 12 GB and what each costs, before the night is spent on one.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
# shellcheck source=/dev/null
source "$(dirname "$0")/cudnn_env.sh"
S=result/OVERNIGHT_STATUS.md
note () { printf '%s\n' "$*" | tee -a "$S"; }

note ""; note "### batch probe $(date '+%H:%M')  (30 steps each, 3 clips, lr 5e-5)"
for b in 2 4 8; do
  out=result/probe_b$b
  rm -rf "$out"; : > "result/probe_b$b.log"
  uv run python - "$b" "$out" <<'PYEOF'
import sys, yaml, pathlib
b, out = int(sys.argv[1]), sys.argv[2]
c = yaml.safe_load(pathlib.Path("configs/probe_overfit.yaml").read_text())
c.pop("version", None)
c["data"]["split"] = {"source": "minival", "n_train": 4, "seed": 42}
c["train"].update(batch=b, out_dir=out, steps=30, log_every=10, decay=1)
pathlib.Path(f"configs/probe_b{b}.yaml").write_text(
    f"# Batch-{b} feasibility probe: does it fit, and what does a step cost?\n"
    + yaml.safe_dump(c, sort_keys=False) + "version: v73\n")
PYEOF
  uv run python scripts/train_depth_refined_tracker.py --config "configs/probe_b$b.yaml" \
    >> "result/probe_b$b.log" 2>&1
  if grep -qai "out of memory" "result/probe_b$b.log"; then
    note "  batch=$b  OUT OF MEMORY"
    continue
  fi
  el=$(tr '\r' '\n' < "result/probe_b$b.log" | grep -aoP 'elapsed=\K[0-9]+' | tail -1)
  st=$(tr '\r' '\n' < "result/probe_b$b.log" | grep -aoP '^\[train\] step\s+\K[0-9]+' | tail -1)
  if [ -n "$el" ] && [ -n "$st" ] && [ "$st" -gt 0 ]; then
    note "  batch=$b  $(uv run python -c "print(f'{$el/$st:.1f} s/step, {$el/$st/$b:.1f} s/clip')")"
  else
    note "  batch=$b  no timing recorded; see result/probe_b$b.log"
  fi
done
note "### batch probe done $(date '+%H:%M')"
