#!/usr/bin/env bash
# What does a longer training window cost, and does it fix the visibility statistics?
#
# The temporal model trains on 8 frames and is evaluated on 300 (adt), so the Mamba-3 layer
# extrapolates 37x beyond its training length. Long windows are ruled out at INFERENCE in this
# work -- that is what stops other methods fitting one 12 GB GPU -- but we already run whole clips
# at inference, so the training crop is a free variable. This measures the price.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
# shellcheck source=/dev/null
source "$(dirname "$0")/cudnn_env.sh"
S=result/OVERNIGHT_STATUS.md
note () { printf '%s\n' "$*" | tee -a "$S"; }

note ""; note "### window probe $(date '+%H:%M')  (20 steps each, vis head only)"
for w in 8 16 32 64; do
  out=result/probe_w$w; cfg=configs/probe_w$w.yaml
  rm -rf "$out"; : > "result/probe_w$w.log"
  uv run python - "$w" "$cfg" "$out" <<'PYEOF'
import sys, yaml, pathlib
w, cfg_path, out = int(sys.argv[1]), sys.argv[2], sys.argv[3]
c = yaml.safe_load(pathlib.Path("configs/v92.yaml").read_text())
c.pop("version", None)
c["train"].update(window=w, out_dir=out, steps=20, log_every=5,
                  val_every=10**6, ckpt_every=10**6, val_at_step0=False)
pathlib.Path(cfg_path).write_text(
    f"# Window-{w} cost probe: 20 steps, visibility head only.\n"
    + yaml.safe_dump(c, sort_keys=False) + "version: v73\n")
PYEOF
  # sample GPU memory while it runs
  ( peak=0
    while :; do
      m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
      [ -n "$m" ] && [ "$m" -gt "$peak" ] && peak=$m
      echo "$peak" > "/tmp/peak_w$w"; sleep 2
    done ) & sampler=$!
  uv run python scripts/train_depth_refined_tracker.py --config "$cfg" >> "result/probe_w$w.log" 2>&1
  kill $sampler 2>/dev/null
  if grep -qai "out of memory" "result/probe_w$w.log"; then
    note "  window=$w  OUT OF MEMORY"; continue
  fi
  el=$(tr '\r' '\n' < "result/probe_w$w.log" | grep -aoP 'elapsed=\K[0-9]+' | tail -1)
  st=$(tr '\r' '\n' < "result/probe_w$w.log" | grep -aoP '^\[train\] step\s+\K[0-9]+' | tail -1)
  vis=$(tr '\r' '\n' < "result/probe_w$w.log" | grep -aoP 'target mean \K[\d.]+' | head -1)
  pk=$(cat "/tmp/peak_w$w" 2>/dev/null || echo "?")
  if [ -n "$el" ] && [ -n "$st" ] && [ "$st" -gt 0 ]; then
    note "  window=$w  $(uv run python -c "print(f'{$el/$st:5.1f} s/step')")  peak ${pk} MiB  target visible ${vis:-?}"
  else
    note "  window=$w  no timing; see result/probe_w$w.log"
  fi
done
note "### window probe done $(date '+%H:%M')   (adt at eval is 300 frames, 0.3923 visible)"
