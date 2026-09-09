#!/usr/bin/env bash
# Can the model reduce its loss on three clips? Run the same 150-step overfit at several rates.
#
# 150 steps of Adam at lr r moves each weight by at most ~150*r, so at 5e-5 the whole probe is
# worth 0.0075 in weight units -- possibly too small to show descent at all. Sweeping upward
# separates "the rate is too high and it is bouncing" from "the rate is too small to move".
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
# shellcheck source=/dev/null
source "$(dirname "$0")/cudnn_env.sh"
S=result/OVERNIGHT_STATUS.md
note () { printf '%s\n' "$*" | tee -a "$S"; }

wait_for () {                      # $1 = config path, wait for that trainer to exit
  while :; do
    local found=0 p a
    for p in $(pgrep -f train_depth_refined_tracker 2>/dev/null); do
      a=$(tr '\0' '\n' < "/proc/$p/cmdline" 2>/dev/null | head -1)
      case "$a" in */.venv/bin/python*) ;; *) continue ;; esac
      grep -qa -- "$1" "/proc/$p/cmdline" 2>/dev/null && found=1
    done
    [ "$found" = 0 ] && return 0
    sleep 20
  done
}

for lr in 5.0e-4 1.5e-4 5.0e-6; do
  tag=${lr//[.-]/}
  cfg=configs/probe_lr_$tag.yaml
  uv run python - "$lr" "$cfg" "result/probe_lr_$tag" <<'PYEOF'
import sys, yaml, pathlib
lr, cfg_path, out = sys.argv[1], sys.argv[2], sys.argv[3]
c = yaml.safe_load(pathlib.Path("configs/probe_overfit.yaml").read_text())
c.pop("version", None)
c["train"].update(lr=float(lr), out_dir=out)
pathlib.Path(cfg_path).write_text(
    f"# Overfit probe at lr={lr}. Same three clips, same 150 steps, as probe_overfit.yaml.\n"
    + yaml.safe_dump(c, sort_keys=False) + "version: v73\n")
PYEOF
  note "### overfit probe lr=$lr  $(date '+%H:%M')"
  uv run python scripts/train_depth_refined_tracker.py --config "$cfg" \
    >> "result/probe_lr_$tag.log" 2>&1
  wait_for "$cfg"
  first=$(tr '\r' '\n' < "result/probe_lr_$tag.log" | grep -aoP 'meanloss=\K[\d.]+' | head -3 | tail -1)
  last=$(tr '\r' '\n' < "result/probe_lr_$tag.log" | grep -aoP 'meanloss=\K[\d.]+' | tail -1)
  note "  lr=$lr  meanloss $first -> $last"
done
note "### overfit probes done $(date '+%H:%M')"
