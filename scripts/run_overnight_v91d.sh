#!/usr/bin/env bash
# Overnight: find a learning rate that actually reduces the loss, then train v91d at it.
#
# WHY A LADDER RATHER THAN ONE GUESS. v91's loss looked flat, but its logger printed one clip's
# loss per 50 steps and the per-clip spread is 21x, so the curve could not have shown a trend
# either way. The logger now reports the window mean. The open question is which direction the
# rate is wrong in: 150 steps of Adam at 5e-5 moves a weight by at most ~0.0075, which may simply
# be too little to see, so the ladder sweeps up as well as down.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
# shellcheck source=/dev/null
source "$(dirname "$0")/cudnn_env.sh"
CACHE=$HOME/data/tapvid3d_baseline_preds/waft_minival_is896_s-1_i4
STOP=$(date -d 'tomorrow 06:20' +%s)     # trainer stops here; leaves 30 min for the evaluation
S=result/OVERNIGHT_STATUS.md
note () { printf '%s\n' "$*" | tee -a "$S"; }

trainer_running () {                      # argv[0], not /proc/exe: under uv the latter resolves
  local p a                               # to the shared interpreter, not this venv
  for p in $(pgrep -f train_depth_refined_tracker 2>/dev/null); do
    [ "$p" = "$$" ] && continue
    a=$(tr '\0' '\n' < "/proc/$p/cmdline" 2>/dev/null | head -1)
    case "$a" in */.venv/bin/python*) return 0 ;; esac
  done
  return 1
}
wait_idle () { while trainer_running; do sleep 20; done; }

# Least-squares slope of the logged window means, per 100 steps. Negative means descending.
# Both fields are taken from one regex on one line: pairing two separate greps produced a
# confident 100.00000 on a log whose lines lacked "meanloss" at all.
slope () {
  tr '\r' '\n' < "$1" | uv run python -c "
import re, sys
rx = re.compile(r'^\[train\] step\s+(\d+)/\S+\s+mean\s*loss=([\d.]+)')
pts = [(float(m[1]), float(m[2])) for ln in sys.stdin if (m := rx.match(ln))]
pts = pts[1:]                      # step 0's window holds a single sample
if len(pts) < 4:
    print('nan'); raise SystemExit
xs, ys = zip(*pts)
n = len(xs); mx = sum(xs)/n; my = sum(ys)/n
den = sum((x-mx)**2 for x in xs)
print(f'{100*sum((x-mx)*(y-my) for x, y in zip(xs, ys))/den:.5f}' if den else 'nan')"
}

wait_idle
note ""; note "## v91d night $(date -Is)   stop $(date -d @"$STOP" '+%H:%M')"

best_lr=""; best_slope=""
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
    f"# Overfit probe at lr={lr}: same three clips and 150 steps as probe_overfit.yaml.\n"
    + yaml.safe_dump(c, sort_keys=False) + "version: v73\n")
PYEOF
  rm -rf "result/probe_lr_$tag"
  uv run python scripts/train_depth_refined_tracker.py --config "$cfg" \
    >> "result/probe_lr_$tag.log" 2>&1 || true
  wait_idle
  sl=$(slope "result/probe_lr_$tag.log")
  note "  probe lr=$lr  slope=${sl}/100 steps  (negative = descending)"
  if [ -n "$sl" ] && [ "$sl" != "nan" ]; then
    if [ -z "$best_slope" ] || uv run python -c "import sys; sys.exit(0 if float('$sl')<float('$best_slope') else 1)"; then
      best_slope=$sl; best_lr=$lr
    fi
  fi
done
sl0=$(slope result/probe_overfit.log)
note "  probe lr=5.0e-5  slope=${sl0}/100 steps  (v91's rate, run earlier)"
if [ -n "$sl0" ] && [ "$sl0" != "nan" ]; then
  if [ -z "$best_slope" ] || uv run python -c "import sys; sys.exit(0 if float('$sl0')<float('$best_slope') else 1)"; then
    best_slope=$sl0; best_lr=5.0e-5
  fi
fi
note "### chosen lr=$best_lr (slope $best_slope). A non-negative slope means no rate tested here"
note "    reduces the loss, and the rate is not the lever."

# --- v91d: continue from v91's last checkpoint at the chosen rate.
rem=$(( (STOP - $(date +%s)) / 12 ))      # ~12 s/step measured on this arm
[ "$rem" -lt 200 ] && rem=200
uv run python - "$best_lr" "$rem" <<'PYEOF'
import sys, yaml, pathlib
lr, steps = float(sys.argv[1]), int(sys.argv[2])
c = yaml.safe_load(pathlib.Path("configs/v91.yaml").read_text())
c.pop("version", None)
c["train"].update(out_dir="result/v91d", init_ckpt="result/v91/ckpt_1200.pt",
                  lr=lr, warmup=0, steps=steps, decay=max(1, steps // 5),
                  log_every=25, val_every=250, ckpt_every=250, val_at_step0=True,
                  early_stop_patience=10**6)
head = f'''# v91d: v91 continued from its last checkpoint at the rate the overfit ladder chose.
#
# lr={lr} was picked by measuring the slope of the window-mean loss on three clips at
# 5e-6, 5e-5, 1.5e-4 and 5e-4 -- not assumed. log_every=25 so each logged point is a 25-step
# mean; v91 logged one clip per 50 steps, and that is why its curve looked flat.
# steps={steps} is what fits before the 06:20 stop at the measured 12 s/step.
'''
pathlib.Path("configs/v91d.yaml").write_text(
    head + yaml.safe_dump(c, sort_keys=False) + "version: v73\n")
PYEOF
note "### v91d start $(date -Is)  lr=$best_lr  steps=$rem"
uv run python scripts/train_depth_refined_tracker.py --config configs/v91d.yaml \
  >> result/v91d.log 2>&1 &
tp=$!
while kill -0 $tp 2>/dev/null; do
  [ "$(date +%s)" -ge "$STOP" ] && { note "  [cut] 06:20 reached; stopping v91d"; kill -INT $tp; break; }
  sleep 60
done
wait $tp 2>/dev/null
note "  v91d slope over the run: $(slope result/v91d.log)/100 steps"

ck=$(ls -1t result/v91d/best/ckpt_*.pt result/v91d/ckpt_*.pt 2>/dev/null | head -1)
if [ -n "$ck" ]; then
  note "  scoring $ck"
  uv run python scripts/eval_metric3d.py --method v73 --ckpt "$ck" --depth da3g \
      --split minival --waft-pred-dir "$CACHE" --out-dir result/v91d/eval || note "  EVAL FAILED"
  [ -f result/v91d/eval/metrics.json ] && uv run python scripts/report_metric_aj.py result/v91d/eval | tee -a "$S"
fi
note "### MORNING $(date '+%H:%M')  bars: v91 best 0.2251 | v63 0.2238 | DELTA 0.2274"
