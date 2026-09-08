#!/usr/bin/env bash
# Overnight controller: cut v90b at a wall-clock time so its number can be read, then continue the
# same trajectory as v90c and cut that in time for its own evaluation to land before 08:00.
#
# WHY WALL-CLOCK CUTS RATHER THAN STEP COUNTS. At the measured 11.35 s/step, v90b's configured 4000
# steps take 12.6 h and would consume the whole window, leaving nothing to continue with and no
# intermediate reading. The cuts are placed just past a checkpoint boundary (ckpt_every=500), so at
# most a few minutes of work is discarded.
#
# WHY pgrep IS NOT TRUSTED HERE. `pgrep -f train_depth_refined_tracker` matches any shell whose
# command line merely mentions the script -- including the diagnostics run while watching this job.
# That misfire has already killed one probe and faked one watchdog in this project, so every
# candidate is confirmed against /proc/<pid>/exe pointing into this repo's venv.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
# shellcheck source=/dev/null
source "$(dirname "$0")/cudnn_env.sh"

CUT_B=$(date -d 'tomorrow 01:20' +%s)   # v90b trainer stopped; ~step 2000
CUT_C=$(date -d 'tomorrow 06:50' +%s)   # v90c trainer stopped; leaves ~25 min for its evaluation
CACHE=$HOME/data/tapvid3d_baseline_preds/waft_minival_is896_s-1_i4
S=result/OVERNIGHT_STATUS.md

note () { printf '%s\n' "$*" | tee -a "$S"; }
aj () { uv run python -c "
import json
try: print(f\"{json.load(open('$1/metrics.json'))['overall']['metric_average_jaccard']:.4f}\")
except Exception: print('nan')"; }
best () { local c; c=$(ls -1t "$1"/best/ckpt_*.pt 2>/dev/null | head -1)
          [ -n "$c" ] || c=$(ls -1t "$1"/ckpt_*.pt 2>/dev/null | head -1); printf '%s' "$c"; }

# A real trainer process for $1, or nothing. The exe check is what makes this reliable.
# argv[0], NOT /proc/<pid>/exe. `exe` is the resolved binary, and for a uv venv that is the shared
# interpreter under ~/.local/share/uv -- so an exe test rejects the real trainer and the controller
# concludes the job already finished. argv[0] keeps the venv path, and it is /bin/bash for the shell
# false positives this exists to reject.
trainer_pid () {
  local p argv0
  for p in $(pgrep -f train_depth_refined_tracker 2>/dev/null); do
    [ "$p" = "$$" ] && continue
    argv0=$(tr '\0' '\n' < "/proc/$p/cmdline" 2>/dev/null | head -1)
    case "$argv0" in *"/.venv/bin/python"*) ;; *) continue ;; esac
    grep -qa -- "$1" "/proc/$p/cmdline" 2>/dev/null && { printf '%s' "$p"; return 0; }
  done
  return 1
}

# Stop the trainer for config $1 at epoch-time $2, or return as soon as it exits on its own.
cut_at () {
  local cfg=$1 deadline=$2 pid
  while :; do
    pid=$(trainer_pid "$cfg") || { note "  [$cfg] trainer gone at $(date '+%H:%M') -- exited on its own"; return 0; }
    if [ "$(date +%s)" -ge "$deadline" ]; then
      note "  [$cfg] deadline reached at $(date '+%H:%M'); stopping pid $pid"
      kill -INT "$pid" 2>/dev/null
      for _ in $(seq 1 60); do sleep 5; kill -0 "$pid" 2>/dev/null || break; done
      kill -0 "$pid" 2>/dev/null && { note "  [$cfg] SIGINT ignored; SIGTERM"; kill -TERM "$pid"; }
      return 0
    fi
    sleep 60
  done
}

note ""; note "## overnight chain armed $(date -Is)  cut_b=$(date -d @"$CUT_B" '+%H:%M')  cut_c=$(date -d @"$CUT_C" '+%H:%M')"

cut_at configs/v90b.yaml "$CUT_B"
# run_v90b.sh runs its own evaluation once the trainer exits; wait for THAT PID, handed in as $1 and
# re-checked against its cmdline each poll so a recycled pid cannot masquerade as it. Waiting on a
# pgrep pattern instead is what matched this session's own diagnostic shells.
RUNB=${1:-}
while [ -n "$RUNB" ] && grep -qa run_v90b.sh "/proc/$RUNB/cmdline" 2>/dev/null; do
  [ "$(date +%s)" -ge "$((CUT_B + 5400))" ] && { note "  run_v90b.sh still alive 90 min after the cut; going on"; break; }
  sleep 60
done
note "  run_v90b.sh finished $(date '+%H:%M')"
note "### v90b = $(aj result/v90b/eval)   $(date '+%H:%M')"
[ -f result/v90b/eval/metrics.json ] && uv run python scripts/report_metric_aj.py result/v90b/eval >> "$S"

# --- v90c: the same trajectory continued. Seeding out_dir with v90b's last checkpoint makes the
# trainer take its RESUME path, which restores optimiser and scheduler state -- a warm start through
# init_ckpt would reset Adam's moments and restart the schedule mid-trajectory.
ck=$(ls -1t result/v90b/ckpt_*.pt 2>/dev/null | head -1)
if [ -z "$ck" ]; then note "### v90c ABORTED: v90b left no resumable checkpoint"; exit 1; fi
mkdir -p result/v90c && cp -n "$ck" result/v90c/
uv run python - "$ck" <<'PYEOF' >> "$S" 2>&1
import sys, yaml, pathlib
p = pathlib.Path("configs/v90c.yaml"); raw = p.read_text()
c = yaml.safe_load(raw); c["train"]["init_ckpt"] = sys.argv[1]; c.pop("version", None)
p.write_text(raw[:raw.index("data:")] + yaml.safe_dump(c, sort_keys=False) + "version: v73\n")
print(f"  v90c resumes {sys.argv[1]} -> {c['train']['steps']} steps")
PYEOF
note "### v90c (v90b continued) $(date -Is)"
uv run python scripts/train_depth_refined_tracker.py --config configs/v90c.yaml >> result/v73_plan.log 2>&1 &
cut_at configs/v90c.yaml "$CUT_C"
wait

ck=$(best result/v90c)
if [ -n "$ck" ] && [ ! -f result/v90c/eval/metrics.json ]; then
  note "  scoring v90c on $ck"
  uv run python scripts/eval_metric3d.py --method v73 --ckpt "$ck" --depth da3g \
      --split minival --waft-pred-dir "$CACHE" --out-dir result/v90c/eval || note "  v90c EVAL FAILED"
fi
[ -f result/v90c/eval/metrics.json ] && uv run python scripts/report_metric_aj.py result/v90c/eval >> "$S"
note "### MORNING $(date '+%H:%M')  DELTA 0.2274 | v63 0.2238 | warm start 0.2191 | v89 $(aj result/v89_scale_only/eval) | v90a $(aj result/v90a/eval) | v90b $(aj result/v90b/eval) | v90c $(aj result/v90c/eval)"
