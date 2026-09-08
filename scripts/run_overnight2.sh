#!/usr/bin/env bash
# Second half of the night: score v87b, then run the V88 path (add the scale stage behind a
# zero-init gate instead of substituting it for the de-flicker).
#
# Why V88 replaces the planned P2 (v63_seed2 resume): V88's warm start is NUMERICALLY v63, so its
# floor is v63's own score and it needs only +0.0036 to pass DELTA. P2 was a second seed of a recipe
# that already lands at 0.2238. V88 is the better use of the hours that remain.
#
# Self-recovering: no exits on recoverable failure, every step idempotent, waits are on FILES or on
# nvidia-smi (never pgrep -- two earlier runners deadlocked matching their own command line).
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
# shellcheck source=/dev/null
source "$(dirname "$0")/cudnn_env.sh"
CACHE=$HOME/data/tapvid3d_baseline_preds/waft_minival_is896_s-1_i4
DA3G=$HOME/data/tapvid3d_da3nested
HARD_STOP=$(date -d "2026-09-08 11:00" +%s)
STATUS=result/OVERNIGHT_STATUS.md

note () { printf '%s\n' "$*" | tee -a "$STATUS"; }
ts   () { date '+%H:%M'; }
aj   () { uv run python -c "
import json
try: print(f\"{json.load(open('$1/metrics.json'))['overall']['metric_average_jaccard']:.4f}\")
except Exception: print('nan')"; }

gpu_busy () { [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ]; }

evaluate () {  # evaluate <ckpt> <method> <outdir> <label>
  local ck=$1 meth=$2 out=$3 lab=$4 try
  if [ -f "$out/metrics.json" ]; then note "  [$(ts)] $lab = $(aj "$out")  (cached)"; return 0; fi
  [ -f "$ck" ] || { note "  [$(ts)] $lab SKIPPED: no checkpoint at $ck"; return 1; }
  for try in 1 2; do
    uv run python scripts/eval_metric3d.py --method "$meth" --ckpt "$ck" \
        --da3-depth-root "$DA3G" --split minival --waft-pred-dir "$CACHE" --out-dir "$out" \
      && { note "  [$(ts)] $lab = $(aj "$out")"
           uv run python scripts/report_metric_aj.py "$out" >> "$STATUS"; return 0; }
    note "  [$(ts)] $lab eval attempt $try failed; retrying in 60s"; rm -rf "$out"; sleep 60
  done
  note "  [$(ts)] $lab EVAL FAILED twice -- continuing"; return 1
}

train_guarded () {  # train_guarded <config> <label>
  local cfg=$1 lab=$2 budget rc try
  for try in 1 2; do
    budget=$(( HARD_STOP - $(date +%s) ))
    if [ "$budget" -lt 1800 ]; then note "  [$(ts)] $lab SKIPPED: <30 min to hard stop"; return 1; fi
    note "  [$(ts)] $lab training, budget $(( budget/3600 ))h$(( (budget%3600)/60 ))m (attempt $try)"
    timeout "${budget}s" uv run python scripts/train_depth_refined_tracker.py --config "$cfg"
    rc=$?
    case $rc in
      0)   note "  [$(ts)] $lab finished"; return 0;;
      124) note "  [$(ts)] $lab hit the wall-clock guard; using best so far"; return 0;;
      *)   note "  [$(ts)] $lab exited $rc on attempt $try";;
    esac
    sleep 60
  done
  note "  [$(ts)] $lab failed twice -- continuing"; return 1
}

best_ckpt () { local c; c=$(ls -1t "$1"/best/ckpt_*.pt 2>/dev/null | head -1)
               [ -n "$c" ] || c=$(ls -1t "$1"/ckpt_*.pt 2>/dev/null | head -1); printf '%s' "$c"; }

note ""
note "## PART 2 started $(date -Is) -- V88 path (gated scale stage ADDED to v63)"

note "  [$(ts)] waiting for the GPU to free (v87b finishing)"
for _ in $(seq 1 720); do gpu_busy || break; sleep 30; done
sleep 20

note "### v87b (substitute-the-deflicker arm) final score"
evaluate "$(best_ckpt result/v87b_joint)" v73 result/v87b_joint/eval "v87b joint"

note "### V88: measure the floor (warm start is numerically v63)"
evaluate result/v88_warmstart/ckpt_0.pt v88 result/v88_warmstart/eval "v88 warm start = FLOOR"

note "### v88a: v63 FROZEN, only the scale stage + gate train"
train_guarded configs/v88a.yaml "v88a (v45 frozen)"
c1=$(best_ckpt result/v88a_gateonly)
evaluate "$c1" v88 result/v88a_gateonly/eval "v88a gate-only"

note "### v88b: all three stages joint, from v88a's best"
if [ -n "$c1" ]; then
  uv run python - "$c1" <<'PY' >> "$STATUS" 2>&1
import sys, yaml, pathlib
p = pathlib.Path("configs/v88b.yaml"); raw = p.read_text()
hdr = raw[:raw.index("version:")]
c = yaml.safe_load(raw); c["train"]["init_ckpt"] = sys.argv[1]
p.write_text(hdr + yaml.safe_dump(c, sort_keys=False))
print(f"  v88b init_ckpt := {sys.argv[1]}")
PY
  train_guarded configs/v88b.yaml "v88b (joint)"
  evaluate "$(best_ckpt result/v88b_joint)" v88 result/v88b_joint/eval "v88b joint"
fi

note ""
note "### FINAL SUMMARY $(date -Is)   TARGET DELTA DA3-g = 0.2274"
note "  0.2238  v63 (best clean 2-pool DA3-g, live iters-5 protocol)"
for d in result/v87_warmstart/eval result/v87a_frozen/eval result/v87b_joint/eval \
         result/v88_warmstart/eval result/v88a_gateonly/eval result/v88b_joint/eval; do
  [ -f "$d/metrics.json" ] && note "  $(aj "$d")  $d"
done
note "### PART 2 ended $(date -Is)"
