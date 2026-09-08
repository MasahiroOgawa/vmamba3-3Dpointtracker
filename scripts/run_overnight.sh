#!/usr/bin/env bash
# Overnight run: beat DELTA on DA3-g (0.2274) with WAFT + (DA3-g + Mamba-3 scale refiner) + 2-pool.
#
# SELF-RECOVERING BY DESIGN. Nothing here exits on a recoverable problem: each step logs its failure,
# records it, and moves to the next thing that can still make progress. `set -e` is deliberately NOT
# used. Every step is idempotent (evals skip when metrics.json exists, training resumes from the
# newest checkpoint in its out_dir), so the whole script can be relaunched and will continue.
#
# Paths, in order, each an independent shot at the goal:
#   P0  re-score v63 on the iters-4 cache            -- exact internal baseline
#   P1  warm start -> v87a (stage 1 FROZEN) -> v87b (both stages joint)
#   P2  resume v63_seed2 from 12000 to 20000         -- second seed, independent of P1
# If stage A or the warm start is unusable, P1 is skipped and the night goes to P2 rather than idling.
#
# Deadline: every training phase runs under `timeout` from a hard stop of 2026-09-08 11:00.
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

aj () { uv run python -c "
import json,sys
try: print(f\"{json.load(open('$1/metrics.json'))['overall']['metric_average_jaccard']:.4f}\")
except Exception: print('nan')"; }

evaluate () {  # evaluate <ckpt> <method> <outdir> <label>
  local ck=$1 meth=$2 out=$3 lab=$4
  if [ -f "$out/metrics.json" ]; then note "  [$(ts)] $lab = $(aj "$out")  (cached)"; return 0; fi
  [ -f "$ck" ] || { note "  [$(ts)] $lab SKIPPED: no checkpoint at $ck"; return 1; }
  local try
  for try in 1 2; do
    uv run python scripts/eval_metric3d.py --method "$meth" --ckpt "$ck" \
        --da3-depth-root "$DA3G" --split minival --waft-pred-dir "$CACHE" --out-dir "$out" \
      && { note "  [$(ts)] $lab = $(aj "$out")"; uv run python scripts/report_metric_aj.py "$out" >> "$STATUS"; return 0; }
    note "  [$(ts)] $lab eval attempt $try failed; retrying after 60s"
    rm -rf "$out"; sleep 60
  done
  note "  [$(ts)] $lab EVAL FAILED twice -- continuing"
  return 1
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
    sleep 60   # transient cuDNN/OOM: resume picks up from the newest checkpoint
  done
  note "  [$(ts)] $lab failed twice -- continuing to the next path"
  return 1
}

best_ckpt () { local c; c=$(ls -1t "$1"/best/ckpt_*.pt 2>/dev/null | head -1)
               [ -n "$c" ] || c=$(ls -1t "$1"/ckpt_*.pt 2>/dev/null | head -1); printf '%s' "$c"; }

note ""
note "## overnight run started $(date -Is)  (hard stop 11:00, target DELTA DA3-g = 0.2274)"

# ---- P0 --------------------------------------------------------------------------------------
note "### P0  (dropped) v63 cannot be re-scored on the iters-4 cache"
note "  v63 trained with flow.iters=5, so scoring it on iters-4 tracks IS a train/eval mismatch and"
note "  eval_metric3d refuses it by design. v63's 0.2238 stands on its own consistent protocol"
note "  (live WAFT, iters 5); v87 stands on its own (iters 4, manifested cache). Both are compared"
note "  against DELTA DA3-g = 0.2274, which is external predictions and protocol-independent."

# ---- P1 --------------------------------------------------------------------------------------
note "### P1  DA3-g scale refiner + vmamba3-2pool"
# Wait for stage A to FINISH, not merely for best.pt to appear: the trainer rewrites best.pt at
# every improvement, so testing existence alone grabbed a half-trained refiner (step 1500,
# 71.9% reduction) while training was still running toward step 3500 (73.3%).
for _ in $(seq 1 240); do
  grep -qE "^\[A\] (done|FAILED)|best val \|err\|" result/stageA_scale_da3g.log 2>/dev/null \
    && [ -f result/scale_standalone_da3g/best.pt ] && break
  sleep 30
done
if [ ! -f result/scale_standalone_da3g/best.pt ]; then
  note "  [$(ts)] stage A produced no best.pt within 2 h -- P1 skipped, going to P2"
else
  if [ ! -f result/v87_warmstart/ckpt_0.pt ]; then
    uv run python scripts/make_v73_warmstart.py \
        --scale-ckpt result/scale_standalone_da3g/best.pt \
        --v45-ckpt   result/v63_waft_live_2pool/ckpt_20000.pt \
        --out        result/v87_warmstart/ckpt_0.pt >> "$STATUS" 2>&1 \
      || note "  [$(ts)] warm start build FAILED"
  fi
  if [ -f result/v87_warmstart/ckpt_0.pt ]; then
    evaluate result/v87_warmstart/ckpt_0.pt v73 result/v87_warmstart/eval "warm start (untrained)"
    # Informational only: phase 1 exists precisely to adapt stage 2 to stage 1's output, so a low
    # warm start is a reason to train, not a reason to stop.
    train_guarded configs/v87a.yaml "v87a (stage 1 FROZEN)"
    c1=$(best_ckpt result/v87a_frozen)
    evaluate "$c1" v73 result/v87a_frozen/eval "v87a frozen"
    src=${c1:-result/v87_warmstart/ckpt_0.pt}
    uv run python - "$src" <<'PY' >> "$STATUS" 2>&1
import sys, yaml, pathlib
p = pathlib.Path("configs/v87b.yaml"); raw = p.read_text()
hdr = raw[:raw.index("version:")]
c = yaml.safe_load(raw); c["train"]["init_ckpt"] = sys.argv[1]
p.write_text(hdr + yaml.safe_dump(c, sort_keys=False))
print(f"  v87b init_ckpt := {sys.argv[1]}")
PY
    train_guarded configs/v87b.yaml "v87b (both stages joint)"
    c2=$(best_ckpt result/v87b_joint)
    evaluate "$c2" v73 result/v87b_joint/eval "v87b joint"
  fi
fi

# ---- P2 --------------------------------------------------------------------------------------
note "### P2  v63_seed2 resumed 12000 -> 20000 (independent second seed)"
if [ ! -f configs/v63_seed2_resume.yaml ]; then
  uv run python - <<'PY' >> "$STATUS" 2>&1
import json, yaml, pathlib
c = json.load(open("result/v63_seed2/cfg.json"))
cfg = {k: c[k] for k in ("version","flow","data","model","train","loss") if k in c}
cfg["loss"] = {"weights": c["loss"].get("weights_raw", {"pos_3D":1.0,"reg_uv":0.1})}
cfg["data"]["split"] = {"source":"official","n_val_monitor":40,"seed":42,"holdout_val":True}
cfg["train"].update({"out_dir":"result/v63_seed2","val_every":1000,"ckpt_every":1000,
                     "val_clips":40,"val_at_step0":True,"early_stop_patience":0})
hdr = ("# v63_seed2 resumed: same recipe as v63 (seed 1000, iters 4), continuing from ckpt_12000\n"
       "# toward 20000. Independent second shot at DELTA's 0.2274 and the only run-to-run variance\n"
       "# estimate available for the best clean DA3-g arm. Resume is automatic: the trainer picks up\n"
       "# the newest ckpt_*.pt in out_dir.\n")
pathlib.Path("configs/v63_seed2_resume.yaml").write_text(hdr + yaml.safe_dump(cfg, sort_keys=False))
print("  wrote configs/v63_seed2_resume.yaml")
PY
fi
train_guarded configs/v63_seed2_resume.yaml "v63_seed2 resume"
cs=$(best_ckpt result/v63_seed2)
evaluate "$cs" v45 result/v63_seed2/eval_full "v63_seed2 (resumed)"

# ---- summary ---------------------------------------------------------------------------------
note ""
note "### SUMMARY $(date -Is)   TARGET DELTA DA3-g = 0.2274"
for d in result/v63_waft_live_2pool/eval_i4cache result/v87_warmstart/eval \
         result/v87a_frozen/eval result/v87b_joint/eval result/v63_seed2/eval_full \
         result/v63_seed2/eval; do
  [ -f "$d/metrics.json" ] && note "  $(aj "$d")  $d"
done
note "### overnight run ended $(date -Is)"
