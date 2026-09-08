#!/usr/bin/env bash
# WAFT + (DA3-g + Mamba-3 depth scale refiner) + vmamba3-2pool, on a NON-BUGGY refiner.
#
# The scale refiner REPLACES the v44 de-flicker. They are duplicates: both per_frame_logscale
# methods return one bounded (B,F,1) log-scale per frame -- the de-flicker from the tracked points,
# the scale refiner from the whole depth map. Two stages doing the same job is not a pipeline.
# Stage 2 (the v35 vmamba3-2pool refiner) is warm-started from v63 so it does not start from zero.
#
# This is v87 rerun on a corrected stage 1. v87's refiner was trained against a target derived from
# DA3-l depth, so it predicted the OPPOSITE SIGN (+0.586 where the truth was -0.927) and the arm
# peaked at 0.2142. Stage 1 is now trained against DA3-g depth sampled at the WAFT uv.
#
# PRE-DECLARED DECISION RULES:
#   G1  stage 1 must REDUCE minival scale error on >=2 of 3 subsets. Its own held-out metric is NOT
#       evidence -- the buggy one improved that by 73.3% while inverting the sign. Fail -> stop.
#   G2  the warm start must beat 0.15. The buggy one scored 0.1184; a correct stage 1 should land
#       far above that. Fail -> stage 1 is still wrong, stop rather than train for hours.
#   Then A (stage 1 frozen, v35 adapts) then B (joint).
#   Outcome vs DELTA DA3-g 0.2274 and v63 0.2238:
#     >0.2274 goal met | 0.2238-0.2274 swap works, not enough | <0.2238 the swap still costs more
#     than the better scale estimate buys, and with v87's 0.2142 that is a reportable negative.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
# shellcheck source=/dev/null
source "$(dirname "$0")/cudnn_env.sh"
CACHE=$HOME/data/tapvid3d_baseline_preds/waft_minival_is896_s-1_i4
S=result/OVERNIGHT_STATUS.md
note () { printf '%s\n' "$*" | tee -a "$S"; }
aj () { uv run python -c "
import json
try: print(f\"{json.load(open('$1/metrics.json'))['overall']['metric_average_jaccard']:.4f}\")
except Exception: print('nan')"; }
evaluate () {
  [ -f "$3/metrics.json" ] && { note "  [$(date +%H:%M)] $4 = $(aj "$3") (cached)"; return 0; }
  [ -f "$1" ] || { note "  [$(date +%H:%M)] $4 SKIPPED: no $1"; return 1; }
  for t in 1 2; do
    uv run python scripts/eval_metric3d.py --method "$2" --ckpt "$1" --depth da3g \
        --split minival --waft-pred-dir "$CACHE" --out-dir "$3" \
      && { note "  [$(date +%H:%M)] $4 = $(aj "$3")"
           uv run python scripts/report_metric_aj.py "$3" >> "$S"; return 0; }
    note "  [$(date +%H:%M)] $4 eval attempt $t failed; retry 60s"; rm -rf "$3"; sleep 60
  done
  note "  [$(date +%H:%M)] $4 EVAL FAILED twice"; return 1
}
best () { local c; c=$(ls -1t "$1"/best/ckpt_*.pt 2>/dev/null | head -1)
          [ -n "$c" ] || c=$(ls -1t "$1"/ckpt_*.pt 2>/dev/null | head -1); printf '%s' "$c"; }

note ""; note "## SWAP PLAN (v73, fixed stage 1) started $(date -Is)  target DELTA = 0.2274"
note "### step 1: waiting for the corrected-target scale refiner"
for _ in $(seq 1 480); do [ -f result/scale_standalone_da3g_fixed/best.pt ] && break; sleep 30; done
[ -f result/scale_standalone_da3g_fixed/best.pt ] || { note "  no refiner after 4h; stopping"; exit 1; }

note "### G1: does stage 1 reduce minival scale error, with the right sign?"
uv run python scripts/check_scale_on_minival.py --ckpt result/scale_standalone_da3g_fixed/best.pt \
  2>&1 | grep -vE "xFormers|Loading|it/s" > /tmp/g1.$$ || true
grep -E "^  (drivetrack|pstudio|adt)|VERDICT" /tmp/g1.$$ | sed 's/^/  /' | tee -a "$S"
ok=$(grep -oE "improves [0-9]+/3" /tmp/g1.$$ | grep -oE "[0-9]+" | head -1)
if [ -z "$ok" ]; then
  note "  G1 INCONCLUSIVE: the checker produced no verdict line -- it crashed rather than
  measuring. Treating this as an error, NOT as a failing refiner. Checker output:"
  tail -12 /tmp/g1.$$ | sed 's/^/    /' | tee -a "$S"
  exit 2
fi
if [ "${ok:-0}" -lt 2 ]; then
  note "  G1 FAILED: improves ${ok:-0}/3. Training on a refiner that worsens depth gave 0.1435 last"
  note "  time. Stopping; stage 1 needs another look before any GPU goes into stage 2."
  exit 1
fi
note "  G1 PASSED: improves $ok/3 subsets"

note "### v89: the FIXED scale refiner ALONE (stage 2 zero-init = identity)"
# Named arm, so the standalone number has a version like every other row. This is
# ray * z_DA3g * exp(ds) with no per-track refinement: it isolates what the depth-scale stage
# contributes by itself. References: no correction at all (v40) = 0.208; the BUGGY refiner scored
# 0.1435 here, i.e. worse than doing nothing, which is what exposed the sign inversion.
if [ ! -f result/v89_scale_only/ckpt_0.pt ]; then
  uv run python - <<'PYEOF' 2>&1 | grep -vE "Loading|it/s" | tee -a "$S"
import torch, sys, json, pathlib
sys.path.insert(0, "src")
from mamba3_tracker.model.depth_refined_tracker import Mamba3V73
v63 = torch.load("result/v63_waft_live_2pool/ckpt_20000.pt", map_location="cpu", weights_only=False)
mc = dict(v63["cfg"]["model"])
m = Mamba3V73(dim=128, state_dim=64, num_heads=4, num_layers=2,
              max_log_correction=float(mc.get("max_log_correction", 2.0)),
              max_delta_uv=float(mc.get("max_delta_uv", 2.0)),
              patch_size=int(mc.get("patch_size", 5)), max_scale_correction=2.5,
              d_proj=int(mc.get("d_proj", 64)),
              dino_model=str(mc.get("dino_model", "facebook/dinov3-vits16-pretrain-lvd1689m")),
              dino_image_size=int(mc.get("dino_image_size", 448)), image_size=896,
              two_pool=True, grid=64, log_ref=2.0, log_std=1.5)
sc = torch.load("result/scale_standalone_da3g_fixed/best.pt", map_location="cpu", weights_only=False)
sd = {k.replace("scale_refiner.", ""): v for k, v in sc.get("model", sc).items()}
r = m.scale_refiner.load_state_dict(sd, strict=False)
assert not r.unexpected_keys, r.unexpected_keys
cfg = {**v63["cfg"], "version": "v73",
       "model": {**mc, "max_scale_correction": 2.5, "grid": 64, "log_ref": 2.0, "log_std": 1.5}}
cfg["flow"] = {**cfg.get("flow", {}), "iters": 4, "source": "waft_live"}
cfg["flow"].pop("_iters_note", None)
d = pathlib.Path("result/v89_scale_only"); d.mkdir(parents=True, exist_ok=True)
torch.save({"model": m.state_dict(), "step": 0, "cfg": cfg}, d / "ckpt_0.pt")
(d / "cfg.json").write_text(json.dumps(cfg, indent=1, sort_keys=True))
print(f"  v89 built: fixed scale refiner + zero-init v35 (missing {len(r.missing_keys)} v35 tensors, expected)")
PYEOF
fi
evaluate result/v89_scale_only/ckpt_0.pt v73 result/v89_scale_only/eval "v89 scale refiner ALONE"
note "  refs: no correction (v40) 0.208 | buggy refiner alone 0.1435 | v63 0.2238 | DELTA 0.2274"

note "### step 2: swap -- stage1 = fixed refiner, stage2 = v63's v35 (vmamba3-2pool)"
uv run python scripts/make_v73_warmstart.py \
    --scale-ckpt result/scale_standalone_da3g_fixed/best.pt \
    --v45-ckpt   result/v63_waft_live_2pool/ckpt_20000.pt \
    --out        result/v73fix_warmstart/ckpt_0.pt 2>&1 | grep -vE "Loading|it/s" | tee -a "$S"
evaluate result/v73fix_warmstart/ckpt_0.pt v73 result/v73fix_warmstart/eval "v90 warm start (untrained)" || exit 1
w=$(aj result/v73fix_warmstart/eval)
note "  buggy stage 1 gave 0.1184 here; v63 (with the de-flicker) is 0.2238"
if [ "$(uv run python -c "print('yes' if $w > 0.15 else 'no')")" != yes ]; then
  note "  G2 FAILED: warm start $w <= 0.15, no better than the buggy refiner. Stopping."
  exit 1
fi
note "  G2 PASSED: warm start = $w"

# v90a only. v90b is a SEPARATE launch (scripts/run_v90b.sh) so the intermediate result can be
# checked before committing more GPU: this line has produced several arms whose numbers only made
# sense after the fact, and v90b costs ~4h.
note "### v90a (stage 1 FROZEN) $(date -Is)"
uv run python scripts/train_depth_refined_tracker.py --config configs/v90a.yaml \
  || note "  v90a exited nonzero; scoring what exists"
evaluate "$(best result/v90a)" v73 result/v90a/eval "v90a"

note ""
note "### STOPPING HERE for confirmation before v90b"
note "  v90a  = $(aj result/v90a/eval)"
note "  bars  : DELTA 0.2274 (goal) | v63 0.2238 | v87b buggy stage 1 0.2142 | v90 warm start above"
note "  v90b (both stages joint, ~4h) is NOT started. Launch it with:"
note "      setsid nohup bash scripts/run_v90b.sh >> result/v73_plan.log 2>&1 &"

note ""; note "### v89/v90 SUMMARY $(date -Is)  DELTA 0.2274 | v63 0.2238 | v87b buggy stage1 0.2142"
for d in result/v89_scale_only/eval result/v73fix_warmstart/eval result/v90a/eval result/v90b/eval; do
  [ -f "$d/metrics.json" ] && note "  $(aj "$d")  $d"
done
note "### SWAP PLAN ended $(date -Is)"
