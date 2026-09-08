#!/usr/bin/env bash
# v90b: stage 1 and stage 2 joint, continued from v90a's best. Launched separately on purpose --
# v90a's number is checked first, because this costs ~4h and several arms in this line only turned
# out to be uninterpretable after the fact.
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
best () { local c; c=$(ls -1t "$1"/best/ckpt_*.pt 2>/dev/null | head -1)
          [ -n "$c" ] || c=$(ls -1t "$1"/ckpt_*.pt 2>/dev/null | head -1); printf '%s' "$c"; }

c1=$(best result/v90a)
[ -n "$c1" ] || { note "[v90b] no v90a checkpoint; nothing to continue from"; exit 1; }
uv run python - "$c1" <<'PYEOF' >> "$S" 2>&1
import sys, yaml, pathlib
p = pathlib.Path("configs/v90b.yaml"); raw = p.read_text()
c = yaml.safe_load(raw); c["train"]["init_ckpt"] = sys.argv[1]
c.pop("version", None)
p.write_text(raw[:raw.index("data:")] + yaml.safe_dump(c, sort_keys=False) + "version: v73\n")
print(f"  v90b init_ckpt := {sys.argv[1]}")
PYEOF
note "### v90b (both stages joint) $(date -Is)"
uv run python scripts/train_depth_refined_tracker.py --config configs/v90b.yaml \
  || note "  v90b exited nonzero; scoring what exists"
ck=$(best result/v90b)
if [ -n "$ck" ] && [ ! -f result/v90b/eval/metrics.json ]; then
  uv run python scripts/eval_metric3d.py --method v73 --ckpt "$ck" --depth da3g \
      --split minival --waft-pred-dir "$CACHE" --out-dir result/v90b/eval \
    || note "  v90b EVAL FAILED"
fi
[ -f result/v90b/eval/metrics.json ] && { note "  v90b = $(aj result/v90b/eval)"
  uv run python scripts/report_metric_aj.py result/v90b/eval >> "$S"; }
note "### FINAL  DELTA 0.2274 | v63 0.2238 | v89 $(aj result/v89_scale_only/eval) | v90a $(aj result/v90a/eval) | v90b $(aj result/v90b/eval)"
