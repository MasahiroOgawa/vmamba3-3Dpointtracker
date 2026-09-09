#!/usr/bin/env bash
# Score v91's ~1-hour checkpoint the moment it lands, rather than waiting for the run to finish.
# The concurrent evaluation costs training roughly a third of its speed; that buys the early read.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
# shellcheck source=/dev/null
source "$(dirname "$0")/cudnn_env.sh"
CK=result/v91/ckpt_$1.pt
CACHE=$HOME/data/tapvid3d_baseline_preds/waft_minival_is896_s-1_i4
S=result/OVERNIGHT_STATUS.md
for _ in $(seq 1 400); do [ -f "$CK" ] && break; sleep 30; done
[ -f "$CK" ] || { printf '  [watch] %s never appeared\n' "$CK" | tee -a "$S"; exit 1; }
sleep 20
printf '  [watch] scoring %s at %s\n' "$CK" "$(date '+%H:%M')" | tee -a "$S"
uv run python scripts/eval_metric3d.py --method v73 --ckpt "$CK" --depth da3g \
    --split minival --waft-pred-dir "$CACHE" --out-dir "result/v91/eval_$1" \
  || { printf '  [watch] eval failed\n' | tee -a "$S"; exit 1; }
uv run python - <<'PYEOF' 2>&1 | tee -a "$S"
import torch, glob
def sd(p):
    st = torch.load(p, map_location="cpu", weights_only=False); return st.get("model", st)
B = sd("result/v73fix_warmstart/ckpt_0.pt")
def rel(p, pre="v35."):
    A = sd(p)
    ks = [k for k in B if k.startswith(pre) and k in A and B[k].dtype.is_floating_point
          and B[k].shape == A[k].shape]
    n = sum(float((B[k].double()-A[k].double()).pow(2).sum()) for k in ks) ** 0.5
    d = sum(float(B[k].double().pow(2).sum()) for k in ks) ** 0.5
    return n/d
for f in sorted(glob.glob("result/v91/ckpt_*.pt"), key=lambda p: int(p.split('_')[-1][:-3])):
    print(f"  tracker displacement: {f.split('/')[-1]:<15} {rel(f):.4%}   (v90a hit 8.68% by step 500)")
PYEOF
uv run python scripts/report_metric_aj.py "result/v91/eval_$1" | tee -a "$S"
