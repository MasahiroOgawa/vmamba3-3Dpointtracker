#!/usr/bin/env bash
# Re-evaluate both VSSD-2pool arms on the collapse path, over the FULL 25814 minival frames.
# The published 0.253 was scored on 18914 frames: 23 adt clips OOMed under the token-level path and
# were dropped, while the 0.246 baseline it is compared against covered all 25814. The exclusion is
# systematic, not random -- all adt clips are T=300, so what decided OOM was query count, meaning
# the clips with the most tracked points were the ones lost.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
CK=result/v50_twopool/ckpt_20000.pt
run () {
  local tag=$1; shift
  local out="result/collapse_2pool/$tag"
  [ -f "$out/metrics.json" ] && { echo "[reeval] $tag done"; return 0; }
  echo "=== [$tag] $(date -Is) ==="
  uv run python scripts/eval_metric3d.py --method v35 --ckpt "$CK" \
      --da3-depth-root "$HOME/data/tapvid3d_da3" --split minival --out-dir "$out" "$@" \
    || { echo "[reeval] $tag FAILED"; return 1; }
  python3 -c "
import json; d=json.load(open('$out/metrics.json'))
f=d.get('failures'); n=len(f) if isinstance(f,list) else f
print(f\"[reeval] $tag frames={d['frames']} failures={n} fps={d['fps']:.2f} mean_metric_AJ={d['overall']['metric_average_jaccard']:.4f}\")"
}
run searaft
run waft --waft-pred-dir "$HOME/data/tapvid3d_baseline_preds/waft"
echo "[reeval] done at $(date -Is); frames must read 25814 for the row to be usable"
