#!/usr/bin/env bash
# Re-evaluate every paper row that ran through the operator, now that the refiners use the collapse
# path instead of the O(T^2) token-level one.
#
# Why: the token-level path was the module default, so every learned arm in the paper was scored
# through it. The two paths compute the same function -- they agree to ~3e-06 relative across
# {bidirectional} x {two_pool} -- so these numbers are expected to reproduce. Expected is not
# measured, and the whole reason this sweep exists is that a difference between the shipped
# artefact and the reported number went unnoticed once already.
#
# Every arm below previously completed all 25814 frames with zero failures, so any row that comes
# back short, or that moves by more than rounding, is a finding and not a re-run to be waved
# through. The two-pool arms are NOT here: result/collapse_2pool covers them, and they are the one
# case where the number is expected to change, because the token-level path lost 33 and 23 adt
# clips to OOM.
#
# Checkpoints and method flags are taken verbatim from each run's own metrics.json, not
# reconstructed from configs.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

DA3L="$HOME/data/tapvid3d_da3"
DA3G="$HOME/data/tapvid3d_da3nested"
WAFT="$HOME/data/tapvid3d_baseline_preds/waft"
LSSM=/home/mas/proj/study/largescale3Dreconstruction_using_SSM
OUT=result/collapse_reeval

# Wait for the two-pool re-evaluation, and anything else, to release the GPU.
for _ in $(seq 1 1440); do
  [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c .)" -eq 0 ] && break
  sleep 60
done
echo "[reeval-all] GPU idle at $(date -Is)"

# tag | method | ckpt | depth root | previous mean metric-AJ | extra args
ARMS=(
  "v35_searaft_da3l|v35|$LSSM/result/20260701_v35/ckpt_20000.pt|$DA3L|0.2263|"
  "v39_waft_da3l|v35|$LSSM/result/20260701_v35/ckpt_20000.pt|$DA3L|0.2458|--waft-pred-dir $WAFT"
  "v41_waft_da3g|v35|result/20260718-1952_v41/ckpt_20000.pt|$DA3G|0.2161|--waft-pred-dir $WAFT"
  "v42_waft_da3g_oversample|v35|result/v42_near/ckpt_20000.pt|$DA3G|0.2246|--waft-pred-dir $WAFT"
  "v43_waft_da3g_scalehead|v35|result/v43_scale/ckpt_20000.pt|$DA3G|0.2292|--waft-pred-dir $WAFT"
  "v44_waft_da3g_deflicker|v44|result/v44_deflicker/ckpt_20000.pt|$DA3G|0.2026|--waft-pred-dir $WAFT"
  "v45_waft_da3g_deflicker_refiner|v45|result/v45_scale/ckpt_20000.pt|$DA3G|0.2320|--waft-pred-dir $WAFT"
)

for row in "${ARMS[@]}"; do
  IFS='|' read -r tag method ckpt depth prev extra <<<"$row"
  out="$OUT/$tag"
  [ -f "$out/metrics.json" ] && { echo "[reeval-all] $tag done; skipping"; continue; }
  [ -f "$ckpt" ] || { echo "[reeval-all] $tag MISSING CKPT $ckpt"; continue; }
  echo "=== [$tag] $(date -Is)  (was $prev) ==="
  # shellcheck disable=SC2086
  uv run python scripts/eval_metric3d.py --method "$method" --ckpt "$ckpt" \
      --da3-depth-root "$depth" --split minival --out-dir "$out" $extra \
    || { echo "[reeval-all] $tag FAILED"; continue; }
  python3 - "$out/metrics.json" "$tag" "$prev" <<'PY'
import json, sys
d = json.load(open(sys.argv[1])); tag, prev = sys.argv[2], float(sys.argv[3])
f = d.get("failures"); n = len(f) if isinstance(f, list) else f
now = d["overall"]["metric_average_jaccard"]
flag = "OK" if abs(now - prev) < 5e-4 and d["frames"] == 25814 and not n else "*** CHECK ***"
print(f"[reeval-all] {tag}: was {prev:.4f} now {now:.4f} (delta {now-prev:+.4f}) "
      f"frames={d['frames']} failures={n} fps={d['fps']:.2f}  {flag}")
PY
done

echo "[reeval-all] done at $(date -Is)"
echo "  Any row marked *** CHECK *** moved by more than rounding, came back short of 25814 frames,"
echo "  or had failures. Those are the only rows that need the paper touched."
