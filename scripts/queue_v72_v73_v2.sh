#!/usr/bin/env bash
# v72 (depth-scale refiner, linear-initialised, early-stopped) then v73 (that stage + vmamba3-2pool,
# 20000 steps, warm-started from v72).
#
# Everything experimental is in the configs. The one thing this script writes into a config is
# v73's train.init_ckpt, because the path depends on where v72's early stopping landed -- and it is
# written into the config rather than passed as a flag so cfg.json records it.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
# shellcheck source=/dev/null
source "$(dirname "$0")/cudnn_env.sh"
CACHE=$HOME/data/tapvid3d_baseline_preds/waft_minival_is896_s-1_i4
INIT=result/scale_linear_init.npz

# The linear initialisation is the point of this attempt; without it, do not start.
for _ in $(seq 1 120); do
  [ -f "$INIT" ] && break
  pgrep -f fit_scale_linear_init >/dev/null || { echo "[queue] linear fit died before writing $INIT"; exit 1; }
  sleep 60
done
[ -f "$INIT" ] || { echo "[queue] $INIT never appeared"; exit 1; }
echo "[queue] linear init present at $(date -Is)"

for _ in $(seq 1 240); do
  [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c .)" -eq 0 ] && break
  sleep 60
done
echo "[queue] GPU idle at $(date -Is)"

# ---- v72 ----------------------------------------------------------------------------------------
V72OUT=$(uv run python -c "import yaml;print(yaml.safe_load(open('configs/v72.yaml'))['train']['out_dir'])")
mkdir -p "$V72OUT"
V72STEPS=$(uv run python -c "import yaml;print(yaml.safe_load(open('configs/v72.yaml'))['train']['steps'])")
# Keyed on the config, not on "any checkpoint exists": a stale checkpoint from a previous
# architecture satisfied the old test, so v72 was skipped and v73 then failed loading it. Early
# stopping means the final step may be below the ceiling, so a best/ checkpoint also counts.
if [ -z "$(ls "$V72OUT"/best/ckpt_*.pt 2>/dev/null)" ] && [ ! -f "$V72OUT/ckpt_${V72STEPS}.pt" ]; then
  echo "[v72] training $(date -Is)"
  uv run python scripts/train_depth_refined_tracker.py --config configs/v72.yaml \
    || { echo "[v72] TRAIN FAILED"; exit 1; }
fi
V72CK=$(ls -t "$V72OUT"/best/ckpt_*.pt "$V72OUT"/ckpt_*.pt 2>/dev/null | head -1)
[ -n "$V72CK" ] || { echo "[v72] no checkpoint"; exit 1; }
echo "[v72] using $V72CK"
if [ ! -f "$V72OUT/eval/metrics.json" ]; then
  uv run python scripts/eval_metric3d.py --method v72 --ckpt "$V72CK" \
      --da3-depth-root "$HOME/data/tapvid3d_da3nested" --split minival \
      --waft-pred-dir "$CACHE" --out-dir "$V72OUT/eval" || echo "[v72] EVAL FAILED"
fi
[ -f "$V72OUT/eval/metrics.json" ] && uv run python -c "
import json; d=json.load(open('$V72OUT/eval/metrics.json')); ps=d['per_subset']
print('[v72] ' + ' '.join(f\"{s}={ps[s]['metric_average_jaccard']:.4f}\" for s in ('drivetrack','pstudio','adt'))
      + f\" mean={d['overall']['metric_average_jaccard']:.4f}\")
print('[v72] against v69 0.1998 (old stage) and the linear map alone 0.3217')"

# ---- v73, warm-started from v72 -----------------------------------------------------------------
uv run python - "$V72CK" <<'PY'
import sys, yaml, pathlib
p = pathlib.Path("configs/v73.yaml"); c = yaml.safe_load(p.read_text())
c["train"]["init_ckpt"] = sys.argv[1]
txt = p.read_text(); hdr = txt[:txt.index("version:")]
p.write_text(hdr + yaml.safe_dump(c, sort_keys=False))
print(f"[queue] configs/v73.yaml train.init_ckpt = {sys.argv[1]}")
PY
V73OUT=$(uv run python -c "import yaml;print(yaml.safe_load(open('configs/v73.yaml'))['train']['out_dir'])")
mkdir -p "$V73OUT"
if [ ! -f "$V73OUT/ckpt_20000.pt" ]; then
  echo "[v73] training $(date -Is)"
  uv run python scripts/train_depth_refined_tracker.py --config configs/v73.yaml \
    || { echo "[v73] TRAIN FAILED"; exit 1; }
fi
if [ ! -f "$V73OUT/eval/metrics.json" ]; then
  uv run python scripts/eval_metric3d.py --method v73 --ckpt "$V73OUT/ckpt_20000.pt" \
      --da3-depth-root "$HOME/data/tapvid3d_da3nested" --split minival \
      --waft-pred-dir "$CACHE" --out-dir "$V73OUT/eval" || { echo "[v73] EVAL FAILED"; exit 1; }
fi
uv run python -c "
import json; d=json.load(open('$V73OUT/eval/metrics.json')); ps=d['per_subset']
print('[v73] ' + ' '.join(f\"{s}={ps[s]['metric_average_jaccard']:.4f}\" for s in ('drivetrack','pstudio','adt'))
      + f\" mean={d['overall']['metric_average_jaccard']:.4f} frames={d['frames']} fail={d.get('failures')}\")
print('[v73] targets: v63 0.2238 | v45 0.2320 | v68 0.2218 | oracle 0.5033')"
echo "[queue] both done at $(date -Is)"
