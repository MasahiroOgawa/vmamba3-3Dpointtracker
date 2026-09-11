#!/usr/bin/env bash
# Stage 1 of v94: wait for the flow cache, train the head both non-causally and causally,
# and print the held-out gate. minival is deliberately NOT cached yet -- if the head loses
# to the forward-backward mask here it cannot help downstream, and the 167 min of WAFT the
# minival cache costs would be wasted.
set -u
cd "$(dirname "$0")/.."
source scripts/cudnn_env.sh
S=result/v94_status.log
note() { echo "$*" | tee -a "$S"; }

note "### v94 stage 1 started $(date '+%H:%M')"
while pgrep -f '[c]ache_flow_vis.py' >/dev/null; do sleep 60; done
note "  flow cache finished $(date '+%H:%M')"
grep -c 'MASK MISMATCH' result/v94_cache.log >/dev/null 2>&1 \
  && note "  WARNING: mask mismatches present" || note "  mask mismatches: 0"
for sp in train heldout; do
  note "  $sp: $(for ss in adt drivetrack pstudio; do
    printf '%s=%s ' "$ss" "$(ls ~/data/tapvid3d_flowvis/$sp/$ss 2>/dev/null | wc -l)"; done)"
done

for mode in true false; do
  tag=$([ "$mode" = true ] && echo noncausal || echo causal)
  cfg=$(mktemp /tmp/v94_XXXX.yaml)
  python3 - "$cfg" "$mode" "$tag" <<'PY'
import sys, yaml, pathlib
out, mode, tag = sys.argv[1], sys.argv[2] == "true", sys.argv[3]
c = yaml.safe_load(pathlib.Path("configs/v94.yaml").read_text())
c["model"]["bidirectional"] = mode
c["train"]["out_dir"] = f"result/v94_{tag}"
pathlib.Path(out).write_text(yaml.safe_dump(c, sort_keys=False))
PY
  note "  training v94_$tag (bidirectional=$mode) $(date '+%H:%M')"
  uv run python scripts/train_flow_vis_head.py "$cfg" >> "result/v94_${tag}.log" 2>&1 \
    || { note "  TRAIN FAILED: $tag"; rm -f "$cfg"; continue; }
  rm -f "$cfg"
  note "    $(grep 'DONE' "result/v94_${tag}.log" | tail -1)"
  note "    $(grep 'VAL' "result/v94_${tag}.log" | tail -1)"
done

note "### GATE 1 $(date '+%H:%M') -- head must beat the mask on held-out accuracy"
for tag in noncausal causal; do
  [ -f "result/v94_$tag/best.json" ] && note "  $tag: $(python3 -c "
import json;d=json.load(open('result/v94_$tag/best.json'))
print('mean head %.4f  mask %.4f  delta %+.4f  |  ' % (d['mean_head'],d['mean_mask'],d['mean_head']-d['mean_mask'])
      + '  '.join('%s %+.4f' % (k, v['head']-v['mask']) for k,v in d['per_subset'].items()))")"
done
note "### stage 1 done $(date '+%H:%M')"
