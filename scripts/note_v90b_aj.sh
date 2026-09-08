#!/usr/bin/env bash
# run_v90b.sh died before scoring and wrote `v90b = nan` into the status file. This appends the real
# number for ckpt_2000 -- the endpoint of the joint phase -- when the evaluation lands, so the
# morning report is correct without needing this session to still be awake.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
S=result/OVERNIGHT_STATUS.md
for _ in $(seq 1 240); do [ -f result/v90b/eval/metrics.json ] && break; sleep 30; done
if [ -f result/v90b/eval/metrics.json ]; then
  a=$(uv run python -c "import json;print(f\"{json.load(open('result/v90b/eval/metrics.json'))['overall']['metric_average_jaccard']:.4f}\")")
  { printf '### v90b (ckpt_2000, the joint-phase endpoint) = %s   %s\n' "$a" "$(date '+%H:%M')"
    printf '    supersedes the "v90b = nan" above: run_v90b.sh exited before its own evaluation.\n'
    printf '    v90b/best is ckpt_0 -- v90a unchanged, already 0.2132 -- so ckpt_2000 is the new information.\n'
  } | tee -a "$S"
  uv run python scripts/report_metric_aj.py result/v90b/eval >> "$S"
else
  printf '### v90b evaluation never produced metrics.json\n' | tee -a "$S"
fi
