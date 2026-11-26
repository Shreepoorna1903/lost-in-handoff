#!/usr/bin/env bash
# Clean-baseline sweep (no faults, no defenses) over the first N non-no-op tasks.
# Usage: scripts/sweep_baseline.sh [model] [limit] [out_dir]
# Records: <out_dir>/<model>/<condition>/task<ID>__trial0.json (resumes if rerun)
# Progress: <out_dir>/sweep.log. Score afterwards with: python score_eval.py <out_dir>/<model>
set -u
cd "$(dirname "$0")/.."
MODEL="${1:-qwen3:8b}"
LIMIT="${2:-10}"
OUT="${3:-runs}"
LOG="$OUT/sweep.log"
mkdir -p "$OUT"

echo "=== sweep start $(date '+%F %T') model=$MODEL limit=$LIMIT" >> "$LOG"
for cond in single-clean clean-D0; do
  echo "=== $cond  $(date '+%T')" >> "$LOG"
  .venv/bin/python run_eval.py --condition "$cond" --model "$MODEL" --limit "$LIMIT" --out "$OUT" 2>&1 \
    | grep -E --line-buffered "^task|Traceback|Error" >> "$LOG"
done
echo "=== sweep end $(date '+%F %T') model=$MODEL" >> "$LOG"
