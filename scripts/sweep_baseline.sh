#!/usr/bin/env bash
# Clean-baseline sweep (no faults, no added constraints) over the first N tasks.
# Usage: scripts/sweep_baseline.sh [model] [limit]
# Progress: one line per task in runs/baseline/sweep.log; full results in runs/baseline/*.json
set -u
cd "$(dirname "$0")/.."
MODEL="${1:-qwen3:8b}"
LIMIT="${2:-10}"
LOG=runs/baseline/sweep.log
mkdir -p runs/baseline

# topology depth format
CONDITIONS=(
  "single 3 free_text"
  "pipeline 3 structured"
)

echo "=== sweep start $(date '+%F %T') model=$MODEL limit=$LIMIT" >> "$LOG"
for spec in "${CONDITIONS[@]}"; do
  read -r topo depth fmt <<< "$spec"
  echo "=== $topo depth=$depth fmt=$fmt  $(date '+%T')" >> "$LOG"
  .venv/bin/python -m handoff.cli baseline --topology "$topo" --depth "$depth" --fmt "$fmt" \
      --model "$MODEL" --limit "$LIMIT" 2>&1 \
    | grep -E --line-buffered "^task|solved|Traceback|Error|Usage:" >> "$LOG"
done
echo "=== sweep end $(date '+%F %T')" >> "$LOG"
