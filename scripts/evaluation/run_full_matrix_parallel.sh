#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

if [[ -f ".venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

export OPENAI_API_KEY="${OPENAI_API_KEY:?Set OPENAI_API_KEY}"
export ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY:?Set ANTHROPIC_API_KEY}"

if [[ -z "${OPENAI_COMPATIBLE_API_KEY:-}" && -n "${OPENROUTER_API_KEY:-}" ]]; then
  export OPENAI_COMPATIBLE_API_KEY="$OPENROUTER_API_KEY"
fi
export OPENAI_COMPATIBLE_API_KEY="${OPENAI_COMPATIBLE_API_KEY:?Set OPENAI_COMPATIBLE_API_KEY (or OPENROUTER_API_KEY as a compatibility alias)}"
export OPENAI_COMPATIBLE_BASE_URL="${OPENAI_COMPATIBLE_BASE_URL:-https://openrouter.ai/api/v1}"

LOG_DIR="$PROJECT_ROOT/logs"
mkdir -p "$LOG_DIR"

CONFIGS=(
  configs/evaluation/prontoqa_full_eval_openai.yaml
  configs/evaluation/prontoqa_full_eval_anthropic.yaml
  configs/evaluation/prontoqa_full_eval_qwen3_next_80b_a3b_instruct.yaml
  configs/evaluation/proofwriter300_openai.yaml
  configs/evaluation/proofwriter300_anthropic.yaml
  configs/evaluation/proofwriter300_openrouter_qwen3_next_80b_a3b_instruct.yaml
)

PIDS=()
for cfg in "${CONFIGS[@]}"; do
  name=$(basename "$cfg" .yaml)
  logfile="$LOG_DIR/${name}.log"
  echo "[$(date '+%H:%M:%S')] Starting: $name -> $logfile"
  python3 -m iga_suite.cli run-eval --config "$cfg" > "$logfile" 2>&1 &
  PIDS+=($!)
  sleep 1
done

echo ""
echo "All 6 runs launched. PIDs: ${PIDS[*]}"
echo "Monitor with: tail -f $LOG_DIR/*.log"
echo ""

FAILED=0
for i in "${!PIDS[@]}"; do
  pid=${PIDS[$i]}
  cfg=${CONFIGS[$i]}
  name=$(basename "$cfg" .yaml)
  if wait "$pid"; then
    echo "[$(date '+%H:%M:%S')] DONE (success): $name"
  else
    echo "[$(date '+%H:%M:%S')] DONE (FAILED): $name"
    FAILED=$((FAILED + 1))
  fi
done

echo ""
if [ "$FAILED" -eq 0 ]; then
  echo "ALL 6 RUNS COMPLETED SUCCESSFULLY"
else
  echo "$FAILED / 6 RUNS FAILED - check logs in $LOG_DIR/"
fi
