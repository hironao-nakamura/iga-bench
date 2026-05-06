#!/usr/bin/env bash
# Build canonicalizer sensitivity artifacts under <release_root>/analysis/.
# Compares conservative (v1.0) vs released (v1.1-final) aggregate metrics on
# ProntoQA-500 and ProofWriter-300 across three model families.
#
# Usage:
#   build_parser_sensitivity.sh [RELEASE_ROOT]
#
# If RELEASE_ROOT is omitted, defaults to release/iga-bench-core-v1.1.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
source .venv/bin/activate

RELEASE_ROOT="${1:-release/iga-bench-core-v1.1}"

# Conservative (v1.0) canonicalizer outputs — same benchmarks as primary
# release. PQ500 legacy roots are named without a _v11 suffix; PW uses the
# proofwriter300_* roots.
CONSERVATIVE_ROOTS=(
  outputs/eval_runs/prontoqa_full_eval_openai
  outputs/eval_runs/prontoqa_full_eval_anthropic
  outputs/eval_runs/prontoqa_full_eval_qwen3_next_80b_a3b_instruct
  outputs/eval_runs/proofwriter300_openai
  outputs/eval_runs/proofwriter300_anthropic
  outputs/eval_runs/proofwriter300_openrouter_qwen3_next_80b_a3b_instruct
)

# Released (v1.1-final) canonicalizer outputs — these feed the primary
# release data/ and must match build_parser_sensitivity.sh exactly.
RELEASED_ROOTS=(
  outputs/eval_runs/prontoqa_full_eval_openai_v11
  outputs/eval_runs/prontoqa_full_eval_anthropic_v11
  outputs/eval_runs/prontoqa_full_eval_qwen3_next_80b_a3b_instruct_v11
  outputs/eval_runs/proofwriter300_openai_v11
  outputs/eval_runs/proofwriter300_anthropic_v11
  outputs/eval_runs/proofwriter300_openrouter_qwen3_next_80b_a3b_instruct_v11
)

PYTHONPATH=src python scripts/evaluation/build_parser_sensitivity.py \
  --release-root "$RELEASE_ROOT" \
  --conservative-roots "${CONSERVATIVE_ROOTS[@]}" \
  --released-roots "${RELEASED_ROOTS[@]}"

echo "=== Parser sensitivity analysis written to $RELEASE_ROOT/analysis/ ==="
