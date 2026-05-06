#!/usr/bin/env bash
# No-API-key smoke run for the iga-suite mock evaluation matrix.
#
# This script exercises the runtime end-to-end with the deterministic
# mock provider only.  It does NOT call materialize-metadata, because
# Croissant materialisation requires real (or example) hosted-archive
# URLs and a SHA-256 digest, which are reviewer-injected values that
# have no meaningful default for an offline smoke run.  Reviewers who
# want to regenerate metadata for an actual release should follow the
# README "Materialize metadata" section instead.
#
# The script is idempotent: it wipes the mock-only output directories
# under ``outputs/eval_runs/`` before each run so repeated invocations
# do not trip the "stale partition leaves" guard inside build-release.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

MOCK_DIRS=(
  outputs/eval_runs/prontoqa_mock
  outputs/eval_runs/prontoqa_mock_companion
  outputs/eval_runs/proofwriter_mock
  outputs/eval_runs/proofwriter_mock_companion
  outputs/eval_runs/release_mock
)
for d in "${MOCK_DIRS[@]}"; do
  rm -rf "$d"
done

PYTHONPATH=src python -m iga_suite.cli run-matrix \
  --matrix-spec configs/evaluation/matrix_mock.yaml
PYTHONPATH=src python -m iga_suite.cli build-release \
  --schema-path schema/parquet_schema_contract.yaml \
  --release-root outputs/eval_runs/release_mock \
  --release-id iga-bench-core-v1.1-mock \
  --output-roots outputs/eval_runs/prontoqa_mock outputs/eval_runs/proofwriter_mock
PYTHONPATH=src python -m iga_suite.cli validate-core \
  --dataset-root outputs/eval_runs/release_mock
