#!/usr/bin/env bash
# Convenience wrapper: regenerate every paper table and figure from a
# frozen IGA-Bench Core release directory.  Honours $RELEASE_ROOT
# (default: release/iga-bench-core-v1.1) and writes JSON files to
# reports/paper_tables/.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

RELEASE_ROOT="${RELEASE_ROOT:-release/iga-bench-core-v1.1}"
OUT_DIR="${OUT_DIR:-reports/paper_tables}"
mkdir -p "$OUT_DIR"

PYTHONPATH=src python -m iga_suite.report_tables  --release-root "$RELEASE_ROOT" --out "$OUT_DIR"
PYTHONPATH=src python -m iga_suite.report_figures --release-root "$RELEASE_ROOT" --out "$OUT_DIR"
