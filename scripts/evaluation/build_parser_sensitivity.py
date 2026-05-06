#!/usr/bin/env python
"""Generate canonicalizer sensitivity analysis artifacts.

Compares aggregate metrics computed with the conservative (v1.0) canonicalizer
against those computed with the released (v1.1-final) canonicalizer, for the
same benchmarks and model families, and writes the result to
``<release_root>/analysis/`` as both Parquet and JSON.

This is explicitly NOT part of the primary release data/ table. It is a
sensitivity artifact for reviewers who want to see how the parser choice
affects coverage and F1 on identical problem sets.

Inputs
------
Two sets of evaluation output roots:

* --conservative-roots : directories produced by the v1.0 (conservative)
  parser. Each root must contain ``data/aggregate_metrics/...``.
* --released-roots : directories produced by the v1.1-final parser.

The script loads aggregate_metrics from each set, runs
``iga_suite.report_tables.build_parser_sensitivity``, and writes:

* ``<release_root>/analysis/parser_sensitivity.parquet``
* ``<release_root>/analysis/parser_sensitivity.json``
* ``<release_root>/analysis/README.md``
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd


_PARTITION_KEYS = ("benchmark_id", "split", "model_family", "config_id")


def _extract_partitions(path: Path, base: Path) -> dict[str, str]:
    """Recover ``key=value`` Hive-style partition fields from the path
    between *base* and *path* (exclusive of the parquet filename).
    """
    rel = path.relative_to(base)
    out: dict[str, str] = {}
    for part in rel.parts[:-1]:
        if "=" in part:
            k, _, v = part.partition("=")
            if k in _PARTITION_KEYS:
                out[k] = v
    return out


def _read_aggregate_metrics(roots: Iterable[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for root in roots:
        agg_dir = root / "data" / "aggregate_metrics"
        if not agg_dir.exists():
            print(f"[warn] skipping {root}: no data/aggregate_metrics/", file=sys.stderr)
            continue
        for pq in agg_dir.rglob("*.parquet"):
            df = pd.read_parquet(pq)
            partitions = _extract_partitions(pq, agg_dir)
            for k, v in partitions.items():
                if k not in df.columns:
                    df[k] = v
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--release-root", required=True, type=Path,
                    help="Primary release root; analysis/ is written under this directory.")
    ap.add_argument("--conservative-roots", nargs="+", required=True, type=Path,
                    help="Output roots processed with the v1.0 conservative canonicalizer.")
    ap.add_argument("--released-roots", nargs="+", required=True, type=Path,
                    help="Output roots processed with the v1.1-final released canonicalizer.")
    args = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from iga_suite.report_tables import build_parser_sensitivity  # noqa: WPS433

    conservative = _read_aggregate_metrics(args.conservative_roots)
    released = _read_aggregate_metrics(args.released_roots)

    if conservative.empty:
        print("[error] no conservative aggregate_metrics loaded", file=sys.stderr)
        return 2
    if released.empty:
        print("[error] no released aggregate_metrics loaded", file=sys.stderr)
        return 2

    sensitivity = build_parser_sensitivity(conservative, released)
    if sensitivity.empty:
        print("[error] build_parser_sensitivity returned empty frame", file=sys.stderr)
        return 3

    analysis_dir = args.release_root / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = analysis_dir / "parser_sensitivity.parquet"
    json_path = analysis_dir / "parser_sensitivity.json"
    readme_path = analysis_dir / "README.md"

    sensitivity.to_parquet(parquet_path, index=False)
    sensitivity.to_json(json_path, orient="records", indent=2, force_ascii=False)

    readme_path.write_text(
        "# Canonicalizer sensitivity analysis\n\n"
        "Files in this directory are NOT part of the primary single source of "
        "truth under `data/`. They exist only to let reviewers see how the "
        "released surface-relaxed canonicalizer (v1.1-final) compares to the "
        "earlier conservative canonicalizer (v1.0) on identical benchmarks and "
        "models.\n\n"
        "- `parser_sensitivity.parquet` / `parser_sensitivity.json`\n"
        "  One row per (benchmark_id, model_family, dependency_mode) with the "
        "coverage and F1 under each parser, their deltas, and an "
        "`in_primary_release` flag marking the (ProntoQA-500, ProofWriter-300) "
        "rows that drive the paper numerics.\n",
        encoding="utf-8",
    )

    summary = {
        "rows": int(len(sensitivity)),
        "benchmarks": sorted(sensitivity["benchmark_id"].dropna().unique().tolist()),
        "model_families": sorted(sensitivity["model_family"].dropna().unique().tolist())
            if "model_family" in sensitivity.columns else [],
        "primary_release_rows": int(sensitivity.get("in_primary_release", pd.Series(dtype=bool)).sum()),
        "parquet": str(parquet_path),
        "json": str(json_path),
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
