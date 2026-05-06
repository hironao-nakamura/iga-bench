#!/usr/bin/env python3
"""Compare parser v1.0 and v1.1 on the same raw traces.

Usage:
    python scripts/analysis/compare_parsers.py [--output-dir DIR] [--limit N]

Scans all companion/raw/**/*.json files, parses every response with both
v1.0 and v1.1, and produces:
  1. summary.json   — aggregate counts and coverage per benchmark/model
  2. recovered.jsonl — steps that were UNPARSEABLE in v1.0 but parsed in v1.1
  3. changed.jsonl   — steps where canonical form changed between versions
  4. report.txt      — human-readable summary
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from iga_suite.normalizer import (
    normalize_step_text as normalize_v10,
    parse_trace_steps as parse_v10,
)
from iga_suite.normalizer_v11 import (
    normalize_step_text as normalize_v11,
    parse_trace_steps as parse_v11,
)


def find_raw_jsons(base: Path, limit: int | None = None) -> list[Path]:
    files = sorted(base.rglob("*.json"))
    if limit:
        files = files[:limit]
    return files


def parse_response_both(response_text: str) -> list[dict]:
    """Parse a raw response with both v1.0 and v1.1, returning per-step comparisons."""
    steps_v10 = parse_v10(response_text)
    steps_v11 = parse_v11(response_text)

    v10_by_idx = {s["step_index"]: s for s in steps_v10}
    v11_by_idx = {s["step_index"]: s for s in steps_v11}

    all_indices = sorted(set(v10_by_idx) | set(v11_by_idx))
    results = []
    for idx in all_indices:
        s10 = v10_by_idx.get(idx, {})
        s11 = v11_by_idx.get(idx, {})
        results.append({
            "step_index": idx,
            "raw_text": s11.get("raw_text") or s10.get("raw_text", ""),
            "v10_canonical": s10.get("canonical_form"),
            "v10_type": s10.get("canonical_type", ""),
            "v10_status": s10.get("parse_status", ""),
            "v11_canonical": s11.get("canonical_form"),
            "v11_type": s11.get("canonical_type", ""),
            "v11_status": s11.get("parse_status", ""),
            "v11_rule": s11.get("canonicalizer_rule_id", ""),
            "recovered": s10.get("parse_status") == "UNPARSEABLE" and s11.get("parse_status") in ("OK", "BREAK_TOKENIZED"),
            "changed": (
                s10.get("canonical_form") != s11.get("canonical_form") and
                s10.get("parse_status") != "UNPARSEABLE"
            ),
        })
    return results


def extract_metadata(path: Path) -> dict:
    """Extract benchmark/model/problem from file path."""
    parts = path.parts
    try:
        raw_idx = parts.index("raw")
        benchmark = parts[raw_idx + 1] if raw_idx + 1 < len(parts) else "unknown"
        model_family = parts[raw_idx + 2] if raw_idx + 2 < len(parts) else "unknown"
        problem_id = parts[raw_idx + 3] if raw_idx + 3 < len(parts) else "unknown"
    except (ValueError, IndexError):
        benchmark = model_family = problem_id = "unknown"
    return {
        "benchmark": benchmark,
        "model_family": model_family,
        "problem_id": problem_id,
        "filename": path.name,
    }


def main():
    parser = argparse.ArgumentParser(description="Compare parser v1.0 and v1.1")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "parser_comparison")
    parser.add_argument("--limit", type=int, default=None, help="Limit total files processed")
    args = parser.parse_args()

    companion_base = ROOT / "outputs" / "eval_runs"
    companion_dirs = sorted(companion_base.glob("*_companion"))

    if not companion_dirs:
        print(f"No companion directories found in {companion_base}")
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_raw_files = []
    for d in companion_dirs:
        raw_dir = d / "raw"
        if raw_dir.exists():
            all_raw_files.extend(find_raw_jsons(raw_dir))

    if args.limit:
        all_raw_files = all_raw_files[:args.limit]

    print(f"Found {len(all_raw_files)} raw JSON files across {len(companion_dirs)} companions")

    stats = defaultdict(lambda: defaultdict(int))
    recovered_steps = []
    changed_steps = []
    total_steps = 0
    total_recovered = 0
    total_changed = 0
    files_processed = 0
    files_with_recovery = 0

    for fpath in all_raw_files:
        try:
            data = json.loads(fpath.read_text())
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue

        response = data.get("response", "")
        if not response:
            continue

        meta = extract_metadata(fpath)
        key = f"{meta['benchmark']}|{meta['model_family']}"

        comparisons = parse_response_both(response)
        files_processed += 1
        file_had_recovery = False

        for comp in comparisons:
            total_steps += 1
            stats[key]["total"] += 1
            stats[key][f"v10_{comp['v10_status']}"] = stats[key].get(f"v10_{comp['v10_status']}", 0) + 1
            stats[key][f"v11_{comp['v11_status']}"] = stats[key].get(f"v11_{comp['v11_status']}", 0) + 1

            if comp["recovered"]:
                total_recovered += 1
                stats[key]["recovered"] += 1
                file_had_recovery = True
                recovered_steps.append({
                    **meta,
                    **comp,
                })

            if comp["changed"]:
                total_changed += 1
                stats[key]["changed"] += 1
                changed_steps.append({
                    **meta,
                    **comp,
                })

        if file_had_recovery:
            files_with_recovery += 1

    # Build summary
    summary = {
        "total_files_processed": files_processed,
        "total_steps": total_steps,
        "total_recovered": total_recovered,
        "total_changed": total_changed,
        "files_with_recovery": files_with_recovery,
        "per_benchmark_model": {},
    }

    for key, counts in sorted(stats.items()):
        benchmark, model = key.split("|")
        total = counts["total"]
        v10_ok = counts.get("v10_OK", 0) + counts.get("v10_BREAK_TOKENIZED", 0)
        v11_ok = counts.get("v11_OK", 0) + counts.get("v11_BREAK_TOKENIZED", 0)
        recovered = counts.get("recovered", 0)
        summary["per_benchmark_model"][key] = {
            "benchmark": benchmark,
            "model_family": model,
            "total_steps": total,
            "v10_parsed": v10_ok,
            "v10_unparseable": total - v10_ok,
            "v10_coverage": round(v10_ok / total, 4) if total else 0,
            "v11_parsed": v11_ok,
            "v11_unparseable": total - v11_ok,
            "v11_coverage": round(v11_ok / total, 4) if total else 0,
            "recovered": recovered,
            "coverage_delta": round((v11_ok - v10_ok) / total, 4) if total else 0,
        }

    # Write outputs
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    with open(args.output_dir / "recovered.jsonl", "w") as f:
        for r in recovered_steps:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(args.output_dir / "changed.jsonl", "w") as f:
        for c in changed_steps:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    # Human-readable report
    lines = [
        "=" * 70,
        "Parser v1.0 vs v1.1 Comparison Report",
        "=" * 70,
        f"Files processed:       {files_processed}",
        f"Total steps:           {total_steps}",
        f"Steps recovered:       {total_recovered}",
        f"Steps changed (non-UNPARSEABLE):  {total_changed}",
        f"Files with >=1 recovery: {files_with_recovery}",
        "",
        "-" * 70,
        f"{'Benchmark/Model':<55} {'v1.0 cov':>8} {'v1.1 cov':>8} {'Delta':>8} {'Recov':>6}",
        "-" * 70,
    ]

    for key, s in sorted(summary["per_benchmark_model"].items()):
        lines.append(
            f"{key:<55} {s['v10_coverage']:>8.1%} {s['v11_coverage']:>8.1%} "
            f"{s['coverage_delta']:>+8.1%} {s['recovered']:>6}"
        )

    lines.extend(["", "-" * 70])

    v10_total_parsed = sum(s["v10_parsed"] for s in summary["per_benchmark_model"].values())
    v11_total_parsed = sum(s["v11_parsed"] for s in summary["per_benchmark_model"].values())
    lines.append(
        f"{'TOTAL':<55} {v10_total_parsed/total_steps:>8.1%} {v11_total_parsed/total_steps:>8.1%} "
        f"{(v11_total_parsed-v10_total_parsed)/total_steps:>+8.1%} {total_recovered:>6}"
    )
    lines.append("=" * 70)

    if recovered_steps:
        lines.extend(["", "Top recovered rule_ids:"])
        rule_counts = defaultdict(int)
        for r in recovered_steps:
            rule_counts[r["v11_rule"]] += 1
        for rule, cnt in sorted(rule_counts.items(), key=lambda x: -x[1])[:15]:
            lines.append(f"  {rule:<40} {cnt:>6}")

    if recovered_steps:
        lines.extend(["", "Sample recovered steps (first 20):"])
        for r in recovered_steps[:20]:
            lines.append(f"  [{r['benchmark']}|{r['model_family']}] {r['raw_text']}")
            lines.append(f"    v1.0: {r['v10_status']:12} → v1.1: {r['v11_status']:12} rule={r['v11_rule']}")
            lines.append(f"    v1.1 canonical: {r['v11_canonical']}")

    report_text = "\n".join(lines)
    (args.output_dir / "report.txt").write_text(report_text)
    print(report_text)


if __name__ == "__main__":
    main()
