"""Reconstruct the ProofWriter-300 split from upstream ProofWriter seeds.

This script is provided for reproducibility of the v1.1
release-time construction: it reads the upstream ProofWriter dataset
under ``data/proofwriter/extracted/...`` and writes the canonical
``examples/current_release/proofwriter_cwa_d3_is_300{,_raw}.jsonl``
files that drive the primary release evaluation configs
(``proofwriter300_*.yaml``).

It is not part of the primary release evaluation path itself —
reviewers who only want to run the v1.1 evaluation should use the
already-shipped
``examples/current_release/proofwriter_cwa_d3_is_300.jsonl`` directly
via the ``proofwriter300_*.yaml`` configs.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "evaluation"))

from _proofwriter_records import build_records  # type: ignore

DATA_DIR = ROOT / "data" / "proofwriter" / "extracted" / "proofwriter-dataset-V2020.12.3" / "CWA" / "depth-3"
OUT_RAW = ROOT / "examples" / "current_release" / "proofwriter_cwa_d3_is_300_raw.jsonl"
OUT_NORM = ROOT / "examples" / "current_release" / "proofwriter_cwa_d3_is_300.jsonl"

TARGET = 300
BENCHMARK_ID = "proofwriter_cwa_d3_is_300"
SPLIT = "analysis"


def main() -> None:
    import tempfile

    sys.path.insert(0, str(ROOT / "src"))
    from iga_suite.benchmarks import bootstrap_proofwriter

    all_raw: list[dict] = []
    seen_ids: set[str] = set()

    for source_file in ["meta-dev.jsonl", "meta-test.jsonl", "meta-train.jsonl"]:
        src = DATA_DIR / source_file
        if not src.exists():
            continue
        with tempfile.NamedTemporaryFile(suffix=".jsonl", mode="w", delete=False) as tmp:
            build_records(src, Path(tmp.name), max_records=1000)
        with open(tmp.name, "r") as f:
            for line in f:
                rec = json.loads(line)
                rid = rec["id"]
                if rid not in seen_ids:
                    seen_ids.add(rid)
                    all_raw.append(rec)
        Path(tmp.name).unlink(missing_ok=True)
        if len(all_raw) >= TARGET:
            break

    all_raw = all_raw[:TARGET]
    print(f"Extracted {len(all_raw)} raw problems")

    OUT_RAW.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_RAW, "w", encoding="utf-8") as f:
        for rec in all_raw:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Wrote raw: {OUT_RAW}")

    count = bootstrap_proofwriter(OUT_RAW, OUT_NORM, benchmark_id=BENCHMARK_ID, split=SPLIT)
    print(f"Wrote normalized: {OUT_NORM} ({count} problems)")

    print(json.dumps({
        "status": "ok",
        "total": count,
        "raw_output": str(OUT_RAW),
        "normalized_output": str(OUT_NORM),
    }, indent=2))


if __name__ == "__main__":
    main()
