from __future__ import annotations

import json
import random
from pathlib import Path


def sample_steps_for_audit(trace_steps_jsonl: str | Path, out_jsonl: str | Path, n: int = 200, seed: int = 42) -> dict:
    rows = []
    p = Path(trace_steps_jsonl)
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    random.Random(seed).shuffle(rows)
    sample = rows[: min(n, len(rows))]
    out = Path(out_jsonl)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w', encoding='utf-8') as f:
        for r in sample:
            payload = {
                'trace_step_id': r.get('trace_step_id'),
                'raw_step_ref': r.get('raw_step_ref'),
                'canonical_pred': r.get('canonical_form'),
                'parse_status_pred': r.get('parse_status'),
                'break_pred': r.get('emits_break_token'),
                'step_type_pred': r.get('canonical_type'),
                'canonical_gold': None,
                'parse_status_gold': None,
                'break_gold': None,
                'step_type_gold': None,
            }
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return {'written': len(sample), 'output': str(out)}


def evaluate_annotations(annot_jsonl: str | Path, out_json: str | Path) -> dict:
    rows = []
    p = Path(annot_jsonl)
    for line in p.read_text(encoding='utf-8').splitlines():
        if line.strip():
            rows.append(json.loads(line))

    def safe_eq(a, b):
        return (a is not None and b is not None and a == b)

    n = len(rows)
    exact = sum(1 for r in rows if safe_eq(r.get('canonical_pred'), r.get('canonical_gold')))
    break_ok = sum(1 for r in rows if safe_eq(r.get('break_pred'), r.get('break_gold')))
    type_ok = sum(1 for r in rows if safe_eq(r.get('step_type_pred'), r.get('step_type_gold')))
    parse_ok = sum(1 for r in rows if safe_eq(r.get('parse_status_pred'), r.get('parse_status_gold')))

    report = {
        'n': n,
        'exact_canonical_accuracy': (exact / n) if n else 0.0,
        'break_detection_accuracy': (break_ok / n) if n else 0.0,
        'step_type_accuracy': (type_ok / n) if n else 0.0,
        'parse_status_accuracy': (parse_ok / n) if n else 0.0,
    }
    outp = Path(out_json)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    return report
