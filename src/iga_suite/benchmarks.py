from __future__ import annotations

import json
from pathlib import Path
import zipfile
from typing import Iterable


def load_generic_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_benchmark(loader: str, path: str | Path) -> list[dict]:
    """Load a benchmark that has already been normalized into the IGA JSONL format.

    Evaluation keeps the runtime loader intentionally small: benchmark-specific heavy lifting
    happens in bootstrap/normalization commands. This keeps the runtime pipeline benchmark-
    agnostic while still allowing us to add new bootstrap utilities.
    """
    if loader in {"generic_jsonl", "proofwriter_normalized_jsonl", "prontoqa_normalized_jsonl"}:
        return load_generic_jsonl(path)
    raise ValueError(f"Unsupported benchmark loader: {loader}")


def bootstrap_iclr_supplementary(supplementary_path: str | Path, output_path: str | Path) -> int:
    supplementary_path = Path(supplementary_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def iter_problem_meta_from_dir(root: Path):
        for p in sorted(root.glob('**/problem_meta.json')):
            if 'evidence' in p.parts:
                yield p.as_posix(), json.loads(p.read_text(encoding='utf-8'))

    def iter_problem_meta_from_zip(zpath: Path):
        with zipfile.ZipFile(zpath) as zf:
            names = sorted([n for n in zf.namelist() if n.endswith('problem_meta.json')])
            for n in names:
                yield n, json.loads(zf.read(n))

    iterator = iter_problem_meta_from_dir(supplementary_path) if supplementary_path.is_dir() else iter_problem_meta_from_zip(supplementary_path)

    seen = set()
    count = 0
    with open(output_path, 'w', encoding='utf-8') as out:
        for source_ref, meta in iterator:
            problem_id = meta['problem_id']
            if problem_id in seen:
                continue
            seen.add(problem_id)
            row = {
                'problem_id': problem_id,
                'benchmark_id': 'prontoqa_iclr50',
                'split': 'analysis',
                'question': meta['question'],
                'answer': meta.get('answer'),
                'premises': meta['premises'],
                'proof_tree': meta.get('proof_tree', []),
                'metadata': {
                    'source_record_ref': source_ref,
                    'difficulty_bin': f"{len(meta.get('proof_tree', []))}hop",
                },
            }
            out.write(json.dumps(row, ensure_ascii=False) + '\n')
            count += 1
    return count


# ---------------------------------------------------------------------------
# ProofWriter bootstrap support (Evaluation)
# ---------------------------------------------------------------------------


def _read_json_or_jsonl(path: Path) -> list[dict]:
    text = path.read_text(encoding='utf-8').strip()
    if not text:
        return []
    if path.suffix.lower() == '.jsonl':
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    payload = json.loads(text)
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        if 'data' in payload and isinstance(payload['data'], list):
            return payload['data']
        return [payload]
    raise ValueError(f'Unsupported JSON payload in {path}')


def _coerce_bool_answer(value) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    s = str(value).strip().lower()
    if s in {'true', 'yes', 'entails', 'supported', '1'}:
        return True
    if s in {'false', 'no', 'contradiction', 'refuted', '0'}:
        return False
    return None


def _normalize_premises(raw) -> list[dict]:
    if raw is None:
        return []
    premises: list[dict] = []
    if isinstance(raw, dict):
        items = list(raw.values())
    else:
        items = list(raw)
    for idx, item in enumerate(items, start=1):
        if isinstance(item, dict):
            pid = item.get('id') or item.get('pid') or f'P{idx}'
            text = item.get('text') or item.get('sentence') or item.get('fact') or item.get('rule')
        else:
            pid = f'P{idx}'
            text = str(item)
        if text is None:
            continue
        premises.append({'id': str(pid), 'text': str(text).strip().rstrip('.')})
    return premises


def _normalize_question(raw_row: dict) -> str:
    for key in ['question', 'query', 'hypothesis', 'statement']:
        if raw_row.get(key):
            return str(raw_row[key]).strip()
    raise ValueError('Could not find a question/hypothesis field in ProofWriter row')


def _normalize_proof_tree(raw_row: dict) -> list[dict]:
    """Normalize a permissive proof-step format into the IGA proof_tree schema.

    Supported already-normalized shape:
      [{'step': 1, 'conclusion': 'is(a,b)', 'depends_on': ['P1', 'P2']}, ...]

    Supported permissive shape:
      [{'id': 'S1', 'conclusion': 'X is a Y', 'premises': ['P1', 'P2']}]
      [{'step': 1, 'text': 'X is a Y', 'supports': ['P1', 'S1']}]

    We intentionally do NOT attempt to infer proof structure from free-form explanations here.
    If a row lacks explicit proof support, the caller should provide a pre-normalized proof_tree.
    """
    candidates = raw_row.get('proof_tree') or raw_row.get('proof') or raw_row.get('proof_steps') or raw_row.get('proofs')
    if candidates is None:
        return []
    if isinstance(candidates, dict) and 'steps' in candidates:
        candidates = candidates['steps']
    if not isinstance(candidates, list):
        return []
    out = []
    for idx, step in enumerate(candidates, start=1):
        if not isinstance(step, dict):
            continue
        sid = step.get('step') or step.get('id') or step.get('sid') or idx
        if isinstance(sid, str) and sid.upper().startswith('S'):
            sid = sid[1:]
        conclusion = step.get('conclusion') or step.get('canonical_conclusion') or step.get('text') or step.get('sentence')
        deps = step.get('depends_on') or step.get('premises') or step.get('supports') or step.get('parents') or []
        norm_deps = []
        for dep in deps:
            d = str(dep)
            if d.upper().startswith('S') or d.upper().startswith('P'):
                norm_deps.append(d.upper())
            else:
                # Assume bare integers refer to steps; everything else becomes a premise ID.
                norm_deps.append(f'S{d}' if d.isdigit() else d)
        if conclusion is None:
            continue
        out.append({'step': int(sid), 'conclusion': str(conclusion).strip(), 'depends_on': norm_deps})
    out.sort(key=lambda x: int(x['step']))
    return out


def _normalize_proofwriter_row(raw_row: dict, *, benchmark_id: str, split: str) -> dict:
    # Already normalized rows are passed through after a minimal sanity rewrite.
    if 'premises' in raw_row and 'question' in raw_row:
        premises = _normalize_premises(raw_row['premises'])
        proof_tree = _normalize_proof_tree(raw_row) or raw_row.get('proof_tree', [])
        answer = _coerce_bool_answer(raw_row['answer']) if 'answer' in raw_row else _coerce_bool_answer(raw_row.get('label'))
        problem_id = str(raw_row.get('problem_id') or raw_row.get('id') or raw_row.get('example_id'))
        if not problem_id:
            raise ValueError('Normalized row is missing problem_id')
        metadata = dict(raw_row.get('metadata') or {})
        metadata.setdefault('source_record_ref', raw_row.get('source_record_ref') or problem_id)
        metadata.setdefault('difficulty_bin', f"{len(proof_tree)}hop" if proof_tree else None)
        return {
            'problem_id': problem_id,
            'benchmark_id': benchmark_id,
            'split': split,
            'question': _normalize_question(raw_row),
            'answer': answer,
            'premises': premises,
            'proof_tree': proof_tree,
            'metadata': metadata,
        }

    # Permissive ProofWriter-like raw shape.
    problem_id = str(raw_row.get('id') or raw_row.get('example_id') or raw_row.get('uid') or raw_row.get('problem_id'))
    if not problem_id:
        raise ValueError('Could not infer problem_id for ProofWriter row')
    facts = raw_row.get('facts') or raw_row.get('triples') or raw_row.get('theory_facts') or []
    rules = raw_row.get('rules') or raw_row.get('theory_rules') or []
    theory = raw_row.get('theory')
    if isinstance(theory, dict):
        facts = theory.get('facts', facts)
        rules = theory.get('rules', rules)
    premises = _normalize_premises(list(facts) + list(rules))
    proof_tree = _normalize_proof_tree(raw_row)
    if 'answer' in raw_row:
        answer = _coerce_bool_answer(raw_row['answer'])
    elif 'label' in raw_row:
        answer = _coerce_bool_answer(raw_row['label'])
    else:
        answer = _coerce_bool_answer(raw_row.get('gold_answer'))
    return {
        'problem_id': problem_id,
        'benchmark_id': benchmark_id,
        'split': split,
        'question': _normalize_question(raw_row),
        'answer': answer,
        'premises': premises,
        'proof_tree': proof_tree,
        'metadata': {
            'source_record_ref': raw_row.get('source_record_ref') or problem_id,
            'difficulty_bin': raw_row.get('difficulty_bin') or (f"{len(proof_tree)}hop" if proof_tree else None),
        },
    }


def bootstrap_proofwriter(input_path: str | Path, output_path: str | Path, *, benchmark_id: str = 'proofwriter', split: str = 'analysis') -> int:
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = _read_json_or_jsonl(input_path)
    count = 0
    seen = set()
    with open(output_path, 'w', encoding='utf-8') as out:
        for raw in rows:
            norm = _normalize_proofwriter_row(raw, benchmark_id=benchmark_id, split=split)
            if norm['problem_id'] in seen:
                continue
            seen.add(norm['problem_id'])
            out.write(json.dumps(norm, ensure_ascii=False) + '\n')
            count += 1
    return count
