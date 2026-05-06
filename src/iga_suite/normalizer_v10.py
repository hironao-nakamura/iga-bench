"""Parser v1.0 — conservative canonicalizer (legacy).

Preserved for parser-sensitivity analysis ONLY. The primary release uses the
v1.1 released canonicalizer (see `iga_suite.normalizer` which re-exports
`iga_suite.normalizer_v11`). Reviewers running `iga-suite run-eval` from the
README get v1.1 by default. `scripts/analysis/reprocess_v10.py` monkey-patches the
default normalizer back to this v1.0 surface when building the
`analysis/parser_sensitivity.parquet` artifact.
"""
from __future__ import annotations

import re

PARSER_VERSION = "1.0"


def _depluralize(word: str) -> str:
    w = word.lower()
    if w.endswith('uses') and len(w) > 4:
        return w[:-2]
    if w.endswith('es') and not w.endswith('us') and len(w) > 2:
        return w[:-2]
    if w.endswith('s') and not w.endswith('us') and len(w) > 1:
        return w[:-1]
    return w


def _normalize_word(word: str) -> str:
    w = word.lower()
    if w.endswith('use') and len(w) > 3:
        return w[:-1]
    if w.endswith('e') and len(w) > 3 and w[:-1].endswith('us'):
        return w[:-1]
    return w


def _tokenize_break(text: str) -> tuple[str, str, str] | None:
    lower = text.lower()

    if re.search(r'there is no (?:premise|rule) (?:linking|connecting)', lower) or re.search(r'(?:no premise connects|nothing links)\b', lower):
        m = re.search(
            r'(?:linking|connecting|connects|links)\s+([a-z][a-z0-9_-]*)\s+(?:to|and)\s+([a-z][a-z0-9_-]*)',
            lower,
        )
        if m:
            left = _normalize_word(_depluralize(m.group(1)))
            right = _normalize_word(_depluralize(m.group(2)))
            return f'break(missing_link:{left}->{right})', 'break', 'BREAK_TOKENIZED'
        return 'break(missing_link)', 'break', 'BREAK_TOKENIZED'

    if re.search(r'\bwe cannot conclude\b', lower) or re.search(r'\bcannot infer\b', lower):
        return 'break(cannot_conclude)', 'break', 'BREAK_TOKENIZED'

    if re.search(r'\bthe chain breaks\b', lower) or re.search(r'\bchain (?:breaks|fails)\b', lower):
        return 'break(chain_break)', 'break', 'BREAK_TOKENIZED'

    return None


def normalize_step_text(step_text: str) -> tuple[str | None, str, str]:
    text = step_text.strip().rstrip('.')
    text = re.sub(r"\s*\((?:given\s+in|from|by|via|per|using|see)\s+premise\s*\d*\)", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*\(premise\s+\d+\)", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*\(given\)", "", text, flags=re.IGNORECASE)
    text = text.strip().rstrip('.')

    break_token = _tokenize_break(text)
    if break_token is not None:
        return break_token

    m = re.search(r",\s*so\s+(.+)$", text, re.I)
    if m:
        text = m.group(1).strip()
    else:
        m = re.match(r"^(?:Since|Because|As|Given that)\s+.+,\s*(.+)$", text, re.I)
        if m:
            text = m.group(1).strip()
    text = re.sub(r"^(?:Therefore|Thus|So|Hence|Consequently),?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^(?:We know that|We can conclude that|It follows that|This means that|This tells us that)\s+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^(?:From|According to)\s+(?:the )?(?:premises?|the facts?|premise \d+),?\s*", "", text, flags=re.IGNORECASE)
    text = text.strip()

    m = re.match(r"(\w+)\s+is\s+not\s+(?:a|an)\s+(\w+)$", text, re.I)
    if m:
        return f"not_is({_normalize_word(m.group(1))}, {_normalize_word(m.group(2))})", 'not_is', 'OK'
    m = re.match(r"(\w+)\s+is\s+(?:a|an)\s+(\w+)$", text, re.I)
    if m:
        return f"is({_normalize_word(m.group(1))}, {_normalize_word(m.group(2))})", 'is', 'OK'
    m = re.match(r"(\w+)\s+is\s+one\s+of\s+the\s+(\w+)$", text, re.I)
    if m:
        return f"is({_normalize_word(m.group(1))}, {_normalize_word(_depluralize(m.group(2)))})", 'is', 'OK'
    m = re.match(r"No\s+(\w+)\s+is\s+(?:a|an)\s+(\w+)$", text, re.I)
    if m:
        return f"not_subtype({_normalize_word(m.group(1))}, {_normalize_word(m.group(2))})", 'not_subtype', 'OK'
    m = re.match(r"(?:All|Every|Each)\s+(\w+)\s+is\s+(?:a|an)\s+(\w+)$", text, re.I)
    if m:
        return f"subtype({_normalize_word(_depluralize(m.group(1)))}, {_normalize_word(m.group(2))})", 'subtype', 'OK'
    m = re.match(r"(?:All|Every|Each)\s+(\w+)\s+are\s+(\w+)$", text, re.I)
    if m:
        return f"subtype({_normalize_word(_depluralize(m.group(1)))}, {_normalize_word(_depluralize(m.group(2)))})", 'subtype', 'OK'
    m = re.match(r"(\w+)\s+are\s+not\s+(\w+)$", text, re.I)
    if m:
        return f"not_subtype({_normalize_word(_depluralize(m.group(1)))}, {_normalize_word(_depluralize(m.group(2)))})", 'not_subtype', 'OK'
    m = re.match(r"(\w+)\s+are\s+(\w+)$", text, re.I)
    if m:
        return f"subtype({_normalize_word(_depluralize(m.group(1)))}, {_normalize_word(_depluralize(m.group(2)))})", 'subtype', 'OK'
    return None, 'free_form', 'UNPARSEABLE'


def parse_trace_steps(raw_response: str) -> list[dict]:
    steps = []
    for line in raw_response.splitlines():
        m = re.match(r"Step\s+(\d+)\s*:\s*(.+)$", line.strip())
        if not m:
            continue
        step_id = int(m.group(1))
        raw_text = m.group(2).strip().rstrip('.')
        canonical, canonical_type, parse_status = normalize_step_text(raw_text)
        steps.append({
            'step_index': step_id,
            'raw_text': raw_text,
            'canonical_form': canonical,
            'canonical_type': canonical_type,
            'parse_status': parse_status,
            'emits_break_token': parse_status == 'BREAK_TOKENIZED',
            'break_token_type': canonical_type if parse_status == 'BREAK_TOKENIZED' else None,
        })
    return steps


def parse_final_answer(raw_response: str) -> str | None:
    m = re.search(r'Answer\s*:\s*(True|False)', raw_response, re.I)
    if m:
        return m.group(1).capitalize()
    return None
