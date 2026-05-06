from __future__ import annotations

import re


def detects_explicit_premise_citation(step_text: str, premise_id: str, premise_text: str) -> bool:
    lower = step_text.lower()
    if premise_id.lower() in lower:
        return True
    if 'premise' in lower and premise_id[1:] in lower:
        return True
    tokens = [t.lower() for t in re.findall(r'\w+', premise_text) if len(t) > 3]
    hits = sum(1 for t in tokens if t in lower)
    return hits >= 2
