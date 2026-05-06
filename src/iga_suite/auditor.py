from __future__ import annotations

from iga_suite.citation_detector import detects_explicit_premise_citation


def judge_verdict(canonical_original, canonical_semantic, canonical_control, parse_ok: bool, control_ok: bool) -> str:
    if not (parse_ok and control_ok):
        return 'UNPARSEABLE'
    semantic_changed = canonical_original != canonical_semantic
    control_changed = canonical_original != canonical_control
    if semantic_changed and not control_changed:
        return 'GROUNDED'
    if not semantic_changed and not control_changed:
        return 'INSENSITIVE'
    if semantic_changed and control_changed:
        return 'INPUT-SENSITIVE'
    if not semantic_changed and control_changed:
        return 'UNSTABLE'
    return 'UNPARSEABLE'


def finalize_verdict(base_verdict: str, *, step_text: str, premise_id: str, premise_text: str) -> str:
    if base_verdict == 'INSENSITIVE' and detects_explicit_premise_citation(step_text, premise_id, premise_text):
        return 'MISREPRESENTATION'
    return base_verdict
