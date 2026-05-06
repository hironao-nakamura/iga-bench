from __future__ import annotations

import re


def _depluralize(word: str) -> str:
    w = word.lower()
    if w.endswith('uses') and len(w) > 4:
        return w[:-2]
    if w.endswith('es') and not w.endswith('us') and len(w) > 2:
        return w[:-2]
    if w.endswith('s') and not w.endswith('us') and len(w) > 1:
        return w[:-1]
    return w


def _pluralize(word: str) -> str:
    w = word.lower()
    if w.endswith('us'):
        return w + 'es'
    if w.endswith('s'):
        return w + 'es'
    return w + 's'


def _token_forms(word: str) -> list[str]:
    singular = word.lower()
    plural = _pluralize(singular)
    return [plural.capitalize(), plural, singular.capitalize(), singular]


def _contains_duplicate_prefix(text: str) -> bool:
    return re.search(r'\bzqzq\w*\b', text.lower()) is not None


def _render_replace(text: str, old: str, new: str) -> str:
    # Single-pass exact-token replacement avoids recursive double application.
    old_forms = _token_forms(old)
    new_forms = _token_forms(new)
    mapping = {src: dst for src, dst in zip(old_forms, new_forms)}
    pattern = r'\b(?:' + '|'.join(re.escape(tok) for tok in old_forms) + r')\b'
    return re.sub(pattern, lambda m: mapping[m.group(0)], text)


def _count_forms(text: str, base: str) -> int:
    forms = _token_forms(base)
    pattern = r'\b(?:' + '|'.join(re.escape(tok) for tok in forms) + r')\b'
    return len(re.findall(pattern, text))


def _validate_modified_premise(
    *,
    original_text: str,
    modified_text: str,
    target_pred: str,
    substitute: str,
    probe_type: str,
    is_target_premise: bool,
):
    if _contains_duplicate_prefix(modified_text):
        raise ValueError(f'probe corruption detected ({probe_type}): duplicate zq-prefix in "{modified_text}"')

    old_count = _count_forms(original_text, target_pred)
    new_count = _count_forms(modified_text, substitute)
    remaining_old = _count_forms(modified_text, target_pred)

    should_apply = probe_type == 'semantic_substitution' or (probe_type == 'local_substitution' and is_target_premise)
    if should_apply:
        if old_count > 0:
            if new_count != old_count:
                raise ValueError(
                    f'probe corruption detected ({probe_type}): expected {old_count} replacement(s), got {new_count} in "{modified_text}"'
                )
            if remaining_old != 0:
                raise ValueError(
                    f'probe corruption detected ({probe_type}): target predicate was only partially replaced in "{modified_text}"'
                )
    elif probe_type == 'local_substitution' and not is_target_premise:
        if modified_text != original_text:
            raise ValueError(
                f'probe corruption detected ({probe_type}): non-target premise changed unexpectedly: "{modified_text}"'
            )


def _get_target_predicate(text: str) -> str | None:
    m = re.match(r'(\w+)\s+are\s+(?:not\s+)?(\w+)$', text, re.I)
    if m:
        return _depluralize(m.group(2))
    m = re.match(r'(\w+)\s+is\s+(?:a|an)\s+(\w+)$', text, re.I)
    if m:
        return m.group(2).lower()
    return None


def _surface_rephrase(text: str) -> str:
    m = re.match(r'(\w+?)(?:es|s)\s+are\s+not\s+(\w+?)(?:es|s)$', text, re.I)
    if m:
        return f"No {m.group(1).lower()} is a {m.group(2).lower()}"
    m = re.match(r'(\w+?)(?:es|s)\s+are\s+(\w+?)(?:es|s)$', text, re.I)
    if m:
        return f"Every {m.group(1).lower()} is a {m.group(2).lower()}"
    m = re.match(r'(\w+)\s+is\s+a\s+(\w+)$', text, re.I)
    if m:
        return f"{m.group(1)} is one of the {_pluralize(m.group(2).lower())}"
    return text


def _null_rephrase(text: str) -> str:
    return text if text.endswith('.') else text + '.'


def _is_entity_premise(text: str) -> bool:
    return re.match(r'(\w+)\s+is\s+a\s+(\w+)$', text, re.I) is not None


def _entity_subject(text: str) -> str | None:
    m = re.match(r'(\w+)\s+is\s+a\s+(\w+)$', text, re.I)
    if not m:
        return None
    return m.group(1)


def _render_entity_replace(text: str, old_entity: str, new_entity: str) -> str:
    pattern = r'\b' + re.escape(old_entity) + r'\b'
    return re.sub(pattern, new_entity, text, count=1)


def generate_probes(problem: dict, include_null_probe: bool = True) -> list[dict]:
    probes = []
    for premise in problem['premises']:
        target_pred = _get_target_predicate(premise['text'])
        if target_pred is None:
            continue
        substitute = 'zq' + target_pred

        modified = []
        for p in problem['premises']:
            new_text = _render_replace(p['text'], target_pred, substitute)
            _validate_modified_premise(
                original_text=p['text'],
                modified_text=new_text,
                target_pred=target_pred,
                substitute=substitute,
                probe_type='semantic_substitution',
                is_target_premise=(p['id'] == premise['id']),
            )
            modified.append({'id': p['id'], 'text': new_text})
        probes.append({
            'probe_id': f"{problem['problem_id']}::{premise['id']}::semantic",
            'problem_id': problem['problem_id'],
            'target_premise': premise['id'],
            'probe_type': 'semantic_substitution',
            'intervention_scope': 'consistent',
            'render_rule_id': 'zq_predicate_substitution_v1',
            'semantic_target_ref': target_pred,
            'modified_premises': modified,
        })

        modified = []
        for p in problem['premises']:
            txt = _render_replace(p['text'], target_pred, substitute) if p['id'] == premise['id'] else p['text']
            _validate_modified_premise(
                original_text=p['text'],
                modified_text=txt,
                target_pred=target_pred,
                substitute=substitute,
                probe_type='local_substitution',
                is_target_premise=(p['id'] == premise['id']),
            )
            modified.append({'id': p['id'], 'text': txt})
        probes.append({
            'probe_id': f"{problem['problem_id']}::{premise['id']}::local",
            'problem_id': problem['problem_id'],
            'target_premise': premise['id'],
            'probe_type': 'local_substitution',
            'intervention_scope': 'local',
            'render_rule_id': 'zq_predicate_substitution_v1',
            'semantic_target_ref': target_pred,
            'modified_premises': modified,
        })

        if _is_entity_premise(premise['text']):
            entity = _entity_subject(premise['text'])
            if entity:
                replacement = f"zq{entity}"
                modified = []
                for p in problem['premises']:
                    if p['id'] == premise['id']:
                        txt = _render_entity_replace(p['text'], entity, replacement)
                    else:
                        txt = p['text']
                    modified.append({'id': p['id'], 'text': txt})
                probes.append({
                    'probe_id': f"{problem['problem_id']}::{premise['id']}::entity",
                    'problem_id': problem['problem_id'],
                    'target_premise': premise['id'],
                    'probe_type': 'entity_substitution',
                    'intervention_scope': 'local',
                    'render_rule_id': 'entity_substitution_v1',
                    'semantic_target_ref': f"entity:{entity}->{replacement}",
                    'modified_premises': modified,
                })

        modified = []
        surface_rule_id = 'surface_rephrase_v1'
        for p in problem['premises']:
            if p['id'] == premise['id']:
                if _is_entity_premise(p['text']):
                    txt = _null_rephrase(p['text'])
                    surface_rule_id = 'surface_control_entity_null_v1'
                else:
                    txt = _surface_rephrase(p['text'])
            else:
                txt = p['text']
            modified.append({'id': p['id'], 'text': txt})
        probes.append({
            'probe_id': f"{problem['problem_id']}::{premise['id']}::surface",
            'problem_id': problem['problem_id'],
            'target_premise': premise['id'],
            'probe_type': 'surface_control',
            'intervention_scope': 'control',
            'render_rule_id': surface_rule_id,
            'semantic_target_ref': target_pred,
            'modified_premises': modified,
        })

        if include_null_probe:
            modified = []
            for p in problem['premises']:
                txt = _null_rephrase(p['text']) if p['id'] == premise['id'] else p['text']
                modified.append({'id': p['id'], 'text': txt})
            probes.append({
                'probe_id': f"{problem['problem_id']}::{premise['id']}::null",
                'problem_id': problem['problem_id'],
                'target_premise': premise['id'],
                'probe_type': 'null_probe',
                'intervention_scope': 'control',
                'render_rule_id': 'null_punctuation_v1',
                'semantic_target_ref': target_pred,
                'modified_premises': modified,
            })
    return probes
