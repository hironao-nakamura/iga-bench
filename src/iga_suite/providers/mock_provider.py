from __future__ import annotations

import re
from collections import OrderedDict

from iga_suite.providers.base import BaseProvider, ProviderResponse


def _depluralize(word: str) -> str:
    w = word.lower()
    if w.endswith('uses') and len(w) > 4:
        return w[:-2]
    if w.endswith('es') and not w.endswith('us') and len(w) > 2:
        return w[:-2]
    if w.endswith('s') and not w.endswith('us') and len(w) > 1:
        return w[:-1]
    return w


def _parse_premise_rule(text: str):
    t = text.strip().rstrip('.')
    m = re.match(r'(\w+)\s+is\s+not\s+(?:a|an)\s+(\w+)$', t, re.I)
    if m:
        return ('neg_fact', m.group(1).lower(), m.group(2).lower())
    m = re.match(r'(\w+)\s+is\s+(?:a|an)\s+(\w+)$', t, re.I)
    if m:
        return ('fact', m.group(1).lower(), m.group(2).lower())
    m = re.match(r'No\s+(\w+)\s+is\s+(?:a|an)\s+(\w+)$', t, re.I)
    if m:
        return ('not_subtype', _depluralize(m.group(1)), _depluralize(m.group(2)))
    m = re.match(r'(?:All|Every|Each)\s+(\w+)\s+(?:is|are)\s+(?:a|an\s+)?(\w+)$', t, re.I)
    if m:
        return ('subtype', _depluralize(m.group(1)), _depluralize(m.group(2)))
    m = re.match(r'(\w+)\s+are\s+not\s+(\w+)$', t, re.I)
    if m:
        return ('not_subtype', _depluralize(m.group(1)), _depluralize(m.group(2)))
    m = re.match(r'(\w+)\s+are\s+(\w+)$', t, re.I)
    if m:
        return ('subtype', _depluralize(m.group(1)), _depluralize(m.group(2)))
    return None


def _parse_question(question: str):
    m = re.match(r'Is\s+(\w+)\s+(?:a|an)\s+(\w+)\?$', question.strip(), re.I)
    if not m:
        raise ValueError(f'Unsupported question format: {question}')
    return m.group(1).lower(), m.group(2).lower()


class MockProntoQAReasoner(BaseProvider):
    def __init__(self):
        self.model_name = 'mock-prontoqa-reasoner'

    def run(self, prompt: str, *, problem=None, premises=None, question=None, temperature_override: float | None = None) -> ProviderResponse:
        if premises is None or question is None:
            if problem is None:
                raise ValueError('Mock provider requires either (problem) or (premises, question)')
            premises = problem['premises']
            question = problem['question']
        raw = self._solve(premises, question)
        return ProviderResponse(raw_response=raw, provider='mock', model_name=self.model_name, temperature=0.0, token_usage=None)

    def _solve(self, premises: list[dict], question: str) -> str:
        rules = []
        known_positive = OrderedDict()
        known_negative = OrderedDict()
        entity = None
        for p in premises:
            parsed = _parse_premise_rule(p['text'])
            if not parsed:
                continue
            kind, a, b = parsed
            if kind == 'fact':
                entity = a
                known_positive[(a, b)] = True
            elif kind == 'neg_fact':
                entity = a
                known_negative[(a, b)] = True
            else:
                rules.append(parsed)

        q_entity, q_pred = _parse_question(question)
        entity = entity or q_entity
        emitted = OrderedDict()
        changed = True
        while changed:
            changed = False
            positives = list(known_positive.keys())
            for ent, pred in positives:
                for kind, src, dst in rules:
                    if src != pred:
                        continue
                    if kind == 'subtype' and (ent, dst) not in known_positive:
                        known_positive[(ent, dst)] = True
                        emitted[(ent, dst, 'pos')] = f'Step {len(emitted)+1}: {ent.capitalize()} is a {dst}.'
                        changed = True
                    elif kind == 'not_subtype' and (ent, dst) not in known_negative:
                        known_negative[(ent, dst)] = True
                        emitted[(ent, dst, 'neg')] = f'Step {len(emitted)+1}: {ent.capitalize()} is not a {dst}.'
                        changed = True

        steps = list(emitted.values())
        answer = 'True' if (q_entity, q_pred) in known_positive else 'False'
        if (q_entity, q_pred) in known_negative:
            answer = 'False'
        if not steps:
            steps = [f'Step 1: {q_entity.capitalize()} is not a {q_pred}.']
        return '\n'.join(steps + [f'Answer: {answer}'])
