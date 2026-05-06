from __future__ import annotations

from functools import lru_cache


def direct_premise_dependencies(problem: dict) -> dict[tuple[int, str], bool]:
    out = {}
    premise_ids = [p['id'] for p in problem['premises']]
    for step in problem.get('proof_tree', []):
        sid = int(step['step'])
        direct = {d for d in step.get('depends_on', []) if d.startswith('P')}
        for pid in premise_ids:
            out[(sid, pid)] = pid in direct
    return out


def transitive_premise_dependencies(problem: dict) -> dict[tuple[int, str], bool]:
    step_map = {int(step['step']): step for step in problem.get('proof_tree', [])}
    premise_ids = [p['id'] for p in problem['premises']]

    @lru_cache(maxsize=None)
    def closure(sid: int) -> frozenset[str]:
        step = step_map[sid]
        acc = set()
        for dep in step.get('depends_on', []):
            if dep.startswith('P'):
                acc.add(dep)
            elif dep.startswith('S'):
                acc |= set(closure(int(dep[1:])))
        return frozenset(acc)

    out = {}
    for sid in step_map:
        deps = set(closure(sid))
        for pid in premise_ids:
            out[(sid, pid)] = pid in deps
    return out


def predicate_determining_dependencies(problem: dict) -> dict[tuple[int, str], bool]:
    direct = direct_premise_dependencies(problem)
    premise_texts = {p['id']: p['text'].lower() for p in problem['premises']}
    out = {}
    for key, val in direct.items():
        pid = key[1]
        text = premise_texts[pid]
        structural = ' is a ' in text and ' are ' not in text
        out[key] = bool(val and not structural)
    return out
