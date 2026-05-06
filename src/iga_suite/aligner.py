from __future__ import annotations

from iga_suite.normalizer import normalize_step_text


def premise_canonical_forms(premises: list[dict]) -> set[str]:
    forms: set[str] = set()
    for premise in premises:
        canonical, _, status = normalize_step_text(str(premise.get('text', '')))
        if status != 'UNPARSEABLE' and canonical:
            forms.add(canonical)
    return forms


def align_steps_to_gold(
    steps: list[dict],
    proof_tree: list[dict],
    premise_forms: set[str],
) -> tuple[dict[int, dict], list[dict]]:
    """
    Align model steps to gold proof steps by canonical conclusion (not raw step id).

    Premise restatement steps are excluded before matching. This prevents
    "premise enumeration" from being treated as derivation evidence.
    """
    usable_steps, skipped_restatements = _derived_steps_without_restatements(steps, premise_forms)

    aligned_lookup: dict[int, dict] = {}
    rows: list[dict] = []
    used_pred_step_idx: set[int] = set()

    for entry in sorted(proof_tree, key=lambda e: int(e['step'])):
        sid = int(entry['step'])
        target = entry.get('conclusion')
        matched = None
        for step in usable_steps:
            idx = int(step.get('step_index', -1))
            if idx in used_pred_step_idx:
                continue
            if step.get('canonical_form') == target:
                matched = step
                break
        if matched is not None:
            used_pred_step_idx.add(int(matched['step_index']))
            aligned_lookup[sid] = matched
            rows.append({
                'step_index': sid,
                'original': None,
                'probed': matched,
                'alignment_status': 'MATCHED',
                'alignment_method': 'semantic_gold_conclusion',
                'alignment_confidence': 1.0,
                'alignment_note': f"matched canonical={target}",
            })
        else:
            rows.append({
                'step_index': sid,
                'original': None,
                'probed': None,
                'alignment_status': 'MISSING',
                'alignment_method': 'semantic_gold_conclusion',
                'alignment_confidence': 0.0,
                'alignment_note': f"no derived-step match for canonical={target}",
            })

    for step in skipped_restatements:
        rows.append({
            'step_index': int(step.get('step_index', -1)),
            'original': None,
            'probed': step,
            'alignment_status': 'SKIPPED_RESTATEMENT',
            'alignment_method': 'premise_restatement_filter',
            'alignment_confidence': 1.0,
            'alignment_note': f"canonical={step.get('canonical_form')}",
        })

    return aligned_lookup, rows


def build_original_alignment_reference(
    original_steps: list[dict],
    proof_tree: list[dict],
    premise_forms: set[str],
) -> tuple[dict[int, dict], dict[int, int], list[dict]]:
    """Build gold-step lookup and derived-rank map from original response."""
    usable_steps, _ = _derived_steps_without_restatements(original_steps, premise_forms)
    rank_by_step_index = {int(s['step_index']): idx for idx, s in enumerate(usable_steps)}
    aligned_lookup, rows = align_steps_to_gold(original_steps, proof_tree, premise_forms)
    rank_by_gold_step: dict[int, int] = {}
    for sid, step in aligned_lookup.items():
        rank = rank_by_step_index.get(int(step['step_index']))
        if rank is not None:
            rank_by_gold_step[sid] = rank
    return aligned_lookup, rank_by_gold_step, rows


def align_steps_by_reference_ranks(
    steps: list[dict],
    premise_forms: set[str],
    rank_by_gold_step: dict[int, int],
    gold_step_ids: list[int],
    expected_type_by_gold_step: dict[int, str] | None = None,
    expected_form_by_gold_step: dict[int, str] | None = None,
) -> tuple[dict[int, dict], list[dict]]:
    """
    Align probe run to gold steps using original-derived rank reference.

    This preserves step semantics even when model outputs extra premise
    restatements or uses non-gold numbering.
    """
    usable_steps, skipped_restatements = _derived_steps_without_restatements(steps, premise_forms)
    aligned_lookup: dict[int, dict] = {}
    rows: list[dict] = []
    used_positions: set[int] = set()
    for sid in gold_step_ids:
        sid_i = int(sid)
        rank = rank_by_gold_step.get(sid_i)
        expected_type = (expected_type_by_gold_step or {}).get(sid_i)
        expected_form = (expected_form_by_gold_step or {}).get(sid_i)
        if rank is None:
            rows.append({
                'step_index': sid_i,
                'original': None,
                'probed': None,
                'alignment_status': 'MISSING',
                'alignment_method': 'derived_rank_reference',
                'alignment_confidence': 0.0,
                'alignment_note': f"missing rank={rank}",
            })
            continue
        chosen_pos = None
        chosen_step = None
        chosen_method = 'derived_rank_reference'
        chosen_conf = 1.0

        if expected_form:
            exact_candidates = []
            for pos, step in enumerate(usable_steps):
                if pos in used_positions:
                    continue
                if step.get('canonical_form') == expected_form:
                    exact_candidates.append((abs(pos - rank), pos, step))
            if exact_candidates:
                _, chosen_pos, chosen_step = sorted(exact_candidates, key=lambda x: x[0])[0]
                chosen_method = 'expected_canonical_exact'
                chosen_conf = 1.0

        if rank < len(usable_steps) and rank not in used_positions:
            step = usable_steps[rank]
            if chosen_step is not None:
                pass
            elif _type_matches(expected_type, _step_type(step)):
                chosen_pos = rank
                chosen_step = step
            else:
                if _is_probe_rule_insertion(expected_type, _step_type(step)):
                    rows.append({
                        'step_index': int(step.get('step_index', -1)),
                        'original': None,
                        'probed': step,
                        'alignment_status': 'SKIPPED_PROBE_RULE_INSERTION',
                        'alignment_method': 'type_aware_filter',
                        'alignment_confidence': 1.0,
                        'alignment_note': f"expected={expected_type}, observed={_step_type(step)}",
                    })
                else:
                    rows.append({
                        'step_index': int(step.get('step_index', -1)),
                        'original': None,
                        'probed': step,
                        'alignment_status': 'SKIPPED_TYPE_MISMATCH',
                        'alignment_method': 'type_aware_filter',
                        'alignment_confidence': 1.0,
                        'alignment_note': f"expected={expected_type}, observed={_step_type(step)}",
                    })

        if chosen_step is None:
            candidates = []
            for pos, step in enumerate(usable_steps):
                if pos in used_positions:
                    continue
                if _type_matches(expected_type, _step_type(step)):
                    candidates.append((abs(pos - rank), pos, step))
            if candidates:
                _, chosen_pos, chosen_step = sorted(candidates, key=lambda x: x[0])[0]
                chosen_method = 'derived_rank_type_aware'
                chosen_conf = 0.8

        if chosen_step is None:
            for pos in range(rank, len(usable_steps)):
                if pos not in used_positions:
                    chosen_pos = pos
                    chosen_step = usable_steps[pos]
                    chosen_method = 'derived_rank_fallback_any_type'
                    chosen_conf = 0.5
                    break

        if chosen_step is None:
            for pos, step in enumerate(usable_steps):
                if pos not in used_positions:
                    chosen_pos = pos
                    chosen_step = step
                    chosen_method = 'derived_rank_fallback_any_type'
                    chosen_conf = 0.5
                    break

        if chosen_step is None:
            rows.append({
                'step_index': sid_i,
                'original': None,
                'probed': None,
                'alignment_status': 'MISSING',
                'alignment_method': 'derived_rank_reference',
                'alignment_confidence': 0.0,
                'alignment_note': f"missing candidate for rank={rank}, expected={expected_type}",
            })
            continue

        used_positions.add(int(chosen_pos))
        aligned_lookup[sid_i] = chosen_step
        rows.append({
            'step_index': sid_i,
            'original': None,
            'probed': chosen_step,
            'alignment_status': 'MATCHED',
            'alignment_method': chosen_method,
            'alignment_confidence': chosen_conf,
            'alignment_note': (
                f"rank={rank}, expected_type={expected_type}, expected_form={expected_form}, "
                f"observed={_step_type(chosen_step)}:{chosen_step.get('canonical_form')}"
            ),
        })
    for step in skipped_restatements:
        rows.append({
            'step_index': int(step.get('step_index', -1)),
            'original': None,
            'probed': step,
            'alignment_status': 'SKIPPED_RESTATEMENT',
            'alignment_method': 'premise_restatement_filter',
            'alignment_confidence': 1.0,
            'alignment_note': f"canonical={step.get('canonical_form')}",
        })
    return aligned_lookup, rows


def _derived_steps_without_restatements(steps: list[dict], premise_forms: set[str]) -> tuple[list[dict], list[dict]]:
    usable_steps = []
    skipped_restatements = []
    for step in sorted(steps, key=lambda s: int(s.get('step_index', 0))):
        canonical = step.get('canonical_form')
        if step.get('parse_status') == 'UNPARSEABLE' or canonical is None:
            continue
        if canonical in premise_forms:
            skipped_restatements.append(step)
            continue
        usable_steps.append(step)
    return usable_steps, skipped_restatements


def _step_type(step: dict | None) -> str | None:
    if not step:
        return None
    t = step.get('canonical_type')
    if t:
        return str(t)
    form = step.get('canonical_form')
    if isinstance(form, str) and '(' in form:
        return form.split('(', 1)[0]
    return None


def _type_matches(expected: str | None, observed: str | None) -> bool:
    if expected is None or observed is None:
        return True
    return expected == observed


def _is_probe_rule_insertion(expected: str | None, observed: str | None) -> bool:
    if expected in {'is', 'not_is'} and observed in {'subtype', 'not_subtype'}:
        return True
    return False


def align_steps(original_steps: list[dict], probed_steps: list[dict]) -> list[dict]:
    orig_map = {s['step_index']: s for s in original_steps}
    prob_map = {s['step_index']: s for s in probed_steps}
    all_ids = sorted(set(orig_map) | set(prob_map))
    out = []
    for sid in all_ids:
        o = orig_map.get(sid)
        p = prob_map.get(sid)
        if o and p:
            status = 'MATCHED'
        elif o and not p:
            status = 'MISSING'
        else:
            status = 'EXTRA'
        out.append({
            'step_index': sid,
            'original': o,
            'probed': p,
            'alignment_status': status,
            'alignment_method': 'step_id',
            'alignment_confidence': 1.0 if status == 'MATCHED' else 0.0,
        })
    return out
