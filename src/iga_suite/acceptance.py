from __future__ import annotations

from pathlib import Path
from collections import defaultdict
import json
import re

import pandas as pd


def _write_subset(rows: list[dict], output_jsonl: Path) -> dict:
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with open(output_jsonl, 'w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    return {
        'output': str(output_jsonl),
        'num_selected': len(rows),
        'problem_ids': [r['problem_id'] for r in rows],
        'difficulty_counts': dict(
            (k, sum(1 for r in rows if r.get('metadata', {}).get('difficulty_bin') == k))
            for k in sorted({r.get('metadata', {}).get('difficulty_bin', 'unknown') for r in rows})
        ),
    }


def _stratified_pick(rows: list[dict], n_plan: list[tuple[str, int]]) -> list[dict]:
    by_hop = defaultdict(list)
    for row in rows:
        hop = row.get('metadata', {}).get('difficulty_bin', 'unknown')
        by_hop[hop].append(row)
    selected = []
    for hop, n in n_plan:
        items = sorted(by_hop.get(hop, []), key=lambda r: r['problem_id'])[:n]
        selected.extend(items)
    if len(selected) < 10:
        used = {r['problem_id'] for r in selected}
        rest = sorted([r for r in rows if r['problem_id'] not in used], key=lambda r: r['problem_id'])
        selected.extend(rest[: 10 - len(selected)])
    return sorted(selected, key=lambda r: r['problem_id'])[:10]


def create_revisioned_acceptance_subset(input_jsonl: str | Path, output_jsonl: str | Path) -> dict:
    input_jsonl = Path(input_jsonl)
    output_jsonl = Path(output_jsonl)
    rows = []
    with open(input_jsonl, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    selected = _stratified_pick(rows, [('3hop', 3), ('4hop', 4), ('5hop', 3)])
    return _write_subset(selected, output_jsonl)


def create_holdout_subset(full_jsonl: str | Path, dev_jsonl: str | Path, output_jsonl: str | Path) -> dict:
    full_rows = []
    with open(full_jsonl, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                full_rows.append(json.loads(line))
    dev_ids = set()
    with open(dev_jsonl, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                dev_ids.add(json.loads(line)['problem_id'])
    pool = [r for r in full_rows if r['problem_id'] not in dev_ids]
    selected = _stratified_pick(pool, [('3hop', 3), ('4hop', 4), ('5hop', 3)])
    result = _write_subset(selected, Path(output_jsonl))
    result['excluded_dev_problem_ids'] = sorted(dev_ids)
    return result


def evaluate_acceptance_gate(output_root: str | Path, companion_root: str | Path | None = None) -> dict:
    output_root = Path(output_root)
    df_metrics = pd.concat(
        [pd.read_parquet(p) for p in (output_root / 'data' / 'aggregate_metrics').rglob('*.parquet')],
        ignore_index=True,
    )
    df_certs = pd.concat(
        [pd.read_parquet(p) for p in (output_root / 'data' / 'audit_certificates').rglob('*.parquet')],
        ignore_index=True,
    )

    direct_iga = df_metrics[(df_metrics['dependency_mode'] == 'direct') & (df_metrics['baseline_name'] == 'iga')]
    if direct_iga.empty:
        raise ValueError('direct scope metrics for baseline=iga not found')
    direct_iga_row = direct_iga.iloc[0]

    direct_sc = df_metrics[(df_metrics['dependency_mode'] == 'direct') & (df_metrics['baseline_name'] == 'self_consistency')]
    if direct_sc.empty:
        raise ValueError('direct scope metrics for baseline=self_consistency not found')
    direct_sc_row = direct_sc.iloc[0]

    direct_certs = df_certs[df_certs['dependency_mode_scored'] == 'direct']
    verdict_counts = direct_certs['verdict_type'].value_counts().to_dict()
    unparseable_rate = float(verdict_counts.get('UNPARSEABLE', 0) / len(direct_certs)) if len(direct_certs) else 1.0
    verdict_total_matches = int(sum(verdict_counts.values()) == len(direct_certs))
    strict_mask = ~direct_certs['canonical_original'].fillna('').str.startswith('break(') & ~direct_certs['canonical_probed'].fillna('').str.startswith('break(')
    strict_direct = direct_certs[(direct_certs['verdict_type'] != 'UNPARSEABLE') & strict_mask]
    break_aware_definitive = direct_certs[direct_certs['verdict_type'] != 'UNPARSEABLE']
    strict_coverage = float(len(strict_direct) / len(direct_certs)) if len(direct_certs) else 0.0
    break_aware_coverage = float(len(break_aware_definitive) / len(direct_certs)) if len(direct_certs) else 0.0

    duplicate_prefix_count = 0
    dup_pat = re.compile(r'\bzqzq\w*\b', re.IGNORECASE)
    if companion_root is not None:
        companion_root = Path(companion_root)
        for p in companion_root.rglob('*.json'):
            try:
                text = p.read_text(encoding='utf-8')
            except Exception:
                continue
            duplicate_prefix_count += len(dup_pat.findall(text))

    f1_gap = float(direct_iga_row['f1'] - direct_sc_row['f1'])
    direct_definitive = direct_certs[direct_certs['verdict_type'] != 'UNPARSEABLE']
    g_tp = g_fp = g_fn = g_tn = 0
    w_tp = w_fp = w_fn = w_tn = 0
    for _, row in direct_definitive.iterrows():
        gold = bool(row.get('gold_dependency_label'))
        pred_grounded = row['verdict_type'] == 'GROUNDED'
        pred_ws = row['verdict_type'] in ('GROUNDED', 'INPUT-SENSITIVE')
        if pred_grounded and gold:
            g_tp += 1
        elif pred_grounded and not gold:
            g_fp += 1
        elif (not pred_grounded) and gold:
            g_fn += 1
        else:
            g_tn += 1
        if pred_ws and gold:
            w_tp += 1
        elif pred_ws and not gold:
            w_fp += 1
        elif (not pred_ws) and gold:
            w_fn += 1
        else:
            w_tn += 1

    def _prf1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        return p, r, f1

    grounded_p, grounded_r, grounded_f1 = _prf1(g_tp, g_fp, g_fn)
    ws_p, ws_r, ws_f1 = _prf1(w_tp, w_fp, w_fn)

    def _scope_pred_positive(verdict: str, scope: str) -> bool:
        if scope == 'direct':
            return verdict == 'GROUNDED'
        return verdict in ('GROUNDED', 'INPUT-SENSITIVE')

    def _recall_for(scope: str, premise_type: str) -> float | None:
        sub = df_certs[
            (df_certs['dependency_mode_scored'] == scope)
            & (df_certs['verdict_type'] != 'UNPARSEABLE')
            & (df_certs.get('premise_canonical_type', pd.Series(index=df_certs.index, dtype=object)).fillna('unknown') == premise_type)
        ]
        if sub.empty:
            return None
        pos = sub[sub['gold_dependency_label'].astype(bool)]
        if pos.empty:
            return None
        hit = pos['verdict_type'].map(lambda v: _scope_pred_positive(v, scope)).sum()
        return float(hit / len(pos))

    direct_recall_is = _recall_for('direct', 'is')
    direct_recall_subtype = _recall_for('direct', 'subtype')
    transitive_recall_is = _recall_for('transitive', 'is')
    transitive_recall_subtype = _recall_for('transitive', 'subtype')

    checks = {
        'duplicate_prefix_zero': duplicate_prefix_count == 0,
        'direct_coverage_gte_075': float(direct_iga_row['coverage']) >= 0.75,
        'direct_grounded_only_f1_gte_060': float(direct_iga_row['f1']) >= 0.60,
        'direct_recall_is_gte_050': (direct_recall_is is not None and direct_recall_is >= 0.50),
        'direct_recall_subtype_gte_095': (direct_recall_subtype is not None and direct_recall_subtype >= 0.95),
        'direct_f1_gap_vs_self_consistency_gte_015': f1_gap >= 0.15,
        'unparseable_rate_lte_025': unparseable_rate <= 0.25,
        'verdict_total_matches_num_certificates': bool(verdict_total_matches),
    }
    passed = all(checks.values())
    return {
        'status': 'PASS' if passed else 'FAIL',
        'checks': checks,
        'stats': {
            'duplicate_prefix_count': duplicate_prefix_count,
            'direct_coverage': float(direct_iga_row['coverage']),
            'direct_f1': float(direct_iga_row['f1']),
            'direct_grounded_only_precision': grounded_p,
            'direct_grounded_only_recall': grounded_r,
            'direct_grounded_only_f1': grounded_f1,
            'direct_ws_compat_precision': ws_p,
            'direct_ws_compat_recall': ws_r,
            'direct_ws_compat_f1': ws_f1,
            'direct_self_consistency_f1': float(direct_sc_row['f1']),
            'direct_f1_gap': f1_gap,
            'direct_unparseable_rate': unparseable_rate,
            'direct_num_certificates': int(len(direct_certs)),
            'direct_verdict_counts': verdict_counts,
            'direct_coverage_break_aware': break_aware_coverage,
            'direct_coverage_strict': strict_coverage,
            'direct_recall_is': direct_recall_is,
            'direct_recall_subtype': direct_recall_subtype,
            'transitive_recall_is': transitive_recall_is,
            'transitive_recall_subtype': transitive_recall_subtype,
        },
    }
