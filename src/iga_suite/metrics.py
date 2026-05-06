from __future__ import annotations

from collections import defaultdict
from statistics import mean


def _prf1(tp: int, fp: int, fn: int):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f1


def _is_dependency_positive(verdict: str, *, mode: str | None, include_input_sensitive: bool) -> bool:
    if include_input_sensitive:
        return verdict in {'GROUNDED', 'INPUT-SENSITIVE'}
    return verdict == 'GROUNDED'


def summarize_problem(certificates: list[dict], final_answer_correct: bool | None) -> dict:
    total = len(certificates)
    definitive = [c for c in certificates if c['verdict_type'] != 'UNPARSEABLE']
    parse_cov = sum(1 for c in certificates if c['parse_ok']) / total if total else 0.0
    align_cov = sum(1 for c in certificates if c['alignment_ok']) / total if total else 0.0
    def_cov = len(definitive) / total if total else 0.0
    misrep = sum(1 for c in certificates if c['verdict_type'] == 'MISREPRESENTATION')
    rawr_direct = bool(final_answer_correct) and any(
        c['dependency_mode_scored'] == 'direct'
        and c['gold_dependency_label']
        and c['verdict_type'] in {'INSENSITIVE', 'MISREPRESENTATION'}
        for c in certificates
    )
    rawr_trans = bool(final_answer_correct) and any(
        c['dependency_mode_scored'] == 'transitive'
        and c['gold_dependency_label']
        and c['verdict_type'] in {'INSENSITIVE', 'MISREPRESENTATION'}
        for c in certificates
    )
    return {
        'definitive_coverage': round(def_cov, 6),
        'parse_coverage': round(parse_cov, 6),
        'alignment_coverage': round(align_cov, 6),
        'num_certificates': total,
        'num_unparseable': total - len(definitive),
        'num_misrepresentation': misrep,
        'rawr_direct': rawr_direct,
        'rawr_transitive': rawr_trans,
    }


def aggregate(certificates: list[dict], problems: list[dict], *, benchmark_id: str, split: str, model_family: str, model_id: str, config_id: str, baseline_name: str = 'iga', voting_k: int = 1, mean_api_calls_per_problem: float | None = None) -> list[dict]:
    rows = []
    by_mode = defaultdict(list)
    for c in certificates:
        by_mode[c['dependency_mode_scored']].append(c)

    mean_num_premises = mean([len(p['premises']) for p in problems]) if problems else 0.0
    for mode, certs in by_mode.items():
        definitive = [c for c in certs if c['verdict_type'] != 'UNPARSEABLE']
        tp = fp = fn = tn = 0
        ws_tp = ws_fp = ws_fn = ws_tn = 0
        lb_tp = lb_fp = lb_fn = lb_tn = 0
        for c in certs:
            gold = bool(c['gold_dependency_label'])
            if c['verdict_type'] == 'UNPARSEABLE':
                pred = False  # lower-bound interpretation
                if pred and gold:
                    lb_tp += 1
                elif pred and not gold:
                    lb_fp += 1
                elif not pred and gold:
                    lb_fn += 1
                else:
                    lb_tn += 1
                continue
            pred = _is_dependency_positive(c['verdict_type'], mode=mode, include_input_sensitive=False)
            ws_pred = _is_dependency_positive(c['verdict_type'], mode=mode, include_input_sensitive=True)
            if pred and gold:
                tp += 1
            elif pred and not gold:
                fp += 1
            elif not pred and gold:
                fn += 1
            else:
                tn += 1
            if ws_pred and gold:
                ws_tp += 1
            elif ws_pred and not gold:
                ws_fp += 1
            elif not ws_pred and gold:
                ws_fn += 1
            else:
                ws_tn += 1
            if pred and gold:
                lb_tp += 1
            elif pred and not gold:
                lb_fp += 1
            elif not pred and gold:
                lb_fn += 1
            else:
                lb_tn += 1

        p, r, f1 = _prf1(tp, fp, fn)
        ws_p, ws_r, ws_f1 = _prf1(ws_tp, ws_fp, ws_fn)
        _, _, lower_f1 = _prf1(lb_tp, lb_fp, lb_fn)
        coverage = len(definitive) / len(certs) if certs else 0.0

        # A (benchmark, model, dependency_mode) cell can legitimately contain
        # zero gold-positive certificates — e.g. ProofWriter-300 under
        # ``predicate_determining`` has no gold-positive pairs by design.  In
        # that regime TP+FN=0 and the classic precision/recall/F1 are
        # mathematically undefined.  Emitting 0.0 there misleads downstream
        # readers into thinking the system scored zero, so we null the
        # metrics out and stamp ``metric_status`` for transparency.
        gold_positives = (tp + fn)
        # When a slice has zero gold-positive certificates, recall (and
        # therefore F1) are mathematically undefined regardless of whether
        # the model predicted anything positive.  Reporting "precision
        # alone" without recall is also ambiguous for downstream readers,
        # so we stamp the whole row as undefined and let the ``notes``
        # column carry the human-readable explanation.
        if gold_positives == 0:
            precision_out: float | None = None
            recall_out: float | None = None
            f1_out: float | None = None
            cov_adj_f1_out: float | None = None
            lower_f1_out: float | None = None
            metric_status = 'undefined:no_gold_positive_pairs'
        else:
            precision_out = round(p, 6)
            recall_out = round(r, 6)
            f1_out = round(f1, 6)
            cov_adj_f1_out = round(f1 * coverage, 6)
            lower_f1_out = round(lower_f1, 6)
            metric_status = 'defined'

        note = None
        if mode == 'direct':
            note = (
                "primary_f1=grounded_only; "
                f"ws_compat_precision={ws_p:.6f};ws_compat_recall={ws_r:.6f};ws_compat_f1={ws_f1:.6f}"
            )
        if metric_status == 'undefined:no_gold_positive_pairs':
            undef_note = (
                "undefined:no_gold_positive_pairs; "
                "precision/recall/F1 are mathematically undefined when the "
                "slice contains zero gold-positive certificates. Numeric "
                "fields are set to null for schema clarity."
            )
            note = f"{note} | {undef_note}" if note else undef_note

        rows.append({
            'benchmark_id': benchmark_id,
            'split': split,
            'model_family': model_family,
            'model_id': model_id,
            'config_id': config_id,
            'dependency_mode': mode,
            'metric_scope': 'all',
            'baseline_name': baseline_name,
            'voting_k': voting_k,
            'num_problems': len(problems),
            'num_certificates': len(certs),
            'precision': precision_out,
            'recall': recall_out,
            'f1': f1_out,
            'coverage': round(coverage, 6),
            'coverage_adjusted_f1': cov_adj_f1_out,
            'lower_bound_f1_all_unresolved_negative': lower_f1_out,
            'mean_num_premises': round(mean_num_premises, 6),
            'mean_api_calls_per_problem': mean_api_calls_per_problem,
            'estimated_cost_usd_per_problem': None,
            'ci95_low': None,
            'ci95_high': None,
            'notes': note,
            'metric_status': metric_status,
        })
    return rows


def compute_scope_metrics(certificates: list[dict]) -> dict:
    definitive = [c for c in certificates if c['verdict_type'] != 'UNPARSEABLE']
    strict_definitive = [
        c
        for c in definitive
        if not (str(c.get('canonical_original') or '').startswith('break(') or str(c.get('canonical_probed') or '').startswith('break('))
    ]
    tp = fp = fn = tn = 0
    mode = certificates[0]['dependency_mode_scored'] if certificates else None
    for c in definitive:
        gold = bool(c['gold_dependency_label'])
        pred = _is_dependency_positive(c['verdict_type'], mode=mode, include_input_sensitive=False)
        if pred and gold:
            tp += 1
        elif pred and not gold:
            fp += 1
        elif not pred and gold:
            fn += 1
        else:
            tn += 1
    p, r, f1 = _prf1(tp, fp, fn)
    stp = sfp = sfn = stn = 0
    ws_tp = ws_fp = ws_fn = ws_tn = 0
    for c in definitive:
        gold = bool(c['gold_dependency_label'])
        ws_pred = _is_dependency_positive(c['verdict_type'], mode=mode, include_input_sensitive=True)
        if ws_pred and gold:
            ws_tp += 1
        elif ws_pred and not gold:
            ws_fp += 1
        elif not ws_pred and gold:
            ws_fn += 1
        else:
            ws_tn += 1
    ws_p, ws_r, ws_f1 = _prf1(ws_tp, ws_fp, ws_fn)
    for c in strict_definitive:
        gold = bool(c['gold_dependency_label'])
        pred = _is_dependency_positive(c['verdict_type'], mode=mode, include_input_sensitive=False)
        if pred and gold:
            stp += 1
        elif pred and not gold:
            sfp += 1
        elif not pred and gold:
            sfn += 1
        else:
            stn += 1
    sp, sr, sf1 = _prf1(stp, sfp, sfn)
    total = len(certificates)
    verdict_counts = defaultdict(int)
    for c in certificates:
        verdict_counts[c['verdict_type']] += 1
    break_token_count = sum(
        1
        for c in definitive
        if str(c.get('canonical_original') or '').startswith('break(') or str(c.get('canonical_probed') or '').startswith('break(')
    )
    by_premise_type = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0, 'n': 0})
    for c in definitive:
        ptype = str(c.get('premise_canonical_type') or 'unknown')
        gold = bool(c['gold_dependency_label'])
        pred = _is_dependency_positive(c['verdict_type'], mode=mode, include_input_sensitive=False)
        by_premise_type[ptype]['n'] += 1
        if pred and gold:
            by_premise_type[ptype]['tp'] += 1
        elif pred and not gold:
            by_premise_type[ptype]['fp'] += 1
        elif (not pred) and gold:
            by_premise_type[ptype]['fn'] += 1
        else:
            by_premise_type[ptype]['tn'] += 1
    premise_type_metrics = {}
    for ptype, cnt in by_premise_type.items():
        pp, rr, ff = _prf1(cnt['tp'], cnt['fp'], cnt['fn'])
        premise_type_metrics[ptype] = {
            'num_certificates': cnt['n'],
            'tp': cnt['tp'],
            'fp': cnt['fp'],
            'fn': cnt['fn'],
            'tn': cnt['tn'],
            'precision': pp,
            'recall': rr,
            'f1': ff,
        }
    return {
        'num_certificates': total,
        'verdict_counts': dict(verdict_counts),
        'verdict_total_matches_certificates': sum(verdict_counts.values()) == total,
        'unparseable_rate': (verdict_counts.get('UNPARSEABLE', 0) / total) if total else 0.0,
        'coverage': (len(definitive) / total) if total else 0.0,
        'coverage_break_aware': (len(definitive) / total) if total else 0.0,
        'coverage_strict': (len(strict_definitive) / total) if total else 0.0,
        'break_tokenized_count': break_token_count,
        'precision': p,
        'recall': r,
        'f1': f1,
        'ws_compat_precision': ws_p,
        'ws_compat_recall': ws_r,
        'ws_compat_f1': ws_f1,
        'strict_precision': sp,
        'strict_recall': sr,
        'strict_f1': sf1,
        'premise_type_metrics': premise_type_metrics,
    }
