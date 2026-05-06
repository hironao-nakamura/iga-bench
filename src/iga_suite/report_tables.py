from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


_MODEL_DISPLAY = {
    'openai': 'GPT-4o',
    'anthropic': 'Claude Sonnet 4.6',
    'openweight': 'Qwen3-Next-80B',
}

# Display name maps.  The primary release of v1.1 is
# ProntoQA-500 + ProofWriter-300 (= 800 problems).  ``_BENCH_DISPLAY``
# is kept as a backwards-compatibility alias for
# ``_BENCH_DISPLAY_PRIMARY`` because some external tooling imports the
# historical name.
_BENCH_DISPLAY_PRIMARY = {
    'prontoqa_full_eval': 'ProntoQA-500',
    'proofwriter_cwa_d3_is_300': 'ProofWriter-300',
}
_BENCH_DISPLAY = dict(_BENCH_DISPLAY_PRIMARY)


def _prf1(tp: int, fp: int, fn: int):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return round(p, 3), round(r, 3), round(f1, 3)


def _read_table(root: Path, name: str) -> pd.DataFrame:
    flat = root / 'data' / f'{name}.parquet'
    if flat.is_file():
        return pd.read_parquet(flat)
    path = root / 'data' / name
    if path.is_dir():
        parquet_files = sorted(path.rglob('*.parquet'))
        if parquet_files:
            return pd.read_parquet(path)
    if path.is_file() and path.suffix == '.parquet':
        return pd.read_parquet(path)
    return pd.DataFrame()


def _derive_benchmark_id(df: pd.DataFrame) -> pd.DataFrame:
    """Add benchmark_id from dataset_problem_id (everything before '::')."""
    if 'benchmark_id' not in df.columns and 'dataset_problem_id' in df.columns:
        df = df.copy()
        df['benchmark_id'] = df['dataset_problem_id'].astype(str).str.split('::').str[0]
    return df


def build_table1(metrics: pd.DataFrame) -> pd.DataFrame:
    """Main evaluation matrix (paper Table 2): direct P/R/F1, coverage, transitive F1."""
    if metrics.empty:
        return pd.DataFrame()
    direct = metrics[metrics['dependency_mode'] == 'direct'].copy()
    trans = metrics[metrics['dependency_mode'] == 'transitive'][
        ['benchmark_id', 'model_family', 'f1']
    ].rename(columns={'f1': 'transitive_f1'})

    merged = direct.merge(trans, on=['benchmark_id', 'model_family'], how='left')
    cols = ['benchmark_id', 'model_family', 'num_problems',
            'precision', 'recall', 'f1', 'coverage', 'transitive_f1']
    return merged[[c for c in cols if c in merged.columns]].reset_index(drop=True)


def build_table2(certs: pd.DataFrame) -> pd.DataFrame:
    """Direct-scope premise-type slices on covered pairs only (paper Table 3)."""
    if certs.empty:
        return pd.DataFrame()
    certs = _derive_benchmark_id(certs)
    direct = certs[certs['dependency_mode_scored'] == 'direct'].copy()
    covered = direct[direct['verdict_type'] != 'UNPARSEABLE'].copy()

    rows = []
    for (bid, mf, ptype), grp in covered.groupby(
        ['benchmark_id', 'model_family', 'premise_canonical_type']
    ):
        gold = grp['gold_dependency_label'].astype(bool)
        pred = grp['verdict_type'] == 'GROUNDED'
        tp = int((pred & gold).sum())
        fp = int((pred & ~gold).sum())
        fn = int((~pred & gold).sum())
        p, r, f1 = _prf1(tp, fp, fn)
        rows.append({
            'benchmark_id': bid,
            'model_family': str(mf),
            'premise_type': str(ptype),
            'n_covered': len(grp),
            'precision': p,
            'recall': r,
            'f1': f1,
        })
    return pd.DataFrame(rows)


def build_table4(run_summaries: pd.DataFrame) -> pd.DataFrame:
    """Problem-level RAWR rates among correctly solved problems (paper Table 5)."""
    if run_summaries.empty:
        return pd.DataFrame()
    rs = _derive_benchmark_id(run_summaries)
    correct = rs[rs['final_answer_correct'] == True]  # noqa: E712

    rows = []
    for (bid, mf), grp in correct.groupby(['benchmark_id', 'model_family']):
        n = len(grp)
        rawr_d = grp['rawr_direct'].sum() / n if n else 0.0
        rawr_t = grp['rawr_transitive'].sum() / n if n else 0.0
        rows.append({
            'benchmark_id': bid,
            'model_family': str(mf),
            'correct_problems': n,
            'rawr_direct_pct': round(rawr_d * 100, 1),
            'rawr_transitive_pct': round(rawr_t * 100, 1),
        })
    return pd.DataFrame(rows)


def build_appendix_a(certs: pd.DataFrame) -> pd.DataFrame:
    """Control-probe change rates on covered pairs, split by gold label (paper Table 6 / Appendix A)."""
    if certs.empty:
        return pd.DataFrame()
    certs = _derive_benchmark_id(certs)
    direct = certs[certs['dependency_mode_scored'] == 'direct'].copy()
    covered = direct[direct['verdict_type'] != 'UNPARSEABLE'].copy()

    rows = []
    for (bid, mf), grp in covered.groupby(['benchmark_id', 'model_family']):
        for gold_label, label_name in [(True, 'gold_positive'), (False, 'gold_negative')]:
            sub = grp[grp['gold_dependency_label'].astype(bool) == gold_label]
            n = len(sub)
            if n == 0:
                continue
            sem = float(sub['semantic_changed'].fillna(False).astype(bool).mean())
            sur = float(sub['surface_changed'].fillna(False).astype(bool).mean())
            nul = float(sub['null_changed'].fillna(False).astype(bool).mean())
            rows.append({
                'benchmark_id': bid,
                'model_family': str(mf),
                'gold_label': label_name,
                'n': n,
                'semantic_rate': round(sem, 3),
                'surface_rate': round(sur, 3),
                'null_rate': round(nul, 3),
            })
    return pd.DataFrame(rows)


def build_appendix_c(metrics: pd.DataFrame, certs: pd.DataFrame) -> pd.DataFrame:
    """Predicate-determining dependency results with FP count (paper Table 8 / Appendix C)."""
    if metrics.empty:
        return pd.DataFrame()
    pred_det = metrics[metrics['dependency_mode'] == 'predicate_determining'].copy()
    if pred_det.empty:
        return pd.DataFrame()

    fp_counts: dict[tuple[str, str], int] = {}
    if not certs.empty:
        c = _derive_benchmark_id(certs)
        pd_certs = c[c['dependency_mode_scored'] == 'predicate_determining']
        covered = pd_certs[pd_certs['verdict_type'] != 'UNPARSEABLE']
        for (bid, mf), grp in covered.groupby(['benchmark_id', 'model_family']):
            gold = grp['gold_dependency_label'].astype(bool)
            pred = grp['verdict_type'] == 'GROUNDED'
            fp_counts[(bid, str(mf))] = int((pred & ~gold).sum())

    pred_det = pred_det.copy()
    pred_det['fp_count'] = pred_det.apply(
        lambda r: fp_counts.get((r['benchmark_id'], str(r['model_family'])), 0), axis=1
    )

    cols = ['benchmark_id', 'model_family', 'coverage', 'precision', 'recall', 'f1', 'fp_count', 'notes']
    return pred_det[[c for c in cols if c in pred_det.columns]].reset_index(drop=True)


def build_appendix_d(certs: pd.DataFrame, run_summaries: pd.DataFrame) -> pd.DataFrame:
    """Coverage-conditioned RAWR variants per model-family (paper Table 9 / Appendix D)."""
    if run_summaries.empty or certs.empty:
        return pd.DataFrame()
    rs = _derive_benchmark_id(run_summaries)
    certs = _derive_benchmark_id(certs)
    correct = rs[rs['final_answer_correct'] == True].copy()  # noqa: E712

    def _model_covered_sets(mode: str, model_family: str):
        """Return (covered_gp_pids, all_gp_pids) for a given mode and model_family."""
        gp = certs[
            (certs['dependency_mode_scored'] == mode)
            & (certs['gold_dependency_label'].astype(bool))
            & (certs['model_family'] == model_family)
        ]
        covered = gp[gp['verdict_type'] != 'UNPARSEABLE']
        return set(covered['dataset_problem_id'].unique()), set(gp['dataset_problem_id'].unique())

    rows = []
    for (bid, mf), grp in correct.groupby(['benchmark_id', 'model_family']):
        n_correct = len(grp)
        pids = set(grp['dataset_problem_id'])

        cov_d_pids, all_d_pids = _model_covered_sets('direct', mf)
        cov_t_pids, all_t_pids = _model_covered_sets('transitive', mf)

        cov_d = pids & cov_d_pids
        cov_t = pids & cov_t_pids
        unres_d = (pids & all_d_pids) - cov_d_pids
        unres_t = (pids & all_t_pids) - cov_t_pids

        n_cov_d = len(cov_d)
        n_cov_t = len(cov_t)
        n_unres_d = len(unres_d)
        n_unres_t = len(unres_t)

        rawr_cov_d = grp[grp['dataset_problem_id'].isin(cov_d)]['rawr_direct'].sum() / n_cov_d if n_cov_d else 0.0
        rawr_cov_t = grp[grp['dataset_problem_id'].isin(cov_t)]['rawr_transitive'].sum() / n_cov_t if n_cov_t else 0.0

        rawr_ub_d = (grp[grp['dataset_problem_id'].isin(cov_d)]['rawr_direct'].sum() + n_unres_d) / (n_cov_d + n_unres_d) if (n_cov_d + n_unres_d) else 0.0
        rawr_ub_t = (grp[grp['dataset_problem_id'].isin(cov_t)]['rawr_transitive'].sum() + n_unres_t) / (n_cov_t + n_unres_t) if (n_cov_t + n_unres_t) else 0.0

        rows.append({
            'benchmark_id': bid,
            'model_family': str(mf),
            'correct_problems': n_correct,
            'n_cov_direct': n_cov_d,
            'n_cov_transitive': n_cov_t,
            'rawr_cov_direct_pct': round(rawr_cov_d * 100, 1),
            'rawr_ub_direct_pct': round(rawr_ub_d * 100, 1),
            'rawr_cov_transitive_pct': round(rawr_cov_t * 100, 1),
            'rawr_ub_transitive_pct': round(rawr_ub_t * 100, 1),
        })
    return pd.DataFrame(rows)


def build_parser_sensitivity(
    conservative_metrics: pd.DataFrame,
    released_metrics: pd.DataFrame,
) -> pd.DataFrame:
    """Canonicalizer sensitivity: compare conservative (v1.0) vs released (v1.1-final)
    aggregate metrics on the same benchmarks/models. One row per
    (benchmark_id, model_family, dependency_mode); reports coverage/F1 under both
    parsers, their deltas, and the primary-release flag.

    This is NOT part of the primary release data/. It is written to analysis/
    to give reviewers a sensitivity-analysis artifact without mixing
    two parser versions into the single source of truth.
    """
    if conservative_metrics.empty or released_metrics.empty:
        return pd.DataFrame()

    key_cols = ['benchmark_id', 'model_family', 'dependency_mode']
    keep_cols = key_cols + ['coverage', 'f1', 'num_problems']

    def _prep(df: pd.DataFrame, suffix: str) -> pd.DataFrame:
        df = df[[c for c in keep_cols if c in df.columns]].copy()
        df = df.rename(columns={
            'coverage': f'coverage_{suffix}',
            'f1': f'f1_{suffix}',
            'num_problems': f'num_problems_{suffix}',
        })
        return df

    cons = _prep(conservative_metrics, 'conservative')
    rel = _prep(released_metrics, 'released')
    merged = cons.merge(rel, on=key_cols, how='outer')

    if 'coverage_conservative' in merged.columns and 'coverage_released' in merged.columns:
        merged['coverage_delta'] = (
            merged['coverage_released'].fillna(0) - merged['coverage_conservative'].fillna(0)
        ).round(4)
    if 'f1_conservative' in merged.columns and 'f1_released' in merged.columns:
        merged['f1_delta'] = (
            merged['f1_released'].fillna(0) - merged['f1_conservative'].fillna(0)
        ).round(4)

    # Flag rows that correspond to the primary v1.1 release (PQ500 + PW300).
    primary_ids = {'prontoqa_full_eval', 'proofwriter_cwa_d3_is_300'}
    merged['in_primary_release'] = merged['benchmark_id'].isin(primary_ids)

    sort_cols = [c for c in ['benchmark_id', 'model_family', 'dependency_mode'] if c in merged.columns]
    if sort_cols:
        merged = merged.sort_values(sort_cols).reset_index(drop=True)
    return merged


def build_tables(release_root: Path, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = _read_table(release_root, 'aggregate_metrics')
    certs = _read_table(release_root, 'audit_certificates')
    run_summaries = _read_table(release_root, 'run_summaries')

    t1 = build_table1(metrics)
    t2 = build_table2(certs)
    t4 = build_table4(run_summaries)
    ta = build_appendix_a(certs)
    tc = build_appendix_c(metrics, certs)
    td = build_appendix_d(certs, run_summaries)

    t1.to_json(out_dir / 'table1_main_matrix.json', orient='records', indent=2, force_ascii=False)
    t2.to_json(out_dir / 'table2_premise_slice.json', orient='records', indent=2, force_ascii=False)
    t4.to_json(out_dir / 'table4_rawr.json', orient='records', indent=2, force_ascii=False)
    ta.to_json(out_dir / 'appendix_a_control_probe.json', orient='records', indent=2, force_ascii=False)
    tc.to_json(out_dir / 'appendix_c_predicate_determining.json', orient='records', indent=2, force_ascii=False)
    td.to_json(out_dir / 'appendix_d_rawr_conditioned.json', orient='records', indent=2, force_ascii=False)

    summary = {
        'table1_rows': len(t1),
        'table2_rows': len(t2),
        'table4_rows': len(t4),
        'appendix_a_rows': len(ta),
        'appendix_c_rows': len(tc),
        'appendix_d_rows': len(td),
        'out_dir': str(out_dir),
    }
    return summary


def main():
    ap = argparse.ArgumentParser(description='Regenerate paper tables from the frozen release.')
    ap.add_argument('--release-root', required=True, help='Path to the release root directory')
    ap.add_argument('--out', required=True, help='Output directory for table JSON files')
    args = ap.parse_args()
    res = build_tables(Path(args.release_root), Path(args.out))
    print(json.dumps(res, indent=2))


if __name__ == '__main__':
    main()
