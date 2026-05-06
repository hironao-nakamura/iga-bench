from __future__ import annotations

from pathlib import Path
import json
import pandas as pd
import re


REQUIRED_TABLES = [
    'benchmarks', 'models', 'problems', 'proof_nodes', 'proof_edges', 'runs',
    'probes', 'trace_steps', 'step_alignments', 'audit_certificates',
    'run_summaries', 'aggregate_metrics',
]


def _read_partitioned_table(path: Path) -> pd.DataFrame:
    flat = path.with_suffix('.parquet')
    if flat.is_file():
        return pd.read_parquet(flat)
    if path.is_file() and path.suffix == '.parquet':
        return pd.read_parquet(path)
    if path.is_file() and path.suffix == '.jsonl':
        return pd.read_json(path, lines=True)
    if path.is_dir():
        parquet_files = sorted(path.rglob('*.parquet'))
        if parquet_files:
            return pd.concat([pd.read_parquet(p) for p in parquet_files], ignore_index=True)
        jsonl_files = sorted(path.rglob('*.jsonl'))
        if jsonl_files:
            return pd.concat([pd.read_json(p, lines=True) for p in jsonl_files], ignore_index=True)
        return pd.DataFrame()
    return pd.DataFrame()


def validate_dataset_root(root: str | Path) -> dict:
    root = Path(root)
    data_root = root / 'data'
    report = {'status': 'PASS', 'missing_tables': [], 'notes': []}
    loaded = {}
    for table in REQUIRED_TABLES:
        df = _read_partitioned_table(data_root / table)
        loaded[table] = df
        if df.empty:
            report['missing_tables'].append(table)
    if report['missing_tables']:
        report['status'] = 'FAIL'
        return report

    problems = set(loaded['problems']['dataset_problem_id'].tolist())
    models = set(loaded['models']['model_id'].tolist())
    runs = set(loaded['runs']['run_id'].tolist())
    probes = set(loaded['probes']['probe_id'].tolist())

    if not set(loaded['runs']['dataset_problem_id']).issubset(problems):
        report['status'] = 'FAIL'
        report['notes'].append('runs reference unknown problems')
    if not set(loaded['runs']['model_id']).issubset(models):
        report['status'] = 'FAIL'
        report['notes'].append('runs reference unknown models')
    if not set(loaded['trace_steps']['run_id']).issubset(runs):
        report['status'] = 'FAIL'
        report['notes'].append('trace_steps reference unknown runs')
    if not set(loaded['step_alignments']['original_run_id']).issubset(runs):
        report['status'] = 'FAIL'
        report['notes'].append('step_alignments reference unknown original runs')
    if not set(loaded['step_alignments']['probed_run_id']).issubset(runs):
        report['status'] = 'FAIL'
        report['notes'].append('step_alignments reference unknown probed runs')
    if not set(loaded['audit_certificates']['probe_id']).issubset(probes):
        report['status'] = 'FAIL'
        report['notes'].append('audit_certificates reference unknown probes')

    certs = loaded['audit_certificates']
    bad = certs[(certs['verdict_type'] == 'GROUNDED') & (certs['control_changed'] == True)]
    if len(bad) > 0:
        report['status'] = 'FAIL'
        report['notes'].append(f'Found {len(bad)} GROUNDED rows with control_changed=True')

    verdict_scope = certs.groupby('dependency_mode_scored')
    for scope, df in verdict_scope:
        scope_total = len(df)
        verdict_total = int(df['verdict_type'].value_counts().sum())
        if scope_total != verdict_total:
            report['status'] = 'FAIL'
            report['notes'].append(
                f'scope={scope}: verdict total mismatch (certificates={scope_total}, verdict_counts={verdict_total})'
            )

    duplicate_prefix_pattern = re.compile(r'\bzqzq\w*\b', re.IGNORECASE)
    bad_canonical = certs['canonical_original'].fillna('').str.contains(duplicate_prefix_pattern) | certs['canonical_probed'].fillna('').str.contains(duplicate_prefix_pattern)
    bad_semantic_ref = loaded['probes']['semantic_target_ref'].fillna('').str.contains(duplicate_prefix_pattern)
    num_bad = int(bad_canonical.sum() + bad_semantic_ref.sum())
    if num_bad > 0:
        report['status'] = 'FAIL'
        report['notes'].append(f'duplicate zq-prefix detected in {num_bad} rows (zqzq*)')

    return report


def write_validation_report(dataset_root: str | Path, output_path: str | Path) -> dict:
    report = validate_dataset_root(dataset_root)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report
