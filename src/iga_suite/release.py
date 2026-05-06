from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable
import pandas as pd

from iga_suite.schema_registry import SchemaRegistry
from iga_suite.table_store import TableStore
from iga_suite.hashing import sha256_file
from iga_suite.validator import validate_dataset_root


_LEGACY_SUBSET_TOKENS: tuple[str, ...] = (
    "smoke", "pilot", "fix", "holdout", "dev",
)
_LEGACY_SUBSET_RE_PARTS = [
    re.escape(t) + (r"\d*" if i == 2 else "")
    for i, t in enumerate(_LEGACY_SUBSET_TOKENS)
]
_LEGACY_SUBSET_RE = re.compile(
    r"(?i)\b(" + "|".join(_LEGACY_SUBSET_RE_PARTS) + r")\b"
)
_CHUNK_SUFFIX_RE = re.compile(r'__chunk\d+$')


def _safe_source_root(root: Path, release_root: Path) -> str:
    # Avoid leaking local absolute paths in released artifacts.
    _ = release_root
    return _CHUNK_SUFFIX_RE.sub('', root.name)


def _sanitize_manifest_value(value):
    if isinstance(value, dict):
        return {k: _sanitize_manifest_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_sanitize_manifest_value(v) for v in value]
    if isinstance(value, str):
        out = value
        out = re.sub(r'(?i)phase[-_ ]*\d+', '', out)
        out = _LEGACY_SUBSET_RE.sub('', out)
        out = re.sub(r'__chunk\d+\b', '', out)
        out = re.sub(r'\s+', ' ', out).strip(' -')
        # Remove local absolute path prefixes from nested manifest strings.
        if out.startswith('/Users/') or out.startswith('/home/') or out.startswith('/tmp/'):
            out = Path(out).name
        return out
    return value


def _assert_single_parquet_per_partition(path: Path) -> None:
    """Fail fast if a hive partition leaf contains multiple parquet files.

    When a partition directory ends up with parquet files from more than one
    pipeline run (e.g. v1.0 conservative + v1.1 released), ``pd.read_parquet``
    silently reads all of them and ``drop_duplicates(keep='last')`` picks one
    based on filesystem iteration order — which is *not* guaranteed to be the
    latest run. This has caused silent v1.0/v1.1 cross-contamination of
    audit_certificates in the past. Fail at build time instead of shipping a
    release whose certificates disagree with its aggregate metrics.
    """
    offenders: list[str] = []
    for leaf in path.rglob('*'):
        if not leaf.is_dir():
            continue
        files = [p for p in leaf.iterdir() if p.is_file() and p.suffix == '.parquet']
        if len(files) > 1:
            offenders.append(f"{leaf}: {len(files)} parquet files")
    if offenders:
        raise RuntimeError(
            "Stale parquet files detected in partition leaves — refuse to "
            "build a release that would silently mix parser versions:\n  "
            + "\n  ".join(offenders)
            + "\nWipe these output roots and re-run the reprocess step before "
            "calling build_release."
        )


def _read_partitioned_table(path: Path) -> pd.DataFrame:
    if path.is_file() and path.suffix == '.parquet':
        return pd.read_parquet(path)
    if path.is_file() and path.suffix == '.jsonl':
        return pd.read_json(path, lines=True)
    if path.is_dir():
        parquet_files = sorted(path.rglob('*.parquet'))
        if parquet_files:
            _assert_single_parquet_per_partition(path)
            # Read from the table root so partition columns (e.g. benchmark_id/split/model_family)
            # are materialized from hive-style directory names.
            return pd.read_parquet(path)
        jsonl_files = sorted(path.rglob('*.jsonl'))
        if jsonl_files:
            return pd.concat([pd.read_json(p, lines=True) for p in jsonl_files], ignore_index=True)
        return pd.DataFrame()
    return pd.DataFrame()


def _records_from_df(df: pd.DataFrame) -> list[dict]:
    if df.empty:
        return []
    df = df.where(pd.notnull(df), None)
    return df.to_dict(orient='records')


def _dedupe_df(df: pd.DataFrame, primary_key: list[str]) -> pd.DataFrame:
    if df.empty or not primary_key:
        return df
    for key in primary_key:
        if key not in df.columns:
            return df
    return df.drop_duplicates(subset=primary_key, keep='last').reset_index(drop=True)


def _normalize_aggregate_metrics_undefined(
    combined_frames: dict[str, pd.DataFrame],
) -> None:
    """Retro-fit ``metric_status`` onto aggregate_metrics rows whose slice has
    no gold-positive certificates.

    ProofWriter-style benchmarks contain zero gold-positive cases in the
    ``predicate_determining`` dependency_mode by design.  When the pipeline
    emits precision/recall/F1 = 0.0 for such slices, downstream readers may
    misread those zeros as "the system scored zero on this slice" — whereas
    the correct semantics is that the metrics are mathematically undefined.

    This normalizer consults ``audit_certificates`` to determine per-slice
    gold-positive counts, and rewrites the corresponding aggregate_metrics
    rows with ``precision=recall=f1=None`` and ``metric_status=
    'undefined:no_gold_positive_pairs'``.  Rows produced by the post-fix
    ``aggregate()`` already carry the correct status; this function is
    idempotent.
    """

    agg_in = combined_frames.get('aggregate_metrics')
    certs = combined_frames.get('audit_certificates')
    if agg_in is None or agg_in.empty or certs is None or certs.empty:
        return

    # ``audit_certificates`` in per-run source roots is partitioned only by
    # model_family / config_id — ``benchmark_id`` is implicit to the source
    # root and not rematerialized as a column.  Derive it from the
    # ``dataset_problem_id`` convention ``<benchmark_id>::<problem_id>``.
    certs = certs.copy()
    if 'benchmark_id' not in certs.columns and 'dataset_problem_id' in certs.columns:
        certs['benchmark_id'] = certs['dataset_problem_id'].astype(str).str.split('::', n=1).str[0]

    slice_keys = ['benchmark_id', 'model_family', 'config_id', 'dependency_mode_scored']
    if not set(slice_keys).issubset(certs.columns):
        return

    gold_series = certs['gold_dependency_label']
    if gold_series.dtype != bool:
        gold_series = gold_series.astype('boolean').fillna(False).astype(bool)
    gold_positive_count = (
        certs.assign(_gold=gold_series)
             .groupby(slice_keys)['_gold'].sum()
             .to_dict()
    )

    # Build a fresh frame explicitly so the new ``metric_status`` column is
    # guaranteed to materialize regardless of pandas Copy-on-Write policy.
    rows_out: list[dict] = []
    for _, row in agg_in.iterrows():
        row_dict = row.to_dict()
        key = (
            row_dict.get('benchmark_id'),
            row_dict.get('model_family'),
            row_dict.get('config_id'),
            row_dict.get('dependency_mode'),
        )
        try:
            gpos = int(gold_positive_count.get(key, 0))
        except (TypeError, ValueError):
            gpos = 0

        if gpos == 0:
            existing_note = row_dict.get('notes')
            undef_note = (
                "undefined:no_gold_positive_pairs; precision/recall/F1 "
                "are mathematically undefined when the slice contains zero "
                "gold-positive certificates. Numeric fields are set to null "
                "for schema clarity."
            )
            if isinstance(existing_note, str) and existing_note and 'undefined:no_gold_positive_pairs' not in existing_note:
                note_out = f"{existing_note} | {undef_note}"
            elif isinstance(existing_note, str) and 'undefined:no_gold_positive_pairs' in existing_note:
                note_out = existing_note
            else:
                note_out = undef_note
            row_dict['precision'] = None
            row_dict['recall'] = None
            row_dict['f1'] = None
            row_dict['coverage_adjusted_f1'] = None
            row_dict['lower_bound_f1_all_unresolved_negative'] = None
            row_dict['metric_status'] = 'undefined:no_gold_positive_pairs'
            row_dict['notes'] = note_out
        else:
            current = row_dict.get('metric_status')
            if not isinstance(current, str) or current in ('', 'nan'):
                row_dict['metric_status'] = 'defined'
        rows_out.append(row_dict)

    combined_frames['aggregate_metrics'] = pd.DataFrame(rows_out)


def _strip_chunk_suffix(value):
    if isinstance(value, str):
        return _CHUNK_SUFFIX_RE.sub('', value)
    return value


def _normalize_chunk_identifiers(table_name: str, df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    if 'config_id' in out.columns:
        out['config_id'] = out['config_id'].map(_strip_chunk_suffix)

    if table_name == 'run_summaries' and {'summary_id', 'dataset_problem_id', 'model_id', 'config_id'}.issubset(out.columns):
        out['summary_id'] = out.apply(
            lambda r: f"{r['dataset_problem_id']}::{r['model_id']}::{r['config_id']}",
            axis=1,
        )

    if table_name == 'aggregate_metrics' and {'metric_id', 'benchmark_id', 'model_family', 'config_id', 'dependency_mode', 'metric_scope'}.issubset(out.columns):
        out['metric_id'] = out.apply(
            lambda r: f"{r['benchmark_id']}::{r['model_family']}::{r['config_id']}::{r['dependency_mode']}::{r['metric_scope']}",
            axis=1,
        )

    if table_name == 'audit_certificates' and {'certificate_id', 'model_id'}.issubset(out.columns):
        def _fix_cert_id(row):
            cid = row['certificate_id']
            mid = row['model_id']
            if f'::{mid}::cert::' not in cid:
                return cid.replace('::cert::', f'::{mid}::cert::', 1)
            return cid
        out['certificate_id'] = out.apply(_fix_cert_id, axis=1)

    return out


def build_release(
    output_roots: Iterable[str | Path],
    *,
    schema_path: str | Path,
    release_root: str | Path,
    release_id: str = 'iga-bench-core-v1.1',
    review_snapshot_id: str | None = 'iga-bench-core-v1.1-review',
) -> dict:
    schema = SchemaRegistry(schema_path)
    roots = [Path(p) for p in output_roots]
    release_root = Path(release_root)
    release_root.mkdir(parents=True, exist_ok=True)

    combined_frames: dict[str, pd.DataFrame] = {}
    source_manifests = []
    for table_name in schema.tables.keys():
        frames = []
        for root in roots:
            df = _read_partitioned_table(root / 'data' / table_name)
            if not df.empty:
                frames.append(df)
        if frames:
            merged = pd.concat(frames, ignore_index=True)
            merged = _normalize_chunk_identifiers(table_name, merged)
            primary_key = schema.raw['tables'][table_name].get('primary_key', [])
            combined_frames[table_name] = _dedupe_df(merged, primary_key)
        else:
            combined_frames[table_name] = pd.DataFrame(columns=schema.column_names(table_name))

    # Retro-fit ``metric_status`` onto aggregate_metrics rows whose slice has
    # no gold-positive certificates (e.g. ProofWriter predicate_determining).
    # Safe to call even when the source parquet already emits the column —
    # the normalizer is idempotent for slices that already carry the
    # correct status and numeric convention.
    _normalize_aggregate_metrics_undefined(combined_frames)

    store = TableStore(schema)
    for table_name, df in combined_frames.items():
        for row in _records_from_df(df):
            store.add(table_name, row)
    store.write(release_root)
    store.write_jsonl_shadow(release_root / 'jsonl_shadow')

    report = validate_dataset_root(release_root)
    (release_root / 'validator_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')

    for root in roots:
        manifest_path = root / 'run_manifest.json'
        legacy_manifest_path = root / 'pilot_manifest.json'
        if manifest_path.exists():
            source_manifests.append(_sanitize_manifest_value(json.loads(manifest_path.read_text(encoding='utf-8'))))
        elif legacy_manifest_path.exists():
            source_manifests.append(_sanitize_manifest_value(json.loads(legacy_manifest_path.read_text(encoding='utf-8'))))
        else:
            source_manifests.append({'output_root': root.name, 'validation_status': 'UNKNOWN'})

    release_manifest = {
        'release_id': release_id,
        # ``review_snapshot_id`` is the logical pointer used by the paper
        # (abstract, dataset card, PROJECT_MAP.md) and by the reviewer-facing
        # bundle to refer back to this exact release.  Keeping it explicit
        # here means the manifest, dataset card, and paper can all be
        # cross-checked by a single ``jq`` on release_manifest.json.
        'review_snapshot_id': review_snapshot_id,
        'source_roots': [_safe_source_root(r, release_root) for r in roots],
        'source_manifests': source_manifests,
        'tables': {
            table: int(len(df)) for table, df in combined_frames.items()
        },
        'validation_status': report['status'],
    }
    (release_root / 'release_manifest.json').write_text(json.dumps(release_manifest, indent=2), encoding='utf-8')

    checksums = {}
    for p in sorted(release_root.rglob('*')):
        if p.is_file():
            checksums[str(p.relative_to(release_root))] = sha256_file(p)
    (release_root / 'checksums.json').write_text(json.dumps(checksums, indent=2), encoding='utf-8')

    return {
        'release_root': str(release_root.resolve()),
        'validation': report,
        'num_source_roots': len(roots),
        'table_counts': release_manifest['tables'],
    }
