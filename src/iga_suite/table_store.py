from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import json
import math
import pandas as pd

from iga_suite.schema_registry import SchemaRegistry


def _sanitize_for_json(value):
    """Replace NaN / pandas-NA with ``None`` for JSON-safe serialization.

    ``json.dumps`` by default emits ``NaN`` for ``float('nan')``, which is not
    valid JSON and breaks downstream JSONL consumers.  ``pd.DataFrame.to_dict``
    can resurrect NaN even when the source frame was populated with ``None``
    (because the column dtype is float).  This helper walks the row and maps
    any such NaN / pd.NA back to ``None`` so the JSONL shadow matches the
    parquet semantics (undefined = null, not NaN).
    """
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    try:
        if value is pd.NA:  # type: ignore[attr-defined]
            return None
    except Exception:
        pass
    return value


def _sanitize_row(row: dict) -> dict:
    return {k: _sanitize_for_json(v) for k, v in row.items()}


class TableStore:
    def __init__(self, schema: SchemaRegistry):
        self.schema = schema
        self.rows = defaultdict(list)

    def add(self, table_name: str, row: dict):
        cols = self.schema.column_names(table_name)
        required = set(self.schema.required(table_name))
        out = {}
        for c in cols:
            out[c] = row.get(c)
        missing = [c for c in required if out.get(c) is None]
        if missing:
            raise ValueError(f'{table_name}: missing required columns {missing} in row {row}')
        self.rows[table_name].append(out)

    def _write_jsonl_partitioned(self, df: pd.DataFrame, table_root: Path, partition_cols: list[str]):
        if not partition_cols:
            table_root.mkdir(parents=True, exist_ok=True)
            path = table_root / 'part-00000.jsonl'
            with open(path, 'w', encoding='utf-8') as f:
                for rec in df.where(pd.notnull(df), None).to_dict(orient='records'):
                    f.write(json.dumps(_sanitize_row(rec), ensure_ascii=False, allow_nan=False) + '\n')
            return
        grouped = df.groupby(partition_cols, dropna=False)
        for keys, sub in grouped:
            if not isinstance(keys, tuple):
                keys = (keys,)
            part_path = table_root
            for col, key in zip(partition_cols, keys):
                part_path = part_path / f"{col}={key}"
            part_path.mkdir(parents=True, exist_ok=True)
            with open(part_path / 'part-00000.jsonl', 'w', encoding='utf-8') as f:
                for rec in sub.where(pd.notnull(sub), None).to_dict(orient='records'):
                    f.write(json.dumps(_sanitize_row(rec), ensure_ascii=False, allow_nan=False) + '\n')

    def write(self, root: str | Path):
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        for table_name, rows in self.rows.items():
            if not rows:
                continue
            df = pd.DataFrame(rows)
            table_root = root / 'data' / table_name
            partition_cols = [c for c in ['benchmark_id', 'split', 'model_family', 'config_id'] if c in df.columns]
            # avoid overpartitioning tiny tables
            partition_cols = [c for c in partition_cols if c in {'benchmark_id', 'split', 'model_family'} or table_name in {'trace_steps', 'step_alignments', 'audit_certificates', 'run_summaries', 'aggregate_metrics', 'runs'}]
            try:
                if partition_cols:
                    df.to_parquet(table_root, engine='pyarrow', index=False, partition_cols=partition_cols)
                else:
                    table_root.mkdir(parents=True, exist_ok=True)
                    df.to_parquet(table_root / 'part-00000.parquet', engine='pyarrow', index=False)
                # Also emit a flat view next to the partitioned directory so
                # the flat-Parquet release layout can resolve each table via
                # a single .parquet file without having to walk hive
                # partitions.
                flat_path = table_root.with_suffix('.parquet')
                df.to_parquet(flat_path, engine='pyarrow', index=False)
            except Exception:
                # Local fallback for environments where pyarrow native libs are unavailable.
                self._write_jsonl_partitioned(df, table_root, partition_cols)

    def write_jsonl_shadow(self, root: str | Path):
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        for table_name, rows in self.rows.items():
            if not rows:
                continue
            path = root / f'{table_name}.jsonl'
            with open(path, 'w', encoding='utf-8') as f:
                for r in rows:
                    f.write(json.dumps(_sanitize_row(r), ensure_ascii=False, allow_nan=False) + '\n')
