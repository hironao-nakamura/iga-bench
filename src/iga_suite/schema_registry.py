from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import yaml


@dataclass
class ColumnSpec:
    name: str
    type: str
    nullable: bool


class SchemaRegistry:
    def __init__(self, schema_path: str | Path):
        with open(schema_path, 'r', encoding='utf-8') as f:
            self.raw = yaml.safe_load(f)
        self.tables = self.raw['tables']

    def columns(self, table_name: str) -> list[ColumnSpec]:
        return [ColumnSpec(c['name'], c['type'], c.get('nullable', True)) for c in self.tables[table_name]['columns']]

    def required(self, table_name: str) -> list[str]:
        return [c.name for c in self.columns(table_name) if not c.nullable]

    def column_names(self, table_name: str) -> list[str]:
        return [c.name for c in self.columns(table_name)]
