from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import os
import yaml


def _expand_env(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    return value


@dataclass
class BenchmarkConfig:
    loader: str
    benchmark_id: str
    benchmark_name: str
    benchmark_version: str
    benchmark_family: str
    source_url: str | None
    upstream_license_status: str
    input_path: str
    split: str
    release_tier: str
    include_benchmark_text_in_core: bool
    supports_extension_modes: bool
    notes: str | None = None


@dataclass
class ModelConfig:
    provider: str
    model_id: str
    provider_name: str
    model_family: str
    model_name: str
    access_type: str
    trace_mode: str
    notes: str | None = None
    api_key_env: str | None = None
    base_url_env: str | None = None
    model_snapshot: str | None = None
    temperature: float = 0.0
    max_tokens: int = 1200


@dataclass
class SelfConsistencyConfig:
    enabled: bool = False
    samples: int = 5
    temperature: float = 0.7


@dataclass
class RunConfig:
    config_id: str
    max_problems: int | None
    repetitions: int
    probe_bundle_id: str
    probe_types: list[str]
    dependency_modes: list[str]
    include_extension_scope_predicate_determining: bool = True
    prompt_mode: str = 'forced_step'
    null_probe_enabled: bool = False
    self_consistency: SelfConsistencyConfig = field(default_factory=SelfConsistencyConfig)


@dataclass
class AppConfig:
    schema_path: str
    output_root: str
    companion_root: str
    raw_companion: bool
    benchmark: BenchmarkConfig
    model: ModelConfig
    run: RunConfig

    @property
    def output_root_path(self) -> Path:
        return Path(self.output_root)

    @property
    def companion_root_path(self) -> Path:
        return Path(self.companion_root)


def load_config(path: str | Path) -> AppConfig:
    with open(path, 'r', encoding='utf-8') as f:
        raw = yaml.safe_load(f)
    raw = _expand_env(raw)
    sc = raw['run'].get('self_consistency', {})
    return AppConfig(
        schema_path=raw['schema_path'],
        output_root=raw['output_root'],
        companion_root=raw['companion_root'],
        raw_companion=bool(raw.get('raw_companion', True)),
        benchmark=BenchmarkConfig(**raw['benchmark']),
        model=ModelConfig(**raw['model']),
        run=RunConfig(
            **{k: v for k, v in raw['run'].items() if k != 'self_consistency'},
            self_consistency=SelfConsistencyConfig(**sc),
        ),
    )
