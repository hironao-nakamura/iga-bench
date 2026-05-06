from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable
import yaml

from iga_suite.config import load_config
from iga_suite.pipeline import run_evaluation


def _load_matrix_spec(path: str | Path) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def resolve_config_paths(matrix_spec_path: str | Path) -> list[Path]:
    spec_path = Path(matrix_spec_path)
    spec = _load_matrix_spec(spec_path)
    base = spec_path.parent
    if 'configs' in spec:
        return [Path(base / p).resolve() for p in spec['configs']]
    if 'config_glob' in spec:
        return sorted(base.glob(spec['config_glob']))
    raise ValueError('Matrix spec must provide `configs` or `config_glob`')


def run_matrix(matrix_spec_path: str | Path, *, continue_on_error: bool = False) -> dict:
    config_paths = resolve_config_paths(matrix_spec_path)
    runs = []
    errors = []
    for cfg_path in config_paths:
        try:
            cfg = load_config(cfg_path)
            result = run_evaluation(cfg)
            runs.append({
                'config_path': str(cfg_path),
                'output_root': result['output_root'],
                'validation_status': result['validation']['status'],
                'num_problems': result['num_problems'],
                'num_certificates': result['num_certificates'],
            })
        except Exception as e:  # pragma: no cover - surfaced to CLI/UI
            errors.append({'config_path': str(cfg_path), 'error': repr(e)})
            if not continue_on_error:
                raise
    return {
        'matrix_spec': str(Path(matrix_spec_path).resolve()),
        'num_runs': len(runs),
        'num_errors': len(errors),
        'runs': runs,
        'errors': errors,
    }
