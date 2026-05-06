from pathlib import Path
import json

from iga_suite.config import load_config
from iga_suite.pipeline import run_evaluation
from iga_suite.validator import validate_dataset_root


def test_mock_pipeline_smoke(tmp_path):
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / 'configs' / 'evaluation' / 'prontoqa_mock.yaml')
    cfg.output_root = str(tmp_path / 'out')
    cfg.companion_root = str(tmp_path / 'companion')
    cfg.run.max_problems = 2
    result = run_evaluation(cfg)
    assert result['validation']['status'] == 'PASS'
    assert result['summary_primary_scope_only'] is True
    assert result['summary_extension_scope_separate'] is True
    for scope, summary in result['scope_summaries'].items():
        assert summary['verdict_total_matches_certificates'] is True, f"scope mismatch: {scope}"
    report = validate_dataset_root(cfg.output_root)
    assert report['status'] == 'PASS'
    manifest_path = Path(cfg.output_root) / 'run_manifest.json'
    if not manifest_path.exists():
        manifest_path = Path(cfg.output_root) / 'pilot_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    assert manifest['num_problems'] == 2
