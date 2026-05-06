from pathlib import Path
import json

from iga_suite.config import load_config
from iga_suite.pipeline import run_evaluation
from iga_suite.release import build_release
from iga_suite.metadata import materialize_dataset_card, materialize_croissant
from iga_suite.validator import validate_dataset_root


def test_evaluation_mock_release_build(tmp_path):
    root = Path(__file__).resolve().parents[1]

    pronto_cfg = load_config(root / 'configs' / 'evaluation' / 'prontoqa_mock.yaml')
    pronto_cfg.output_root = str(tmp_path / 'prontoqa_out')
    pronto_cfg.companion_root = str(tmp_path / 'prontoqa_comp')
    pronto_cfg.run.max_problems = 2
    r1 = run_evaluation(pronto_cfg)
    assert r1['validation']['status'] == 'PASS'

    proof_cfg = load_config(root / 'configs' / 'evaluation' / 'proofwriter_mock.yaml')
    proof_cfg.output_root = str(tmp_path / 'proofwriter_out')
    proof_cfg.companion_root = str(tmp_path / 'proofwriter_comp')
    proof_cfg.run.max_problems = 2
    r2 = run_evaluation(proof_cfg)
    assert r2['validation']['status'] == 'PASS'

    release_root = tmp_path / 'release'
    report = build_release(
        [pronto_cfg.output_root, proof_cfg.output_root],
        schema_path=root / 'schema' / 'parquet_schema_contract.yaml',
        release_root=release_root,
        release_id='iga-bench-core-v1.1-test',
    )
    assert report['validation']['status'] == 'PASS'

    card_out = release_root / 'IGA-Bench-Core-v1.1.dataset-card.md'
    croissant_out = release_root / 'iga-bench-core-v1.1.croissant.json'
    stats = materialize_dataset_card(release_root, card_out)
    assert stats['num_benchmarks'] == 2
    materialize_croissant(release_root, root / 'templates' / 'croissant_template.json', croissant_out)
    assert card_out.exists()
    assert croissant_out.exists()
    assert validate_dataset_root(release_root)['status'] == 'PASS'
