from pathlib import Path
import json

from iga_suite.benchmarks import bootstrap_proofwriter, load_generic_jsonl


def test_bootstrap_proofwriter_tiny(tmp_path):
    root = Path(__file__).resolve().parents[1]
    raw = root / 'examples' / 'tiny' / 'proofwriter_tiny_raw.jsonl'
    out = tmp_path / 'proofwriter_norm.jsonl'
    count = bootstrap_proofwriter(raw, out, benchmark_id='proofwriter_tiny', split='analysis')
    rows = load_generic_jsonl(out)
    assert count == 4
    assert len(rows) == 4
    assert rows[0]['benchmark_id'] == 'proofwriter_tiny'
    assert rows[0]['premises'][0]['id'] == 'P1'
    assert rows[1]['answer'] is False
    assert rows[2]['proof_tree'][0]['conclusion'].startswith('is(')
