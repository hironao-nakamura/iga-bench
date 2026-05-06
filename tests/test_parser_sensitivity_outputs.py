"""Smoke tests for the canonicalizer sensitivity analysis artifacts.

The parser sensitivity tables live under `<release>/analysis/` and are
deliberately *not* part of the primary single source of truth in `data/`.
These tests verify that the analysis outputs exist, cover the expected
(benchmark, family, dependency_mode) cells, and carry the columns the
paper's sensitivity appendix relies on.
"""

from pathlib import Path
import json

import pandas as pd
import pytest

RELEASE_ROOT = Path(__file__).resolve().parents[1] / 'release' / 'iga-bench-core-v1.1'
ANALYSIS_DIR = RELEASE_ROOT / 'analysis'

pytestmark = pytest.mark.skipif(
    not ANALYSIS_DIR.exists(),
    reason='Analysis directory not present; skipping sensitivity tests',
)


REQUIRED_COLUMNS = {
    'benchmark_id',
    'model_family',
    'dependency_mode',
    'coverage_conservative',
    'f1_conservative',
    'num_problems_conservative',
    'coverage_released',
    'f1_released',
    'num_problems_released',
    'coverage_delta',
    'f1_delta',
    'in_primary_release',
}


def _load_parquet() -> pd.DataFrame:
    return pd.read_parquet(ANALYSIS_DIR / 'parser_sensitivity.parquet')


def _load_json() -> list:
    with (ANALYSIS_DIR / 'parser_sensitivity.json').open() as f:
        return json.load(f)


class TestArtifactsExist:
    def test_parquet_present(self):
        assert (ANALYSIS_DIR / 'parser_sensitivity.parquet').is_file()

    def test_json_present(self):
        assert (ANALYSIS_DIR / 'parser_sensitivity.json').is_file()

    def test_readme_present(self):
        assert (ANALYSIS_DIR / 'README.md').is_file()


class TestSchema:
    def test_columns(self):
        df = _load_parquet()
        missing = REQUIRED_COLUMNS - set(df.columns)
        assert not missing, f'Missing columns: {missing}'

    def test_json_matches_parquet(self):
        df = _load_parquet()
        js = _load_json()
        assert len(df) == len(js)

    def test_value_ranges(self):
        df = _load_parquet()
        for col in ['coverage_conservative', 'coverage_released',
                    'f1_conservative', 'f1_released']:
            assert (df[col] >= 0.0).all()
            assert (df[col] <= 1.0).all()
        # Deltas: released - conservative, bounded in [-1, 1].
        for col in ['coverage_delta', 'f1_delta']:
            assert (df[col] >= -1.0).all()
            assert (df[col] <= 1.0).all()


class TestCoverage:
    def test_primary_benchmarks_covered(self):
        df = _load_parquet()
        primary = df[df['in_primary_release']]
        bids = set(primary['benchmark_id'])
        assert bids == {'prontoqa_full_eval', 'proofwriter_cwa_d3_is_300'}

    def test_three_families_per_primary_benchmark(self):
        df = _load_parquet()
        primary = df[df['in_primary_release']]
        # 2 benchmarks × 3 families × 3 dependency modes = 18 rows
        assert len(primary) == 18
        for bid in ['prontoqa_full_eval', 'proofwriter_cwa_d3_is_300']:
            sub = primary[primary['benchmark_id'] == bid]
            assert set(sub['model_family']) == {'openai', 'anthropic', 'openweight'}
            assert set(sub['dependency_mode']) == {
                'direct', 'transitive', 'predicate_determining',
            }

    def test_legacy_proofwriter_isolated_if_present(self):
        # The retired ProofWriter slice from the v1.0 timeline is
        # explicitly not part of the primary v1.1 release.  If a
        # sensitivity-mode row for it ever appears under analysis/, it
        # must be flagged ``in_primary_release == False``.
        legacy_bid = 'proofwriter_cwa_d3_is_100'
        df = _load_parquet()
        legacy = df[df['benchmark_id'] == legacy_bid]
        if len(legacy) > 0:
            assert not legacy['in_primary_release'].any()


class TestKnownDeltas:
    """Sanity anchors on observable parser sensitivity shifts.

    The surface-relaxed canonicalizer (v1.1-final) is known to dramatically
    improve coverage on Anthropic Claude traces (which use non-canonical
    surface forms for `is`-predicates). We pin the qualitative direction
    rather than exact numerics to stay robust to future reprocessing.
    """

    def test_anthropic_prontoqa_coverage_up(self):
        df = _load_parquet()
        row = df[
            (df['benchmark_id'] == 'prontoqa_full_eval')
            & (df['model_family'] == 'anthropic')
            & (df['dependency_mode'] == 'direct')
        ]
        assert len(row) == 1
        # Conservative parser misses most Anthropic surface-forms; released
        # parser should recover ≥ 50 percentage points of coverage.
        assert row.iloc[0]['coverage_delta'] > 0.5

    def test_openai_prontoqa_coverage_stable(self):
        df = _load_parquet()
        row = df[
            (df['benchmark_id'] == 'prontoqa_full_eval')
            & (df['model_family'] == 'openai')
            & (df['dependency_mode'] == 'direct')
        ]
        assert len(row) == 1
        # OpenAI already emits canonical forms; delta should be small.
        assert abs(row.iloc[0]['coverage_delta']) < 0.1
