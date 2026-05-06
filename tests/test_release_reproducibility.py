"""Smoke test that paper tables can be regenerated from the shipped release.

Target: IGA-Bench Core v1.1 (ProntoQA-500 + ProofWriter-300, parser v1.1-final).
"""

from pathlib import Path

import pandas as pd
import pytest

from iga_suite.metadata import materialize_dataset_card
from iga_suite.report_tables import (
    build_tables, build_table1, build_table2, build_table4,
    build_appendix_a, build_appendix_c, build_appendix_d,
)
from iga_suite.report_figures import build_figures

RELEASE_ROOT = Path(__file__).resolve().parents[1] / 'release' / 'iga-bench-core-v1.1'

pytestmark = pytest.mark.skipif(
    not RELEASE_ROOT.exists(),
    reason='Release root not present; skipping reproducibility tests',
)


def _read_table(name: str) -> pd.DataFrame:
    flat = RELEASE_ROOT / 'data' / f'{name}.parquet'
    if flat.is_file():
        return pd.read_parquet(flat)
    path = RELEASE_ROOT / 'data' / name
    if path.is_dir():
        return pd.read_parquet(path)
    return pd.DataFrame()


class TestMetadata:
    def test_dataset_card_fields(self, tmp_path):
        card_out = tmp_path / 'card.md'
        stats = materialize_dataset_card(RELEASE_ROOT, card_out)

        assert stats['num_benchmarks'] == 2
        assert stats['num_models'] == 6
        assert stats['num_problems'] == 800
        assert stats['num_certificates'] == 150624
        assert 'direct' in stats['dependency_modes']
        assert 'transitive' in stats['dependency_modes']
        assert 'predicate_determining' in stats['dependency_modes']
        assert set(stats['benchmark_ids']) == {
            'prontoqa_full_eval', 'proofwriter_cwa_d3_is_300',
        }

        text = card_out.read_text()
        assert 'predicate-determining' in text or 'predicate_determining' in text
        assert 'v1.1-review' in text


class TestTable1:
    def test_cell_count(self):
        metrics = _read_table('aggregate_metrics')
        t1 = build_table1(metrics)
        assert len(t1) == 6

    def test_expected_benchmarks(self):
        metrics = _read_table('aggregate_metrics')
        t1 = build_table1(metrics)
        bids = set(t1['benchmark_id'])
        assert 'prontoqa_full_eval' in bids
        assert 'proofwriter_cwa_d3_is_300' in bids
        # The retired ProofWriter slice from the v1.0 timeline is held back
        # from the primary release and must not appear in the primary table
        # alongside the headline (300-problem) ProofWriter row.
        legacy_bid = 'proofwriter_cwa_d3_is_100'
        assert legacy_bid not in bids

    def test_expected_models(self):
        metrics = _read_table('aggregate_metrics')
        t1 = build_table1(metrics)
        mfs = set(t1['model_family'])
        assert mfs == {'openai', 'anthropic', 'openweight'}

    def test_f1_range(self):
        metrics = _read_table('aggregate_metrics')
        t1 = build_table1(metrics)
        for _, row in t1.iterrows():
            assert 0.0 <= row['f1'] <= 1.0
            if not pd.isna(row.get('transitive_f1')):
                assert 0.0 <= row['transitive_f1'] <= 1.0


class TestTable2:
    def test_covered_pairs_only(self):
        certs = _read_table('audit_certificates')
        t2 = build_table2(certs)
        assert len(t2) > 0
        for _, row in t2.iterrows():
            assert 0.0 <= row['recall'] <= 1.0
            assert 0.0 <= row['f1'] <= 1.0

    def test_spot_check_gpt4o_prontoqa_is(self):
        certs = _read_table('audit_certificates')
        t2 = build_table2(certs)
        row = t2[
            (t2['benchmark_id'] == 'prontoqa_full_eval')
            & (t2['model_family'] == 'openai')
            & (t2['premise_type'] == 'is')
        ]
        assert len(row) == 1
        assert abs(row.iloc[0]['recall'] - 0.811) < 0.01


class TestTable4:
    def test_rawr_rows(self):
        rs = _read_table('run_summaries')
        t4 = build_table4(rs)
        assert len(t4) == 6

    def test_gpt4o_prontoqa_direct_rawr(self):
        rs = _read_table('run_summaries')
        t4 = build_table4(rs)
        row = t4[
            (t4['benchmark_id'] == 'prontoqa_full_eval')
            & (t4['model_family'] == 'openai')
        ]
        assert len(row) == 1
        assert abs(row.iloc[0]['rawr_direct_pct'] - 2.4) < 0.5


class TestAppendixA:
    def test_control_probe_rows(self):
        certs = _read_table('audit_certificates')
        ta = build_appendix_a(certs)
        assert len(ta) > 0
        assert 'gold_positive' in ta['gold_label'].values
        assert 'gold_negative' in ta['gold_label'].values

    def test_spot_check_gpt4o_prontoqa(self):
        certs = _read_table('audit_certificates')
        ta = build_appendix_a(certs)
        row = ta[
            (ta['benchmark_id'] == 'prontoqa_full_eval')
            & (ta['model_family'] == 'openai')
            & (ta['gold_label'] == 'gold_positive')
        ]
        assert len(row) == 1
        assert abs(row.iloc[0]['semantic_rate'] - 0.799) < 0.01
        assert abs(row.iloc[0]['surface_rate'] - 0.161) < 0.01
        assert abs(row.iloc[0]['null_rate'] - 0.097) < 0.01


class TestAppendixC:
    def test_predicate_determining_rows(self):
        metrics = _read_table('aggregate_metrics')
        certs = _read_table('audit_certificates')
        tc = build_appendix_c(metrics, certs)
        assert len(tc) == 6
        assert 'fp_count' in tc.columns

    def test_fp_count_gpt4o_prontoqa(self):
        metrics = _read_table('aggregate_metrics')
        certs = _read_table('audit_certificates')
        tc = build_appendix_c(metrics, certs)
        row = tc[
            (tc['benchmark_id'] == 'prontoqa_full_eval')
            & (tc['model_family'] == 'openai')
        ]
        assert len(row) == 1
        assert row.iloc[0]['fp_count'] == 693

    def test_fp_count_proofwriter_has_values(self):
        metrics = _read_table('aggregate_metrics')
        certs = _read_table('audit_certificates')
        tc = build_appendix_c(metrics, certs)
        pw = tc[tc['benchmark_id'] == 'proofwriter_cwa_d3_is_300']
        assert len(pw) == 3
        assert all(pw['fp_count'] > 0)


class TestAppendixD:
    def test_rawr_conditioned_rows(self):
        certs = _read_table('audit_certificates')
        rs = _read_table('run_summaries')
        td = build_appendix_d(certs, rs)
        assert len(td) == 6

    def test_gpt4o_prontoqa_rawr_cov(self):
        certs = _read_table('audit_certificates')
        rs = _read_table('run_summaries')
        td = build_appendix_d(certs, rs)
        row = td[
            (td['benchmark_id'] == 'prontoqa_full_eval')
            & (td['model_family'] == 'openai')
        ]
        assert len(row) == 1
        assert abs(row.iloc[0]['rawr_cov_direct_pct'] - 2.6) < 1.0
        assert row.iloc[0]['n_cov_direct'] == 462

    def test_claude_prontoqa_n_cov(self):
        # v1.1 snapshot: Claude-on-ProntoQA now yields 413 covered certificates
        # under the released (v1.1-final) surface-relaxed canonicalizer.  The
        # earlier expected value (92) was produced by the conservative v1.0
        # parser, which v1.1 retires as a sensitivity condition under
        # analysis/ rather than as the single source of truth.
        certs = _read_table('audit_certificates')
        rs = _read_table('run_summaries')
        td = build_appendix_d(certs, rs)
        row = td[
            (td['benchmark_id'] == 'prontoqa_full_eval')
            & (td['model_family'] == 'anthropic')
        ]
        assert len(row) == 1
        assert row.iloc[0]['n_cov_direct'] == 413


class TestFigure2:
    def test_figure2_data(self, tmp_path):
        result = build_figures(RELEASE_ROOT, tmp_path / 'figs')
        assert result['figure2_points'] == 12
        assert (tmp_path / 'figs' / 'figure2_coverage_vs_f1.json').exists()


class TestBuildTables:
    def test_full_pipeline(self, tmp_path):
        result = build_tables(RELEASE_ROOT, tmp_path / 'tables')
        assert result['table1_rows'] == 6
        assert result['table2_rows'] > 0
        assert result['table4_rows'] == 6
        assert result['appendix_a_rows'] > 0
        assert result['appendix_d_rows'] == 6
        assert (tmp_path / 'tables' / 'table1_main_matrix.json').exists()
        assert (tmp_path / 'tables' / 'table2_premise_slice.json').exists()
        assert (tmp_path / 'tables' / 'table4_rawr.json').exists()
        assert (tmp_path / 'tables' / 'appendix_a_control_probe.json').exists()
        assert (tmp_path / 'tables' / 'appendix_c_predicate_determining.json').exists()
        assert (tmp_path / 'tables' / 'appendix_d_rawr_conditioned.json').exists()
