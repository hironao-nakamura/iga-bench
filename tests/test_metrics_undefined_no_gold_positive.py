"""Regression tests for the ``metric_status`` convention.

Some benchmark/dependency-mode slices contain zero gold-positive
certificates by design — most notably ProofWriter-style benchmarks under
the ``predicate_determining`` dependency mode.  In that regime, the
classic precision/recall/F1 are mathematically undefined (division by
zero in both precision and recall denominators for the positive class).

Historically the pipeline emitted ``0.0`` in those cells, which invited
the misreading "the system scored zero here" when the correct reading
is "the metric is not defined here at all".  These tests pin down the
new contract: such cells surface with numeric fields set to ``None`` and
a ``metric_status`` field set to ``'undefined:no_gold_positive_pairs'``.
"""

from iga_suite.metrics import aggregate


def _cert(mode: str, verdict: str, gold: bool) -> dict:
    return {
        "dependency_mode_scored": mode,
        "verdict_type": verdict,
        "gold_dependency_label": gold,
        "parse_ok": True,
        "alignment_ok": True,
        "canonical_original": "is(a, b)",
        "canonical_probed": "is(a, c)",
    }


def test_predicate_determining_slice_without_gold_positives_is_undefined():
    # ProofWriter-style: every predicate_determining cert is gold-negative
    # (no designated "dependency-positive" premise exists for this regime).
    certs = [
        _cert("predicate_determining", "GROUNDED", False),
        _cert("predicate_determining", "INSENSITIVE", False),
        _cert("predicate_determining", "INPUT-SENSITIVE", False),
        _cert("predicate_determining", "MISREPRESENTATION", False),
    ]
    rows = aggregate(
        certs,
        problems=[{"premises": [{"id": "P1"}]}],
        benchmark_id="proofwriter_cwa_d3_is_300",
        split="analysis",
        model_family="anthropic",
        model_id="mid",
        config_id="cfg",
    )
    assert len(rows) == 1
    row = rows[0]
    assert row["dependency_mode"] == "predicate_determining"
    assert row["metric_status"] == "undefined:no_gold_positive_pairs"
    assert row["precision"] is None
    assert row["recall"] is None
    assert row["f1"] is None
    assert row["coverage_adjusted_f1"] is None
    assert row["lower_bound_f1_all_unresolved_negative"] is None
    assert row["num_certificates"] == len(certs)
    assert "undefined:no_gold_positive_pairs" in (row["notes"] or "")


def test_slice_with_gold_positives_is_defined():
    certs = [
        _cert("direct", "GROUNDED", True),
        _cert("direct", "GROUNDED", False),
        _cert("direct", "INSENSITIVE", True),
    ]
    rows = aggregate(
        certs,
        problems=[{"premises": [{"id": "P1"}]}],
        benchmark_id="prontoqa_full_eval",
        split="full_eval",
        model_family="openai",
        model_id="mid",
        config_id="cfg",
    )
    row = rows[0]
    assert row["metric_status"] == "defined"
    assert row["precision"] is not None
    assert row["recall"] is not None
    assert row["f1"] is not None


def test_undefined_slice_still_reports_numeric_counts_and_coverage():
    # Even when metrics are undefined, counts/coverage should remain
    # numerically meaningful so audit scripts can sanity-check the slice.
    certs = [
        _cert("predicate_determining", "GROUNDED", False),
        _cert("predicate_determining", "UNPARSEABLE", False),
    ]
    rows = aggregate(
        certs,
        problems=[{"premises": [{"id": "P1"}]}],
        benchmark_id="proofwriter_cwa_d3_is_300",
        split="analysis",
        model_family="openweight",
        model_id="mid",
        config_id="cfg",
    )
    row = rows[0]
    assert row["metric_status"] == "undefined:no_gold_positive_pairs"
    assert row["num_certificates"] == 2
    # coverage is (# definitive) / (# certs) = 1/2 = 0.5 irrespective of
    # whether F1 is defined for this slice.
    assert row["coverage"] == 0.5
