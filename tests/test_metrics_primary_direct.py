from iga_suite.metrics import aggregate, compute_scope_metrics


def _base_cert(mode: str, verdict: str, gold: bool) -> dict:
    return {
        "dependency_mode_scored": mode,
        "verdict_type": verdict,
        "gold_dependency_label": gold,
        "parse_ok": True,
        "alignment_ok": True,
        "canonical_original": "is(a, b)",
        "canonical_probed": "is(a, c)",
    }


def test_aggregate_direct_primary_is_grounded_only():
    certs = [
        _base_cert("direct", "GROUNDED", True),          # TP
        _base_cert("direct", "INPUT-SENSITIVE", True),   # FN in primary direct
        _base_cert("direct", "INPUT-SENSITIVE", False),  # TN in primary direct
    ]
    rows = aggregate(
        certs,
        problems=[{"premises": [{"id": "P1"}, {"id": "P2"}]}],
        benchmark_id="b",
        split="s",
        model_family="m",
        model_id="mid",
        config_id="cfg",
    )
    direct = [r for r in rows if r["dependency_mode"] == "direct"][0]
    assert direct["precision"] == 1.0
    assert direct["recall"] == 0.5
    assert direct["f1"] == 0.666667


def test_scope_metrics_exposes_ws_compat_for_direct():
    certs = [
        _base_cert("direct", "GROUNDED", True),
        _base_cert("direct", "INPUT-SENSITIVE", True),
        _base_cert("direct", "INPUT-SENSITIVE", False),
    ]
    summary = compute_scope_metrics(certs)
    assert summary["f1"] == 0.6666666666666666
    assert summary["ws_compat_f1"] == 0.8

