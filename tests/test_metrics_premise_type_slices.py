from iga_suite.metrics import compute_scope_metrics


def _cert(scope: str, verdict: str, gold: bool, ptype: str):
    return {
        "dependency_mode_scored": scope,
        "verdict_type": verdict,
        "gold_dependency_label": gold,
        "premise_canonical_type": ptype,
        "canonical_original": "is(a, b)",
        "canonical_probed": "is(a, c)",
    }


def test_scope_metrics_contains_premise_type_slices():
    certs = [
        _cert("direct", "GROUNDED", True, "is"),
        _cert("direct", "INSENSITIVE", True, "is"),
        _cert("direct", "GROUNDED", True, "subtype"),
        _cert("direct", "GROUNDED", False, "subtype"),
    ]
    s = compute_scope_metrics(certs)
    assert "premise_type_metrics" in s
    assert "is" in s["premise_type_metrics"]
    assert "subtype" in s["premise_type_metrics"]
    assert s["premise_type_metrics"]["is"]["recall"] == 0.5
    assert s["premise_type_metrics"]["subtype"]["recall"] == 1.0
