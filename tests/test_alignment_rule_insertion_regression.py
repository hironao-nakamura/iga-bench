from iga_suite.aligner import align_steps_by_reference_ranks


def _revisionture_probe_steps_with_inserted_rule():
    # A probe run where subtype rule sentences are inserted before is(...) derivations.
    return [
        {"step_index": 1, "canonical_form": "subtype(a, zqb)", "canonical_type": "subtype", "parse_status": "OK"},
        {"step_index": 2, "canonical_form": "subtype(zqb, c)", "canonical_type": "subtype", "parse_status": "OK"},
        {"step_index": 3, "canonical_form": "is(x, zqb)", "canonical_type": "is", "parse_status": "OK"},
        {"step_index": 4, "canonical_form": "is(x, c)", "canonical_type": "is", "parse_status": "OK"},
    ]


def _run_alignment():
    gold_step_ids = [1, 2]
    rank_by_gold_step = {1: 0, 2: 1}
    expected_type_by_gold_step = {1: "is", 2: "is"}
    # Simulate hallucinated rule insertion: subtype lines are NOT probe premises.
    premise_forms = {"is(x, a)"}
    return align_steps_by_reference_ranks(
        _revisionture_probe_steps_with_inserted_rule(),
        premise_forms,
        rank_by_gold_step,
        gold_step_ids,
        expected_type_by_gold_step,
    )


def test_regression_p017_p2_rule_insertion_not_aligned():
    lookup, rows = _run_alignment()
    assert lookup[1]["canonical_type"] == "is"
    assert lookup[2]["canonical_type"] == "is"
    assert any(r["alignment_status"] == "SKIPPED_PROBE_RULE_INSERTION" for r in rows)


def test_regression_p024_p3_rule_insertion_not_aligned():
    lookup, rows = _run_alignment()
    assert lookup[1]["canonical_type"] == "is"
    assert lookup[2]["canonical_type"] == "is"
    assert any(r["alignment_status"] == "SKIPPED_PROBE_RULE_INSERTION" for r in rows)


def test_regression_p026_p2_rule_insertion_not_aligned():
    lookup, rows = _run_alignment()
    assert lookup[1]["canonical_type"] == "is"
    assert lookup[2]["canonical_type"] == "is"
    assert any(r["alignment_status"] == "SKIPPED_PROBE_RULE_INSERTION" for r in rows)

