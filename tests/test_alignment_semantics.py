from iga_suite.aligner import align_steps_to_gold, premise_canonical_forms
from iga_suite.aligner import align_steps_by_reference_ranks


def test_align_steps_to_gold_skips_premise_restatements():
    premises = [
        {"id": "P1", "text": "All boltuses are biltuses"},
        {"id": "P2", "text": "Max is a boltus"},
    ]
    proof_tree = [
        {"step": 1, "conclusion": "is(max, biltus)"},
    ]
    steps = [
        {"step_index": 1, "canonical_form": "subtype(boltus, biltus)", "parse_status": "OK"},
        {"step_index": 2, "canonical_form": "is(max, boltus)", "parse_status": "OK"},
        {"step_index": 3, "canonical_form": "is(max, biltus)", "parse_status": "OK"},
    ]
    lookup, rows = align_steps_to_gold(steps, proof_tree, premise_canonical_forms(premises))
    assert lookup[1]["step_index"] == 3
    skipped = [r for r in rows if r["alignment_status"] == "SKIPPED_RESTATEMENT"]
    assert len(skipped) == 2


def test_align_steps_to_gold_marks_missing_without_semantic_match():
    premises = [{"id": "P1", "text": "All a are b"}]
    proof_tree = [{"step": 1, "conclusion": "is(x, b)"}]
    steps = [{"step_index": 1, "canonical_form": "is(y, b)", "parse_status": "OK"}]
    lookup, rows = align_steps_to_gold(steps, proof_tree, premise_canonical_forms(premises))
    assert 1 not in lookup
    gold_row = [r for r in rows if r["step_index"] == 1][0]
    assert gold_row["alignment_status"] == "MISSING"


def test_probe_specific_premise_forms_filter_mutated_restatements():
    """
    Regression for p005/P1-style bug:
    mutated premise restatements in probe runs must be skipped using probe-specific premises.
    """
    # Gold derivation steps correspond to rank 0..4 from original run.
    rank_by_gold_step = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4}
    gold_steps = [1, 2, 3, 4, 5]

    # Probe run includes two mutated premise restatements before actual derivations.
    steps = [
        {"step_index": 1, "canonical_form": "subtype(boltus, zqbiltus)", "parse_status": "OK"},
        {"step_index": 2, "canonical_form": "subtype(zqbiltus, bultus)", "parse_status": "OK"},
        {"step_index": 3, "canonical_form": "is(max, zqbiltus)", "parse_status": "OK"},
        {"step_index": 4, "canonical_form": "is(max, bultus)", "parse_status": "OK"},
        {"step_index": 5, "canonical_form": "is(max, biltus)", "parse_status": "OK"},
        {"step_index": 6, "canonical_form": "is(max, boltus)", "parse_status": "OK"},
        {"step_index": 7, "canonical_form": "is(max, zqbultus)", "parse_status": "OK"},
    ]

    original_forms = {"subtype(boltus, biltus)", "subtype(biltus, bultus)", "is(max, boltus)"}
    probe_forms = {"subtype(boltus, zqbiltus)", "subtype(zqbiltus, bultus)", "is(max, boltus)"}

    # Buggy behavior (original forms): mutated restatements are not skipped.
    buggy_lookup, _ = align_steps_by_reference_ranks(steps, original_forms, rank_by_gold_step, gold_steps)
    assert buggy_lookup[1]["step_index"] == 1

    # Revisioned behavior (probe-specific forms): mutated restatements are skipped.
    revisioned_lookup, revisioned_rows = align_steps_by_reference_ranks(steps, probe_forms, rank_by_gold_step, gold_steps)
    assert revisioned_lookup[1]["step_index"] == 3
    assert revisioned_lookup[2]["step_index"] == 4
    skipped = [r for r in revisioned_rows if r["alignment_status"] == "SKIPPED_RESTATEMENT"]
    assert len(skipped) >= 2


def test_expected_canonical_exact_has_priority_over_rank():
    steps = [
        {"step_index": 1, "canonical_form": "is(max, boltus)", "canonical_type": "is", "parse_status": "OK"},
        {"step_index": 2, "canonical_form": "is(max, zqbiltus)", "canonical_type": "is", "parse_status": "OK"},
    ]
    lookup, rows = align_steps_by_reference_ranks(
        steps=steps,
        premise_forms=set(),
        rank_by_gold_step={1: 0},
        gold_step_ids=[1],
        expected_type_by_gold_step={1: "is"},
        expected_form_by_gold_step={1: "is(max, zqbiltus)"},
    )
    assert lookup[1]["step_index"] == 2
    matched = [r for r in rows if r["alignment_status"] == "MATCHED" and r["step_index"] == 1][0]
    assert matched["alignment_method"] == "expected_canonical_exact"

