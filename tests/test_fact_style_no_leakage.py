from iga_suite.pipeline import _fact_style_verdict, _infer_fact_child_sid


def test_inferred_fact_child_sid_is_trace_derived_and_stable():
    # Same original trace => same inferred child sid, independent of any gold map.
    original_step_lookup = {
        1: {"canonical_form": "is(alex, wumpus)", "canonical_type": "is", "parse_status": "OK"},
        2: {"canonical_form": "is(alex, tumpus)", "canonical_type": "is", "parse_status": "OK"},
        3: {"canonical_form": "is(alex, vumpus)", "canonical_type": "is", "parse_status": "OK"},
    }
    sid_a = _infer_fact_child_sid(
        original_step_lookup=original_step_lookup,
        premise_canonical="is(alex, wumpus)",
        premise_type="is",
    )
    sid_b = _infer_fact_child_sid(
        original_step_lookup=original_step_lookup,
        premise_canonical="is(alex, wumpus)",
        premise_type="is",
    )
    assert sid_a == 2
    assert sid_b == 2


def test_non_child_step_not_grounded_even_if_entity_changed():
    # child is sid=2, so sid=3 should not be direct positive.
    verdict, _, _ = _fact_style_verdict(
        mode="direct",
        sid=3,
        inferred_fact_child_sid=2,
        parse_ok_orig=True,
        parse_ok_entity=True,
        entity_step={"canonical_form": "is(alex, vumpus)", "canonical_type": "is", "parse_status": "OK"},
        entity_delta=True,
        surface_delta=False,
        null_delta=False,
    )
    assert verdict == "INSENSITIVE"


def test_direct_fact_style_uses_inferred_child_only():
    # With same probe effect, only inferred child sid is positive in direct mode.
    v_child, _, _ = _fact_style_verdict(
        mode="direct",
        sid=2,
        inferred_fact_child_sid=2,
        parse_ok_orig=True,
        parse_ok_entity=True,
        entity_step={"canonical_form": "break(cannot_conclude)", "canonical_type": "break", "parse_status": "BREAK_TOKENIZED"},
        entity_delta=True,
        surface_delta=False,
        null_delta=False,
    )
    v_other, _, _ = _fact_style_verdict(
        mode="direct",
        sid=4,
        inferred_fact_child_sid=2,
        parse_ok_orig=True,
        parse_ok_entity=True,
        entity_step={"canonical_form": "break(cannot_conclude)", "canonical_type": "break", "parse_status": "BREAK_TOKENIZED"},
        entity_delta=True,
        surface_delta=False,
        null_delta=False,
    )
    assert v_child == "GROUNDED"
    assert v_other == "INSENSITIVE"

