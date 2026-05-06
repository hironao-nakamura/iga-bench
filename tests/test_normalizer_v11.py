"""Comprehensive tests for parser v1.1 (normalizer_v11).

Covers all 5 functional requirements from plan.md:
  1. article-optional copular parsing
  2. no-article negative subtype
  3. trailing provenance stripping
  4. terminal conclusion extraction
  5. break phrase expansion

Also tests non-functional requirement D (rule_id traceability)
and backward compatibility with v1.0 patterns.
"""
import pytest
from iga_suite.normalizer_v11 import (
    normalize_step_text,
    normalize_step_text_compat,
    parse_trace_steps,
    parse_final_answer,
    PARSER_VERSION,
)

# Re-import v1.0 for comparison tests
from iga_suite.normalizer import normalize_step_text as normalize_v10


# ═══════════════════════════════════════════════════════════════════════
# 1. article-optional copular parsing
# ═══════════════════════════════════════════════════════════════════════

class TestArticleOptionalCopular:
    """Feature 1: X is Y / X is not Y without articles."""

    def test_bare_is(self):
        canon, ctype, status, rule = normalize_step_text("Rex is cold")
        assert ctype == "is"
        assert status == "OK"
        assert canon == "is(rex, cold)"
        assert "V11" in rule

    def test_bare_is_not(self):
        canon, ctype, status, rule = normalize_step_text("Rex is not cold")
        assert ctype == "not_is"
        assert status == "OK"
        assert canon == "not_is(rex, cold)"
        assert "V11" in rule

    def test_is_indeed(self):
        canon, ctype, status, rule = normalize_step_text("Rex is indeed cold")
        assert ctype == "is"
        assert canon == "is(rex, cold)"
        assert "ADVERB" in rule

    def test_is_definitely(self):
        canon, ctype, status, rule = normalize_step_text("Rex is definitely cold")
        assert ctype == "is"
        assert canon == "is(rex, cold)"

    def test_is_certainly(self):
        canon, ctype, status, rule = normalize_step_text("Rex is certainly cold")
        assert ctype == "is"
        assert canon == "is(rex, cold)"

    def test_is_also(self):
        canon, ctype, status, rule = normalize_step_text("Rex is also cold")
        assert ctype == "is"
        assert canon == "is(rex, cold)"

    def test_is_actually(self):
        canon, ctype, status, rule = normalize_step_text("Rex is actually cold")
        assert ctype == "is"
        assert canon == "is(rex, cold)"

    def test_is_indeed_with_article(self):
        """Adverb variant with article should still work."""
        canon, ctype, status, rule = normalize_step_text("Rex is indeed a bompus")
        assert ctype == "is"
        assert canon == "is(rex, bompus)"

    def test_bare_is_excludes_stopwords(self):
        """'X is not' / 'X is that' etc. should not match bare-is."""
        _, ctype, status, _ = normalize_step_text("Rex is that")
        assert ctype == "free_form"

    def test_v10_article_is_still_works(self):
        canon, ctype, status, rule = normalize_step_text("Rex is a bompus")
        assert ctype == "is"
        assert canon == "is(rex, bompus)"
        assert "V10" in rule

    def test_v10_article_not_is_still_works(self):
        canon, ctype, status, rule = normalize_step_text("Rex is not a bompus")
        assert ctype == "not_is"
        assert canon == "not_is(rex, bompus)"
        assert "V10" in rule

    def test_trailing_period_stripped(self):
        canon, ctype, status, _ = normalize_step_text("Rex is cold.")
        assert ctype == "is"
        assert canon == "is(rex, cold)"


# ═══════════════════════════════════════════════════════════════════════
# 2. no-article negative subtype
# ═══════════════════════════════════════════════════════════════════════

class TestNoArticleNegativeSubtype:
    """Feature 2: No X is Y / No X are Y without articles."""

    def test_no_x_is_y(self):
        canon, ctype, status, rule = normalize_step_text("No wumpus is cold")
        assert ctype == "not_subtype"
        assert canon == "not_subtype(wumpus, cold)"
        assert "BARE" in rule

    def test_no_x_are_y(self):
        canon, ctype, status, rule = normalize_step_text("No wumpuses are cold")
        assert ctype == "not_subtype"
        assert "not_subtype(" in canon
        assert "ARE" in rule

    def test_v10_no_x_is_a_y_still_works(self):
        canon, ctype, status, rule = normalize_step_text("No wumpus is a bompus")
        assert ctype == "not_subtype"
        assert canon == "not_subtype(wumpus, bompus)"
        assert "V10" in rule


# ═══════════════════════════════════════════════════════════════════════
# 3. trailing provenance stripping
# ═══════════════════════════════════════════════════════════════════════

class TestTrailingProvenanceStripping:
    """Feature 3: Broader provenance stripping patterns."""

    def test_from_step(self):
        canon, ctype, _, _ = normalize_step_text("Rex is a bompus (from step 2)")
        assert ctype == "is"
        assert canon == "is(rex, bompus)"

    def test_using_premise(self):
        canon, ctype, _, _ = normalize_step_text("Rex is a bompus (using premise 3)")
        assert ctype == "is"
        assert canon == "is(rex, bompus)"

    def test_by_premise(self):
        canon, ctype, _, _ = normalize_step_text("Rex is a bompus (by premise 1)")
        assert ctype == "is"
        assert canon == "is(rex, bompus)"

    def test_given_by_premise(self):
        canon, ctype, _, _ = normalize_step_text("Rex is a bompus (given by premise 4)")
        assert ctype == "is"
        assert canon == "is(rex, bompus)"

    def test_because_of_premise(self):
        canon, ctype, _, _ = normalize_step_text("Rex is a bompus (because of premise 2)")
        assert ctype == "is"
        assert canon == "is(rex, bompus)"

    def test_by_step(self):
        canon, ctype, _, _ = normalize_step_text("Rex is a bompus (by step 5)")
        assert ctype == "is"
        assert canon == "is(rex, bompus)"

    def test_trailing_comma_by_premise(self):
        """Non-parenthesized trailing provenance."""
        canon, ctype, _, _ = normalize_step_text("Rex is a bompus, by premise 3")
        assert ctype == "is"
        assert canon == "is(rex, bompus)"

    def test_trailing_from_step(self):
        """Non-parenthesized trailing provenance."""
        canon, ctype, _, _ = normalize_step_text("Rex is a bompus from step 1")
        assert ctype == "is"
        assert canon == "is(rex, bompus)"

    def test_v10_given_in_premise(self):
        canon, ctype, _, _ = normalize_step_text("Rex is a bompus (given in premise 1)")
        assert ctype == "is"
        assert canon == "is(rex, bompus)"

    def test_v10_premise_only(self):
        canon, ctype, _, _ = normalize_step_text("Rex is a bompus (premise 1)")
        assert ctype == "is"
        assert canon == "is(rex, bompus)"

    # ── compound provenance patch ──────────────────────────────────────
    def test_compound_from_step_and_premise(self):
        canon, ctype, status, _ = normalize_step_text(
            "Alex is a bimpus (from step 1 and premise 1)."
        )
        assert status == "OK"
        assert canon == "is(alex, bimpus)"

    def test_compound_from_step_and_premise_with_explanation(self):
        canon, ctype, status, _ = normalize_step_text(
            "Alex is a bimpus (from step 1 and premise 1: bempuses are bimpuses)."
        )
        assert status == "OK"
        assert canon == "is(alex, bimpus)"

    def test_compound_from_Step_capital(self):
        canon, ctype, status, _ = normalize_step_text(
            "Alex is a bimpus (from Step 1 and premise 1)."
        )
        assert status == "OK"
        assert canon == "is(alex, bimpus)"

    def test_compound_from_Step_and_Premise_both_capital(self):
        canon, ctype, status, _ = normalize_step_text(
            "Alex is a bimpus (from Step 1 and Premise 1)."
        )
        assert status == "OK"
        assert canon == "is(alex, bimpus)"

    def test_compound_using_step_and_premise(self):
        canon, ctype, status, _ = normalize_step_text(
            "Eve is a bloltus (using step 2 and premise 3: bleltuses are bloltuses)."
        )
        assert status == "OK"
        assert canon == "is(eve, bloltus)"

    def test_compound_not_is_with_explanation(self):
        canon, ctype, status, _ = normalize_step_text(
            "Alex is not a belpus (from step 5 and premise 5: balpuses are not belpuses)."
        )
        assert status == "OK"
        assert canon == "not_is(alex, belpus)"

    def test_compound_from_premise_since(self):
        canon, ctype, status, _ = normalize_step_text(
            "Nova is a branpus (from premise 1, since brurpuses are branpuses)."
        )
        assert status == "OK"
        assert canon == "is(nova, branpus)"

    def test_compound_because_by_premise(self):
        canon, ctype, status, _ = normalize_step_text(
            "Fae is a banpus (because burpuses are banpuses, by premise 4)."
        )
        assert status == "OK"
        assert canon == "is(fae, banpus)"

    def test_compound_since_by_premise(self):
        canon, ctype, status, _ = normalize_step_text(
            "Eve is a bleltus (since blenduses are bleltuses, by premise 1)."
        )
        assert status == "OK"
        assert canon == "is(eve, bleltus)"

    def test_compound_by_transitivity_from_step_and_premise(self):
        canon, ctype, status, _ = normalize_step_text(
            "Sam is a blunkus (by transitivity from Step 1 and premise 1)."
        )
        assert status == "OK"
        assert canon == "is(sam, blunkus)"

    def test_compound_premise_so_conclusion(self):
        """'(premise N), so conclusion' — conclusion extraction via terminal clause."""
        canon, ctype, status, _ = normalize_step_text(
            "Balpuses are not belpuses (premise 5), so Alex is not a belpus."
        )
        assert status == "OK"
        assert canon == "not_is(alex, belpus)"

    def test_compound_no_article_negative_with_provenance(self):
        canon, ctype, status, _ = normalize_step_text(
            "Alex is not a belpus (from step 5 and premise 5)."
        )
        assert status == "OK"
        assert canon == "not_is(alex, belpus)"

    def test_compound_subtype_with_provenance(self):
        canon, ctype, status, _ = normalize_step_text(
            "Every bempus is a bimpus (from premise 1)."
        )
        assert status == "OK"
        assert canon == "subtype(bempus, bimpus)"


# ═══════════════════════════════════════════════════════════════════════
# 4. terminal conclusion extraction
# ═══════════════════════════════════════════════════════════════════════

class TestTerminalConclusionExtraction:
    """Feature 4: Extract proposition from compound sentences."""

    def test_comma_so(self):
        canon, ctype, _, _ = normalize_step_text(
            "All bompuses are cold, so Rex is a bompus"
        )
        assert ctype == "is"
        assert canon == "is(rex, bompus)"

    def test_comma_therefore(self):
        canon, ctype, _, _ = normalize_step_text(
            "All bompuses are cold, therefore Rex is cold"
        )
        assert ctype == "is"
        assert canon == "is(rex, cold)"

    def test_comma_hence(self):
        canon, ctype, _, _ = normalize_step_text(
            "All bompuses are cold, hence Rex is cold"
        )
        assert ctype == "is"
        assert canon == "is(rex, cold)"

    def test_comma_thus(self):
        canon, ctype, _, _ = normalize_step_text(
            "All bompuses are cold, thus Rex is cold"
        )
        assert ctype == "is"
        assert canon == "is(rex, cold)"

    def test_we_can_conclude_that(self):
        canon, ctype, _, _ = normalize_step_text(
            "We can conclude that Rex is a bompus"
        )
        assert ctype == "is"
        assert canon == "is(rex, bompus)"

    def test_it_follows_that(self):
        canon, ctype, _, _ = normalize_step_text(
            "It follows that Rex is a bompus"
        )
        assert ctype == "is"
        assert canon == "is(rex, bompus)"

    def test_thus_leading(self):
        canon, ctype, _, _ = normalize_step_text("Thus, Rex is a bompus")
        assert ctype == "is"
        assert canon == "is(rex, bompus)"

    def test_therefore_leading(self):
        canon, ctype, _, _ = normalize_step_text("Therefore Rex is a bompus")
        assert ctype == "is"
        assert canon == "is(rex, bompus)"

    def test_since_prefix(self):
        canon, ctype, _, _ = normalize_step_text(
            "Since all bompuses are cold, Rex is a bompus"
        )
        assert ctype == "is"
        assert canon == "is(rex, bompus)"

    def test_because_prefix(self):
        canon, ctype, _, _ = normalize_step_text(
            "Because Rex is a bompus, Rex is cold"
        )
        assert ctype == "is"
        assert canon == "is(rex, cold)"

    def test_we_determine_that(self):
        canon, ctype, _, _ = normalize_step_text(
            "We can determine that Rex is a bompus"
        )
        assert ctype == "is"
        assert canon == "is(rex, bompus)"

    def test_we_infer_that(self):
        canon, ctype, _, _ = normalize_step_text(
            "We can infer that Rex is a bompus"
        )
        assert ctype == "is"
        assert canon == "is(rex, bompus)"

    def test_we_deduce_that(self):
        canon, ctype, _, _ = normalize_step_text(
            "We can deduce that Rex is a bompus"
        )
        assert ctype == "is"
        assert canon == "is(rex, bompus)"

    def test_and_so(self):
        canon, ctype, _, _ = normalize_step_text(
            "Rex is a bompus, and so Rex is cold"
        )
        assert ctype == "is"
        assert canon == "is(rex, cold)"


# ═══════════════════════════════════════════════════════════════════════
# 5. break phrase expansion
# ═══════════════════════════════════════════════════════════════════════

class TestBreakPhraseExpansion:
    """Feature 5: Expanded break-phrase vocabulary."""

    def test_no_premise_states(self):
        canon, ctype, status, rule = normalize_step_text(
            "No premise states that bompuses are cold"
        )
        assert ctype == "break"
        assert status == "BREAK_TOKENIZED"
        assert "V11" in rule

    def test_no_premise_implies(self):
        _, ctype, status, _ = normalize_step_text(
            "No premise implies that Rex is cold"
        )
        assert ctype == "break"
        assert status == "BREAK_TOKENIZED"

    def test_no_premise_states_or_implies(self):
        _, ctype, status, _ = normalize_step_text(
            "No premise states or implies that Rex is cold"
        )
        assert ctype == "break"
        assert status == "BREAK_TOKENIZED"

    def test_there_is_no_premise_stating(self):
        _, ctype, status, _ = normalize_step_text(
            "There is no premise stating that Rex is a bompus"
        )
        assert ctype == "break"
        assert status == "BREAK_TOKENIZED"

    def test_there_is_no_rule_that(self):
        _, ctype, status, _ = normalize_step_text(
            "There is no rule that connects bompuses to wumpuses"
        )
        assert ctype == "break"
        assert status == "BREAK_TOKENIZED"

    def test_cannot_determine(self):
        _, ctype, status, _ = normalize_step_text(
            "Cannot determine whether Rex is a bompus"
        )
        assert ctype == "break"
        assert status == "BREAK_TOKENIZED"

    def test_not_enough_information(self):
        _, ctype, status, _ = normalize_step_text(
            "Not enough information to conclude that Rex is cold"
        )
        assert ctype == "break"
        assert status == "BREAK_TOKENIZED"

    def test_proof_chain_stops(self):
        _, ctype, status, _ = normalize_step_text(
            "The proof chain stops here"
        )
        assert ctype == "break"
        assert status == "BREAK_TOKENIZED"

    def test_proof_chain_breaks(self):
        _, ctype, status, _ = normalize_step_text(
            "The proof chain breaks at this point"
        )
        assert ctype == "break"
        assert status == "BREAK_TOKENIZED"

    def test_proof_chain_fails(self):
        _, ctype, status, _ = normalize_step_text(
            "The proof chain fails here"
        )
        assert ctype == "break"
        assert status == "BREAK_TOKENIZED"

    def test_proof_stops(self):
        _, ctype, status, _ = normalize_step_text(
            "The proof stops at step 3"
        )
        assert ctype == "break"
        assert status == "BREAK_TOKENIZED"

    def test_cannot_be_concluded(self):
        _, ctype, status, _ = normalize_step_text(
            "It cannot be concluded from the premises"
        )
        assert ctype == "break"
        assert status == "BREAK_TOKENIZED"

    def test_cannot_be_derived(self):
        _, ctype, status, _ = normalize_step_text(
            "This cannot be derived from the given information"
        )
        assert ctype == "break"
        assert status == "BREAK_TOKENIZED"

    def test_no_direct_path(self):
        _, ctype, status, _ = normalize_step_text(
            "There is no direct path from bompuses to wumpuses"
        )
        assert ctype == "break"
        assert status == "BREAK_TOKENIZED"

    def test_no_valid_connection(self):
        _, ctype, status, _ = normalize_step_text(
            "No valid connection exists between Rex and bompus"
        )
        assert ctype == "break"
        assert status == "BREAK_TOKENIZED"

    # ── v1.0 break patterns still work ────────────────────────────────

    def test_v10_no_premise_linking(self):
        _, ctype, status, rule = normalize_step_text(
            "There is no premise linking bompuses to wumpuses"
        )
        assert ctype == "break"
        assert status == "BREAK_TOKENIZED"
        assert "V10" in rule

    def test_v10_we_cannot_conclude(self):
        _, ctype, status, _ = normalize_step_text(
            "We cannot conclude Rex is a bompus"
        )
        assert ctype == "break"
        assert status == "BREAK_TOKENIZED"

    def test_v10_chain_breaks(self):
        _, ctype, status, _ = normalize_step_text(
            "The chain breaks after this step"
        )
        assert ctype == "break"
        assert status == "BREAK_TOKENIZED"


# ═══════════════════════════════════════════════════════════════════════
# Non-functional: rule_id traceability (requirement D)
# ═══════════════════════════════════════════════════════════════════════

class TestRuleIdTraceability:
    """Every result carries a canonicalizer_rule_id."""

    def test_rule_id_on_is(self):
        _, _, _, rule = normalize_step_text("Rex is a bompus")
        assert rule is not None and len(rule) > 0

    def test_rule_id_on_break(self):
        _, _, _, rule = normalize_step_text("We cannot conclude Rex is a bompus")
        assert rule is not None and "BRK" in rule

    def test_rule_id_on_unparseable(self):
        _, _, _, rule = normalize_step_text("This is a very long rambling sentence about many things happening")
        assert rule == "NO_MATCH"


# ═══════════════════════════════════════════════════════════════════════
# Backward compatibility
# ═══════════════════════════════════════════════════════════════════════

class TestBackwardCompatibility:
    """v1.0 patterns produce the same canonical form in v1.1."""

    V10_FIXTURES = [
        ("Rex is a bompus", "is", "is(rex, bompus)"),
        ("Rex is not a bompus", "not_is", "not_is(rex, bompus)"),
        ("No wumpus is a bompus", "not_subtype", "not_subtype(wumpus, bompus)"),
        ("All wumpuses are bompuses", "subtype", "subtype(wumpus, bompus)"),
        ("Every wumpus is a bompus", "subtype", "subtype(wumpus, bompus)"),
        ("Wumpuses are bompuses", "subtype", "subtype(wumpus, bompus)"),
        ("Wumpuses are not bompuses", "not_subtype", "not_subtype(wumpus, bompus)"),
    ]

    @pytest.mark.parametrize("text,expected_type,expected_canon", V10_FIXTURES)
    def test_v10_pattern_produces_same_output(self, text, expected_type, expected_canon):
        canon_v11, ctype_v11, status_v11, _ = normalize_step_text(text)
        assert ctype_v11 == expected_type
        assert canon_v11 == expected_canon
        assert status_v11 == "OK"

    def test_compat_signature(self):
        """normalize_step_text_compat returns 3-tuple like v1.0."""
        result = normalize_step_text_compat("Rex is a bompus")
        assert len(result) == 3
        canon, ctype, status = result
        assert ctype == "is"
        assert canon == "is(rex, bompus)"
        assert status == "OK"


# ═══════════════════════════════════════════════════════════════════════
# parse_trace_steps and parse_final_answer
# ═══════════════════════════════════════════════════════════════════════

class TestParseTraceSteps:
    """Full trace parsing with v1.1 features."""

    TRACE = (
        "Step 1: Rex is a bompus (from step 2).\n"
        "Step 2: All bompuses are cold, therefore Rex is cold.\n"
        "Step 3: Rex is indeed cold.\n"
        "Step 4: No premise states that Rex is happy.\n"
        "Answer: True\n"
    )

    def test_step_count(self):
        steps = parse_trace_steps(self.TRACE)
        assert len(steps) == 4

    def test_step_has_rule_id(self):
        steps = parse_trace_steps(self.TRACE)
        for s in steps:
            assert "canonicalizer_rule_id" in s
            assert s["canonicalizer_rule_id"] is not None

    def test_step_has_parser_version(self):
        steps = parse_trace_steps(self.TRACE)
        for s in steps:
            assert s["parser_version"] == PARSER_VERSION

    def test_step1_provenance_stripped(self):
        steps = parse_trace_steps(self.TRACE)
        assert steps[0]["canonical_type"] == "is"
        assert steps[0]["canonical_form"] == "is(rex, bompus)"

    def test_step2_conclusion_extracted(self):
        steps = parse_trace_steps(self.TRACE)
        assert steps[1]["canonical_type"] == "is"
        assert steps[1]["canonical_form"] == "is(rex, cold)"

    def test_step3_adverb(self):
        steps = parse_trace_steps(self.TRACE)
        assert steps[2]["canonical_type"] == "is"

    def test_step4_break(self):
        steps = parse_trace_steps(self.TRACE)
        assert steps[3]["canonical_type"] == "break"
        assert steps[3]["parse_status"] == "BREAK_TOKENIZED"
        assert steps[3]["emits_break_token"] is True

    def test_final_answer(self):
        assert parse_final_answer(self.TRACE) == "True"


# ═══════════════════════════════════════════════════════════════════════
# Recovery: things that were UNPARSEABLE in v1.0 but OK in v1.1
# ═══════════════════════════════════════════════════════════════════════

class TestV10UnparseableRecoveredByV11:
    """Inputs that v1.0 returns UNPARSEABLE but v1.1 can parse."""

    RECOVERY_CASES = [
        ("Rex is cold", "is"),
        ("Rex is not cold", "not_is"),
        ("No wumpus is cold", "not_subtype"),
        ("No wumpuses are cold", "not_subtype"),
        ("Rex is indeed cold", "is"),
        ("Rex is definitely cold", "is"),
        ("Rex is a bompus (given by premise 4)", "is"),
        ("Rex is a bompus (because of premise 2)", "is"),
        ("Rex is a bompus (from step 2)", "is"),
        ("All bompuses are cold, therefore Rex is cold", "is"),
        ("All bompuses are cold, hence Rex is cold", "is"),
        ("No premise states that bompuses are cold", "break"),
        ("Not enough information to conclude that Rex is cold", "break"),
        ("Cannot determine whether Rex is a bompus", "break"),
        ("The proof chain stops here", "break"),
    ]

    @pytest.mark.parametrize("text,expected_v11_type", RECOVERY_CASES)
    def test_v10_unparseable_v11_recovered(self, text, expected_v11_type):
        v10_canon, v10_type, v10_status = normalize_v10(text)
        v11_canon, v11_type, v11_status, v11_rule = normalize_step_text(text)

        if v10_status == "UNPARSEABLE":
            assert v11_type == expected_v11_type, (
                f"v1.0 returned UNPARSEABLE, v1.1 should recover as {expected_v11_type} "
                f"but got {v11_type} (canon={v11_canon}, rule={v11_rule})"
            )
            assert v11_status in ("OK", "BREAK_TOKENIZED")
        else:
            assert v11_type == expected_v11_type
