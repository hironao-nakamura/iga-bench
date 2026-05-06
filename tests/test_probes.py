from iga_suite.probes import _render_replace, generate_probes


def test_render_replace_plural_case():
    assert _render_replace("Bimpuses are crimpuses", "bimpus", "zqbimpus") == "Zqbimpuses are crimpuses"


def test_render_replace_singular_case():
    assert _render_replace("Alex is a bimpus", "bimpus", "zqbimpus") == "Alex is a zqbimpus"


def test_render_replace_self_mapping():
    assert _render_replace("Bimpuses are bimpuses", "bimpus", "zqbimpus") == "Zqbimpuses are zqbimpuses"


def test_render_replace_idempotent_for_prerevisioned_tokens():
    text = "Zqbimpuses are crimpuses"
    once = _render_replace(text, "bimpus", "zqbimpus")
    twice = _render_replace(once, "bimpus", "zqbimpus")
    assert once == text
    assert twice == text


def test_render_replace_does_not_touch_unrelated_token():
    assert _render_replace("Crimpus is a crimpus", "bimpus", "zqbimpus") == "Crimpus is a crimpus"


def test_generate_probes_never_contains_duplicate_prerevision():
    problem = {
        "problem_id": "p_test",
        "premises": [
            {"id": "P1", "text": "Bimpuses are crimpuses"},
            {"id": "P2", "text": "Alex is a bimpus"},
        ],
    }
    probes = generate_probes(problem)
    for probe in probes:
        for premise in probe["modified_premises"]:
            assert "zqzq" not in premise["text"].lower()


def test_surface_control_uses_null_for_entity_premise():
    problem = {
        "problem_id": "p_entity",
        "premises": [
            {"id": "P1", "text": "Alex is a bimpus"},
        ],
    }
    probes = generate_probes(problem)
    surface = [p for p in probes if p["probe_type"] == "surface_control"][0]
    assert surface["render_rule_id"] == "surface_control_entity_null_v1"
    assert surface["modified_premises"][0]["text"] == "Alex is a bimpus."


def test_entity_substitution_probe_added_for_is_premise():
    problem = {
        "problem_id": "p_entity_sub",
        "premises": [
            {"id": "P1", "text": "Alex is a bimpus"},
            {"id": "P2", "text": "All bimpuses are cimpuses"},
        ],
    }
    probes = generate_probes(problem)
    entity_probes = [p for p in probes if p["probe_type"] == "entity_substitution" and p["target_premise"] == "P1"]
    assert len(entity_probes) == 1
    ep = entity_probes[0]
    assert ep["semantic_target_ref"] == "entity:Alex->zqAlex"
    changed = [x for x in ep["modified_premises"] if x["id"] == "P1"][0]["text"]
    assert changed == "zqAlex is a bimpus"


def test_entity_substitution_not_added_for_rule_premise():
    problem = {
        "problem_id": "p_rule_only",
        "premises": [
            {"id": "P1", "text": "All bimpuses are cimpuses"},
        ],
    }
    probes = generate_probes(problem)
    assert not any(p["probe_type"] == "entity_substitution" for p in probes)
