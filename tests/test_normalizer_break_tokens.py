from iga_suite.normalizer import normalize_step_text


CLAUDE_META_FIXTURES = [
    "There is no premise linking zqzqbimpuses to zqzqzqbimpuses.",
    "There is no rule linking bimpuses to bompuses.",
    "There is no premise connecting bimpuses and bompuses.",
    "We cannot conclude Alex is a bompus.",
    "Cannot infer Alex is a bompus from these premises.",
    "The chain breaks after this step.",
    "No premise connects bimpuses to bompuses.",
    "Nothing links bimpuses to bompuses.",
    "There is no premise that connects bimpuses to bompuses.",
    "Since the chain breaks here because missing link, we cannot conclude Alex is a bompus.",
]


def test_break_tokenization_rate_for_claude_meta_reasoning():
    tokenized = 0
    for text in CLAUDE_META_FIXTURES:
        canonical, canonical_type, status = normalize_step_text(text)
        if status == "BREAK_TOKENIZED" and canonical_type == "break" and canonical and canonical.startswith("break("):
            tokenized += 1
    assert tokenized / len(CLAUDE_META_FIXTURES) >= 0.7
