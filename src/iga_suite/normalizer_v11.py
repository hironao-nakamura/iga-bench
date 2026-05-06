"""Parser v1.1 — surface-relaxed canonicalizer (with compound provenance patch).

Drop-in replacement for normalizer.py (v1.0) that recovers surface-level
false negatives identified in the human audit.  All new rules are:

  - label-blind  (no gold label / proof outcome / F1 used)
  - deterministic (no LLM-as-parser, no learned parser)
  - benchmark-agnostic (English surface syntax only)
  - explainable  (each recovered step carries a canonicalizer_rule_id)

v1.1 additions over v1.0
────────────────────────
1. article-optional copular:  "X is Y" without a/an
2. no-article negative subtype:  "No X is Y" / "No X are Y"
3. broader trailing provenance stripping (incl. compound provenance patch)
4. terminal conclusion extraction (so/therefore/hence/thus)
5. expanded break-phrase vocabulary
"""
from __future__ import annotations

import re
from typing import Optional

PARSER_VERSION = "1.1"

# ── word-level helpers (unchanged from v1.0) ──────────────────────────

def _depluralize(word: str) -> str:
    w = word.lower()
    if w.endswith('uses') and len(w) > 4:
        return w[:-2]
    if w.endswith('es') and not w.endswith('us') and len(w) > 2:
        return w[:-2]
    if w.endswith('s') and not w.endswith('us') and len(w) > 1:
        return w[:-1]
    return w


def _normalize_word(word: str) -> str:
    w = word.lower()
    if w.endswith('use') and len(w) > 3:
        return w[:-1]
    if w.endswith('e') and len(w) > 3 and w[:-1].endswith('us'):
        return w[:-1]
    return w


# ── break tokenizer (v1.1: expanded vocabulary) ──────────────────────

def _tokenize_break(text: str) -> tuple[str, str, str, str] | None:
    """Return (canonical_form, canonical_type, parse_status, rule_id) or None."""
    lower = text.lower()

    # v1.0 patterns
    if (re.search(r'there is no (?:premise|rule) (?:linking|connecting)', lower) or
            re.search(r'(?:no premise connects|nothing links)\b', lower)):
        m = re.search(
            r'(?:linking|connecting|connects|links)\s+([a-z][a-z0-9_-]*)\s+(?:to|and)\s+([a-z][a-z0-9_-]*)',
            lower,
        )
        if m:
            left = _normalize_word(_depluralize(m.group(1)))
            right = _normalize_word(_depluralize(m.group(2)))
            return f'break(missing_link:{left}->{right})', 'break', 'BREAK_TOKENIZED', 'BRK_LINK_V10'
        return 'break(missing_link)', 'break', 'BREAK_TOKENIZED', 'BRK_LINK_V10'

    if re.search(r'\bwe cannot conclude\b', lower) or re.search(r'\bcannot infer\b', lower):
        return 'break(cannot_conclude)', 'break', 'BREAK_TOKENIZED', 'BRK_CANNOT_V10'

    if re.search(r'\bthe chain breaks\b', lower) or re.search(r'\bchain (?:breaks|fails)\b', lower):
        return 'break(chain_break)', 'break', 'BREAK_TOKENIZED', 'BRK_CHAIN_V10'

    # ── v1.1 additions ────────────────────────────────────────────────

    if re.search(r'\bno premise (?:states?|implies?|mentions?|establishes?)\b', lower):
        return 'break(no_premise_states)', 'break', 'BREAK_TOKENIZED', 'BRK_NO_PREMISE_STATES_V11'

    if re.search(r'there is no (?:premise|rule) (?:stating|that)\b', lower):
        return 'break(no_premise_states)', 'break', 'BREAK_TOKENIZED', 'BRK_NO_PREMISE_THAT_V11'

    if re.search(r'\bcannot determine\b', lower):
        return 'break(cannot_determine)', 'break', 'BREAK_TOKENIZED', 'BRK_CANNOT_DET_V11'

    if re.search(r'\bnot enough information\b', lower):
        return 'break(insufficient_info)', 'break', 'BREAK_TOKENIZED', 'BRK_INSUFF_V11'

    if re.search(r'\bthe proof (?:chain\s+)?(?:stops|breaks|fails)\b', lower):
        return 'break(chain_break)', 'break', 'BREAK_TOKENIZED', 'BRK_PROOF_CHAIN_V11'

    if re.search(r'\bcannot (?:be )?(?:concluded|derived|determined|established)\b', lower):
        return 'break(cannot_conclude)', 'break', 'BREAK_TOKENIZED', 'BRK_CANNOT_PASSIVE_V11'

    if re.search(r'\bno (?:direct|valid|known) (?:path|connection|link)\b', lower):
        return 'break(missing_link)', 'break', 'BREAK_TOKENIZED', 'BRK_NO_PATH_V11'

    return None


# ── provenance stripping (v1.1: broader patterns) ────────────────────

_PROVENANCE_PATTERNS = [
    # v1.1+patch: compound provenance — must come FIRST (greedy match)
    # "(from step N and premise N: bempuses are bimpuses)"
    # "(from Step 1 and premise 1)"
    # "(using step 2 and premise 3: ...)"
    re.compile(r"\s*\((?:from|by|using|via)(?:\s+transitivity\s+from)?\s+[Ss]tep\s*\d+\s+and\s+premise\s*\d*(?:\s*:\s*[^)]+)?\)", re.I),
    # "(from premise N, since bempuses are bimpuses)"
    re.compile(r"\s*\((?:from|by|using|via)\s+premise\s*\d+\s*,\s*since\s+[^)]+\)", re.I),
    # "(since bempuses are bimpuses, by premise N)"
    # "(because burpuses are banpuses, by premise N)"
    re.compile(r"\s*\((?:since|because)\s+[^,]+,\s*(?:by|from|per|via)\s+premise\s*\d+\)", re.I),
    # v1.0 patterns
    re.compile(r"\s*\((?:given\s+in|from|by|via|per|using|see)\s+premise\s*\d*\)", re.I),
    re.compile(r"\s*\(premise\s+\d+\)", re.I),
    re.compile(r"\s*\(given\)", re.I),
    # v1.1: step-based provenance
    re.compile(r"\s*\((?:from|by|using|via|see)\s+step\s*\d+\)", re.I),
    # v1.1: broader premise references
    re.compile(r"\s*\((?:given\s+by|because\s+of|based\s+on|according\s+to)\s+premise\s*\d*\)", re.I),
    # v1.1: trailing ", by premise N" / ", from step N" without parens
    re.compile(r",?\s+(?:by|from|using|via)\s+(?:premise|step)\s+\d+\s*$", re.I),
]


def _strip_provenance(text: str) -> str:
    for pat in _PROVENANCE_PATTERNS:
        text = pat.sub("", text)
    return text.strip()


# ── terminal conclusion extraction (v1.1: richer) ───────────────────

def _extract_conclusion(text: str) -> str:
    """Extract the core proposition from compound sentences."""
    # ", so X is Y" / ", and so X is Y"
    m = re.search(r",\s*(?:and\s+)?so\s+(.+)$", text, re.I)
    if m:
        return m.group(1).strip()

    # ", therefore X is Y" / ", hence X is Y"
    m = re.search(r",\s*(?:therefore|hence|thus|consequently|accordingly)\s+(.+)$", text, re.I)
    if m:
        return m.group(1).strip()

    # "Since/Because ..., X is Y"
    m = re.match(r"^(?:Since|Because|As|Given that)\s+.+,\s*(.+)$", text, re.I)
    if m:
        return m.group(1).strip()

    return text


def _strip_leading_connectives(text: str) -> str:
    """Remove leading discourse markers and conclusion phrases."""
    text = re.sub(
        r"^(?:Therefore|Thus|So|Hence|Consequently|Accordingly|As a result),?\s*",
        "", text, flags=re.I,
    )
    text = re.sub(
        r"^(?:We (?:can |know we can )?conclude(?: that)?|"
        r"It follows that|"
        r"This means(?: that)?|"
        r"This tells us(?: that)?|"
        r"We (?:can )?(?:determine|infer|deduce)(?: that)?)\s+",
        "", text, flags=re.I,
    )
    text = re.sub(
        r"^(?:From|According to)\s+(?:the )?(?:premises?|the facts?|premise \d+),?\s*",
        "", text, flags=re.I,
    )
    return text.strip()


# ── copular proposition matching (v1.1: article-optional) ────────────

def _match_proposition(text: str) -> tuple[str | None, str, str, str] | None:
    """Try to match a canonical proposition pattern.

    Returns (canonical_form, canonical_type, parse_status, rule_id) or None.
    """
    # ── not_is ────────────────────────────────────────────────────────
    # "X is not a/an Y"  (v1.0)
    m = re.match(r"(\w+)\s+is\s+not\s+(?:a|an)\s+(\w+)$", text, re.I)
    if m:
        return (f"not_is({_normalize_word(m.group(1))}, {_normalize_word(m.group(2))})",
                'not_is', 'OK', 'NOT_IS_ART_V10')

    # v1.1: "X is not Y" (no article)
    m = re.match(r"(\w+)\s+is\s+not\s+(\w+)$", text, re.I)
    if m:
        return (f"not_is({_normalize_word(m.group(1))}, {_normalize_word(m.group(2))})",
                'not_is', 'OK', 'NOT_IS_BARE_V11')

    # ── is ────────────────────────────────────────────────────────────
    # "X is a/an Y"  (v1.0)
    m = re.match(r"(\w+)\s+is\s+(?:a|an)\s+(\w+)$", text, re.I)
    if m:
        return (f"is({_normalize_word(m.group(1))}, {_normalize_word(m.group(2))})",
                'is', 'OK', 'IS_ART_V10')

    # v1.1: "X is indeed/definitely/certainly/also/actually Y"
    m = re.match(r"(\w+)\s+is\s+(?:indeed|definitely|certainly|also|actually)\s+(?:a\s+|an\s+)?(\w+)$", text, re.I)
    if m:
        return (f"is({_normalize_word(m.group(1))}, {_normalize_word(m.group(2))})",
                'is', 'OK', 'IS_ADVERB_V11')

    # v1.1: "X is Y" (no article) — must come AFTER adverb variant
    m = re.match(r"(\w+)\s+is\s+(\w+)$", text, re.I)
    if m:
        subj, obj = m.group(1), m.group(2)
        obj_lower = obj.lower()
        if obj_lower not in ('not', 'true', 'false', 'that', 'the', 'this', 'it', 'what', 'which'):
            return (f"is({_normalize_word(subj)}, {_normalize_word(obj)})",
                    'is', 'OK', 'IS_BARE_V11')

    # "X is one of the Y"  (v1.0)
    m = re.match(r"(\w+)\s+is\s+one\s+of\s+the\s+(\w+)$", text, re.I)
    if m:
        return (f"is({_normalize_word(m.group(1))}, {_normalize_word(_depluralize(m.group(2)))})",
                'is', 'OK', 'IS_ONE_OF_V10')

    # ── not_subtype ───────────────────────────────────────────────────
    # "No X is a/an Y"  (v1.0)
    m = re.match(r"No\s+(\w+)\s+is\s+(?:a|an)\s+(\w+)$", text, re.I)
    if m:
        return (f"not_subtype({_normalize_word(m.group(1))}, {_normalize_word(m.group(2))})",
                'not_subtype', 'OK', 'NOT_SUB_ART_V10')

    # v1.1: "No X is Y" (no article)
    m = re.match(r"No\s+(\w+)\s+is\s+(\w+)$", text, re.I)
    if m:
        return (f"not_subtype({_normalize_word(m.group(1))}, {_normalize_word(m.group(2))})",
                'not_subtype', 'OK', 'NOT_SUB_BARE_V11')

    # v1.1: "No X are Y" (no article)
    m = re.match(r"No\s+(\w+)\s+are\s+(\w+)$", text, re.I)
    if m:
        return (f"not_subtype({_normalize_word(_depluralize(m.group(1)))}, {_normalize_word(_depluralize(m.group(2)))})",
                'not_subtype', 'OK', 'NOT_SUB_ARE_V11')

    # ── subtype ───────────────────────────────────────────────────────
    # "All/Every/Each X is a/an Y"  (v1.0)
    m = re.match(r"(?:All|Every|Each)\s+(\w+)\s+is\s+(?:a|an)\s+(\w+)$", text, re.I)
    if m:
        return (f"subtype({_normalize_word(_depluralize(m.group(1)))}, {_normalize_word(m.group(2))})",
                'subtype', 'OK', 'SUB_ART_V10')

    # v1.1: "All/Every/Each X is Y" (no article)
    m = re.match(r"(?:All|Every|Each)\s+(\w+)\s+is\s+(\w+)$", text, re.I)
    if m:
        return (f"subtype({_normalize_word(_depluralize(m.group(1)))}, {_normalize_word(m.group(2))})",
                'subtype', 'OK', 'SUB_BARE_V11')

    # "All/Every/Each X are Y"  (v1.0)
    m = re.match(r"(?:All|Every|Each)\s+(\w+)\s+are\s+(\w+)$", text, re.I)
    if m:
        return (f"subtype({_normalize_word(_depluralize(m.group(1)))}, {_normalize_word(_depluralize(m.group(2)))})",
                'subtype', 'OK', 'SUB_ARE_V10')

    # "X are not Y"  (v1.0)
    m = re.match(r"(\w+)\s+are\s+not\s+(\w+)$", text, re.I)
    if m:
        return (f"not_subtype({_normalize_word(_depluralize(m.group(1)))}, {_normalize_word(_depluralize(m.group(2)))})",
                'not_subtype', 'OK', 'NOT_SUB_ARE_NOT_V10')

    # "X are Y"  (v1.0)
    m = re.match(r"(\w+)\s+are\s+(\w+)$", text, re.I)
    if m:
        return (f"subtype({_normalize_word(_depluralize(m.group(1)))}, {_normalize_word(_depluralize(m.group(2)))})",
                'subtype', 'OK', 'SUB_ARE_BARE_V10')

    return None


# ── main entry point ─────────────────────────────────────────────────

def normalize_step_text(step_text: str) -> tuple[str | None, str, str, str]:
    """Normalize a single step's text into canonical form.

    Returns (canonical_form, canonical_type, parse_status, rule_id).
    rule_id tracks which canonicalization rule fired.
    """
    text = step_text.strip().rstrip('.')
    text = _strip_provenance(text)
    text = text.strip().rstrip('.')

    brk = _tokenize_break(text)
    if brk is not None:
        return brk

    text = _extract_conclusion(text)
    text = _strip_leading_connectives(text)
    text = text.strip().rstrip('.')

    result = _match_proposition(text)
    if result is not None:
        return result

    return None, 'free_form', 'UNPARSEABLE', 'NO_MATCH'


def normalize_step_text_compat(step_text: str) -> tuple[str | None, str, str]:
    """v1.0-compatible signature (without rule_id)."""
    canonical, ctype, status, _rule = normalize_step_text(step_text)
    return canonical, ctype, status


def parse_trace_steps(raw_response: str) -> list[dict]:
    steps = []
    for line in raw_response.splitlines():
        m = re.match(r"Step\s+(\d+)\s*:\s*(.+)$", line.strip())
        if not m:
            continue
        step_id = int(m.group(1))
        raw_text = m.group(2).strip().rstrip('.')
        canonical, canonical_type, parse_status, rule_id = normalize_step_text(raw_text)
        steps.append({
            'step_index': step_id,
            'raw_text': raw_text,
            'canonical_form': canonical,
            'canonical_type': canonical_type,
            'parse_status': parse_status,
            'emits_break_token': parse_status == 'BREAK_TOKENIZED',
            'break_token_type': canonical_type if parse_status == 'BREAK_TOKENIZED' else None,
            'canonicalizer_rule_id': rule_id,
            'parser_version': PARSER_VERSION,
        })
    return steps


def parse_final_answer(raw_response: str) -> str | None:
    m = re.search(r'Answer\s*:\s*(True|False)', raw_response, re.I)
    if m:
        return m.group(1).capitalize()
    return None
