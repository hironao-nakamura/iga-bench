"""Default canonicalizer — re-exports the released v1.1 parser.

This is the module that the rest of the pipeline (`aligner`, `pipeline`,
`cli run-eval`, ...) and third-party reviewers hit when they run
`iga-suite run-eval` from the README. It must always be the surface-relaxed,
released canonicalizer that the primary release is built from.

    `normalize_step_text(step)` returns `(canonical, canonical_type, status)`
    (v1.0-compatible 3-tuple). The full `(canonical, canonical_type, status,
    rule_id)` 4-tuple is available as
    `iga_suite.normalizer_v11.normalize_step_text` when audit trails need
    the rule id.

For parser sensitivity analysis we keep the conservative v1.0 parser in
`iga_suite.normalizer_v10`. `scripts/analysis/reprocess_v10.py` monkey-patches this
module at import time to swap in v1.0 for that run only.
"""
from __future__ import annotations

from iga_suite.normalizer_v11 import (  # noqa: F401  (re-exports)
    PARSER_VERSION,
    normalize_step_text_compat as normalize_step_text,
    parse_trace_steps,
    parse_final_answer,
)

__all__ = [
    "PARSER_VERSION",
    "normalize_step_text",
    "parse_trace_steps",
    "parse_final_answer",
]
