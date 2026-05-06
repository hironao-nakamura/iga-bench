from __future__ import annotations

import json
from pathlib import Path

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


def main() -> None:
    total = len(CLAUDE_META_FIXTURES)
    hits = 0
    rows = []
    for text in CLAUDE_META_FIXTURES:
        canonical, ctype, status = normalize_step_text(text)
        ok = bool(status == "BREAK_TOKENIZED" and ctype == "break" and canonical and canonical.startswith("break("))
        hits += int(ok)
        rows.append(
            {
                "text": text,
                "canonical": canonical,
                "canonical_type": ctype,
                "parse_status": status,
                "is_break_tokenized": ok,
            }
        )
    out = {
        "dataset": "claude_meta_reasoning_fixture",
        "n": total,
        "break_tokenized_n": hits,
        "break_tokenized_rate": float(hits / total if total else 0.0),
        "details": rows,
    }
    out_dir = Path("reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "normalizer_eval.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"status": "ok", "out": str(out_dir / "normalizer_eval.json"), "break_tokenized_rate": out["break_tokenized_rate"]}, indent=2))


if __name__ == "__main__":
    main()

