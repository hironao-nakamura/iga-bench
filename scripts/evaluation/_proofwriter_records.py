"""Internal helper for ``prepare_proofwriter_300.py``.

Reads upstream ProofWriter metadata files (``meta-*.jsonl``) and emits
the canonical raw-records JSONL used to drive the ProofWriter-300
analysis split.  This module is consumed by
``prepare_proofwriter_300.py`` and is not a CLI entrypoint of its own.
"""
from __future__ import annotations

import json
import re
from pathlib import Path


def _num_key(s: str) -> int:
    m = re.search(r"(\d+)$", s)
    return int(m.group(1)) if m else 0


def _clean_token(x: str) -> str:
    x = x.strip().lower()
    x = re.sub(r"[^a-z0-9_]+", "", x)
    return x


def _parse_atom_repr(rep: str) -> str | None:
    m = re.match(r'^\("([^"]+)"\s+"is"\s+"([^"]+)"\s+"([+-])"\)$', rep.strip())
    if not m:
        return None
    ent = _clean_token(m.group(1))
    pred = _clean_token(m.group(2))
    sign = m.group(3)
    if not ent or not pred:
        return None
    return f"is({ent}, {pred})" if sign == "+" else f"not_is({ent}, {pred})"


def _extract_dep_tokens(proof_repr: str) -> tuple[list[str], list[str], list[str]]:
    triples = sorted(set(re.findall(r"\btriple(\d+)\b", proof_repr)))
    rules = sorted(set(re.findall(r"\brule(\d+)\b", proof_repr)))
    ints = sorted(set(re.findall(r"\bint(\d+)\b", proof_repr)))
    return triples, rules, ints


def _is_unary_is_rep(rep: str) -> bool:
    return bool(re.match(r'^\("([^"]+)"\s+"is"\s+"([^"]+)"\s+"[+-]"\)$', rep.strip()))


def _parse_is_atom(rep: str) -> tuple[str, str, str] | None:
    m = re.match(r'^\("([^"]+)"\s+"is"\s+"([^"]+)"\s+"([+-])"\)$', rep.strip())
    if not m:
        return None
    return _clean_token(m.group(1)), _clean_token(m.group(2)), m.group(3)


def _render_fact_from_rep(rep: str) -> str | None:
    atom = _parse_is_atom(rep)
    if atom is None:
        return None
    subj, pred, sign = atom
    if sign != "+":
        return None
    return f"{subj} is a {pred}"


def _render_rule_from_rep(rep: str) -> str | None:
    tuples = re.findall(r'\("([^"]+)"\s+"is"\s+"([^"]+)"\s+"([+-])"\)', rep)
    if len(tuples) != 2:
        return None
    ant_subj, ant_pred, ant_sign = tuples[0]
    cons_subj, cons_pred, cons_sign = tuples[1]
    ant_subj = _clean_token(ant_subj)
    cons_subj = _clean_token(cons_subj)
    ant_pred = _clean_token(ant_pred)
    cons_pred = _clean_token(cons_pred)
    if ant_sign != "+" or cons_sign != "+":
        return None
    if ant_subj != cons_subj:
        return None
    if ant_subj not in {"something", "someone"}:
        return None
    return f"All {ant_pred} are {cons_pred}"


def build_records(
    input_jsonl: Path,
    output_raw_jsonl: Path,
    max_records: int = 1000,
) -> int:
    """Convert upstream ProofWriter ``meta-*.jsonl`` rows to the canonical
    raw-records JSONL format used by the IGA-Bench bootstrapper.

    Returns the number of records written.
    """
    rows = []
    with input_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                rows.append(json.loads(s))

    out = []
    for row in rows:
        triples = row.get("triples", {})
        rules = row.get("rules", {})
        questions = row.get("questions", {})
        if not triples or not rules or not questions:
            continue
        if any(not _is_unary_is_rep(str(v.get("representation", ""))) for v in triples.values()):
            continue
        rendered_rules = {}
        rules_ok = True
        for rk, rv in rules.items():
            rr = _render_rule_from_rep(str(rv.get("representation", "")))
            if rr is None:
                rules_ok = False
                break
            rendered_rules[rk] = rr
        if not rules_ok:
            continue

        triple_keys = sorted(triples.keys(), key=_num_key)
        rule_keys = sorted(rules.keys(), key=_num_key)

        premises = []
        premise_map = {}
        idx = 1
        for k in triple_keys:
            txt = _render_fact_from_rep(str(triples[k].get("representation", ""))) or str(triples[k].get("text", "")).strip().rstrip(".")
            if not txt:
                continue
            pid = f"P{idx}"
            premise_map[k] = pid
            premises.append({"id": pid, "text": txt})
            idx += 1
        for k in rule_keys:
            txt = rendered_rules.get(k) or str(rules[k].get("text", "")).strip().rstrip(".")
            if not txt:
                continue
            pid = f"P{idx}"
            premise_map[k] = pid
            premises.append({"id": pid, "text": txt})
            idx += 1

        q_items = sorted(questions.items(), key=lambda kv: _num_key(kv[0]))
        for qk, qv in q_items:
            if str(qv.get("strategy", "")).strip() not in {"proof", "inv-proof"}:
                continue
            if not _is_unary_is_rep(str(qv.get("representation", ""))):
                continue
            pwi = qv.get("proofsWithIntermediates") or []
            if not pwi:
                continue
            proof_obj = pwi[0]
            proof_repr = str(proof_obj.get("representation", ""))
            intermediates = proof_obj.get("intermediates") or {}
            if not proof_repr:
                continue

            proof_tree = []
            emitted_int_to_step: dict[int, int] = {}
            int_keys = sorted(intermediates.keys(), key=_num_key)
            for ik in int_keys:
                iobj = intermediates[ik]
                c = _parse_atom_repr(str(iobj.get("representation", "")))
                if c is None:
                    continue
                int_num = _num_key(ik)
                step_num = len(proof_tree) + 1
                triples_used, rules_used, ints_used = _extract_dep_tokens(proof_repr)
                deps = []
                for t in triples_used:
                    k = f"triple{t}"
                    if k in premise_map:
                        deps.append(premise_map[k])
                for r in rules_used:
                    k = f"rule{r}"
                    if k in premise_map:
                        deps.append(premise_map[k])
                for iv in ints_used:
                    iv_i = int(iv)
                    mapped = emitted_int_to_step.get(iv_i)
                    if mapped is not None and iv_i < int_num:
                        deps.append(f"S{mapped}")
                deps = sorted(set(deps), key=lambda x: (x[0], int(re.findall(r"\d+", x)[0])))
                proof_tree.append({"step": step_num, "conclusion": c, "depends_on": deps})
                emitted_int_to_step[int_num] = step_num

            q_repr = str(qv.get("representation", ""))
            q_canon = _parse_atom_repr(q_repr)
            if q_canon:
                step_n = len(proof_tree) + 1
                triples_used, rules_used, ints_used = _extract_dep_tokens(proof_repr)
                deps = []
                for t in triples_used:
                    k = f"triple{t}"
                    if k in premise_map:
                        deps.append(premise_map[k])
                for r in rules_used:
                    k = f"rule{r}"
                    if k in premise_map:
                        deps.append(premise_map[k])
                for iv in ints_used:
                    iv_i = int(iv)
                    mapped = emitted_int_to_step.get(iv_i)
                    if mapped is not None:
                        deps.append(f"S{mapped}")
                deps = sorted(set(deps), key=lambda x: (x[0], int(re.findall(r"\d+", x)[0])))
                if not proof_tree or proof_tree[-1]["conclusion"] != q_canon:
                    proof_tree.append({"step": step_n, "conclusion": q_canon, "depends_on": deps})

            rec = {
                "id": f"{row.get('id', 'pw')}__{qk}",
                "facts": [prem["text"] for prem in premises if int(prem["id"][1:]) <= len(triple_keys)],
                "rules": [prem["text"] for prem in premises if int(prem["id"][1:]) > len(triple_keys)],
                "question": str(qv.get("question", "")).strip(),
                "answer": bool(qv.get("answer")),
                "proof_tree": proof_tree,
                "source_record_ref": f"{row.get('id')}::{qk}",
            }
            if rec["question"] and rec["proof_tree"]:
                out.append(rec)
            if len(out) >= max_records:
                break
        if len(out) >= max_records:
            break

    output_raw_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_raw_jsonl.open("w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(out)
