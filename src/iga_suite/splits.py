from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path


def _read_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            rows.append(json.loads(s))
    return rows


def _write_jsonl(path: str | Path, rows: list[dict]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _hop_bin(row: dict) -> str:
    proof = row.get("proof_tree") or []
    return f"{len(proof)}hop"


def _label_bin(row: dict) -> str:
    ans = row.get("answer")
    return "true" if bool(ans) else "false"


def _clone_with_new_id(row: dict, *, new_id: str, split: str) -> dict:
    out = json.loads(json.dumps(row))
    out["problem_id"] = new_id
    out["split"] = split
    meta = dict(out.get("metadata") or {})
    meta["source_record_ref"] = meta.get("source_record_ref") or row.get("problem_id")
    meta["synthetic_augmented"] = True
    out["metadata"] = meta
    return out


def build_prontoqa_splits(
    input_jsonl: str | Path,
    out_dir: str | Path,
    *,
    target_full_eval_n: int = 500,
    dev_n: int = 50,
    holdout_n: int = 50,
    seed: int = 42,
    allow_upsample: bool = True,
) -> dict:
    rows = _read_jsonl(input_jsonl)
    if not rows:
        raise ValueError("input_jsonl contains no rows")

    # Stratify by (hop, label).
    buckets: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        buckets[(_hop_bin(r), _label_bin(r))].append(r)

    rng = random.Random(seed)
    for arr in buckets.values():
        rng.shuffle(arr)

    full_eval = list(rows)
    if len(full_eval) < target_full_eval_n:
        if not allow_upsample:
            raise ValueError(f"Need {target_full_eval_n} rows but only {len(full_eval)} available")
        # Balanced upsample across strata to reach requested size.
        strata = list(buckets.keys())
        i = 0
        while len(full_eval) < target_full_eval_n:
            key = strata[i % len(strata)]
            src = buckets[key][i % len(buckets[key])]
            new_id = f"{src['problem_id']}__aug{len(full_eval)+1}"
            full_eval.append(_clone_with_new_id(src, new_id=new_id, split="full_eval"))
            i += 1

    # Build dev/holdout from non-augmented source rows when possible.
    source_rows = [r for r in rows]
    rng.shuffle(source_rows)

    # Build dev/holdout from source; if source is small, upsample to requested sizes.
    dev = source_rows[: min(dev_n, len(source_rows))]
    if len(dev) < dev_n and allow_upsample:
        i = 0
        while len(dev) < dev_n:
            src = source_rows[i % len(source_rows)]
            dev.append(_clone_with_new_id(src, new_id=f"{src['problem_id']}__devaug{len(dev)+1}", split="dev"))
            i += 1

    remain = source_rows[len(dev):] if len(source_rows) > len(dev) else []
    holdout = remain[: min(holdout_n, len(remain))]
    if len(holdout) < holdout_n and allow_upsample:
        base = remain if remain else source_rows
        i = 0
        while len(holdout) < holdout_n:
            src = base[i % len(base)]
            holdout.append(_clone_with_new_id(src, new_id=f"{src['problem_id']}__holdaug{len(holdout)+1}", split="holdout"))
            i += 1

    for r in full_eval:
        r["split"] = "full_eval"
    for r in dev:
        r["split"] = "dev"
    for r in holdout:
        r["split"] = "holdout"

    out_root = Path(out_dir)
    _write_jsonl(out_root / "prontoqa_full_eval.jsonl", full_eval)
    _write_jsonl(out_root / "prontoqa_dev.jsonl", dev)
    _write_jsonl(out_root / "prontoqa_holdout.jsonl", holdout)

    report = {
        "input_n": len(rows),
        "full_eval_n": len(full_eval),
        "dev_n": len(dev),
        "holdout_n": len(holdout),
        "out_dir": str(out_root),
    }
    (out_root / "split_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report

