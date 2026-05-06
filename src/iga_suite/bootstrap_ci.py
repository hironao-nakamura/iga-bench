from __future__ import annotations

import json
import random
from pathlib import Path

import pandas as pd


def _f1(tp: int, fp: int, fn: int) -> float:
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    return (2 * p * r / (p + r)) if (p + r) else 0.0


def _metrics_on_sample(df: pd.DataFrame) -> dict[str, float]:
    # Primary direct metric: GROUNDED-only positive
    pred_pos = df["verdict_type"].eq("GROUNDED")
    gold_pos = df["gold_dependency_label"].fillna(False).astype(bool)
    tp = int((pred_pos & gold_pos).sum())
    fp = int((pred_pos & ~gold_pos).sum())
    fn = int((~pred_pos & gold_pos).sum())
    f1 = _f1(tp, fp, fn)
    rawr = float(df.groupby("dataset_problem_id")["verdict_type"].apply(lambda s: (s == "MISREPRESENTATION").any()).mean())
    return {"direct_grounded_f1": f1, "rawr_rate": rawr}


def bootstrap_problem_level_ci(
    certs_parquet_root: str | Path,
    out_json: str | Path,
    *,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> dict:
    certs = pd.read_parquet(certs_parquet_root)
    certs = certs[certs["dependency_mode_scored"].eq("direct")]
    if certs.empty:
        report = {"n_problems": 0, "n_bootstrap": n_bootstrap}
        Path(out_json).write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report

    problem_ids = sorted(certs["dataset_problem_id"].dropna().unique().tolist())
    grouped = {pid: certs[certs["dataset_problem_id"].eq(pid)] for pid in problem_ids}
    rng = random.Random(seed)

    f1_vals = []
    rawr_vals = []
    for _ in range(n_bootstrap):
        sample = [grouped[rng.choice(problem_ids)] for _ in problem_ids]
        df = pd.concat(sample, ignore_index=True)
        m = _metrics_on_sample(df)
        f1_vals.append(m["direct_grounded_f1"])
        rawr_vals.append(m["rawr_rate"])

    f1_vals_sorted = sorted(f1_vals)
    rawr_vals_sorted = sorted(rawr_vals)
    lo = int(0.025 * len(f1_vals_sorted))
    hi = int(0.975 * len(f1_vals_sorted)) - 1

    report = {
        "n_problems": len(problem_ids),
        "n_bootstrap": n_bootstrap,
        "direct_grounded_f1_mean": sum(f1_vals) / len(f1_vals),
        "direct_grounded_f1_ci95_low": f1_vals_sorted[lo],
        "direct_grounded_f1_ci95_high": f1_vals_sorted[hi],
        "rawr_rate_mean": sum(rawr_vals) / len(rawr_vals),
        "rawr_rate_ci95_low": rawr_vals_sorted[lo],
        "rawr_rate_ci95_high": rawr_vals_sorted[hi],
    }
    outp = Path(out_json)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report

