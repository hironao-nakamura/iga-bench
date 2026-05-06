"""Problem-level bootstrap 95% CIs for IGA-Bench main metrics.

For each (benchmark, model_family, dependency_mode) cell this script
reports 95% percentile bootstrap confidence intervals for:

* direct / transitive / predicate-determining F1 (Grounded-only)
* definitive coverage
* observed RAWR rate (problem-level event rate, conditioned on
  final_answer_correct)

The resampling unit is the problem.  For each bootstrap iteration we
sample ``N`` problems with replacement (where ``N`` is the size of the
benchmark split), re-aggregate all certificates belonging to the
resampled problems, and recompute the metrics.  We report the mean and
the 2.5 / 97.5 percentile bounds over ``B=10,000`` iterations.

This script also emits a small "severity-ish" supplementary metric:
the per-problem mean / median ``insensitive_required_pair_rate`` among
covered correctly-answered problems, with the same bootstrap CIs.  The
severity metric complements the event-rate RAWR by counting within a
problem how many of the required-covered pairs were insensitive or
misrepresented, rather than just flagging whether any such pair exists.

Outputs:

    outputs/bootstrap_ci/bootstrap_ci.json
    outputs/bootstrap_ci/bootstrap_ci.tsv
    outputs/bootstrap_ci/bootstrap_ci_severity.tsv
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[2]
RELEASE_DIR = PROJECT / "release" / "iga-bench-core-v1.1"
CERTS_PARQUET = RELEASE_DIR / "data" / "audit_certificates.parquet"
SUMMARIES_PARQUET = RELEASE_DIR / "data" / "run_summaries.parquet"
PROBLEMS_PARQUET = RELEASE_DIR / "data" / "problems.parquet"

OUT_DIR_DEFAULT = PROJECT / "outputs" / "bootstrap_ci"

BENCHMARKS = ["prontoqa_full_eval", "proofwriter_cwa_d3_is_300"]
MODEL_FAMILIES = ["openai", "anthropic", "openweight"]
MODES = ["direct", "transitive", "predicate_determining"]

BENCHMARK_DISPLAY = {
    "prontoqa_full_eval":         "ProntoQA-500",
    "proofwriter_cwa_d3_is_300":  "ProofWriter-300",
}
MODEL_DISPLAY = {
    "openai":     "GPT-4o",
    "anthropic":  "Claude Sonnet 4.6",
    "openweight": "Qwen3-Next-80B",
}


def _f1(tp: int, fp: int, fn: int) -> float:
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    return 2 * p * r / (p + r) if (p + r) else 0.0


def _slice_f1_coverage(verdict: np.ndarray, gold: np.ndarray) -> tuple[float, float]:
    """Compute primary Grounded-only F1 and definitive coverage on a flat slice.

    Matches :mod:`iga_suite.metrics.aggregate`: ``UNPARSEABLE`` certificates
    are **excluded** from the primary TP/FP/FN tally (they are only counted
    in the lower-bound variant) and coverage is the fraction of definitive
    (non-UNPARSEABLE) certificates over all certificates in the slice.
    """
    n = verdict.size
    if n == 0:
        return float("nan"), float("nan")
    definitive = verdict != "UNPARSEABLE"
    v_def = verdict[definitive]
    g_def = gold[definitive]
    pred_pos = v_def == "GROUNDED"
    tp = int(np.sum(pred_pos & g_def))
    fp = int(np.sum(pred_pos & ~g_def))
    fn = int(np.sum(~pred_pos & g_def))
    if (tp + fn) == 0:
        f1 = float("nan")
    else:
        f1 = _f1(tp, fp, fn)
    coverage = float(np.sum(definitive) / n)
    return f1, coverage


def _percentile_ci(arr: np.ndarray) -> tuple[float, float, float]:
    """Return (mean, 2.5pct, 97.5pct) ignoring NaNs."""
    clean = arr[~np.isnan(arr)]
    if clean.size == 0:
        return float("nan"), float("nan"), float("nan")
    return (
        float(np.mean(clean)),
        float(np.percentile(clean, 2.5)),
        float(np.percentile(clean, 97.5)),
    )


def _load() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    certs = pd.read_parquet(CERTS_PARQUET)
    summaries = pd.read_parquet(SUMMARIES_PARQUET)
    problems = pd.read_parquet(PROBLEMS_PARQUET)
    return certs, summaries, problems


def _build_per_cell_index(
    certs: pd.DataFrame, summaries: pd.DataFrame, problems: pd.DataFrame
) -> dict:
    """Group certificate / summary rows by (benchmark, model_family).

    Returns a nested dict ``idx[benchmark][family] = {...}`` where each
    leaf entry caches NumPy arrays keyed by problem id for fast
    resampling.
    """
    problems = problems[["dataset_problem_id", "benchmark_id"]]
    certs_m = certs.merge(problems, on="dataset_problem_id", how="inner")
    summ_m = summaries.merge(problems, on="dataset_problem_id", how="inner")

    idx: dict = {}
    for bench in BENCHMARKS:
        idx[bench] = {}
        for fam in MODEL_FAMILIES:
            c_bf = certs_m[
                (certs_m["benchmark_id"] == bench)
                & (certs_m["model_family"] == fam)
            ].copy()
            s_bf = summ_m[
                (summ_m["benchmark_id"] == bench)
                & (summ_m["model_family"] == fam)
            ].copy()
            problem_ids = sorted(s_bf["dataset_problem_id"].unique().tolist())

            cell: dict = {"problem_ids": problem_ids}
            for mode in MODES:
                c_mode = c_bf[c_bf["dependency_mode_scored"] == mode]
                grouped: dict[str, dict] = {}
                for pid, g in c_mode.groupby("dataset_problem_id"):
                    grouped[pid] = {
                        "verdict":  g["verdict_type"].to_numpy(),
                        "gold":     g["gold_dependency_label"].fillna(False)
                                        .astype(bool).to_numpy(),
                    }
                cell[mode] = grouped
            # RAWR is stored per problem in run_summaries, already gated
            # on final_answer_correct in the aggregation code (rawr_*
            # is False for incorrect answers by construction).
            rawr_direct = s_bf.set_index("dataset_problem_id")["rawr_direct"].to_dict()
            rawr_trans  = s_bf.set_index("dataset_problem_id")["rawr_transitive"].to_dict()
            correct     = s_bf.set_index("dataset_problem_id")["final_answer_correct"].to_dict()
            cell["rawr_direct"]     = rawr_direct
            cell["rawr_transitive"] = rawr_trans
            cell["correct"]         = correct
            idx[bench][fam] = cell
    return idx


def _bootstrap_cell(cell: dict, *, n_boot: int, rng: np.random.Generator) -> dict:
    problem_ids = np.asarray(cell["problem_ids"])
    n = len(problem_ids)
    if n == 0:
        return {}

    out: dict[str, list[float]] = {
        "direct_f1": [], "transitive_f1": [], "predicate_determining_f1": [],
        "direct_coverage": [],
        "rawr_direct": [], "rawr_transitive": [],
    }

    correct_lookup = cell["correct"]

    for _ in range(n_boot):
        sample_idx = rng.integers(0, n, size=n)
        sample_pids = problem_ids[sample_idx]

        # F1 / coverage per mode
        for mode, key in [
            ("direct", "direct_f1"),
            ("transitive", "transitive_f1"),
            ("predicate_determining", "predicate_determining_f1"),
        ]:
            mode_map = cell[mode]
            verdicts = []
            golds = []
            for pid in sample_pids:
                g = mode_map.get(pid)
                if g is None:
                    continue
                verdicts.append(g["verdict"])
                golds.append(g["gold"])
            if verdicts:
                v = np.concatenate(verdicts)
                gl = np.concatenate(golds)
            else:
                v = np.array([], dtype=object)
                gl = np.array([], dtype=bool)
            f1, cov = _slice_f1_coverage(v, gl)
            out[key].append(f1)
            if mode == "direct":
                out["direct_coverage"].append(cov)

        # Observed RAWR event rates (among correct-answer problems).
        # Denominator is the number of correct-answer problems in the
        # resample; matches the definition in Section 8.
        correct_mask = np.array(
            [bool(correct_lookup.get(pid, False)) for pid in sample_pids],
            dtype=bool,
        )
        n_correct = int(correct_mask.sum())
        if n_correct == 0:
            out["rawr_direct"].append(float("nan"))
            out["rawr_transitive"].append(float("nan"))
        else:
            rd = np.array(
                [bool(cell["rawr_direct"].get(pid, False))
                 for pid in sample_pids[correct_mask]],
                dtype=bool,
            )
            rt = np.array(
                [bool(cell["rawr_transitive"].get(pid, False))
                 for pid in sample_pids[correct_mask]],
                dtype=bool,
            )
            out["rawr_direct"].append(float(rd.mean()))
            out["rawr_transitive"].append(float(rt.mean()))

    stats: dict[str, dict[str, float]] = {}
    for key, vals in out.items():
        mean, lo, hi = _percentile_ci(np.asarray(vals, dtype=float))
        stats[key] = {"mean": mean, "ci95_low": lo, "ci95_high": hi}
    return stats


# ---- severity-ish: insensitive_required_pair_rate (direct mode) ------

def _severity_per_cell(cell: dict, *, n_boot: int, rng: np.random.Generator) -> dict:
    problem_ids = np.asarray(cell["problem_ids"])
    n = len(problem_ids)
    if n == 0:
        return {}

    # Precompute per-problem severity score (direct mode).
    # insensitive_required_pair_rate =
    #   (# covered required pairs with verdict in {INSENSITIVE, MISREPRESENTATION}) /
    #   (# covered required pairs)
    # where covered = verdict != UNPARSEABLE, required = gold_dependency_label.
    per_problem_rate: dict[str, float] = {}
    direct_map = cell["direct"]
    for pid, g in direct_map.items():
        v = g["verdict"]
        gl = g["gold"]
        if v.size == 0:
            continue
        covered = v != "UNPARSEABLE"
        required_covered = covered & gl
        denom = int(required_covered.sum())
        if denom == 0:
            continue
        insensitive = np.isin(v, ["INSENSITIVE", "MISREPRESENTATION"])
        numer = int((required_covered & insensitive).sum())
        per_problem_rate[pid] = numer / denom

    correct_lookup = cell["correct"]

    means = []
    medians = []
    for _ in range(n_boot):
        sample_idx = rng.integers(0, n, size=n)
        sample_pids = problem_ids[sample_idx]
        vals = [
            per_problem_rate[pid]
            for pid in sample_pids
            if pid in per_problem_rate and bool(correct_lookup.get(pid, False))
        ]
        if not vals:
            means.append(float("nan"))
            medians.append(float("nan"))
            continue
        arr = np.asarray(vals)
        means.append(float(arr.mean()))
        medians.append(float(np.median(arr)))

    stats: dict[str, dict[str, float]] = {}
    mean, lo, hi = _percentile_ci(np.asarray(means))
    stats["insensitive_required_pair_rate_mean"] = {
        "mean": mean, "ci95_low": lo, "ci95_high": hi
    }
    mean, lo, hi = _percentile_ci(np.asarray(medians))
    stats["insensitive_required_pair_rate_median"] = {
        "mean": mean, "ci95_low": lo, "ci95_high": hi
    }
    stats["num_problems_with_required_covered_pairs_and_correct_answer"] = {
        "mean": float(sum(
            1 for pid in problem_ids
            if pid in per_problem_rate and bool(correct_lookup.get(pid, False))
        )),
        "ci95_low": float("nan"),
        "ci95_high": float("nan"),
    }
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-bootstrap", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=20260415)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR_DEFAULT)
    args = ap.parse_args()

    certs, summaries, problems = _load()
    print(f"[bootstrap] loaded certs={len(certs)}  summaries={len(summaries)} "
          f" problems={len(problems)}")

    idx = _build_per_cell_index(certs, summaries, problems)

    results: dict = {
        "release_id":    "iga-bench-core-v1.1",
        "snapshot_id":   "iga-bench-core-v1.1-review",
        "resampling_unit": "problem",
        "n_bootstrap":   args.n_bootstrap,
        "seed":          args.seed,
        "ci":            "95% percentile",
        "cells":         [],
    }
    severity_rows = []
    metric_rows = []

    rng = np.random.default_rng(args.seed)
    for bench in BENCHMARKS:
        for fam in MODEL_FAMILIES:
            cell = idx[bench][fam]
            print(f"[bootstrap] {bench} / {fam}: "
                  f"{len(cell['problem_ids'])} problems")
            stats = _bootstrap_cell(cell, n_boot=args.n_bootstrap, rng=rng)
            sev = _severity_per_cell(cell, n_boot=args.n_bootstrap, rng=rng)
            results["cells"].append({
                "benchmark_id": bench,
                "model_family": fam,
                "num_problems": len(cell["problem_ids"]),
                "metrics":      stats,
                "severity":     sev,
            })
            # Flat rows for TSV
            for key, v in stats.items():
                metric_rows.append({
                    "benchmark_id": bench,
                    "benchmark_display": BENCHMARK_DISPLAY[bench],
                    "model_family": fam,
                    "model_display": MODEL_DISPLAY[fam],
                    "metric": key,
                    "mean": v["mean"],
                    "ci95_low": v["ci95_low"],
                    "ci95_high": v["ci95_high"],
                })
            for key, v in sev.items():
                severity_rows.append({
                    "benchmark_id": bench,
                    "benchmark_display": BENCHMARK_DISPLAY[bench],
                    "model_family": fam,
                    "model_display": MODEL_DISPLAY[fam],
                    "metric": key,
                    "mean": v["mean"],
                    "ci95_low": v["ci95_low"],
                    "ci95_high": v["ci95_high"],
                })

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "bootstrap_ci.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    pd.DataFrame(metric_rows).to_csv(
        out_dir / "bootstrap_ci.tsv", sep="\t", index=False,
    )
    pd.DataFrame(severity_rows).to_csv(
        out_dir / "bootstrap_ci_severity.tsv", sep="\t", index=False,
    )
    print(f"[bootstrap] wrote {out_dir / 'bootstrap_ci.json'}")
    print(f"[bootstrap] wrote {out_dir / 'bootstrap_ci.tsv'}")
    print(f"[bootstrap] wrote {out_dir / 'bootstrap_ci_severity.tsv'}")


if __name__ == "__main__":
    main()
