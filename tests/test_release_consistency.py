"""Release-level internal consistency gates.

These tests exist because earlier validator passes did not catch the case
where a partition directory contained stale parquet files from a previous
parser version: the `aggregate_metrics` / `run_summaries` tables can be
freshly regenerated from v1.1 parser output while the `audit_certificates`
table still contains v1.0 parser records, yielding a release where derived
metrics disagree with the underlying certificates.

Gate A — every `aggregate_metrics` row matches a recompute from
          `audit_certificates` (precision, recall, F1, coverage).
Gate B — every `run_summaries.rawr_direct / rawr_transitive` matches a
          recompute from `audit_certificates` (per problem).
Gate C — `analysis/parser_sensitivity.parquet` "released" columns
          (coverage_released / f1_released) agree with
          `data/aggregate_metrics.parquet` primary values.
Gate D — `from iga_suite.normalizer import normalize_step_text` is the
          v1.1 released canonicalizer (smoke test on a canonical form
          that only v1.1 accepts).
Gate E — each partition directory under `data/<table>/...` contains at
          most one parquet file so that no stale file can silently
          override a freshly reprocessed one.

If you ever need to intentionally add multiple parquet files per
partition, change Gate E to validate on row-level content identity
rather than disabling the check.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest


RELEASE_ROOT = Path(__file__).resolve().parents[1] / "release" / "iga-bench-core-v1.1"
ANALYSIS_DIR = RELEASE_ROOT / "analysis"

pytestmark = pytest.mark.skipif(
    not RELEASE_ROOT.exists(),
    reason="Release root not present; skipping consistency gates",
)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
TOL = 1e-5


def _read_flat(name: str) -> pd.DataFrame:
    flat = RELEASE_ROOT / "data" / f"{name}.parquet"
    if flat.is_file():
        return pd.read_parquet(flat)
    partitioned = RELEASE_ROOT / "data" / name
    if partitioned.is_dir():
        return pd.read_parquet(partitioned)
    raise FileNotFoundError(f"{name} not found under {RELEASE_ROOT / 'data'}")


def _prf1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f1


def _recompute_primary_metrics(sub: pd.DataFrame) -> dict[str, float]:
    """Match src/iga_suite/metrics.py `aggregate` definition.

    Primary F1 is computed over *definitive* certificates only; UNPARSEABLE
    is excluded from the TP/FP/FN accounting. Coverage is the fraction of
    definitive certificates over all certificates for that (config, mode).
    """
    total = len(sub)
    defin = sub[sub["verdict_type"] != "UNPARSEABLE"]
    gold = defin["gold_dependency_label"].astype(bool)
    pred = defin["verdict_type"] == "GROUNDED"
    tp = int((pred & gold).sum())
    fp = int((pred & ~gold).sum())
    fn = int((~pred & gold).sum())
    p, r, f1 = _prf1(tp, fp, fn)
    coverage = len(defin) / total if total else 0.0
    return {
        "precision": p,
        "recall": r,
        "f1": f1,
        "coverage": coverage,
        "num_certificates": total,
    }


# --------------------------------------------------------------------------
# Gate A — aggregate_metrics ⇔ audit_certificates
# --------------------------------------------------------------------------
class TestGateA_AggregateMatchesCertificates:
    """Every row in aggregate_metrics must be reproducible from certificates."""

    @pytest.fixture(scope="class")
    def agg(self) -> pd.DataFrame:
        return _read_flat("aggregate_metrics")

    @pytest.fixture(scope="class")
    def ac(self) -> pd.DataFrame:
        return _read_flat("audit_certificates")

    def test_every_cell_is_consistent(self, agg, ac):
        mismatches: list[str] = []
        for _, row in agg.iterrows():
            cfg = row["config_id"]
            mode = row["dependency_mode"]
            sub = ac[
                (ac["config_id"] == cfg)
                & (ac["dependency_mode_scored"] == mode)
            ]
            computed = _recompute_primary_metrics(sub)
            for field in ("precision", "recall", "f1", "coverage"):
                got = float(row[field])
                exp = computed[field]
                if abs(got - exp) > TOL:
                    mismatches.append(
                        f"{cfg} / {mode} / {field}: agg={got:.6f}  "
                        f"cert-recomputed={exp:.6f}  Δ={got - exp:+.6f}"
                    )
            if row["num_certificates"] != computed["num_certificates"]:
                mismatches.append(
                    f"{cfg} / {mode} / num_certificates: agg={row['num_certificates']}  "
                    f"cert-count={computed['num_certificates']}"
                )
        assert not mismatches, (
            "aggregate_metrics disagree with audit_certificates:\n  "
            + "\n  ".join(mismatches)
        )


# --------------------------------------------------------------------------
# Gate B — run_summaries.rawr_* ⇔ audit_certificates per problem
# --------------------------------------------------------------------------
class TestGateB_RunSummaryRawrMatchesCertificates:
    """Per-problem RAWR flags in run_summaries must be reproducible from
    certificates. RAWR := final_answer_correct AND ∃ cert in mode with
    gold_dependency_label=True AND verdict_type ∈ {INSENSITIVE, MISREPRESENTATION}.
    """

    @pytest.fixture(scope="class")
    def rs(self) -> pd.DataFrame:
        return _read_flat("run_summaries")

    @pytest.fixture(scope="class")
    def ac(self) -> pd.DataFrame:
        return _read_flat("audit_certificates")

    def _recompute_rawr(self, ac_cfg: pd.DataFrame, fa: dict[str, bool], mode: str) -> dict[str, bool]:
        sub = ac_cfg[ac_cfg["dependency_mode_scored"] == mode]
        out: dict[str, bool] = {}
        for pid, g in sub.groupby("dataset_problem_id", observed=True):
            pid = str(pid)
            if not fa.get(pid, False):
                out[pid] = False
                continue
            hit = (
                g["gold_dependency_label"].astype(bool)
                & g["verdict_type"].isin(["INSENSITIVE", "MISREPRESENTATION"])
            ).any()
            out[pid] = bool(hit)
        return out

    def test_rawr_matches_each_config(self, rs, ac):
        mismatches: list[str] = []
        for cfg, rs_cfg in rs.groupby("config_id", observed=True):
            ac_cfg = ac[ac["config_id"] == cfg]
            fa = {
                str(pid): bool(v)
                for pid, v in zip(rs_cfg["dataset_problem_id"], rs_cfg["final_answer_correct"])
            }
            for mode, col in (("direct", "rawr_direct"), ("transitive", "rawr_transitive")):
                expected = self._recompute_rawr(ac_cfg, fa, mode)
                for _, r in rs_cfg.iterrows():
                    pid = str(r["dataset_problem_id"])
                    got = bool(r[col])
                    exp = expected.get(pid, False)
                    if got != exp:
                        mismatches.append(f"{cfg}/{pid}/{col}: rs={got} cert={exp}")
                        if len(mismatches) > 20:
                            break
                if len(mismatches) > 20:
                    break
            if len(mismatches) > 20:
                break
        assert not mismatches, (
            "run_summaries.rawr_* disagree with audit_certificates (first 20 shown):\n  "
            + "\n  ".join(mismatches)
        )


# --------------------------------------------------------------------------
# Gate C — parser_sensitivity.released ⇔ aggregate_metrics primary
# --------------------------------------------------------------------------
class TestGateC_ParserSensitivityReleasedMatchesPrimary:
    """The "released" column family of the sensitivity table is by
    construction the primary aggregate_metrics. They must agree exactly."""

    @pytest.fixture(scope="class")
    def sens(self) -> pd.DataFrame:
        path = ANALYSIS_DIR / "parser_sensitivity.parquet"
        if not path.is_file():
            pytest.skip("Parser sensitivity artifacts not present")
        return pd.read_parquet(path)

    @pytest.fixture(scope="class")
    def agg(self) -> pd.DataFrame:
        return _read_flat("aggregate_metrics")

    def test_released_columns_match_primary(self, sens, agg):
        primary = sens[sens["in_primary_release"]]
        mismatches: list[str] = []
        for _, row in primary.iterrows():
            bid = row["benchmark_id"]
            fam = row["model_family"]
            mode = row["dependency_mode"]
            m = agg[
                (agg["benchmark_id"] == bid)
                & (agg["model_family"] == fam)
                & (agg["dependency_mode"] == mode)
            ]
            if len(m) != 1:
                mismatches.append(f"{bid}/{fam}/{mode}: expected 1 primary row, got {len(m)}")
                continue
            a = m.iloc[0]
            for sens_col, prim_col in (
                ("coverage_released", "coverage"),
                ("f1_released", "f1"),
            ):
                s = float(row[sens_col])
                p = float(a[prim_col])
                if abs(s - p) > TOL:
                    mismatches.append(
                        f"{bid}/{fam}/{mode} {sens_col}: sens={s:.6f} "
                        f"aggregate.{prim_col}={p:.6f} Δ={s - p:+.6f}"
                    )
        assert not mismatches, (
            "parser_sensitivity.released disagrees with aggregate_metrics primary:\n  "
            + "\n  ".join(mismatches)
        )


# --------------------------------------------------------------------------
# Gate D — default parser is the v1.1 released canonicalizer
# --------------------------------------------------------------------------
class TestGateD_DefaultParserIsReleased:
    """`iga_suite.normalizer` is what the user hits when running
    `iga-suite run-eval` from the README. It must be the released v1.1
    canonicalizer, not the v1.0 conservative one."""

    def test_bare_is_predicate_parses(self):
        from iga_suite.normalizer import normalize_step_text

        out = normalize_step_text("Rex is cold")
        # v1.1 → ("is(rex, cold)", "is", "OK", ...)
        # v1.0 → (None, "free_form", "UNPARSEABLE")
        assert out[2] == "OK", (
            f"Default parser returned {out!r}; expected status='OK' "
            "(= v1.1 released canonicalizer). If this fails, the README "
            "`iga-suite run-eval` flow is using the v1.0 conservative parser."
        )
        assert out[0] == "is(rex, cold)", (
            f"Default parser produced unexpected canonical form: {out[0]!r}"
        )


# --------------------------------------------------------------------------
# Gate E — no stale parquet files in partition directories
# --------------------------------------------------------------------------
class TestGateE_NoStalePartitionFiles:
    """Each hive partition directory must contain exactly one .parquet
    file. Multiple files in the same leaf partition caused silent
    v1.0/v1.1 cross-contamination of audit_certificates in the past."""

    TABLES = [
        "audit_certificates",
        "aggregate_metrics",
        "run_summaries",
        "benchmarks",
        "problems",
        "models",
    ]

    @pytest.mark.parametrize("table", TABLES)
    def test_one_parquet_per_partition(self, table):
        root = RELEASE_ROOT / "data" / table
        if not root.is_dir():
            pytest.skip(f"{table} is not partitioned in this release")
        offenders: list[str] = []
        for leaf in root.rglob("*"):
            if not leaf.is_dir():
                continue
            parquets = [p for p in leaf.iterdir() if p.is_file() and p.suffix == ".parquet"]
            if len(parquets) > 1:
                offenders.append(
                    f"{leaf.relative_to(RELEASE_ROOT)}: "
                    f"{len(parquets)} parquet files — "
                    + ", ".join(p.name for p in parquets)
                )
        assert not offenders, (
            "Stale parquet files detected in partition leaves "
            "(previous rebuilds left old parser output behind):\n  "
            + "\n  ".join(offenders)
        )
