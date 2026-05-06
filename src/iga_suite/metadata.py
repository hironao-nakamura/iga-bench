"""Single source of truth for dataset card / Croissant metadata.

The reviewer-facing regeneration entrypoint — the
``iga-suite materialize-metadata`` CLI defined in
``src/iga_suite/cli.py`` — formats the dataset card and applies the
four-URL Croissant identity quad exclusively through the helpers
exposed here.

Four-URL identity quad
----------------------

A final-hosted Croissant must record four mutually-consistent fields
so the published metadata always points to (and verifies) the actually
hosted dataset archive:

    url                          ← top-level canonical landing page
                                   (anonymized Dataverse Preview URL
                                    during review, DOI URL after
                                    publication).
    distribution[0].contentUrl   ← direct-bytes URL for the archive
                                   (anonymized HF mirror during review).
    distribution[0].sha256       ← SHA-256 hex digest of the archive.
    distribution[0].contentSize  ← archive size in bytes (as ``"N B"``).

The contract disallows ``url`` and ``contentUrl`` collapsing onto the
same value, because reviewers cannot fetch raw bytes from a Dataverse
anonymized Preview URL — the bytes URL must be a separate direct
mirror.  The CLI enforces these invariants by delegating to
``apply_croissant_url_quad`` / ``check_hosted_archive_invariants``.

Per-file FileObject sha256 / contentSize
----------------------------------------

In addition to the top-level archive identity quad, every
``distribution`` entry whose ``contentUrl`` is a *local* path inside
the release (e.g. ``data/benchmarks.parquet``) must carry a
``sha256`` and ``contentSize`` that match the actual bytes of that
file in the release root.  ``materialize_croissant`` recomputes both
fields from the release tree on every invocation, so a reviewer who
re-runs ``iga-suite materialize-metadata`` against the released
dataset gets a Croissant whose per-file integrity claims agree with
the dataset bytes byte-for-byte.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Iterable, Mapping

import pandas as pd


# ---------------------------------------------------------------------------
# Croissant identity invariants
# ---------------------------------------------------------------------------

# Substrings that must never appear inside a final ``contentUrl``.  The
# Dataverse Preview URL (which legitimately appears as the *top-level*
# ``url``) does not serve raw bytes via the file-download API, so it
# cannot be the direct-bytes ``contentUrl``.
_FORBIDDEN_CONTENT_URL_SUBSTRINGS = (
    "previewurl.xhtml",
    "PLACEHOLDER",
    "TBD",
    "__HOSTED_ARCHIVE_CONTENT_URL_REQUIRED__",
)

# Substrings that must never appear inside the top-level ``url``.
# Sentinels are forbidden, but ``previewurl.xhtml`` *is* allowed here
# because the canonical landing page during double-blind review may
# legitimately be an anonymized Dataverse Preview URL.
_FORBIDDEN_TOP_LEVEL_URL_SUBSTRINGS = (
    "PLACEHOLDER",
    "TBD",
    "__HOSTED_ARCHIVE_CONTENT_URL_REQUIRED__",
)

_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_CONTENT_SIZE = re.compile(r"^[0-9]+ B$")


class CroissantInvariantError(ValueError):
    """Raised when a Croissant document violates an identity-quad invariant."""


def check_hosted_archive_invariants(croissant: Mapping) -> list[str]:
    """Return a list of identity-quad invariant violations for ``croissant``.

    The function does *not* raise; the caller decides whether a violation
    is fatal (e.g. final-hosted regeneration) or merely a warning.
    The returned list is empty iff the Croissant satisfies every
    identity-quad invariant.
    """
    violations: list[str] = []

    # conformsTo: accept either the bare core profile string
    # ``"http://mlcommons.org/croissant/1.1"`` or an array that includes
    # it (the RAI-augmented form
    # ``["http://mlcommons.org/croissant/1.1",
    #    "http://mlcommons.org/croissant/RAI/1.0"]`` is also valid).
    raw_ct = croissant.get("conformsTo")
    ct_list = raw_ct if isinstance(raw_ct, list) else [raw_ct] if raw_ct else []
    if "http://mlcommons.org/croissant/1.1" not in ct_list:
        violations.append(
            f"conformsTo must declare 'http://mlcommons.org/croissant/1.1' "
            f"(string or array element); got {raw_ct!r}"
        )

    top_level_url = str(croissant.get("url", ""))
    if not top_level_url:
        violations.append("top-level 'url' is empty")
    elif not top_level_url.startswith("https://"):
        violations.append(f"top-level 'url' must be https://, got {top_level_url!r}")
    for marker in _FORBIDDEN_TOP_LEVEL_URL_SUBSTRINGS:
        if marker in top_level_url:
            violations.append(
                f"top-level 'url' contains forbidden marker {marker!r}: {top_level_url!r}"
            )

    distributions: Iterable[Mapping] = croissant.get("distribution", []) or []
    core = next(
        (d for d in distributions if d.get("@id") == "core-archive"),
        None,
    )
    if core is None:
        violations.append("distribution[@id=core-archive] is missing")
        return violations

    content_url = str(core.get("contentUrl", ""))
    if not content_url:
        violations.append("core-archive 'contentUrl' is empty")
    elif not content_url.startswith("https://"):
        violations.append(
            f"core-archive 'contentUrl' must be https://, got {content_url!r}"
        )
    for marker in _FORBIDDEN_CONTENT_URL_SUBSTRINGS:
        if marker in content_url:
            violations.append(
                f"core-archive 'contentUrl' contains forbidden marker {marker!r}: "
                f"{content_url!r} (the contract requires a direct-bytes URL "
                "distinct from the anonymized Preview URL)"
            )
    if content_url and top_level_url and content_url == top_level_url:
        violations.append(
            "core-archive 'contentUrl' must not equal the top-level 'url' "
            "(the contract separates the canonical landing page from the "
            "direct-bytes URL)."
        )

    sha = str(core.get("sha256", ""))
    if not sha:
        violations.append("core-archive 'sha256' is missing")
    elif not _HEX64.match(sha):
        violations.append(
            f"core-archive 'sha256' must be 64 hex chars, got {sha!r}"
        )

    content_size = str(core.get("contentSize", ""))
    if not content_size:
        violations.append("core-archive 'contentSize' is missing")
    elif not _CONTENT_SIZE.match(content_size):
        violations.append(
            f"core-archive 'contentSize' must match '<bytes> B', got {content_size!r}"
        )

    return violations


def apply_croissant_url_quad(
    croissant: dict,
    *,
    top_level_url: str,
    content_url: str,
    archive_sha256: str,
    archive_size: int,
    description: str | None = None,
) -> dict:
    """Apply the four-URL identity quad to ``croissant`` in place.

    All four fields are required; partial application is not allowed
    because that would produce a half-finalized Croissant whose URLs
    do not match the actual hosted bytes.  Use
    ``check_hosted_archive_invariants`` afterwards if the caller wants to
    fail fast on any residual violation (e.g. ``conformsTo`` left at
    a stale value by an older template).
    """
    # Preserve a richer ``conformsTo`` declaration if the template
    # already lists Croissant 1.1 (potentially together with RAI 1.0);
    # only force-set it when the existing value omits Croissant 1.1.
    _ct = croissant.get("conformsTo")
    _ct_list = _ct if isinstance(_ct, list) else [_ct] if _ct else []
    if "http://mlcommons.org/croissant/1.1" not in _ct_list:
        croissant["conformsTo"] = "http://mlcommons.org/croissant/1.1"
    croissant["url"] = top_level_url
    distributions = croissant.setdefault("distribution", [])
    if not distributions:
        raise CroissantInvariantError(
            "Croissant has no `distribution` entries; cannot apply identity quad"
        )
    core = next(
        (d for d in distributions if d.get("@id") == "core-archive"),
        None,
    )
    if core is None:
        raise CroissantInvariantError(
            "Croissant has no `distribution[@id=core-archive]` entry"
        )
    core["contentUrl"] = content_url
    core["sha256"] = archive_sha256
    core["contentSize"] = f"{int(archive_size)} B"
    if description is not None:
        core["description"] = description
    return croissant


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------

def _read_partitioned_table(path: Path) -> pd.DataFrame:
    flat = path.with_suffix('.parquet')
    if flat.is_file():
        return pd.read_parquet(flat)
    if path.is_file() and path.suffix == '.parquet':
        return pd.read_parquet(path)
    if path.is_file() and path.suffix == '.jsonl':
        return pd.read_json(path, lines=True)
    if path.is_dir():
        parquet_files = sorted(path.rglob('*.parquet'))
        if parquet_files:
            return pd.read_parquet(path)
        jsonl_files = sorted(path.rglob('*.jsonl'))
        if jsonl_files:
            return pd.concat([pd.read_json(p, lines=True) for p in jsonl_files], ignore_index=True)
        return pd.DataFrame()
    return pd.DataFrame()


def _stats_from_dataframes(
    *,
    benchmarks: pd.DataFrame,
    models: pd.DataFrame,
    problems: pd.DataFrame,
    certs: pd.DataFrame,
    metrics: pd.DataFrame,
) -> dict:
    if not models.empty:
        num_model_configurations = int(len(models))
        model_family_values = models['model_family'].dropna().unique().tolist()
        num_model_families = int(len(model_family_values))
        model_families_sorted = sorted(model_family_values)
    else:
        num_model_configurations = 0
        num_model_families = 0
        model_families_sorted = []
    return {
        'num_benchmarks': int(len(benchmarks)) if not benchmarks.empty else 0,
        # ``num_models`` is preserved for backwards compatibility with
        # earlier callers and equals ``num_model_configurations``.
        'num_models': num_model_configurations,
        'num_model_configurations': num_model_configurations,
        'num_model_families': num_model_families,
        'num_problems': int(len(problems)) if not problems.empty else 0,
        'num_certificates': int(len(certs)) if not certs.empty else 0,
        'benchmark_ids': sorted(benchmarks['benchmark_id'].dropna().unique().tolist()) if not benchmarks.empty else [],
        'model_families': model_families_sorted,
        'dependency_modes': sorted(metrics['dependency_mode'].dropna().unique().tolist()) if not metrics.empty else [],
    }


def _release_stats(release_root: str | Path) -> dict:
    root = Path(release_root)
    data = root / 'data'
    return _stats_from_dataframes(
        benchmarks=_read_partitioned_table(data / 'benchmarks'),
        models=_read_partitioned_table(data / 'models'),
        problems=_read_partitioned_table(data / 'problems'),
        certs=_read_partitioned_table(data / 'audit_certificates'),
        metrics=_read_partitioned_table(data / 'aggregate_metrics'),
    )


def stats_from_zip_parquet(read_parquet_bytes: callable, contract_tables: Mapping) -> dict:
    """Compute release stats from in-memory parquet bytes.

    ``read_parquet_bytes(rel_path)`` must return ``bytes`` for the given
    table relative path (e.g. ``"data/models.parquet"``) or raise
    ``KeyError``.  ``contract_tables`` is a mapping describing the
    flat-Parquet release layout (one entry per top-level table).

    This is the entry point used by release-build tooling that
    inspects the dataset zip *contents* rather than a release root, so
    the stats it derives are byte-for-byte identical to those produced
    by ``_release_stats`` from the release root.
    """
    import io
    import pyarrow.parquet as pq

    def _maybe_read(rel: str) -> pd.DataFrame:
        try:
            raw = read_parquet_bytes(rel)
        except KeyError:
            return pd.DataFrame()
        return pq.read_table(io.BytesIO(raw)).to_pandas()

    return _stats_from_dataframes(
        benchmarks=_maybe_read(contract_tables.get('benchmarks', '')),
        models=_maybe_read(contract_tables.get('models', '')),
        problems=_maybe_read(contract_tables.get('problems', '')),
        certs=_maybe_read(contract_tables.get('audit_certificates', '')),
        metrics=_maybe_read(contract_tables.get('aggregate_metrics', '')),
    )


# ---------------------------------------------------------------------------
# Dataset card (formatter + materializer)
# ---------------------------------------------------------------------------

DATASET_CARD_TEMPLATE = """# {dataset_name}

## What this release contains

This is the **derived audit dataset** for IGA-Bench. It does **not** claim ownership of upstream benchmark corpora; instead it releases benchmark-referenced audit annotations, canonicalized traces, probe specifications, certificates, and aggregate metrics.

- Number of benchmarks: **{num_benchmarks}**
- Number of model families: **{num_model_families}** ({model_families})
- Number of benchmark-specific model configurations: **{num_model_configurations}** ({num_model_families} model families x {num_benchmarks} benchmarks)
- Number of problems: **{num_problems}**
- Number of certificates: **{num_certificates}**
- Benchmarks: {benchmark_ids}
- Dependency modes: {dependency_modes}

## Intended use

IGA-Bench is intended for evaluating **step-level premise dependency** in chain-of-thought traces under controlled interventions. Primary claims target formal or semi-formal reasoning settings with recoverable canonical structure.

## Review snapshot

**IGA-Bench Core v1.1-review** is the frozen review snapshot of the IGA-Bench Core v1.1 release family.

Relative to the earlier v1.0 snapshot, v1.1 (i) adopts the released surface-relaxed canonicalizer (v1.1-final) as the single source of truth for all numerics and (ii) expands the ProofWriter slice from 100 to 300 problems (ProofWriter-300) to reduce small-sample variance. The conservative (v1.0) canonicalizer is retained only as a sensitivity analysis under `analysis/` and is not part of the primary data/.

## Important limitations

- Coverage depends on canonicalization and alignment quality.
- The current release reports three dependency conventions: direct, transitive, and predicate-determining.
- Direct and transitive are defined for all included benchmarks; predicate-determining is reported where supported and explicitly marked undefined otherwise.
- Upstream benchmark text may be referenced rather than redistributed depending on license status.

## Licensing

We distinguish three licensing layers:

- **Upstream source-benchmark corpora.** To the best of our knowledge at release time: **ProntoQA** (Saparov & He, 2023; [https://github.com/asaparov/prontoqa](https://github.com/asaparov/prontoqa)) is distributed under the **Apache License 2.0**, and **ProofWriter** (Tafjord et al., 2021; [https://allenai.org/data/proofwriter](https://allenai.org/data/proofwriter)) is distributed under **CC BY 4.0**. The `data/benchmarks/` table conservatively records `upstream_license_status = unknown` because the IGA-Bench release itself does **not** redistribute upstream benchmark text---problem and step records reference upstream items by identifier only (`upstream_problem_id`, `upstream_record_locator`). Reviewers and downstream users who also consume the upstream corpora must follow the license terms published by the upstream authors at the URLs above.
- **Derived IGA-Bench audit dataset (this release).** All files shipped in this dataset zip---derived audit records, canonicalized trace steps, alignment decisions, per-certificate verdicts, aggregate metrics, integrity manifests, and Croissant metadata---are licensed under **CC0 1.0** (matching the `"license"` field in `iga-bench-core-v1.1.croissant.json`). This license applies only to the derived records introduced by IGA-Bench, not to any upstream-benchmark text they reference.
- **Code (evaluation suite).** The accompanying evaluation suite (the IGA-Bench code release) is distributed under the **MIT License** (see the repository `LICENSE` file).

If you distribute derivative datasets that combine this release with upstream-benchmark text, you must separately comply with the upstream license terms; those terms are not modified by this release.

## Release structure

The release contains partitioned Parquet tables under `data/`, plus a validator report, release manifest, checksums, and an optional `analysis/` directory with canonicalizer sensitivity artifacts (not part of the primary single source of truth).
"""


def format_dataset_card_from_stats(stats: Mapping, *, dataset_name: str = 'IGA-Bench Core v1.1') -> str:
    """Format the canonical dataset card text from a stats dict.

    All regeneration entrypoints (``stage_f_package.py`` building from
    a zip, ``materialize-metadata`` building from a release root) call
    this helper so the resulting markdown is byte-for-byte identical
    regardless of source.  This guarantees that the v1.1 wording
    (``Number of model families: **N** ({list})`` / ``Number of
    benchmark-specific model configurations: **M** (N model families
    x K benchmarks)``) never silently regresses to the v1.0-era
    single-count wording, and that the Licensing section that the
    paper refers to from the dataset card always renders.
    """
    return DATASET_CARD_TEMPLATE.format(
        dataset_name=dataset_name,
        num_benchmarks=stats['num_benchmarks'],
        num_model_families=stats['num_model_families'],
        num_model_configurations=stats['num_model_configurations'],
        num_problems=stats['num_problems'],
        num_certificates=stats['num_certificates'],
        benchmark_ids=', '.join(stats.get('benchmark_ids') or []) or 'TBD',
        model_families=', '.join(stats.get('model_families') or []) or 'TBD',
        dependency_modes=', '.join(stats.get('dependency_modes') or []) or 'TBD',
    )


def materialize_dataset_card(release_root: str | Path, output_path: str | Path, *, dataset_name: str = 'IGA-Bench Core v1.1') -> dict:
    stats = _release_stats(release_root)
    content = format_dataset_card_from_stats(stats, dataset_name=dataset_name)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding='utf-8')
    return stats


# ---------------------------------------------------------------------------
# Croissant materializer
# ---------------------------------------------------------------------------

def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _refresh_local_file_objects(template: dict, release_root: Path) -> None:
    """Update ``sha256`` / ``contentSize`` on every local FileObject.

    A local FileObject is any ``distribution`` entry whose
    ``contentUrl`` is a relative path (i.e. not an ``http(s)://`` URL)
    and whose ``@id`` is not the top-level archive.

    Path resolution accommodates two equally valid conventions for
    ``contentUrl``:

    * release-root-relative (e.g. ``data/benchmarks.parquet``) — the
      legacy convention; resolves directly under ``release_root``.
    * archive-relative (e.g. ``iga-bench-core-v1.1/data/benchmarks.parquet``)
      — the convention required by the MLCommons Croissant validator
      so that ``containedIn: core-archive`` lookups can locate the
      file inside the unzipped archive tree.  This convention prefixes
      every internal ``contentUrl`` with the archive's inner directory
      name (= ``release_root.name``), and resolves under the parent of
      ``release_root``.

    For each local FileObject we try the release-root-relative path
    first; if it does not exist we strip the inner-archive prefix and
    retry under ``release_root.parent``.  Whichever resolves wins;
    both conventions yield the same on-disk bytes and therefore the
    same ``sha256`` / ``contentSize``.  Files that neither convention
    can locate are left untouched (the dataset validator gate is the
    right place to flag missing files).
    """
    inner_root = release_root.name  # e.g. "iga-bench-core-v1.1"
    for dist in template.get('distribution', []) or []:
        if dist.get('@id') == 'core-archive':
            continue
        cu = str(dist.get('contentUrl') or '')
        if not cu or cu.startswith(('http://', 'https://')):
            continue
        candidates = [release_root / cu]
        if cu.startswith(inner_root + "/"):
            stripped = cu[len(inner_root) + 1:]
            candidates.append(release_root / stripped)
        path = next((p for p in candidates if p.is_file()), None)
        if path is None:
            continue
        dist['sha256'] = _sha256_file(path)
        dist['contentSize'] = f"{path.stat().st_size} B"


def materialize_croissant(release_root: str | Path,
                          template_path: str | Path,
                          output_path: str | Path,
                          *,
                          content_url: str | None = None,
                          top_level_url: str | None = None,
                          archive_sha256: str | None = None,
                          archive_size: int | None = None) -> dict:
    """Materialize a final Croissant JSON from a built release.

    See module docstring for the four-URL identity-quad contract and
    the per-file FileObject contract.  This function:

    * applies the four identity-quad fields to ``core-archive``
      (top-level ``url``, ``contentUrl``, ``sha256``, ``contentSize``),
      and
    * recomputes ``sha256`` and ``contentSize`` for every other
      ``distribution`` entry from the corresponding bytes inside
      ``release_root``.

    The second step is what guarantees that a reviewer who runs
    ``iga-suite materialize-metadata`` against the released dataset
    obtains a Croissant whose per-file integrity claims agree with the
    actual dataset bytes byte-for-byte.

    Older callers passed only ``content_url``; that path remains
    available so the function does not break tests that materialize an
    intermediate, not-yet-hosted Croissant. The ``cli.py`` entrypoint
    however *requires* the three identity fields (top-level URL,
    content URL, archive SHA-256) so the CLI cannot accidentally
    produce a half-finalized Croissant for a real release.
    """
    release_root = Path(release_root)
    stats = _release_stats(release_root)
    template = json.loads(Path(template_path).read_text(encoding='utf-8'))
    template['version'] = '1.1.0'
    # ``conformsTo`` may declare core Croissant alone (string) or core +
    # RAI (array).  Preserve whatever the template ships; the contract
    # we enforce is "Croissant 1.1 must be among the declared profiles".
    _ct = template.get('conformsTo')
    _ct_list = _ct if isinstance(_ct, list) else [_ct] if _ct else []
    if 'http://mlcommons.org/croissant/1.1' not in _ct_list:
        template['conformsTo'] = 'http://mlcommons.org/croissant/1.1'
    if str(template.get('license', '')).upper().startswith('TBD'):
        template['license'] = "CC-BY-4.0 (derived metadata); see benchmark provenance for upstream terms"
    if top_level_url is not None:
        template['url'] = top_level_url
    description = (
        f"Primary release archive with {stats['num_problems']} "
        f"problems and {stats['num_certificates']} certificates."
    )
    for dist in template.get('distribution', []):
        if dist.get('@id') == 'core-archive':
            if content_url is not None:
                dist['contentUrl'] = content_url
            if archive_sha256 is not None:
                dist['sha256'] = archive_sha256
            if archive_size is not None:
                dist['contentSize'] = f"{int(archive_size)} B"
            dist['description'] = description
    _refresh_local_file_objects(template, release_root)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(template, indent=2, ensure_ascii=False), encoding='utf-8')
    return stats


__all__ = [
    'CroissantInvariantError',
    'apply_croissant_url_quad',
    'check_hosted_archive_invariants',
    'format_dataset_card_from_stats',
    'materialize_croissant',
    'materialize_dataset_card',
    'stats_from_zip_parquet',
]
