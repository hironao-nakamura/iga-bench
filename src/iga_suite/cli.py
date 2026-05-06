from __future__ import annotations

import argparse
import json
from pathlib import Path

from iga_suite.benchmarks import bootstrap_iclr_supplementary, bootstrap_proofwriter
from iga_suite.config import load_config
from iga_suite.pipeline import run_evaluation
from iga_suite.validator import validate_dataset_root
from iga_suite.acceptance import create_revisioned_acceptance_subset, create_holdout_subset, evaluate_acceptance_gate
from iga_suite.matrix import run_matrix
from iga_suite.release import build_release
from iga_suite.metadata import materialize_dataset_card, materialize_croissant
from iga_suite.splits import build_prontoqa_splits
from iga_suite.normalizer_eval import sample_steps_for_audit, evaluate_annotations
from iga_suite.bootstrap_ci import bootstrap_problem_level_ci
from iga_suite.report_tables import build_tables
from iga_suite.report_figures import build_figures


def cmd_bootstrap_iclr(args):
    count = bootstrap_iclr_supplementary(args.supplementary, args.output)
    print(json.dumps({'status': 'ok', 'count': count, 'output': str(Path(args.output).resolve())}, indent=2))


def cmd_bootstrap_proofwriter(args):
    count = bootstrap_proofwriter(args.input, args.output, benchmark_id=args.benchmark_id, split=args.split)
    print(json.dumps({'status': 'ok', 'count': count, 'output': str(Path(args.output).resolve())}, indent=2))


def cmd_run_eval(args):
    cfg = load_config(args.config)
    result = run_evaluation(cfg)
    print(json.dumps(result, indent=2))


def cmd_validate_core(args):
    report = validate_dataset_root(args.dataset_root)
    print(json.dumps(report, indent=2))


def cmd_make_acceptance_subset(args):
    result = create_revisioned_acceptance_subset(args.input_jsonl, args.output_jsonl)
    print(json.dumps(result, indent=2))


def cmd_make_holdout_subset(args):
    result = create_holdout_subset(args.full_jsonl, args.dev_jsonl, args.output_jsonl)
    print(json.dumps(result, indent=2))


def cmd_eval_acceptance_gate(args):
    report = evaluate_acceptance_gate(args.output_root, args.companion_root)
    print(json.dumps(report, indent=2))


def cmd_run_matrix(args):
    report = run_matrix(args.matrix_spec, continue_on_error=args.continue_on_error)
    print(json.dumps(report, indent=2))


def cmd_build_release(args):
    report = build_release(
        args.output_roots,
        schema_path=args.schema_path,
        release_root=args.release_root,
        release_id=args.release_id,
        review_snapshot_id=args.review_snapshot_id,
    )
    print(json.dumps(report, indent=2))


def cmd_materialize_metadata(args):
    # Refuse to bake a half-finalized Croissant.  All three identity
    # fields (top-level URL, content URL, archive SHA-256) must be
    # supplied together; otherwise reviewers may regenerate metadata
    # whose ``url`` / ``contentUrl`` / ``sha256`` do not match the
    # actual hosted dataset zip.  ``--archive-size`` is optional but
    # strongly recommended; if omitted, ``contentSize`` is left at
    # whatever the template specifies (which is usually missing).
    missing = [
        flag for flag, value in (
            ('--top-level-url',  args.top_level_url),
            ('--content-url',    args.content_url),
            ('--archive-sha256', args.archive_sha256),
        ) if not value
    ]
    if missing:
        raise SystemExit(
            "Error: " + ", ".join(missing) +
            " are required for final hosted metadata. "
            "These three identity fields must be set together so the "
            "regenerated Croissant matches the actually hosted dataset "
            "archive (top-level dataset landing page, direct-bytes "
            "contentUrl, and archive SHA-256)."
        )
    card_stats = materialize_dataset_card(args.release_root, args.dataset_card_out, dataset_name=args.dataset_name)
    croissant_stats = materialize_croissant(
        args.release_root,
        args.croissant_template,
        args.croissant_out,
        content_url=args.content_url,
        top_level_url=args.top_level_url,
        archive_sha256=args.archive_sha256,
        archive_size=args.archive_size,
    )
    print(json.dumps({
        'status': 'ok',
        'dataset_card_out': str(Path(args.dataset_card_out).resolve()),
        'croissant_out': str(Path(args.croissant_out).resolve()),
        'dataset_card_stats': card_stats,
        'croissant_stats': croissant_stats,
    }, indent=2))


def cmd_build_prontoqa_splits(args):
    report = build_prontoqa_splits(
        args.input_jsonl,
        args.out_dir,
        target_full_eval_n=args.target_full_eval_n,
        dev_n=args.dev_n,
        holdout_n=args.holdout_n,
        seed=args.seed,
        allow_upsample=args.allow_upsample,
    )
    print(json.dumps(report, indent=2))


def cmd_normalizer_sample(args):
    report = sample_steps_for_audit(args.trace_steps_jsonl, args.out_jsonl, n=args.n, seed=args.seed)
    print(json.dumps(report, indent=2))


def cmd_normalizer_eval(args):
    report = evaluate_annotations(args.annotation_jsonl, args.out_json)
    print(json.dumps(report, indent=2))


def cmd_bootstrap_ci(args):
    report = bootstrap_problem_level_ci(
        args.certs_parquet_root,
        args.out_json,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
    )
    print(json.dumps(report, indent=2))


def cmd_report_tables(args):
    report = build_tables(Path(args.release_root), Path(args.out))
    print(json.dumps(report, indent=2))


def cmd_report_figures(args):
    report = build_figures(Path(args.release_root), Path(args.out))
    print(json.dumps(report, indent=2))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog='iga-suite', description='IGA-Bench evaluation suite')
    sub = p.add_subparsers(dest='cmd', required=True)

    p_boot = sub.add_parser('bootstrap-iclr', help='Extract benchmark problem JSONL from an ICLR supplementary zip or directory')
    p_boot.add_argument('--supplementary', required=True)
    p_boot.add_argument('--output', required=True)
    p_boot.set_defaults(func=cmd_bootstrap_iclr)

    p_boot_pw = sub.add_parser('bootstrap-proofwriter', help='Normalize a local ProofWriter-like JSON/JSONL file into IGA benchmark JSONL')
    p_boot_pw.add_argument('--input', required=True)
    p_boot_pw.add_argument('--output', required=True)
    p_boot_pw.add_argument('--benchmark-id', default='proofwriter')
    p_boot_pw.add_argument('--split', default='analysis')
    p_boot_pw.set_defaults(func=cmd_bootstrap_proofwriter)

    p_run = sub.add_parser('run-eval', help='Run the evaluation pipeline for one config')
    p_run.add_argument('--config', required=True)
    p_run.set_defaults(func=cmd_run_eval)

    p_matrix = sub.add_parser('run-matrix', help='Run a matrix spec containing multiple configs')
    p_matrix.add_argument('--matrix-spec', required=True)
    p_matrix.add_argument('--continue-on-error', action='store_true')
    p_matrix.set_defaults(func=cmd_run_matrix)

    p_val = sub.add_parser('validate-core', help='Validate an emitted IGA-Bench Core dataset root')
    p_val.add_argument('--dataset-root', required=True)
    p_val.set_defaults(func=cmd_validate_core)

    p_subset = sub.add_parser('make-acceptance-subset', help='Create revisioned 10-problem acceptance subset (stratified by hop)')
    p_subset.add_argument('--input-jsonl', required=True)
    p_subset.add_argument('--output-jsonl', required=True)
    p_subset.set_defaults(func=cmd_make_acceptance_subset)
    p_subset_dev = sub.add_parser('make-dev-subset', help='Create revisioned 10-problem development subset (stratified by hop)')
    p_subset_dev.add_argument('--input-jsonl', required=True)
    p_subset_dev.add_argument('--output-jsonl', required=True)
    p_subset_dev.set_defaults(func=cmd_make_acceptance_subset)
    p_subset_holdout = sub.add_parser('make-holdout-subset', help='Create holdout 10-problem subset excluding dev set')
    p_subset_holdout.add_argument('--full-jsonl', required=True)
    p_subset_holdout.add_argument('--dev-jsonl', required=True)
    p_subset_holdout.add_argument('--output-jsonl', required=True)
    p_subset_holdout.set_defaults(func=cmd_make_holdout_subset)

    p_gate = sub.add_parser('eval-acceptance-gate', help='Evaluate acceptance gate on an evaluation output')
    p_gate.add_argument('--output-root', required=True)
    p_gate.add_argument('--companion-root', required=False, default=None)
    p_gate.set_defaults(func=cmd_eval_acceptance_gate)

    p_release = sub.add_parser('build-release', help='Merge multiple run outputs into one IGA-Bench Core release root')
    p_release.add_argument('--schema-path', required=True)
    p_release.add_argument('--release-root', required=True)
    p_release.add_argument('--release-id', default='iga-bench-core-v1.1')
    p_release.add_argument(
        '--review-snapshot-id',
        default='iga-bench-core-v1.1-review',
        help=(
            'Logical review-snapshot identifier written into release_manifest.json. '
            'The paper, dataset card, and reviewer-facing bundle all reference this id.'
        ),
    )
    p_release.add_argument('--output-roots', nargs='+', required=True)
    p_release.set_defaults(func=cmd_build_release)

    p_meta = sub.add_parser('materialize-metadata', help='Generate a dataset card and Croissant JSON from a built release')
    p_meta.add_argument('--release-root', required=True)
    p_meta.add_argument('--dataset-card-out', required=True)
    p_meta.add_argument('--croissant-template', required=True)
    p_meta.add_argument('--croissant-out', required=True)
    p_meta.add_argument('--dataset-name', default='IGA-Bench Core v1.1')
    # All three of --top-level-url, --content-url, --archive-sha256 must
    # be supplied together. They are nominally optional at argparse
    # level so the cli can emit a single grouped error message in
    # ``cmd_materialize_metadata`` listing every missing flag at once
    # (instead of argparse's default one-at-a-time complaint).
    p_meta.add_argument('--top-level-url',
                        help='Croissant top-level "url" (canonical landing page; e.g. Dataverse Preview URL).')
    p_meta.add_argument('--content-url',
                        help='Croissant distribution[0].contentUrl (direct-bytes URL for the archive; e.g. Hugging Face direct-download).')
    p_meta.add_argument('--archive-sha256',
                        help='SHA-256 hex digest of the hosted dataset archive; written to distribution[0].sha256.')
    p_meta.add_argument('--archive-size', type=int, default=None,
                        help='Size in bytes of the hosted dataset archive; written as distribution[0].contentSize ("N B"). Strongly recommended.')
    p_meta.set_defaults(func=cmd_materialize_metadata)

    p_splits = sub.add_parser('build-prontoqa-splits', help='Build dev/holdout/full_eval splits for ProntoQA')
    p_splits.add_argument('--input-jsonl', required=True)
    p_splits.add_argument('--out-dir', required=True)
    p_splits.add_argument('--target-full-eval-n', type=int, default=500)
    p_splits.add_argument('--dev-n', type=int, default=50)
    p_splits.add_argument('--holdout-n', type=int, default=50)
    p_splits.add_argument('--seed', type=int, default=42)
    p_splits.add_argument('--allow-upsample', action='store_true')
    p_splits.set_defaults(func=cmd_build_prontoqa_splits)

    p_norm_sample = sub.add_parser('normalizer-sample', help='Sample trace steps for normalizer human audit')
    p_norm_sample.add_argument('--trace-steps-jsonl', required=True)
    p_norm_sample.add_argument('--out-jsonl', required=True)
    p_norm_sample.add_argument('--n', type=int, default=200)
    p_norm_sample.add_argument('--seed', type=int, default=42)
    p_norm_sample.set_defaults(func=cmd_normalizer_sample)

    p_norm_eval = sub.add_parser('normalizer-eval', help='Evaluate annotated normalizer audit file')
    p_norm_eval.add_argument('--annotation-jsonl', required=True)
    p_norm_eval.add_argument('--out-json', required=True)
    p_norm_eval.set_defaults(func=cmd_normalizer_eval)

    p_boot_ci = sub.add_parser('bootstrap-ci', help='Compute problem-level bootstrap CI from audit certificates')
    p_boot_ci.add_argument('--certs-parquet-root', required=True)
    p_boot_ci.add_argument('--out-json', required=True)
    p_boot_ci.add_argument('--n-bootstrap', type=int, default=1000)
    p_boot_ci.add_argument('--seed', type=int, default=42)
    p_boot_ci.set_defaults(func=cmd_bootstrap_ci)

    p_report_t = sub.add_parser('report-tables', help='Regenerate paper tables (Table 2, 3, 5, Appendices A/C/D) from a frozen release')
    p_report_t.add_argument('--release-root', required=True, help='Path to the release root directory')
    p_report_t.add_argument('--out', required=True, help='Output directory for table JSON files')
    p_report_t.set_defaults(func=cmd_report_tables)

    p_report_f = sub.add_parser('report-figures', help='Regenerate paper figures (Figure 2 coverage-vs-F1) from a frozen release')
    p_report_f.add_argument('--release-root', required=True, help='Path to the release root directory')
    p_report_f.add_argument('--out', required=True, help='Output directory for figure JSON files')
    p_report_f.set_defaults(func=cmd_report_figures)
    return p


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == '__main__':
    main()
