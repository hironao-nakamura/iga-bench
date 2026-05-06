# iga-suite — IGA-Bench reference implementation

`iga-suite` is the Python library and CLI that backs the IGA-Bench
paper.  It implements the validator, canonicalizer, auditor, dependency
metrics, RAWR computation, and bootstrap CIs that produce every numeric
result reported in the paper.

## Install

```bash
unzip iga-suite-code.zip
cd code/
python -m pip install -e .
```

This installs the `iga-suite` console script.

## Smoke test (no API keys, < 1 minute)

The validator takes an *extracted* dataset root, so unzip the dataset
ZIP that ships in the same submission first.  Replace
`<path-to-dataset-zip>` with the location of
`IGA-Bench-Core-v1.1-dataset.zip` (e.g. the sibling `dataset/`
directory inside the submission bundle):

```bash
pytest -q                                          # functional tests, all offline
bash scripts/evaluation/run_mock_matrix.sh tiny    # end-to-end mock evaluator
unzip -q <path-to-dataset-zip> -d /tmp/iga-bench-data
iga-suite validate-core \
    --dataset-root /tmp/iga-bench-data/iga-bench-core-v1.1
```

## Full matrix reproduction

To reproduce the matrix used in the paper:

```bash
# 1. Set provider keys (the matrix calls live OpenAI/Anthropic/OpenRouter):
export OPENAI_API_KEY=...
export ANTHROPIC_API_KEY=...
export OPENROUTER_API_KEY=...

# 2. Run the matrix (configs reference `configs/evaluation/*.yaml`):
bash scripts/evaluation/run_full_matrix_parallel.sh

# 3. Render the paper tables/figures from the materialised release tree:
iga-suite report-tables  --release-root <release-root> --out tables_out/
iga-suite report-figures --release-root <release-root> --out figures_out/
```

## Reproducing the dataset card and Croissant

The dataset card and Croissant record that ship in this submission are
both regenerable from the released dataset bytes alone.  After
extracting `IGA-Bench-Core-v1.1-dataset.zip` to `<release-root>`:

```bash
iga-suite materialize-metadata \
    --release-root         <release-root> \
    --croissant-template   templates/croissant_template.json \
    --dataset-card-out     <release-root>/IGA-Bench-Core-v1.1.dataset-card.md \
    --croissant-out        iga-bench-core-v1.1.croissant.json \
    --top-level-url        "https://dataverse.harvard.edu/previewurl.xhtml?token=89486fac-ef5a-42ba-8201-17a389a699af" \
    --content-url          "https://huggingface.co/datasets/anon-eval3/iga-bench-core-v1.1/resolve/main/IGA-Bench-Core-v1.1-dataset.zip" \
    --archive-sha256       <archive-sha256> \
    --archive-size         <archive-size>
```

Both outputs are deterministic functions of the release bytes plus the
four hosted-archive identity fields (`top-level-url`, `content-url`,
`archive-sha256`, `archive-size`); supplying the same inputs produces
the byte-identical Croissant that ships at
`croissant/iga-bench-core-v1.1.croissant.json` in the submission
bundle.  The per-FileObject `sha256` and `contentSize` for every
in-archive parquet are recomputed from the actual file bytes on every
invocation.

## What the paper draws from this codebase

- **`src/iga_suite/**/*.py`** — iga_suite library: validator, canonicalizer, auditor, metrics, RAWR, report tables.
- **`configs/evaluation/*.yaml`** — Per-(benchmark, model) evaluation configs that reproduce the matrix.
- **`configs/models/*.yaml`** — Provider-side model configs referenced by the evaluation configs.
- **`configs/quickcheck_*.yaml`** — Three-problem quickcheck configs to smoke-test the evaluator.
- **`schema/parquet_schema_contract.yaml`** — Canonical parquet schema contract enforced by `iga-suite validate-core`.
- **`scripts/analysis/*.py`** — Analysis utilities (parser comparison, bootstrap CIs, reprocess).
- **`scripts/analysis/*.sh`** — Analysis driver scripts (paper-table rendering).
- **`scripts/evaluation/*.py`** — Evaluation utilities (proofwriter prep, parser sensitivity).
- **`scripts/evaluation/run_mock_matrix.sh`** — Documented no-API-key smoke flow that exercises the evaluator end-to-end.
- **`scripts/evaluation/run_full_matrix_parallel.sh`** — Driver script that reproduces the matrix at the (benchmark, model) granularity.
- **`scripts/evaluation/build_parser_sensitivity.sh`** — Driver for the canonicalizer-sensitivity reprocess used in Appendix E.
- **`splits/prontoqa_*.jsonl`** — Frozen ProntoQA splits (full_eval, dev, holdout) used by the matrix.
- **`splits/split_report.json`** — Split-construction provenance report.
- **`examples/current_release/*.jsonl`** — ProofWriter-300 release-aligned examples (single source of truth for the v1.1 ProofWriter slice).
- **`examples/tiny/*.jsonl`** — Three-problem tiny fixtures used by `run_mock_matrix.sh`.
- **`templates/croissant_template.json`** — Croissant template used by `iga-suite materialize-metadata`.
- **`pyproject.toml`** — Library packaging metadata; declares the `iga-suite` console script.
- **`LICENSE`** — License under which the code is released.
- **`README.md`** — Reviewer-facing entry point: install, smoke run, full matrix, paper tables.

## Project layout

```
code/
├── src/iga_suite/        # importable library
│   └── providers/        # provider adapters (OpenAI, Anthropic, mock)
├── configs/              # evaluation/model configs
├── scripts/
│   ├── analysis/         # bootstrap CIs, parser comparison, table rendering
│   └── evaluation/       # matrix drivers, parser-sensitivity rebuild
├── splits/               # frozen evaluation splits
├── examples/             # tiny + release-aligned example records
├── tests/                # functional test suite
├── schema/               # parquet schema contract
└── templates/            # Croissant template
```

## License

See `LICENSE` for terms.
