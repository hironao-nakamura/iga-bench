#!/usr/bin/env python3
"""Deterministic reprocessing of existing raw traces with parser v1.0 (conservative).

Mirror of ``scripts/analysis/reprocess_v11.py``. Because the default
canonicalizer (``iga_suite.normalizer``) is now the released v1.1
parser, this script monkey-patches the default normalizer back to the
legacy v1.0 surface in ``iga_suite.normalizer_v10`` for this run only.
Outputs are written back to the original (non-_v11) output roots so
that downstream tooling (``build_parser_sensitivity.sh``) picks up the
refreshed conservative aggregate_metrics (including
predicate_determining mode).

Usage::

    python scripts/analysis/reprocess_v10.py configs/evaluation/<config>.yaml
    python scripts/analysis/reprocess_v10.py --all   # reprocess prontoqa_full_eval_* and proofwriter300_*
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

# ── Monkey-patch normalizer to v1.0 BEFORE importing pipeline ──────────
# (the default `iga_suite.normalizer` now re-exports v1.1 — for parser
# sensitivity we need to restore v1.0 explicitly)
import iga_suite.normalizer as _normalizer_mod
import iga_suite.normalizer_v10 as _v10

_normalizer_mod.normalize_step_text = _v10.normalize_step_text
_normalizer_mod.parse_trace_steps = _v10.parse_trace_steps
_normalizer_mod.parse_final_answer = _v10.parse_final_answer

import iga_suite.aligner as _aligner_mod
_aligner_mod.normalize_step_text = _v10.normalize_step_text

from iga_suite.config import load_config
from iga_suite.pipeline import run_evaluation
from iga_suite.providers.base import BaseProvider, ProviderResponse
from iga_suite.probes import generate_probes


class ReplayProvider(BaseProvider):
    """Reads saved model responses from companion JSON files."""

    def __init__(self, companion_root: Path, benchmark_id: str,
                 model_family: str, provider_name: str, model_name: str,
                 probe_types: list[str], null_probe_enabled: bool):
        self.companion_root = companion_root
        self.benchmark_id = benchmark_id
        self.model_family = model_family
        self.provider_name = provider_name
        self.model_name = model_name
        self.temperature = 0.0
        self.probe_types = probe_types
        self.null_probe_enabled = null_probe_enabled
        self._queues: dict[str, list[str]] = {}
        self._idx: dict[str, int] = {}

    def _build_queue(self, problem: dict) -> list[str]:
        pid = problem["problem_id"]
        raw_dir = (self.companion_root / "raw" / self.benchmark_id
                   / self.model_family / pid)

        responses: list[str] = []
        orig = raw_dir / "original.json"
        if orig.exists():
            responses.append(json.loads(orig.read_text())["response"])
        else:
            responses.append("")

        probes = [
            p for p in generate_probes(problem, include_null_probe=self.null_probe_enabled)
            if p["probe_type"] in self.probe_types
        ]
        for probe in probes:
            fname = f"{probe['probe_type']}__{probe['target_premise']}.json"
            fpath = raw_dir / fname
            if fpath.exists():
                responses.append(json.loads(fpath.read_text())["response"])
            else:
                responses.append("")
        return responses

    def run(self, prompt: str, *, problem: dict | None = None,
            premises: list[dict] | None = None,
            question: str | None = None,
            temperature_override: float | None = None) -> ProviderResponse:
        pid = problem["problem_id"]
        if pid not in self._queues:
            self._queues[pid] = self._build_queue(problem)
            self._idx[pid] = 0

        idx = self._idx[pid]
        self._idx[pid] += 1

        raw = self._queues[pid][idx] if idx < len(self._queues[pid]) else ""
        return ProviderResponse(
            raw_response=raw,
            provider=self.provider_name,
            model_name=self.model_name,
            temperature=0.0,
            token_usage=None,
        )


def reprocess_config(config_path: Path):
    config = load_config(config_path)

    v10_output = Path(config.output_root)
    # Write back to the same v1.0 root (the config already points there).
    original_companion = Path(config.companion_root)
    if not original_companion.exists():
        for d in (ROOT / "outputs" / "eval_runs").iterdir():
            if d.name.endswith("_companion") and v10_output.name in d.name and "_v11" not in d.name:
                original_companion = d
                break

    print(f"{'=' * 70}")
    print(f"Reprocessing (v1.0 conservative): {config_path.name}")
    print(f"  Output:    {v10_output}")
    print(f"  Companion: {original_companion}")
    print(f"  Parser:    v{_v10.PARSER_VERSION} (monkey-patched from v1.1 default)")
    print(f"{'=' * 70}")

    provider = ReplayProvider(
        companion_root=original_companion,
        benchmark_id=config.benchmark.benchmark_id,
        model_family=config.model.model_family,
        provider_name=config.model.provider_name,
        model_name=config.model.model_name,
        probe_types=config.run.probe_types,
        null_probe_enabled=config.run.null_probe_enabled,
    )

    run_evaluation(config, provider_override=provider)
    print(f"\nDone: {v10_output}")
    return v10_output


def main():
    parser = argparse.ArgumentParser(description="Reprocess traces with parser v1.0 (conservative)")
    parser.add_argument("configs", nargs="*", help="Config YAML paths")
    parser.add_argument("--all", action="store_true",
                        help="Reprocess all prontoqa_full_eval_* and proofwriter300_* configs")
    args = parser.parse_args()

    if args.all:
        config_dir = ROOT / "configs" / "evaluation"
        configs = sorted(config_dir.glob("prontoqa_full_eval_*.yaml")) + \
                  sorted(config_dir.glob("proofwriter300_*.yaml"))
        configs = [c for c in configs
                   if "mock" not in c.name
                   and "gemini" not in c.name]
    else:
        configs = [Path(c) for c in args.configs]

    if not configs:
        print("No configs specified. Use --all or pass config paths.")
        sys.exit(1)

    print(f"Will reprocess {len(configs)} configs with parser v1.0 (conservative)")
    outputs = []
    for cfg in configs:
        out = reprocess_config(cfg)
        outputs.append(out)

    print(f"\n{'=' * 70}")
    print(f"All done. {len(outputs)} configs reprocessed.")
    for o in outputs:
        print(f"  {o}")


if __name__ == "__main__":
    main()
