from __future__ import annotations

from pathlib import Path
from collections import defaultdict
import json
import re
import time

from iga_suite.config import AppConfig
from iga_suite.prompts import build_prompt
from iga_suite.providers import build_provider
from iga_suite.benchmarks import load_benchmark
from iga_suite.probes import generate_probes
from iga_suite.normalizer import parse_trace_steps, parse_final_answer, normalize_step_text
from iga_suite.aligner import (
    align_steps_by_reference_ranks,
    build_original_alignment_reference,
    premise_canonical_forms,
)
from iga_suite.dependency import direct_premise_dependencies, transitive_premise_dependencies, predicate_determining_dependencies
from iga_suite.auditor import judge_verdict, finalize_verdict
from iga_suite.citation_detector import detects_explicit_premise_citation
from iga_suite.metrics import summarize_problem, aggregate
from iga_suite.metrics import compute_scope_metrics
from iga_suite.hashing import sha256_json, sha256_text, sha256_file
from iga_suite.schema_registry import SchemaRegistry
from iga_suite.table_store import TableStore
from iga_suite.validator import write_validation_report
from iga_suite.baselines import run_baseline_suite
from iga_suite.providers.base import ProviderResponse


_LEGACY_SUBSET_TOKENS: tuple[str, ...] = (
    "smoke", "pilot", "fix", "holdout", "dev",
)
_LEGACY_SUBSET_RE_PARTS = [
    re.escape(t) + (r"\d*" if i == 2 else "")
    for i, t in enumerate(_LEGACY_SUBSET_TOKENS)
]
_LEGACY_SUBSET_RE = re.compile(
    r"(?i)\b(" + "|".join(_LEGACY_SUBSET_RE_PARTS) + r")\b"
)


def _write_json(base: Path, rel: str, payload: dict) -> str:
    path = base / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')
    return str(path.relative_to(base))


def _source_hash(problem: dict) -> str:
    return sha256_json(problem)


def _problem_note(problem: dict) -> str | None:
    return problem.get('metadata', {}).get('difficulty_bin')


def _sanitize_model_notes(note: str | None) -> str | None:
    if not note:
        return note
    note = re.sub(r'\b[A-Z][A-Z0-9_]*API_KEY\b', 'API_KEY', note)
    note = re.sub(r'\b[A-Z][A-Z0-9_]*BASE_URL\b', 'BASE_URL', note)
    return note


def _sanitize_identifier(value: str | None) -> str | None:
    if value is None:
        return None
    out = value
    out = re.sub(r'(?i)phase[-_ ]*\d+', '', out)
    out = _LEGACY_SUBSET_RE.sub('', out)
    out = re.sub(r'_+', '_', out)
    out = re.sub(r'-+', '-', out)
    out = out.strip('_- ')
    return out or value


def _sanitize_note_text(note: str | None) -> str | None:
    if note is None:
        return None
    out = note
    out = re.sub(r'(?i)phase[-_ ]*\d+', '', out)
    out = _LEGACY_SUBSET_RE.sub('', out)
    out = re.sub(r'\s+', ' ', out).strip(' -')
    return out


def _gold_maps(problem: dict) -> dict[str, dict[tuple[int, str], bool]]:
    return {
        'direct': direct_premise_dependencies(problem),
        'transitive': transitive_premise_dependencies(problem),
        'predicate_determining': predicate_determining_dependencies(problem),
    }


def _canonical_type_from_form(canonical_form: str | None) -> str | None:
    if not canonical_form or '(' not in canonical_form:
        return None
    return canonical_form.split('(', 1)[0]


def _parse_semantic_target_ref(semantic_target_ref: str | None) -> tuple[str, str] | None:
    if not semantic_target_ref or '->' not in semantic_target_ref:
        return None
    left, right = semantic_target_ref.split('->', 1)
    left = left.strip()
    right = right.strip()
    if not left or not right:
        return None
    return left, right


def _expected_form_map_for_probe(proof_tree: list[dict], probe: dict) -> dict[int, str]:
    ptype = probe.get('probe_type')
    mapping = _parse_semantic_target_ref(probe.get('semantic_target_ref'))
    out: dict[int, str] = {}
    for entry in proof_tree:
        sid = int(entry['step'])
        c = str(entry.get('conclusion') or '')
        if ptype in {'semantic_substitution', 'local_substitution'} and mapping is not None:
            src, dst = mapping
            c = c.replace(f"({src},", f"({dst},").replace(f", {src})", f", {dst})").replace(f",{src})", f",{dst})")
        out[sid] = c
    return out


def _record_trace_steps(store: TableStore, *, run_id: str, steps: list[dict], raw_ref_prefix: str | None) -> list[str]:
    trace_step_ids = []
    for s in steps:
        trace_step_id = f"{run_id}::step::{s['step_index']}"
        trace_step_ids.append(trace_step_id)
        store.add('trace_steps', {
            'trace_step_id': trace_step_id,
            'run_id': run_id,
            'step_index': s['step_index'],
            'raw_step_ref': f"{raw_ref_prefix}#step={s['step_index']}" if raw_ref_prefix else None,
            'raw_step_sha256': sha256_text(s['raw_text']),
            'parse_status': s['parse_status'],
            'canonical_form': s['canonical_form'],
            'canonical_type': s['canonical_type'],
            'emits_break_token': s['emits_break_token'],
            'break_token_type': s['break_token_type'],
            'notes': None,
        })
    return trace_step_ids


def _mean(xs: list[float]) -> float | None:
    return round(sum(xs) / len(xs), 6) if xs else None


def _write_progress_heartbeat(
    output_root: Path,
    *,
    benchmark_id: str,
    model_id: str,
    config_id: str,
    total_problems: int,
    completed_problems: int,
    current_problem_id: str | None,
    stage: str,
) -> None:
    payload = {
        'benchmark_id': benchmark_id,
        'model_id': model_id,
        'config_id': config_id,
        'total_problems': int(total_problems),
        'completed_problems': int(completed_problems),
        'current_problem_id': current_problem_id,
        'stage': stage,
        'updated_epoch_s': int(time.time()),
    }
    (output_root / 'progress_heartbeat.json').write_text(
        json.dumps(payload, indent=2),
        encoding='utf-8',
    )


def _safe_provider_call(provider, prompt: str, *, problem: dict, premises: list[dict], question: str) -> tuple[ProviderResponse, str | None]:
    try:
        return provider.run(prompt, problem=problem, premises=premises, question=question), None
    except Exception as e:
        # Keep the pipeline progressing problem-by-problem even when one provider call fails.
        provider_name = getattr(provider, 'provider_name', provider.__class__.__name__)
        model_name = getattr(provider, 'model_name', 'unknown')
        temperature = float(getattr(provider, 'temperature', 0.0) or 0.0)
        return (
            ProviderResponse(
                raw_response='',
                provider=str(provider_name),
                model_name=str(model_name),
                temperature=temperature,
                token_usage=None,
            ),
            f"{type(e).__name__}: {e}",
        )


def _delta(orig_step: dict | None, other_step: dict | None) -> bool | None:
    if orig_step is None or other_step is None:
        return None
    if orig_step.get('parse_status') == 'UNPARSEABLE' or other_step.get('parse_status') == 'UNPARSEABLE':
        return None
    if orig_step.get('canonical_form') is None or other_step.get('canonical_form') is None:
        return None
    return orig_step['canonical_form'] != other_step['canonical_form']


def _step_type_from_step(step: dict | None) -> str | None:
    if not step:
        return None
    t = step.get('canonical_type')
    if t:
        return str(t)
    return _canonical_type_from_form(step.get('canonical_form'))


def _parse_is_like_args(canonical_form: str | None) -> tuple[str, str] | None:
    if not canonical_form:
        return None
    if canonical_form.startswith('is(') and canonical_form.endswith(')'):
        inner = canonical_form[3:-1]
    elif canonical_form.startswith('not_is(') and canonical_form.endswith(')'):
        inner = canonical_form[7:-1]
    else:
        return None
    parts = [p.strip() for p in inner.split(',')]
    if len(parts) != 2:
        return None
    return parts[0], parts[1]


def _infer_fact_child_sid(
    *,
    original_step_lookup: dict[int, dict],
    premise_canonical: str | None,
    premise_type: str | None,
) -> int | None:
    # Trace-derived inference only (no proof-tree dependency labels).
    if premise_type not in {'is', 'not_is'}:
        return None
    premise_args = _parse_is_like_args(premise_canonical)
    if premise_args is None:
        return None
    premise_entity, premise_predicate = premise_args
    candidates: list[tuple[int, bool]] = []
    for sid in sorted(original_step_lookup):
        step = original_step_lookup.get(sid)
        if not step:
            continue
        if _step_type_from_step(step) not in {'is', 'not_is'}:
            continue
        args = _parse_is_like_args(step.get('canonical_form'))
        if args is None:
            continue
        entity, predicate = args
        if entity != premise_entity:
            continue
        is_restatement = (step.get('canonical_form') == premise_canonical)
        if is_restatement:
            continue
        predicate_changed = (predicate != premise_predicate)
        candidates.append((sid, predicate_changed))
    if not candidates:
        return None
    # Prefer first entity-matching non-restatement step where predicate changed.
    changed = [sid for sid, pred_changed in candidates if pred_changed]
    if changed:
        return min(changed)
    return min(sid for sid, _ in candidates)


def _mode_verdict(
    mode: str,
    *,
    sid: int,
    first_local_change_sid: int | None,
    sem_delta: bool | None,
    local_delta: bool | None,
    surface_delta: bool | None,
    null_delta: bool | None,
    parse_ready: bool,
) -> tuple[str, str, str, bool, bool]:
    if not parse_ready:
        return 'UNPARSEABLE', f'R_{mode.upper()}_UNPARSEABLE', 'none', False, False

    use_consistent = bool(sem_delta)
    if mode == 'transitive':
        use_local = bool(local_delta)
    else:
        use_local = bool(local_delta) and (first_local_change_sid == sid)

    grounded = use_consistent or use_local
    control_changed = bool(surface_delta) or bool(null_delta)

    if use_consistent and use_local:
        evidence_mode = 'both'
    elif use_consistent:
        evidence_mode = 'consistent_only'
    elif use_local:
        evidence_mode = 'local_first_change'
    else:
        evidence_mode = 'none'

    if grounded and not control_changed:
        return 'GROUNDED', f'R_{mode.upper()}_GROUNDED', evidence_mode, use_consistent, use_local
    if (not grounded) and (not control_changed):
        return 'INSENSITIVE', f'R_{mode.upper()}_INSENSITIVE', evidence_mode, use_consistent, use_local
    if grounded and control_changed:
        return 'INPUT-SENSITIVE', f'R_{mode.upper()}_INPUT_SENSITIVE', evidence_mode, use_consistent, use_local
    return 'UNSTABLE', f'R_{mode.upper()}_UNSTABLE', evidence_mode, use_consistent, use_local


def _fact_style_verdict(
    *,
    mode: str,
    sid: int,
    inferred_fact_child_sid: int | None,
    parse_ok_orig: bool,
    parse_ok_entity: bool,
    entity_step: dict | None,
    entity_delta: bool | None,
    surface_delta: bool | None,
    null_delta: bool | None,
) -> tuple[str, str, str]:
    control_changed = bool(surface_delta) or bool(null_delta)
    entity_missing = parse_ok_orig and (entity_step is None)
    entity_break = bool(entity_step and str(entity_step.get('canonical_form') or '').startswith('break('))
    same_type = _step_type_from_step(entity_step) in {'is', 'not_is'}
    entity_changed = (entity_delta is True) and same_type and parse_ok_entity
    if mode == 'direct':
        evidence = (inferred_fact_child_sid is not None and sid == inferred_fact_child_sid) and (entity_missing or entity_break or entity_changed)
    else:
        evidence = entity_missing or entity_break or entity_changed

    if entity_missing:
        evidence_mode = 'entity_child_missing'
    elif entity_break:
        evidence_mode = 'entity_break'
    elif entity_changed:
        evidence_mode = 'entity_changed'
    else:
        evidence_mode = 'entity_none'

    if evidence and not control_changed:
        return 'GROUNDED', f'R_{mode.upper()}_ENTITY_GROUNDED', evidence_mode
    if (not evidence) and (not control_changed):
        return 'INSENSITIVE', f'R_{mode.upper()}_ENTITY_INSENSITIVE', evidence_mode
    if evidence and control_changed:
        return 'INPUT-SENSITIVE', f'R_{mode.upper()}_ENTITY_INPUT_SENSITIVE', evidence_mode
    return 'UNSTABLE', f'R_{mode.upper()}_ENTITY_UNSTABLE', evidence_mode


def run_evaluation(config: AppConfig, *, provider_override=None) -> dict:
    provider = provider_override if provider_override is not None else build_provider(config.model)
    problems = load_benchmark(config.benchmark.loader, config.benchmark.input_path)
    if config.run.max_problems is not None:
        problems = problems[: config.run.max_problems]

    schema = SchemaRegistry(config.schema_path)
    store = TableStore(schema)
    output_root = Path(config.output_root)
    companion_root = Path(config.companion_root)
    output_root.mkdir(parents=True, exist_ok=True)
    companion_root.mkdir(parents=True, exist_ok=True)

    benchmark_id = config.benchmark.benchmark_id
    split = config.benchmark.split
    config_id = _sanitize_identifier(config.run.config_id)
    model_id = _sanitize_identifier(config.model.model_id)

    store.add('benchmarks', {
        'benchmark_id': benchmark_id,
        'benchmark_name': config.benchmark.benchmark_name,
        'benchmark_version': config.benchmark.benchmark_version,
        'benchmark_family': config.benchmark.benchmark_family,
        'source_url': config.benchmark.source_url,
        'upstream_license_status': config.benchmark.upstream_license_status,
        'text_in_core_default': config.benchmark.include_benchmark_text_in_core,
        'gold_proof_available': True,
        'supports_extension_modes': config.benchmark.supports_extension_modes,
        'notes': _sanitize_note_text(config.benchmark.notes),
    })
    store.add('models', {
        'model_id': model_id,
        'provider': config.model.provider_name,
        'model_family': config.model.model_family,
        'model_name': config.model.model_name,
        'model_snapshot': config.model.model_snapshot,
        'access_type': config.model.access_type,
        'default_trace_mode': config.model.trace_mode,
        'notes': _sanitize_note_text(_sanitize_model_notes(config.model.notes)),
    })

    all_primary_certificates = []
    all_certificates = []
    api_calls_per_problem = []
    total_problems = len(problems)
    completed_problems = 0

    _write_progress_heartbeat(
        output_root,
        benchmark_id=benchmark_id,
        model_id=model_id,
        config_id=config_id,
        total_problems=total_problems,
        completed_problems=completed_problems,
        current_problem_id=None,
        stage='starting',
    )

    for problem in problems:
        pid = problem['problem_id']
        print(f"[run-eval] start problem {completed_problems + 1}/{total_problems}: {pid}", flush=True)
        _write_progress_heartbeat(
            output_root,
            benchmark_id=benchmark_id,
            model_id=model_id,
            config_id=config_id,
            total_problems=total_problems,
            completed_problems=completed_problems,
            current_problem_id=str(pid),
            stage='running_problem',
        )
        dataset_problem_id = f"{benchmark_id}::{pid}"
        source_ref = problem.get('metadata', {}).get('source_record_ref', pid)
        proof_tree = problem.get('proof_tree', [])

        store.add('problems', {
            'dataset_problem_id': dataset_problem_id,
            'benchmark_id': benchmark_id,
            'benchmark_problem_id': pid,
            'split': split,
            'release_tier': config.benchmark.release_tier,
            'source_record_ref': source_ref,
            'benchmark_text_in_core': config.benchmark.include_benchmark_text_in_core,
            'gold_answer': str(problem.get('answer')) if problem.get('answer') is not None else None,
            'answer_type': 'boolean',
            'num_premises': len(problem['premises']),
            'num_gold_steps': len(proof_tree),
            'max_gold_depth': len(proof_tree) if proof_tree else None,
            'difficulty_bin': problem.get('metadata', {}).get('difficulty_bin'),
            'source_hash': _source_hash(problem),
            'notes': _problem_note(problem),
        })

        proof_graph_id = f"{dataset_problem_id}::proof::main"
        for idx, premise in enumerate(problem['premises'], start=1):
            canonical_form, canonical_type, _ = normalize_step_text(premise['text'])
            store.add('proof_nodes', {
                'proof_node_id': f"{dataset_problem_id}::node::{premise['id']}",
                'dataset_problem_id': dataset_problem_id,
                'proof_graph_id': proof_graph_id,
                'node_role': 'premise',
                'node_index': idx,
                'source_node_ref': premise['id'],
                'canonical_form': canonical_form,
                'canonical_type': canonical_type,
                'extension_group': None,
                'notes': None,
            })
        for step in proof_tree:
            sid = int(step['step'])
            store.add('proof_nodes', {
                'proof_node_id': f"{dataset_problem_id}::node::S{sid}",
                'dataset_problem_id': dataset_problem_id,
                'proof_graph_id': proof_graph_id,
                'node_role': 'intermediate_step',
                'node_index': sid,
                'source_node_ref': f"S{sid}",
                'canonical_form': step['conclusion'],
                'canonical_type': step['conclusion'].split('(')[0] if '(' in step['conclusion'] else 'free_form',
                'extension_group': None,
                'notes': None,
            })
            for dep in step.get('depends_on', []):
                store.add('proof_edges', {
                    'proof_edge_id': f"{dataset_problem_id}::edge::{dep}->S{sid}",
                    'dataset_problem_id': dataset_problem_id,
                    'proof_graph_id': proof_graph_id,
                    'src_proof_node_id': f"{dataset_problem_id}::node::{dep}",
                    'dst_proof_node_id': f"{dataset_problem_id}::node::S{sid}",
                    'edge_role': 'direct_dependency',
                    'notes': None,
                })

        gold_maps = _gold_maps(problem)
        calls_this_problem = 0

        original_prompt = build_prompt(problem['premises'], problem['question'], config.run.prompt_mode)
        original_response, original_error = _safe_provider_call(
            provider,
            original_prompt,
            problem=problem,
            premises=problem['premises'],
            question=problem['question'],
        )
        calls_this_problem += 1
        original_run_id = f"{dataset_problem_id}::{model_id}::orig::r0"
        original_companion_rel = None
        if config.raw_companion:
            original_companion_rel = _write_json(
                companion_root,
                f"raw/{benchmark_id}/{config.model.model_family}/{pid}/original.json",
                {
                    'prompt': original_prompt,
                    'response': original_response.raw_response,
                    'provider': original_response.provider,
                    'model_name': original_response.model_name,
                    'usage': original_response.token_usage,
                },
            )
        original_answer = parse_final_answer(original_response.raw_response)
        original_answer_correct = None
        if original_answer is not None and problem.get('answer') is not None:
            original_answer_correct = (original_answer == ('True' if problem['answer'] else 'False'))

        store.add('runs', {
            'run_id': original_run_id,
            'dataset_problem_id': dataset_problem_id,
            'model_id': model_id,
            'model_family': config.model.model_family,
            'config_id': config_id,
            'trace_mode': config.run.prompt_mode,
            'probe_bundle_id': config.run.probe_bundle_id,
            'repetition_index': 0,
            'source_response_ref': original_companion_rel,
            'raw_output_in_companion': bool(config.raw_companion),
            'final_answer': original_answer,
            'final_answer_correct': original_answer_correct,
            'notes': f"provider_error={original_error[:500]}" if original_error else None,
        })
        original_steps = parse_trace_steps(original_response.raw_response)
        _record_trace_steps(store, run_id=original_run_id, steps=original_steps, raw_ref_prefix=original_companion_rel)
        premise_forms = premise_canonical_forms(problem['premises'])
        original_step_lookup, original_rank_map, _ = build_original_alignment_reference(
            original_steps, proof_tree, premise_forms
        )
        expected_type_by_gold_step = {
            int(entry['step']): _canonical_type_from_form(entry.get('conclusion'))
            for entry in proof_tree
        }

        probes = [p for p in generate_probes(problem, include_null_probe=config.run.null_probe_enabled) if p['probe_type'] in config.run.probe_types]
        probe_lookup = defaultdict(dict)
        problem_primary_certs = []

        for probe in probes:
            probe_id = probe['probe_id']
            probe_store_rel = None
            if config.raw_companion:
                probe_store_rel = _write_json(
                    companion_root,
                    f"probes/{benchmark_id}/{pid}/{probe_id.replace('::', '__')}.json",
                    {
                        'problem_id': pid,
                        'target_premise': probe['target_premise'],
                        'probe_type': probe['probe_type'],
                        'intervention_scope': probe['intervention_scope'],
                        'semantic_target_ref': probe['semantic_target_ref'],
                        'modified_premises': probe['modified_premises'],
                    },
                )
            store.add('probes', {
                'probe_id': probe_id,
                'dataset_problem_id': dataset_problem_id,
                'target_premise_ref': probe['target_premise'],
                'probe_type': probe['probe_type'],
                'intervention_scope': probe['intervention_scope'],
                'render_rule_id': probe['render_rule_id'],
                'rendered_probe_in_companion': bool(config.raw_companion),
                'semantic_target_ref': probe['semantic_target_ref'],
                'notes': probe_store_rel,
            })

            prompt = build_prompt(probe['modified_premises'], problem['question'], config.run.prompt_mode)
            resp, probe_error = _safe_provider_call(
                provider,
                prompt,
                problem=problem,
                premises=probe['modified_premises'],
                question=problem['question'],
            )
            calls_this_problem += 1
            run_id = f"{dataset_problem_id}::{model_id}::{probe['probe_type']}::{probe['target_premise']}::r0"
            run_companion_rel = None
            if config.raw_companion:
                run_companion_rel = _write_json(
                    companion_root,
                    f"raw/{benchmark_id}/{config.model.model_family}/{pid}/{probe['probe_type']}__{probe['target_premise']}.json",
                    {
                        'prompt': prompt,
                        'response': resp.raw_response,
                        'provider': resp.provider,
                        'model_name': resp.model_name,
                        'usage': resp.token_usage,
                        'probe_id': probe_id,
                    },
                )
            final_answer = parse_final_answer(resp.raw_response)
            final_answer_correct = None
            if final_answer is not None and problem.get('answer') is not None:
                final_answer_correct = (final_answer == ('True' if problem['answer'] else 'False'))
            store.add('runs', {
                'run_id': run_id,
                'dataset_problem_id': dataset_problem_id,
                'model_id': model_id,
                'model_family': config.model.model_family,
                'config_id': config_id,
                'trace_mode': config.run.prompt_mode,
                'probe_bundle_id': config.run.probe_bundle_id,
                'repetition_index': 0,
                'source_response_ref': run_companion_rel,
                'raw_output_in_companion': bool(config.raw_companion),
                'final_answer': final_answer,
                'final_answer_correct': final_answer_correct,
                'notes': f"{probe_id} | provider_error={probe_error[:500]}" if probe_error else probe_id,
            })
            steps = parse_trace_steps(resp.raw_response)
            _record_trace_steps(store, run_id=run_id, steps=steps, raw_ref_prefix=run_companion_rel)
            run_premise_forms = premise_canonical_forms(probe['modified_premises'])
            step_lookup_direct, _ = align_steps_by_reference_ranks(
                steps,
                run_premise_forms,
                original_rank_map,
                sorted(int(e['step']) for e in proof_tree),
                expected_type_by_gold_step,
                _expected_form_map_for_probe(proof_tree, probe),
            )
            step_lookup_relaxed, _ = align_steps_by_reference_ranks(
                steps,
                run_premise_forms,
                original_rank_map,
                sorted(int(e['step']) for e in proof_tree),
            )
            expected_step_ids = sorted(int(e['step']) for e in proof_tree)
            for sid in expected_step_ids:
                o = original_step_lookup.get(sid)
                p = step_lookup_direct.get(sid)
                status = 'MATCHED' if (o and p) else ('MISSING' if (o and not p) else ('EXTRA' if (p and not o) else 'MISSING'))
                original_step_id = f"{original_run_id}::step::{o['step_index']}" if o else None
                probed_step_id = f"{run_id}::step::{p['step_index']}" if p else None
                store.add('step_alignments', {
                    'alignment_id': f"{run_id}::align::goldS{sid}",
                    'dataset_problem_id': dataset_problem_id,
                    'original_run_id': original_run_id,
                    'probed_run_id': run_id,
                    'original_trace_step_id': original_step_id,
                    'probed_trace_step_id': probed_step_id,
                    'alignment_status': status,
                    'alignment_method': 'derived_rank_reference',
                    'alignment_confidence': 1.0 if status == 'MATCHED' else 0.0,
                    'notes': None,
                })
            probe_lookup[probe['target_premise']][probe['probe_type']] = {
                'probe': probe,
                'run_id': run_id,
                'steps_direct': step_lookup_direct,
                'steps_relaxed': step_lookup_relaxed,
            }

        # Primary certificates use semantic/local substitutions + surface/null controls.
        for premise in problem['premises']:
            prem_id = premise['id']
            premise_canonical, premise_type, _ = normalize_step_text(premise['text'])
            inferred_fact_child_sid = _infer_fact_child_sid(
                original_step_lookup=original_step_lookup,
                premise_canonical=premise_canonical,
                premise_type=premise_type,
            )
            semantic_pack = probe_lookup.get(prem_id, {}).get('semantic_substitution')
            local_pack = probe_lookup.get(prem_id, {}).get('local_substitution')
            entity_pack = probe_lookup.get(prem_id, {}).get('entity_substitution')
            surface_pack = probe_lookup.get(prem_id, {}).get('surface_control')
            null_pack = probe_lookup.get(prem_id, {}).get('null_probe')
            if semantic_pack is None or local_pack is None or surface_pack is None:
                continue
            semantic_probe_id = semantic_pack['probe']['probe_id']
            semantic_run_id = semantic_pack['run_id']
            semantic_steps_direct = semantic_pack['steps_direct']
            semantic_steps_relaxed = semantic_pack['steps_relaxed']
            local_probe_id = local_pack['probe']['probe_id']
            local_run_id = local_pack['run_id']
            local_steps_direct = local_pack['steps_direct']
            local_steps_relaxed = local_pack['steps_relaxed']
            surface_steps_direct = surface_pack['steps_direct']
            surface_steps_relaxed = surface_pack['steps_relaxed']
            entity_probe_id = entity_pack['probe']['probe_id'] if entity_pack is not None else None
            entity_run_id = entity_pack['run_id'] if entity_pack is not None else None
            entity_steps_direct = entity_pack['steps_direct'] if entity_pack is not None else {}
            entity_steps_relaxed = entity_pack['steps_relaxed'] if entity_pack is not None else {}
            null_steps_direct = null_pack['steps_direct'] if null_pack is not None else {}
            null_steps_relaxed = null_pack['steps_relaxed'] if null_pack is not None else {}
            expected_step_ids = sorted(int(e['step']) for e in proof_tree) if proof_tree else sorted(set(original_step_lookup) | set(semantic_steps_direct) | set(surface_steps_direct))
            all_step_ids = expected_step_ids
            first_local_change_sid = None
            for sid in all_step_ids:
                d = _delta(original_step_lookup.get(sid), local_steps_direct.get(sid))
                if d is True:
                    first_local_change_sid = sid
                    break
            for sid in all_step_ids:
                o = original_step_lookup.get(sid)
                explicit_citation = detects_explicit_premise_citation(o['raw_text'] if o else '', prem_id, premise['text'])
                for mode in config.run.dependency_modes:
                    use_direct_alignment = mode == 'direct'
                    s = semantic_steps_direct.get(sid) if use_direct_alignment else semantic_steps_relaxed.get(sid)
                    l = local_steps_direct.get(sid) if use_direct_alignment else local_steps_relaxed.get(sid)
                    e = entity_steps_direct.get(sid) if use_direct_alignment else entity_steps_relaxed.get(sid)
                    c = surface_steps_direct.get(sid) if use_direct_alignment else surface_steps_relaxed.get(sid)
                    n = (null_steps_direct.get(sid) if use_direct_alignment else null_steps_relaxed.get(sid)) if null_pack is not None else None

                    parse_ok_orig = bool(o and o['parse_status'] != 'UNPARSEABLE')
                    parse_ok_sem = bool(s and s['parse_status'] != 'UNPARSEABLE')
                    parse_ok_local = bool(l and l['parse_status'] != 'UNPARSEABLE')
                    parse_ok_entity = bool(e and e['parse_status'] != 'UNPARSEABLE')
                    parse_ok_surface = bool(c and c['parse_status'] != 'UNPARSEABLE')
                    parse_ok_null = True if null_pack is None else bool(n and n['parse_status'] != 'UNPARSEABLE')
                    parse_ok = bool(parse_ok_orig and parse_ok_surface and parse_ok_null and (parse_ok_sem or parse_ok_local))

                    sem_delta = _delta(o, s)
                    local_delta = _delta(o, l)
                    entity_delta = _delta(o, e)
                    surface_delta = _delta(o, c)
                    null_delta = _delta(o, n) if null_pack is not None else None

                    is_fact_premise = premise_type in {'is', 'not_is'}
                    gold = gold_maps[mode].get((sid, prem_id))
                    if is_fact_premise and entity_pack is not None:
                        parse_ready_fact = bool(parse_ok_orig and parse_ok_surface and parse_ok_null)
                        if parse_ready_fact:
                            base_verdict, verdict_rule, evidence_mode = _fact_style_verdict(
                                mode=mode,
                                sid=sid,
                                inferred_fact_child_sid=inferred_fact_child_sid,
                                parse_ok_orig=parse_ok_orig,
                                parse_ok_entity=parse_ok_entity,
                                entity_step=e,
                                entity_delta=entity_delta,
                                surface_delta=surface_delta,
                                null_delta=null_delta,
                            )
                        else:
                            base_verdict, verdict_rule, evidence_mode = (
                                'UNPARSEABLE',
                                f'R_{mode.upper()}_ENTITY_UNPARSEABLE',
                                'entity_none',
                            )
                        use_consistent = False
                        use_local = False
                    else:
                        mode_sem_delta = sem_delta
                        mode_local_delta = local_delta
                        if mode == 'direct':
                            orig_type = _step_type_from_step(o)
                            sem_type = _step_type_from_step(s)
                            local_type = _step_type_from_step(l)
                            # Hardening: in direct mode, reject evidence if probe step type differs from original.
                            if (orig_type is not None and sem_type is not None and orig_type != sem_type):
                                mode_sem_delta = None
                            if (orig_type is not None and local_type is not None and orig_type != local_type):
                                mode_local_delta = None
                        base_verdict, verdict_rule, evidence_mode, use_consistent, use_local = _mode_verdict(
                            mode,
                            sid=sid,
                            first_local_change_sid=first_local_change_sid,
                            sem_delta=mode_sem_delta,
                            local_delta=mode_local_delta,
                            surface_delta=surface_delta,
                            null_delta=null_delta,
                            parse_ready=parse_ok,
                        )
                    final_verdict = finalize_verdict(
                        base_verdict,
                        step_text=o['raw_text'] if o else '',
                        premise_id=prem_id,
                        premise_text=premise['text'],
                    )
                    selected_probe_id = semantic_probe_id
                    selected_run_id = semantic_run_id
                    selected_step = s
                    selected_probe_type = 'semantic_substitution'
                    evidence_stream = 'rule_style'
                    if is_fact_premise and entity_pack is not None:
                        selected_probe_id = entity_probe_id or semantic_probe_id
                        selected_run_id = entity_run_id or semantic_run_id
                        selected_step = e
                        selected_probe_type = 'entity_substitution'
                        evidence_stream = 'fact_style_entity'
                    elif use_local and not use_consistent:
                        selected_probe_id = local_probe_id
                        selected_run_id = local_run_id
                        selected_step = l
                        selected_probe_type = 'local_substitution'
                    control_changed = None
                    if surface_delta is not None or null_delta is not None:
                        flags = [x for x in (surface_delta, null_delta) if x is not None]
                        control_changed = any(flags) if flags else None
                    cert = {
                        'certificate_id': f"{dataset_problem_id}::{model_id}::cert::{mode}::{prem_id}::S{sid}",
                        'dataset_problem_id': dataset_problem_id,
                        'model_id': model_id,
                        'model_family': config.model.model_family,
                        'config_id': config_id,
                        'dependency_mode_scored': mode,
                        'probe_id': selected_probe_id,
                        'premise_canonical_type': premise_type,
                        'target_premise_ref': prem_id,
                        'original_run_id': original_run_id,
                        'probed_run_id': selected_run_id,
                        'original_trace_step_id': f"{original_run_id}::step::{o['step_index']}" if o else None,
                        'probed_trace_step_id': f"{selected_run_id}::step::{selected_step['step_index']}" if selected_step else None,
                        'probe_type': selected_probe_type,
                        'verdict_type': final_verdict,
                        'explicit_premise_citation': explicit_citation,
                        'semantic_changed': sem_delta,
                        'control_changed': control_changed,
                        'consistent_changed': sem_delta,
                        'local_changed': local_delta,
                        'surface_changed': surface_delta,
                        'null_changed': null_delta,
                        'verdict_rule': verdict_rule,
                        'evidence_mode': evidence_mode,
                        'evidence_stream': evidence_stream,
                        'canonical_consistent': s['canonical_form'] if s else None,
                        'canonical_local': l['canonical_form'] if l else None,
                        'canonical_entity': e['canonical_form'] if e else None,
                        'canonical_surface': c['canonical_form'] if c else None,
                        'canonical_null': n['canonical_form'] if n else None,
                        'parse_ok': parse_ok,
                        'alignment_ok': bool(o and selected_step),
                        'gold_dependency_label': gold,
                        'unresolved_reason': None if final_verdict != 'UNPARSEABLE' else ('missing_step' if not (o and selected_step and c and (n or null_pack is None)) else 'parse_failure_or_control_missing'),
                        'canonical_original': o['canonical_form'] if o else None,
                        'canonical_probed': selected_step['canonical_form'] if selected_step else None,
                        'certificate_sha256': None,
                        'notes': None,
                    }
                    cert['certificate_sha256'] = sha256_json({k: cert[k] for k in cert if k != 'certificate_sha256'})
                    store.add('audit_certificates', cert)
                    problem_primary_certs.append(cert)
                    all_primary_certificates.append(cert)
                    all_certificates.append(cert)
                if config.run.include_extension_scope_predicate_determining:
                    s = semantic_steps_relaxed.get(sid)
                    l = local_steps_relaxed.get(sid)
                    c = surface_steps_relaxed.get(sid)
                    n = null_steps_relaxed.get(sid) if null_pack is not None else None
                    parse_ok_orig = bool(o and o['parse_status'] != 'UNPARSEABLE')
                    parse_ok_sem = bool(s and s['parse_status'] != 'UNPARSEABLE')
                    parse_ok_local = bool(l and l['parse_status'] != 'UNPARSEABLE')
                    parse_ok_surface = bool(c and c['parse_status'] != 'UNPARSEABLE')
                    parse_ok_null = True if null_pack is None else bool(n and n['parse_status'] != 'UNPARSEABLE')
                    parse_ok = bool(parse_ok_orig and parse_ok_surface and parse_ok_null and (parse_ok_sem or parse_ok_local))
                    sem_delta = _delta(o, s)
                    local_delta = _delta(o, l)
                    surface_delta = _delta(o, c)
                    null_delta = _delta(o, n) if null_pack is not None else None
                    gold = gold_maps['predicate_determining'].get((sid, prem_id))
                    base_verdict, verdict_rule, evidence_mode, use_consistent, use_local = _mode_verdict(
                        'predicate_determining',
                        sid=sid,
                        first_local_change_sid=first_local_change_sid,
                        sem_delta=sem_delta,
                        local_delta=local_delta,
                        surface_delta=surface_delta,
                        null_delta=null_delta,
                        parse_ready=parse_ok,
                    )
                    final_verdict = finalize_verdict(
                        base_verdict,
                        step_text=o['raw_text'] if o else '',
                        premise_id=prem_id,
                        premise_text=premise['text'],
                    )
                    selected_probe_id = semantic_probe_id
                    selected_run_id = semantic_run_id
                    selected_step = s
                    selected_probe_type = 'semantic_substitution'
                    if use_local and not use_consistent:
                        selected_probe_id = local_probe_id
                        selected_run_id = local_run_id
                        selected_step = l
                        selected_probe_type = 'local_substitution'
                    control_changed = None
                    if surface_delta is not None or null_delta is not None:
                        flags = [x for x in (surface_delta, null_delta) if x is not None]
                        control_changed = any(flags) if flags else None
                    cert = {
                        'certificate_id': f"{dataset_problem_id}::{model_id}::cert::predicate_determining::{prem_id}::S{sid}",
                        'dataset_problem_id': dataset_problem_id,
                        'model_id': model_id,
                        'model_family': config.model.model_family,
                        'config_id': config_id,
                        'dependency_mode_scored': 'predicate_determining',
                        'probe_id': selected_probe_id,
                        'premise_canonical_type': premise_type,
                        'target_premise_ref': prem_id,
                        'original_run_id': original_run_id,
                        'probed_run_id': selected_run_id,
                        'original_trace_step_id': f"{original_run_id}::step::{o['step_index']}" if o else None,
                        'probed_trace_step_id': f"{selected_run_id}::step::{selected_step['step_index']}" if selected_step else None,
                        'probe_type': selected_probe_type,
                        'verdict_type': final_verdict,
                        'explicit_premise_citation': explicit_citation,
                        'semantic_changed': sem_delta,
                        'control_changed': control_changed,
                        'consistent_changed': sem_delta,
                        'local_changed': local_delta,
                        'surface_changed': surface_delta,
                        'null_changed': null_delta,
                        'verdict_rule': verdict_rule,
                        'evidence_mode': evidence_mode,
                        'evidence_stream': 'rule_style_extension',
                        'canonical_consistent': s['canonical_form'] if s else None,
                        'canonical_local': l['canonical_form'] if l else None,
                        'canonical_entity': None,
                        'canonical_surface': c['canonical_form'] if c else None,
                        'canonical_null': n['canonical_form'] if n else None,
                        'parse_ok': parse_ok,
                        'alignment_ok': bool(o and selected_step),
                        'gold_dependency_label': gold,
                        'unresolved_reason': None if final_verdict != 'UNPARSEABLE' else ('missing_step' if not (o and selected_step and c and (n or null_pack is None)) else 'parse_failure_or_control_missing'),
                        'canonical_original': o['canonical_form'] if o else None,
                        'canonical_probed': selected_step['canonical_form'] if selected_step else None,
                        'certificate_sha256': None,
                        'notes': 'extension_scope=predicate_determining',
                    }
                    cert['certificate_sha256'] = sha256_json({k: cert[k] for k in cert if k != 'certificate_sha256'})
                    store.add('audit_certificates', cert)
                    all_certificates.append(cert)

        problem_summary = summarize_problem(problem_primary_certs, original_answer_correct)
        store.add('run_summaries', {
            'summary_id': f"{dataset_problem_id}::{model_id}::{config_id}",
            'dataset_problem_id': dataset_problem_id,
            'model_id': model_id,
            'model_family': config.model.model_family,
            'config_id': config_id,
            'final_answer_correct': original_answer_correct,
            'rawr_direct': problem_summary['rawr_direct'],
            'rawr_transitive': problem_summary['rawr_transitive'],
            'definitive_coverage': problem_summary['definitive_coverage'],
            'parse_coverage': problem_summary['parse_coverage'],
            'alignment_coverage': problem_summary['alignment_coverage'],
            'num_certificates': problem_summary['num_certificates'],
            'num_unparseable': problem_summary['num_unparseable'],
            'num_misrepresentation': problem_summary['num_misrepresentation'],
            'notes': None,
        })
        api_calls_per_problem.append(float(calls_this_problem))
        completed_problems += 1
        _write_progress_heartbeat(
            output_root,
            benchmark_id=benchmark_id,
            model_id=model_id,
            config_id=config_id,
            total_problems=total_problems,
            completed_problems=completed_problems,
            current_problem_id=str(pid),
            stage='problem_completed',
        )
        print(f"[run-eval] completed problem {completed_problems}/{total_problems}: {pid}", flush=True)

    # Use all_certificates (primary direct+transitive + extension predicate_determining)
    # so aggregate_metrics covers all three dependency modes shown in Appendix C.
    aggregate_rows = aggregate(
        all_certificates,
        problems,
        benchmark_id=benchmark_id,
        split=split,
        model_family=config.model.model_family,
        model_id=model_id,
        config_id=config_id,
        baseline_name='iga',
        voting_k=1,
        mean_api_calls_per_problem=_mean(api_calls_per_problem),
    )
    for row in aggregate_rows:
        row['metric_id'] = f"{row['benchmark_id']}::{row['model_family']}::{row['config_id']}::{row['dependency_mode']}::{row['metric_scope']}"
        store.add('aggregate_metrics', row)

    if config.run.self_consistency.enabled:
        baseline = run_baseline_suite(
            provider,
            problems,
            n_samples=config.run.self_consistency.samples,
            temperature=config.run.self_consistency.temperature,
            prompt_mode=config.run.prompt_mode,
            companion_root=companion_root if config.raw_companion else None,
            benchmark_id=benchmark_id,
            model_family=config.model.model_family,
        )
        for mode in ['direct', 'transitive']:
            br = baseline[mode]
            coverage = 1.0
            mean_num_premises = round(sum(len(p['premises']) for p in problems) / len(problems), 6) if problems else None
            store.add('aggregate_metrics', {
                'metric_id': f"{benchmark_id}::{config.model.model_family}::{config_id}::{mode}::self_consistency",
                'benchmark_id': benchmark_id,
                'split': split,
                'model_family': config.model.model_family,
                'model_id': model_id,
                'config_id': config_id,
                'dependency_mode': mode,
                'metric_scope': 'all',
                'baseline_name': 'self_consistency',
                'voting_k': config.run.self_consistency.samples,
                'num_problems': len(problems),
                'num_certificates': br['tp'] + br['fp'] + br['fn'] + br['tn'],
                'precision': br['precision'],
                'recall': br['recall'],
                'f1': br['f1'],
                'coverage': coverage,
                'coverage_adjusted_f1': br['f1'],
                'lower_bound_f1_all_unresolved_negative': br['f1'],
                'mean_num_premises': mean_num_premises,
                'mean_api_calls_per_problem': float(config.run.self_consistency.samples),
                'estimated_cost_usd_per_problem': None,
                'ci95_low': None,
                'ci95_high': None,
                'notes': 'Generous passive baseline: consistent steps predict all premises as dependencies.',
                'metric_status': 'defined',
            })

    store.write(output_root)
    store.write_jsonl_shadow(output_root / 'jsonl_shadow')
    report = write_validation_report(output_root, output_root / 'validator_report.json')

    manifest = {
        'benchmark_id': benchmark_id,
        'model_id': model_id,
        'config_id': config_id,
        'num_problems': len(problems),
        'output_root': output_root.name,
        'companion_root': companion_root.name,
        'validation_status': report['status'],
    }
    # Write a neutral manifest name for release-facing artifacts.
    (output_root / 'run_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')

    checksums = {}
    for p in sorted(output_root.rglob('*')):
        if p.is_file():
            checksums[str(p.relative_to(output_root))] = sha256_file(p)
    (output_root / 'checksums.json').write_text(json.dumps(checksums, indent=2), encoding='utf-8')

    by_scope = defaultdict(list)
    for cert in all_certificates:
        by_scope[cert['dependency_mode_scored']].append(cert)
    scope_summaries = {scope: compute_scope_metrics(certs) for scope, certs in by_scope.items()}

    primary_scope_only = {
        scope: stats
        for scope, stats in scope_summaries.items()
        if scope in {'direct', 'transitive'}
    }
    extension_scope_separate = {
        'primary': primary_scope_only,
        'extensions': {
            scope: stats
            for scope, stats in scope_summaries.items()
            if scope not in {'direct', 'transitive'}
        },
    }
    (output_root / 'summary_primary_scope_only.json').write_text(
        json.dumps({'primary_scope_only': True, 'scope_summaries': primary_scope_only}, indent=2),
        encoding='utf-8',
    )
    (output_root / 'summary_extension_scope_separate.json').write_text(
        json.dumps({'extension_scope_separate': True, 'scope_summaries': extension_scope_separate}, indent=2),
        encoding='utf-8',
    )
    _write_progress_heartbeat(
        output_root,
        benchmark_id=benchmark_id,
        model_id=model_id,
        config_id=config_id,
        total_problems=total_problems,
        completed_problems=completed_problems,
        current_problem_id=None,
        stage='finished',
    )

    return {
        'output_root': str(output_root),
        'validation': report,
        'num_problems': len(problems),
        'num_certificates': len(all_primary_certificates),
        'num_certificates_all_scopes': len(all_certificates),
        'scope_summaries': scope_summaries,
        'summary_primary_scope_only': True,
        'summary_extension_scope_separate': True,
    }
