from __future__ import annotations

from pathlib import Path
import json

from iga_suite.prompts import build_prompt
from iga_suite.normalizer import parse_trace_steps
from iga_suite.metrics import _prf1
from iga_suite.dependency import direct_premise_dependencies, transitive_premise_dependencies


def run_self_consistency(provider, problem: dict, *, n_samples: int, temperature: float, prompt_mode: str = 'forced_step') -> dict:
    # Temperature is provider-dependent; for now reuse provider as configured and note temperature in metadata.
    prompt = build_prompt(problem['premises'], problem['question'], prompt_mode)
    samples = []
    for _ in range(n_samples):
        resp = provider.run(
            prompt,
            problem=problem,
            premises=problem['premises'],
            question=problem['question'],
            temperature_override=temperature,
        )
        steps = parse_trace_steps(resp.raw_response)
        samples.append({'raw_response': resp.raw_response, 'steps': steps})

    proof_steps = [int(e['step']) for e in problem.get('proof_tree', [])]
    step_verdicts = {}
    for sid in proof_steps:
        forms = []
        for s in samples:
            step = next((x for x in s['steps'] if x['step_index'] == sid and x['canonical_form'] is not None), None)
            if step is not None:
                forms.append(step['canonical_form'])
        consistent = len(forms) > 0 and len(set(forms)) == 1
        step_verdicts[sid] = {'consistent': consistent, 'canonical_forms': forms}
    return {'n_samples': n_samples, 'temperature': temperature, 'step_verdicts': step_verdicts}


def evaluate_baseline(baseline_results: dict, problems: list[dict], gt_type: str = 'direct') -> dict:
    """Convert self-consistency results to P/R/F1.

    Generous baseline:
      - If a step is consistent across samples, predict ALL premises are dependencies.
      - If a step is inconsistent, predict NO premises are dependencies.
    """
    tp = fp = fn = tn = 0
    for problem in problems:
        pid = problem['problem_id']
        br = baseline_results.get(pid)
        if br is None:
            continue
        gt = transitive_premise_dependencies(problem) if gt_type == 'transitive' else direct_premise_dependencies(problem)
        proof_steps = [int(e['step']) for e in problem.get('proof_tree', [])]
        premise_ids = [p['id'] for p in problem['premises']]
        for sid in proof_steps:
            sv = br['step_verdicts'].get(sid) or br['step_verdicts'].get(str(sid))
            for prem_id in premise_ids:
                gold = gt.get((sid, prem_id), False)
                predicted = bool(sv and sv['consistent'])
                if predicted and gold:
                    tp += 1
                elif predicted and not gold:
                    fp += 1
                elif not predicted and gold:
                    fn += 1
                else:
                    tn += 1
    p, r, f1 = _prf1(tp, fp, fn)
    return {'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn, 'precision': round(p, 6), 'recall': round(r, 6), 'f1': round(f1, 6)}


def run_baseline_suite(provider, problems: list[dict], *, n_samples: int, temperature: float, prompt_mode: str, companion_root: str | Path | None = None, benchmark_id: str = '', model_family: str = '') -> dict:
    companion_root = Path(companion_root) if companion_root else None
    baseline_results = {}
    for problem in problems:
        pid = problem['problem_id']
        br = run_self_consistency(provider, problem, n_samples=n_samples, temperature=temperature, prompt_mode=prompt_mode)
        baseline_results[pid] = br
        if companion_root is not None:
            path = companion_root / 'baselines' / 'self_consistency' / benchmark_id / model_family / f'{pid}.json'
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(br, indent=2), encoding='utf-8')
    return {
        'results': baseline_results,
        'direct': evaluate_baseline(baseline_results, problems, gt_type='direct'),
        'transitive': evaluate_baseline(baseline_results, problems, gt_type='transitive'),
    }
