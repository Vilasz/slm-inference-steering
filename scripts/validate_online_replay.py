from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

# Load CUDA DLLs before numpy/scipy on Windows.
if all(flag not in sys.argv for flag in ('--help', '-h')):
    import torch  # noqa: F401
import joblib
import numpy as np

from slm_steering.datasets import CodingProblem
from slm_steering.generation import render_code_prompt
from slm_steering.research.evaluation import EvalPlusClient
from slm_steering.research.features import observable_features
from slm_steering.research.generation import PrefixGenerator
from slm_steering.research.protocol import file_hash, load_config, now, read_bank, write_json
from slm_steering.verifier import build_candidate_source


def main():
    parser = argparse.ArgumentParser(description='Gate 5: reproduce early-abort replay on real candidate seeds')
    parser.add_argument('--bank-dir', type=Path, required=True)
    parser.add_argument('--probes-dir', type=Path, default=ROOT / 'runs/probes/protocol_v0.2')
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/protocol_v0.2.yaml')
    parser.add_argument('--wsl-python')
    parser.add_argument('--max-tasks', type=int, default=1)
    args = parser.parse_args()
    config = load_config(args.config)
    manifest = json.loads((args.bank_dir / 'manifest.json').read_text(encoding='utf-8'))
    probe_manifest = json.loads((args.probes_dir / 'manifest.json').read_text(encoding='utf-8'))
    if not manifest.get('bank_frozen') or probe_manifest['status'] != 'completed':
        raise ValueError('Online validation requires a frozen bank and completed grouped probes')
    if manifest['config_hash'] != file_hash(args.config) or manifest['bank_sha256'] != probe_manifest['bank_sha256']:
        raise ValueError('Bank/probe/config provenance differs')
    records = read_bank(args.bank_dir / 'trajectories.jsonl')
    audit = json.loads((args.probes_dir / 'selection_audit.json').read_text(encoding='utf-8'))
    offline = json.loads((args.probes_dir / 'policy_replay.json').read_text(encoding='utf-8'))
    # Deterministic task order, never select examples based on whether replay saved tokens.
    tasks = sorted({r['task_id'] for r in records})[:args.max_tasks]
    generator = PrefixGenerator(config)
    if generator.template_hash != manifest['prompt_template_hash']:
        raise ValueError('Online prompt/stopping template differs from the stored bank')
    validations = []
    first = next(record for record in records if record['task_id'] == tasks[0])
    warmup = generator.generate(first['prompt'], first['seed'], first['decoding_config']['temperature'])
    with EvalPlusClient(config['benchmark']['name'], wsl_python=args.wsl_python) as verifier:
        for task_index, task_id in enumerate(tasks):
            attempts = sorted([r for r in records if r['task_id'] == task_id], key=lambda r: r['attempt_id'])
            families = ['verifier', 'B', 'HB']
            rotation = task_index % len(families)
            for family in families[rotation:] + families[:rotation]:
                expected = next(row for row in offline if row['task_id'] == task_id and row['budget'] == 20 and row['policy'] == family)
                fold = expected['fold']
                saved, t, q = None, 32, 0.0
                if family != 'verifier':
                    selection = next(row for row in audit if row.get('fold') == fold and row.get('policy') == family)
                    t, q = selection['checkpoint'], selection['threshold']
                    saved = joblib.load(args.probes_dir / 'models' / f'fold{fold}_{family}_t{t}.joblib')
                    if task_id in saved['training_tasks']:
                        raise ValueError('Online example was used to train its predictor')
                actual_tokens, fully_generated, resolved, identities = 0, 0, False, []
                seconds_generation, seconds_verification, prompt_tokens = 0.0, 0.0, 0
                score_seconds, abort_count = 0.0, 0
                for stored in attempts:
                    def abort(offset, captures, logprobs, entropies, margins):
                        nonlocal score_seconds
                        if offset != t:
                            return False
                        started = time.perf_counter()
                        prefix = {**stored, 'token_logprobs': logprobs, 'token_entropies': entropies,
                                  'token_margins': margins, 'generated_tokens': offset}
                        observable = observable_features(prefix, t)
                        features = observable if family == 'B' else np.concatenate((captures[saved['layer']], observable))
                        raw = saved['model'].decision_function(features.reshape(1, -1))
                        probability = saved['calibrator'].predict_proba(raw.reshape(-1, 1))[0, 1]
                        score_seconds += time.perf_counter() - started
                        return probability < q
                    sample = generator.generate(stored['prompt'], stored['seed'], stored['decoding_config']['temperature'],
                                                abort=None if family == 'verifier' else abort)
                    abort_count += sample['finish_reason'] == 'abort'
                    if sample['token_ids'] != stored['token_ids'][:sample['generated_tokens']]:
                        raise ValueError('Online token prefix differs from independently seeded stored attempt')
                    actual_tokens += sample['generated_tokens']
                    prompt_tokens += len(generator.tokenizer.encode(render_code_prompt(generator.tokenizer, stored['prompt'])))
                    seconds_generation += sample['generation_time_seconds']
                    identities.append({'task_id': task_id, 'attempt_id': stored['attempt_id'], 'seed': stored['seed']})
                    if sample['finish_reason'] != 'abort':
                        fully_generated += 1
                        problem = CodingProblem(task_id=task_id, prompt=stored['prompt'], entry_point=next(
                            p['entry_point'] for p in verifier.dataset['tasks'] if p['task_id'] == task_id), test='')
                        verified = verifier.verify(task_id, build_candidate_source(problem, sample['raw_completion']))
                        seconds_verification += verified['verification_time_seconds']
                        resolved = verified['passed']
                        if resolved:
                            break
                actual = {'completion_tokens': actual_tokens, 'fully_generated_attempts': fully_generated,
                          'resolved': int(resolved), 'attempt_identities': identities}
                for key, value in actual.items():
                    if value != expected[key]:
                        raise ValueError(f'Offline/online mismatch for {task_id} {family}: {key}')
                validations.append({'task_id': task_id, 'policy': family, 'passed': True, **actual,
                                    'generation_seconds': seconds_generation, 'verification_seconds': seconds_verification,
                                    'score_seconds_included_in_generation': score_seconds, 'aborted_attempts': abort_count,
                                    'total_seconds': seconds_generation + seconds_verification,
                                    'prompt_prefill_tokens': prompt_tokens})
    write_json(args.probes_dir / 'online_validation.json', {'passed': True, 'created_at': now(),
               'predictions_sha256': probe_manifest['predictions_sha256'], 'bank_sha256': manifest['bank_sha256'],
               'validations': validations, 'hardware': generator.hardware(),
               'has_actual_abort': any(row['aborted_attempts'] for row in validations),
               'tasks': len(tasks), 'task_selection': 'Lexicographic task IDs; no outcome filtering',
               'warmup_excluded_seconds': warmup['generation_time_seconds'],
               'policy_order': 'Deterministic rotation across tasks',
               'timing_scope': 'Descriptive paired instrumented execution; score included, all families use the same capture implementation',
               'smoke_only': len(tasks) == 1})
    print('Online replay validated against the verifier baseline and both adaptive policies', flush=True)


if __name__ == '__main__':
    main()
