"""Atomic per-candidate collection, resumable by identity with an explicit budget."""
from __future__ import annotations

import ast
import importlib.metadata
from pathlib import Path
import time

import numpy as np

from slm_steering.datasets import CodingProblem
from slm_steering.final_extension.common import (ROOT, REPORT, RUN, atomic_json, assert_history, canonical,
                                                digest, file_hash, now, read_json, validate_freeze, writer_lock)
from slm_steering.final_extension.generation import ExtensionGenerator
from slm_steering.final_extension.sandbox import SandboxedEvalPlus, probe_sandbox
from slm_steering.research.protocol import attempt_seed
from slm_steering.verifier import build_candidate_source, normalize_completion


def records(directory):
    result = []
    for path in sorted((Path(directory) / 'records').glob('*.json')):
        record = read_json(path)
        if file_hash(Path(directory) / record['activation_path']) != record['artifact_hashes']['activations']:
            raise ValueError(f'Activation integrity failure: {path}')
        if digest(canonical(record['token_ids'])) != record['artifact_hashes']['tokens']:
            raise ValueError('Token integrity failure')
        if digest(record['extracted_code']) != record['artifact_hashes']['code']:
            raise ValueError('Code integrity failure')
        result.append(record)
    identities = {(r['task_id'], r['attempt_id']) for r in result}
    if len(identities) != len(result):
        raise ValueError('Duplicate candidate identity')
    return result


def collect(name, phase, *, session_seconds=3600):
    from slm_steering.final_extension.workflow import charged_hours
    if session_seconds <= 0:
        raise ValueError('Session duration must be positive')
    already_charged = charged_hours()
    if already_charged >= 60:
        raise RuntimeError('Local extension budget exhausted')
    if phase == 'external':
        validate_freeze()
    assert_history()
    inventory = read_json(REPORT / 'task_inventory.json')
    if phase == 'external':
        plan = read_json(REPORT / 'final_design.json')
        ids, attempts, benchmark = plan['external_ids'], 20, 'mbpp'
    elif phase == 'source':
        if not read_json(REPORT / 'pilot_results.json')['feasible']:
            raise ValueError('Pilot viability failed')
        ids, attempts, benchmark = inventory['source_ids'], 20, 'humaneval'
    elif phase in ('pilot', 'technical'):
        ids, attempts, benchmark = inventory['pilot_ids'], 20 if phase == 'pilot' else 1, 'humaneval'
    else:
        raise ValueError('Unknown collection phase')
    directory = RUN / name / phase
    contract = {'name': name, 'phase': phase, 'ids': ids, 'attempts': attempts,
                'generator': file_hash(ROOT / 'src/slm_steering/final_extension/generation.py'),
                'collector': file_hash(Path(__file__)), 'sandbox': file_hash(ROOT / 'src/slm_steering/final_extension/sandbox.py'),
                'worker': file_hash(ROOT / 'scripts/final_extension_worker.py'),
                'historical_worker': file_hash(ROOT / 'scripts/evalplus_worker.py'),
                'metadata': file_hash(REPORT / 'model_metadata.json')}
    with writer_lock(directory):
        manifest_path = directory / 'manifest.json'
        manifest = read_json(manifest_path) if manifest_path.exists() else {
            'created_at': now(), 'contract': contract, 'sessions': [], 'expected_attempts': len(ids) * attempts}
        if manifest['contract'] != contract:
            raise ValueError('Collection contract changed; cannot mix procedures on resume')
        existing = records(directory)
        expected = {(task, j) for task in ids for j in range(attempts)}
        done = {(r['task_id'], r['attempt_id']) for r in existing}
        if done - expected:
            raise ValueError('Unexpected candidate in bank')
        if done == expected:
            return manifest
        start = time.monotonic()
        manifest.update(status='running', completed_attempts=len(existing), last_session_started_at=now())
        atomic_json(manifest_path, manifest)
        new, engine = 0, None
        try:
            sandbox = probe_sandbox()
            hardware_contract = read_json(REPORT / 'hardware.json')
            for package, version in hardware_contract['packages'].items():
                if importlib.metadata.version(package) != version:
                    raise ValueError(f'Runtime library changed: {package}')
            engine = ExtensionGenerator(name)
            if engine.hardware()['device'] != hardware_contract['gpu']:
                raise ValueError('Accessible GPU changed; prospectively remeasure before collection')
            with SandboxedEvalPlus(benchmark) as verifier:
                if manifest.get('benchmark') and manifest['benchmark'] != verifier.dataset['metadata']:
                    raise ValueError('Verifier dataset/runtime changed on resume')
                if phase == 'external' and verifier.dataset['metadata']['dataset_sha256'] != inventory['dataset']['dataset_sha256']:
                    raise ValueError('External dataset changed')
                manifest['benchmark'] = verifier.dataset['metadata']
                tasks = {t['task_id']: t for t in verifier.dataset['tasks']}
                if not manifest.get('verifier_controls'):
                    # Source control only; do not expose a new external task for controls.
                    control = ids[0] if benchmark == 'humaneval' else 'Mbpp/2'
                    positive = verifier.request({'operation': 'verify', 'task_id': control, 'canonical_control': True})
                    negative = verifier.verify(control, f"def {tasks[control]['entry_point']}(*args, **kwargs):\n    raise RuntimeError()\n")
                    if not positive['passed'] or negative['passed']:
                        raise ValueError('EvalPlus positive/negative control failed')
                    manifest['verifier_controls'] = {'positive': positive, 'negative': negative, 'sandbox': sandbox}
                for attempt in range(attempts):
                    for task_id in ids:
                        if (task_id, attempt) in done:
                            continue
                        if already_charged + (time.monotonic() - start) / 3600 >= 60:
                            break
                        if new and time.monotonic() - start >= session_seconds:
                            break
                        task = tasks[task_id]
                        seed = attempt_seed('final-cross-model-v1-' + name, verifier.dataset['metadata']['version'], task_id, attempt)
                        sample = engine.generate(task['prompt'], seed)
                        if not manifest.get('numerical_checks'):
                            reference = engine.generate(task['prompt'], seed, capture_states=False)
                            repeat = engine.generate(task['prompt'], seed)
                            if sample['token_ids'] != repeat['token_ids'] or sample['token_ids'] != reference['token_ids']:
                                raise ValueError('Seed/capture determinism failed')
                            if any(not np.array_equal(a, repeat['activations'][k]) for k, a in sample['activations'].items()):
                                raise ValueError('Activation seed reproduction failed')
                            manifest['numerical_checks'] = engine.numerical_checks(sample)
                            manifest['capture_benchmark'] = {
                                'tokens': sample['generated_tokens'], 'instrumented_seconds': sample['generation_time_seconds'],
                                'uninstrumented_seconds': reference['generation_time_seconds'],
                                'copy_seconds': sample['capture_copy_seconds'],
                                'limitation': 'One technical pair; overhead estimate has substantial timing noise.'}
                        code = build_candidate_source(CodingProblem(task_id=task_id, prompt=task['prompt'],
                                                                   entry_point=task['entry_point'], test=''), sample['raw_completion'])
                        parsing_error = None
                        try:
                            ast.parse(code)
                            if not normalize_completion(sample['raw_completion']).strip():
                                parsing_error = 'empty_extraction'
                        except SyntaxError:
                            parsing_error = 'syntax_error'
                        label = verifier.verify(task_id, code)
                        arrays = sample.pop('activations')
                        if any(not np.isfinite(a).all() for a in arrays.values()):
                            raise ValueError('Nonfinite activation; do not append')
                        stem = f'{digest(task_id)[:16]}_{attempt:02d}'
                        activation_path = directory / 'activations' / f'{stem}.npz'
                        activation_path.parent.mkdir(exist_ok=True)
                        tmp = activation_path.with_suffix('.tmp.npz')
                        np.savez_compressed(tmp, **arrays, layers=np.array(engine.layers))
                        tmp.replace(activation_path)
                        record = {**sample, **label, 'task_id': task_id, 'attempt_id': attempt, 'seed': seed,
                                  'protocol_version': 'final-cross-model-v1', 'model': name,
                                  'model_id': engine.config['model']['id'], 'model_revision': engine.config['model']['revision'],
                                  'tokenizer_revision': engine.config['model']['revision'], 'prompt_template_hash': engine.template_hash,
                                  'prompt': task['prompt'], 'entry_point': task['entry_point'], 'extracted_code': code,
                                  'benchmark_version': verifier.dataset['metadata']['version'],
                                  'label_definition': 'evalplus_base_and_plus_pass', 'parsing_error': parsing_error,
                                  'activation_path': activation_path.relative_to(directory).as_posix(),
                                  'created_at': now(), 'contract_sha256': digest(canonical(contract)),
                                  'artifact_hashes': {'activations': file_hash(activation_path), 'code': digest(code),
                                                      'tokens': digest(canonical(sample['token_ids']))}}
                        # A single atomic JSON is the commit marker; orphan .npz files
                        # after interruption are harmless and deterministically replaced.
                        atomic_json(directory / 'records' / f'{stem}.json', record)
                        done.add((task_id, attempt))
                        new += 1
                        manifest.update(status='running', completed_attempts=len(done), updated_at=now(), hardware=engine.hardware())
                        atomic_json(manifest_path, manifest)
                        print(f'{name}/{phase} {len(done)}/{len(expected)} {task_id} attempt={attempt} '
                              f'tokens={record["generated_tokens"]} pass={record["passed"]} '
                              f'seconds={record["generation_time_seconds"]:.2f}', flush=True)
                    else:
                        continue
                    break
            manifest['status'] = 'completed' if done == expected else 'partial'
        except BaseException as exc:
            manifest.update(status='failed', failure={'type': type(exc).__name__, 'message': str(exc), 'at': now()})
            raise
        finally:
            manifest['sessions'].append({'ended_at': now(), 'new_attempts': new, 'wall_seconds': time.monotonic() - start,
                                         'hardware': engine.hardware() if engine else None})
            atomic_json(manifest_path, manifest)
        assert_history()
        return manifest


def summarize_pilot():
    ds = records(RUN / 'deepseek/pilot')
    qw = records(RUN / 'qwen/technical')
    ids = read_json(REPORT / 'task_inventory.json')['pilot_ids']
    if len(ds) != len(ids) * 20 or len(qw) != len(ids):
        raise ValueError('Pilot/technical sample incomplete; do not extrapolate incomplete task outcomes')
    counts = [sum(r['passed'] for r in ds if r['task_id'] == task) for task in ids]
    mixed = sum(0 < c < 20 for c in counts)
    result = {'tasks': len(ids), 'attempts': len(ds), 'all_success_tasks': counts.count(20),
              'all_failure_tasks': counts.count(0), 'mixed_tasks': mixed,
              'success_rate': float(np.mean([r['passed'] for r in ds])),
              'mean_tokens': float(np.mean([r['generated_tokens'] for r in ds])),
              'truncation_rate': float(np.mean([r['truncated'] for r in ds])),
              'parsing_error_rate': float(np.mean([r['parsing_error'] is not None for r in ds])),
              'mean_seconds_deepseek': float(np.mean([r['generation_time_seconds'] for r in ds])),
              'mean_seconds_qwen': float(np.mean([r['generation_time_seconds'] for r in qw])),
              'tokens_per_second_deepseek': sum(r['generated_tokens'] for r in ds) / sum(r['generation_time_seconds'] for r in ds),
              'mean_activation_copy_seconds': float(np.mean([r['capture_copy_seconds'] for r in ds])),
              'feasible': bool(mixed >= 4 and np.mean([r['passed'] for r in ds]) >= .05
                               and np.mean([r['truncated'] for r in ds]) <= .25
                               and np.mean([r['parsing_error'] is not None for r in ds]) <= .15)}
    atomic_json(REPORT / 'pilot_results.json', result)
    return result


def load_features(name, phase='source'):
    from slm_steering.research.features import observable_features
    if name == 'qwen' and phase == 'source':
        from slm_steering.excellence.audit import checkpoint_data
        data = checkpoint_data(ROOT)
        # Historical block index 20 is the predeclared 75% relative depth.
        data['hidden'] = data['hidden'][:, data['layers'].index(20)]
        data['contract'] = {'source_bank': file_hash(ROOT / 'runs/candidate_bank/definitive_v0.2/trajectories.jsonl'),
                            'layer': 20, 'checkpoint': 32}
        return data
    directory = RUN / name / phase
    selected = [r for r in records(directory) if r['generated_tokens'] > 32]
    hidden = []
    for r in selected:
        with np.load(directory / r['activation_path'], allow_pickle=False) as a:
            hidden.append(a['32'][0].astype(float))
    return {'records': selected, 'hidden': np.array(hidden),
            'observable': np.array([observable_features(r, 32) for r in selected]),
            'y': np.array([int(r['passed']) for r in selected]),
            'groups': np.array([r['task_id'] for r in selected]),
            'identities': [(r['task_id'], r['attempt_id']) for r in selected],
            'contract': {'manifest': file_hash(directory / 'manifest.json'), 'checkpoint': 32}}
