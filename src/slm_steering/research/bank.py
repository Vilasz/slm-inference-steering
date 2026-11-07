from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np

from slm_steering.datasets import CodingProblem
from slm_steering.research.evaluation import EvalPlusClient
from slm_steering.research.generation import PrefixGenerator
from slm_steering.research.protocol import (attempt_seed, canonical, digest, file_hash, load_config, now,
                                          pilot_task_ids, provenance, read_bank, require_freeze, validate_bank, write_json)
from slm_steering.verifier import build_candidate_source, normalize_completion


def select_temperature(records: list[dict], config: dict, ids: list[str]) -> dict:
    grid = config['pilot']['temperatures']
    n = config['pilot']['attempts_per_task']
    rows = []
    complete = True
    for temperature in grid:
        subset = [r for r in records if r['decoding_config']['temperature'] == temperature]
        by_task = {task: [r for r in subset if r['task_id'] == task] for task in ids}
        complete &= all(len(attempts) == n for attempts in by_task.values())
        mixed = sum(any(r['passed'] for r in attempts) and any(not r['passed'] for r in attempts)
                    for attempts in by_task.values())
        invalid = [r['truncated'] or r['error_type'] in ('syntax_error', 'empty_extraction') for r in subset]
        rows.append({'temperature': temperature, 'attempts': len(subset), 'mixed_tasks': mixed,
                     'pass_rate': float(np.mean([r['passed'] for r in subset])) if subset else None,
                     'invalid_truncated_rate': float(np.mean(invalid)) if subset else None,
                     'mean_tokens': float(np.mean([r['generated_tokens'] for r in subset])) if subset else None})
    selected = None
    if complete:
        reference = min(row['invalid_truncated_rate'] for row in rows)
        eligible = [row for row in rows if 0.1 < row['pass_rate'] < 0.9 and
                    row['invalid_truncated_rate'] <= reference + config['pilot']['max_invalid_truncated_excess']]
        if eligible:
            selected = min(eligible, key=lambda r: (-r['mixed_tasks'], abs(r['pass_rate'] - 0.5),
                                                   r['mean_tokens'], r['temperature']))['temperature']
    return {'complete': bool(complete), 'selected_temperature': selected, 'temperatures': rows}


def build_bank(config_path: Path, output: Path, *, phase: str, time_budget: float | None,
               local_files_only: bool, wsl_python: str | None, pilot_manifest_path: Path | None = None,
               max_attempts: int | None = None, transfer_bundle: Path | None = None) -> dict:
    config_path = config_path.resolve()
    config = load_config(config_path)
    if config['protocol_version'] != '0.2':
        raise ValueError('Generation now implements v0.2; preserve v0.1 banks as historical artifacts')
    if phase not in {'pilot', 'definitive', 'transfer'}:
        raise ValueError('Unknown candidate-bank phase')
    if phase in {'definitive', 'transfer'}:
        import json
        if pilot_manifest_path is None:
            raise ValueError('Definitive generation needs --pilot-manifest')
        pilot_manifest = json.loads(pilot_manifest_path.read_text(encoding='utf-8'))
        require_freeze(config_path, pilot_manifest)
    if phase == 'transfer':
        from slm_steering.research.transfer import validate_bundle
        validate_bundle(transfer_bundle, config_path)
    output.mkdir(parents=True, exist_ok=True)
    lock = output / '.writer.lock'
    # Exclusive writer prevents duplicate append if two sessions target the same run.
    with lock.open('x', encoding='utf-8') as stream:
        stream.write(str(os.getpid()))
    try:
        return _build_locked(config_path, output, config=config, phase=phase, time_budget=time_budget,
                             local_files_only=local_files_only, wsl_python=wsl_python,
                             pilot_manifest_path=pilot_manifest_path, max_attempts=max_attempts,
                             transfer_bundle=transfer_bundle)
    except Exception as exc:
        import json
        manifest_path = output / 'manifest.json'
        if manifest_path.exists():
            failed = json.loads(manifest_path.read_text(encoding='utf-8'))
            failed.setdefault('failures', []).append({'at': now(), 'exception': type(exc).__name__, 'message': str(exc)})
            failed['status'] = 'failed'
            write_json(manifest_path, failed)
        raise
    finally:
        lock.unlink(missing_ok=True)


def _build_locked(config_path, output, *, config, phase, time_budget, local_files_only, wsl_python,
                  pilot_manifest_path, max_attempts, transfer_bundle=None):
    import json
    bank_path, manifest_path = output / 'trajectories.jsonl', output / 'manifest.json'
    records = read_bank(bank_path)
    manifest = provenance(config_path)
    session_provenance = dict(manifest)
    manifest.update({'protocol_version': config['protocol_version'], 'run_id': output.name, 'phase': phase,
                     'status': 'running', 'config': config, 'sessions': []})
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding='utf-8'))
        for key in ('config_hash', 'protocol_hash', 'phase'):
            if previous[key] != manifest[key]:
                raise ValueError(f'Cannot resume after changing {key}; use a new version')
        generation_source = 'src/slm_steering/research/generation.py'
        allowed_worker = previous.get('compatible_worker_updates', [{}])[-1].get(
            'new_sha256', previous['evalplus_worker_hash'])
        if (previous['source_hashes'][generation_source] != manifest['source_hashes'][generation_source]
                or allowed_worker != manifest['evalplus_worker_hash']):
            raise ValueError('Generation/verifier source changed; do not mix numerical or label regimes')
        # Preserve the run's original provenance; record each resumed environment
        # separately instead of overwriting the history with the latest session.
        manifest = previous
        manifest['status'] = 'running'
    session = {'started_at': now(), 'git_commit': manifest['git_commit'], 'git_dirty': manifest['git_dirty'],
               'time_budget_seconds': time_budget, 'new_attempts': 0, 'provenance': session_provenance}
    start = time.monotonic()
    benchmark_name = config['robustness']['benchmark'] if phase == 'transfer' else config['benchmark']['name']
    with EvalPlusClient(benchmark_name, wsl_python=wsl_python) as verifier:
        metadata = verifier.dataset['metadata']
        if metadata['evalplus_version'] != config['benchmark']['evalplus_version']:
            raise ValueError('Verifier version differs from frozen configuration')
        tasks = {task['task_id']: task for task in verifier.dataset['tasks']}
        pilot_ids = pilot_task_ids(tasks, config) if phase != 'transfer' else []
        ids = pilot_ids if phase == 'pilot' else sorted(set(tasks) - set(pilot_ids))
        if phase == 'transfer':
            ids = sorted(tasks, key=lambda task: (digest(config['robustness']['selection_prefix'] + task), task))[:config['robustness']['tasks']]
            bundle_hash = file_hash(transfer_bundle)
            if manifest.get('transfer_bundle_sha256', bundle_hash) != bundle_hash:
                raise ValueError('Frozen transfer model changed during collection')
            manifest['transfer_bundle_sha256'] = bundle_hash
            manifest['transfer_mode'] = 'frozen_transfer_no_target_fitting'
        grid = config['pilot']['temperatures'] if phase == 'pilot' else [
            json.loads(pilot_manifest_path.read_text(encoding='utf-8'))['selected_temperature']]
        n = config['pilot']['attempts_per_task'] if phase == 'pilot' else config['candidate_bank']['attempts_per_task']
        expected = {(task, index * n + attempt): temperature for index, temperature in enumerate(grid)
                    for task in ids for attempt in range(n)}
        for record in records:
            identity = (record['task_id'], record['attempt_id'])
            if identity not in expected or record['decoding_config']['temperature'] != expected[identity]:
                raise ValueError('Stored sample differs from planned grid')
            if record['benchmark_version'] != metadata['version'] or record['seed'] != attempt_seed(
                    config['protocol_version'], metadata['version'], *identity):
                raise ValueError('Stored benchmark/seed differs from frozen provenance')
        manifest.update({'benchmark': metadata, 'pilot_task_ids': pilot_ids, 'task_ids': ids,
                         'expected_attempts': len(expected), 'attempts_per_temperature': n,
                         'development_only': phase == 'pilot'})
        if len(records) == len(expected):
            manifest.update(json.loads(manifest_path.read_text(encoding='utf-8')))
            return manifest
        generator = PrefixGenerator(config, local_files_only=local_files_only)
        manifest.update({'layers': generator.layers, 'prompt_template_hash': generator.template_hash})
        # Positive and negative verifier controls precede outcome generation.
        control = verifier.request({'operation': 'verify', 'task_id': ids[0], 'canonical_control': True})
        negative = verifier.verify(ids[0], 'def ' + tasks[ids[0]]['entry_point'] + '(*args):\n    raise RuntimeError()\n')
        if not control['passed'] or negative['passed']:
            raise ValueError('EvalPlus positive/negative control failed')
        manifest['verifier_controls'] = {'positive': control, 'negative': negative}
        write_json(manifest_path, manifest)
        done = {(r['task_id'], r['attempt_id']) for r in records}
        # Interleave temperatures by task/attempt so partial sessions are balanced.
        jobs = [(task, index * n + attempt, temperature) for attempt in range(n)
                for task in ids for index, temperature in enumerate(grid)]
        (output / 'activations').mkdir(exist_ok=True)
        for task_id, attempt_id, temperature in jobs:
            if (task_id, attempt_id) in done:
                continue
            if session['new_attempts'] and ((time_budget and time.monotonic() - start >= time_budget) or
                    (max_attempts and session['new_attempts'] >= max_attempts)):
                break
            task = tasks[task_id]
            seed = attempt_seed(config['protocol_version'], metadata['version'], task_id, attempt_id)
            sample = generator.generate(task['prompt'], seed, temperature)
            if 'causal_invariance' not in manifest:
                manifest['causal_invariance'] = generator.invariance_test(sample)
                repeat = generator.generate(task['prompt'], seed, temperature)
                if sample['token_ids'] != repeat['token_ids']:
                    raise ValueError('Independent attempt seed did not reproduce generated token IDs')
                manifest['seed_reproduction_passed'] = True
            if any(not np.isfinite(array).all() for array in sample['activations'].values()):
                raise ValueError('Non-finite activation artifact; no record can be appended')
            problem = CodingProblem(task_id=task_id, prompt=task['prompt'], entry_point=task['entry_point'], test='')
            code = build_candidate_source(problem, sample['raw_completion'])
            verification = verifier.verify(task_id, code)
            if not normalize_completion(sample['raw_completion']).strip():
                verification.update({'passed': False, 'error_type': 'empty_extraction'})
            verification['passed_nontruncated'] = bool(verification['passed'] and not sample['truncated'])
            verification['label_definition'] = 'evalplus_base_and_plus_pass'
            verification['margin_definition'] = 'temperature_scaled_pre_top_p'
            activation_path = output / 'activations' / f'{digest(task_id)[:12]}_{attempt_id:03d}.npz'
            temporary = activation_path.with_suffix('.tmp.npz')
            np.savez_compressed(temporary, **sample.pop('activations'), layers=generator.layers)
            temporary.replace(activation_path)
            record = {**sample, **verification, 'protocol_version': config['protocol_version'], 'run_id': output.name,
                      'benchmark_name': benchmark_name, 'benchmark_version': metadata['version'],
                      'benchmark_hash': metadata['dataset_sha256'], 'task_id': task_id, 'attempt_id': attempt_id,
                      'seed': seed, 'development_only': phase == 'pilot', 'model_id': config['model']['id'],
                      'model_revision': config['model']['revision'], 'tokenizer_revision': config['model']['tokenizer_revision'],
                      'prompt_template_hash': generator.template_hash, 'decoding_config': {
                          **config['decoding'], 'temperature': temperature}, 'prompt': task['prompt'],
                      'extracted_code': code, 'verifier_version': metadata['evalplus_version'],
                      'activation_path': activation_path.relative_to(output).as_posix(), 'artifact_hashes': {
                          'completion': digest(sample['raw_completion']), 'code': digest(code),
                          'tokens': digest(canonical(sample['token_ids'])), 'activations': file_hash(activation_path)}}
            validate_bank([record])
            with bank_path.open('a', encoding='utf-8') as stream:
                stream.write(canonical(record) + '\n')
                stream.flush()
                os.fsync(stream.fileno())
            records.append(record)
            session['new_attempts'] += 1
            manifest['completed_attempts'] = len(records)
            manifest['hardware'] = generator.hardware()
            write_json(manifest_path, manifest)
            print(f'{len(records)}/{len(expected)} {task_id} attempt={attempt_id} T={temperature} '
                  f'tokens={sample["generated_tokens"]} pass={record["passed"]}', flush=True)
        session['ended_at'] = now()
        session['wall_seconds'] = time.monotonic() - start
        session['hardware'] = generator.hardware()
        manifest['sessions'].append(session)
        manifest['completed_attempts'] = len(records)
        manifest['status'] = 'completed' if len(records) == len(expected) else 'partial'
        manifest['bank_sha256'] = file_hash(bank_path) if bank_path.exists() else None
        manifest['bank_frozen'] = phase in {'definitive', 'transfer'} and manifest['status'] == 'completed'
        if phase == 'pilot':
            selection = select_temperature(records, config, ids)
            manifest.update(selection)
            if selection['complete'] and selection['selected_temperature'] is None:
                manifest['status'] = 'no_eligible_regime'
        manifest['ended_at'] = now()
        write_json(manifest_path, manifest)
        return manifest
