"""Approval-gated, append-only independent extension, separate from historical v0.2."""
from __future__ import annotations

import importlib.metadata
import json
import os
from pathlib import Path
import time

import numpy as np

from slm_steering.excellence.audit import assert_preserved
from slm_steering.excellence.approvals import approved_design
from slm_steering.research.protocol import attempt_seed, canonical, digest, file_hash, now, read_bank, validate_bank, write_json


def collect(root: Path, phase: str, *, wsl_python: str, time_budget=600):
    design = approved_design(root)
    assert_preserved(root)
    if phase not in ('technical', 'main') or time_budget <= 0:
        raise ValueError('Invalid phase or session time budget')
    ids = design['technical_pilot_task_ids' if phase == 'technical' else 'main_task_ids']
    if phase == 'main':
        technical = json.loads((root / 'runs/excellence/independent_v1/technical/manifest.json').read_text(encoding='utf-8'))
        if technical['status'] != 'completed' or not technical.get('numerical_checks_passed'):
            raise ValueError('Complete the disjoint technical pilot and numerical gates before main collection')
    import torch  # caller initializes this before scientific libraries on Windows
    from slm_steering.datasets import CodingProblem
    from slm_steering.excellence.benchmark import selective_generate
    from slm_steering.research.evaluation import EvalPlusClient
    from slm_steering.research.generation import PrefixGenerator
    from slm_steering.research.protocol import load_config
    from slm_steering.verifier import build_candidate_source, normalize_completion
    directory = root / f'runs/excellence/independent_v1/{phase}'
    directory.mkdir(parents=True, exist_ok=True)
    lock = directory / '.writer.lock'
    with lock.open('x', encoding='utf-8') as stream:
        stream.write(str(os.getpid()))
    started = time.monotonic()
    try:
        records = read_bank(directory / 'trajectories.jsonl')
        config = load_config(root / 'configs/protocol_v0.2.yaml')
        contract = {'protocol_sha256': file_hash(root / 'docs/protocols/EXCELLENCE_EXTENSION_v1.md'),
                    'design_sha256': file_hash(root / 'reports/excellence/proposal_design.json'),
                    'collector_sha256': file_hash(Path(__file__)),
                    'generator_sha256': file_hash(root / 'src/slm_steering/research/generation.py'),
                    'selective_capture_sha256': file_hash(root / 'src/slm_steering/excellence/benchmark.py'),
                    'bundle_sha256': file_hash(root / 'runs/transfer/protocol_v0.2/models.joblib'),
                    'inference_config_sha256': file_hash(root / 'configs/protocol_v0.2.yaml'),
                    'packages': {name: importlib.metadata.version(name) for name in ('torch', 'transformers', 'numpy')}}
        manifest_path = directory / 'manifest.json'
        manifest = json.loads(manifest_path.read_text(encoding='utf-8')) if manifest_path.exists() else {
            'created_at': now(), 'contract': contract, 'phase': phase, 'expected_attempts': len(ids) * 20, 'sessions': []}
        if manifest['contract'] != contract:
            raise ValueError('Cannot mix collection sources or design versions on resume')
        expected = {(task, attempt) for task in ids for attempt in range(20)}
        if any((r['task_id'], r['attempt_id']) not in expected for r in records):
            raise ValueError('Existing candidate outside approved sample')
        for r in records:
            if file_hash(directory / r['activation_path']) != r['artifact_hashes']['activations']:
                raise ValueError('Stored activation changed')
        engine = PrefixGenerator(config)
        if engine.hardware()['device'] != design['approved_hardware_device']:
            raise ValueError('GPU differs from reviewed hardware; run preflight and amend before collection')
        metadata = json.loads((root / 'runs/transfer/protocol_v0.2/models.json').read_text(encoding='utf-8'))
        selection = next(r for r in metadata['selections'] if r['family'] == 'HB' and r['checkpoint'] == 8)
        index = engine.layers.index(selection['layer'])
        done = {(r['task_id'], r['attempt_id']) for r in records}
        new_attempts = 0
        with EvalPlusClient('mbpp', wsl_python=wsl_python) as verifier:
            if verifier.dataset['metadata']['dataset_sha256'] != design['benchmark_sha256']:
                raise ValueError('Benchmark differs from proposed/reviewed version')
            tasks = {t['task_id']: t for t in verifier.dataset['tasks']}
            positive = verifier.request({'operation': 'verify', 'task_id': 'Mbpp/2', 'canonical_control': True})
            negative = verifier.verify('Mbpp/2', 'def similar_elements(*args):\n    raise RuntimeError()\n')
            if not positive['passed'] or negative['passed']:
                raise ValueError('Verifier controls failed')
            for attempt in range(20):
                for task_id in ids:
                    if (task_id, attempt) in done:
                        continue
                    if new_attempts and time.monotonic() - started >= time_budget:
                        break
                    seed = attempt_seed('excellence-extension-v1', verifier.dataset['metadata']['version'], task_id, attempt)
                    sample = selective_generate(engine, tasks[task_id]['prompt'], seed, 1.)
                    if not manifest.get('numerical_checks_passed'):
                        reference = engine.generate(tasks[task_id]['prompt'], seed, 1.)
                        if sample['token_ids'] != reference['token_ids'] or any(not np.array_equal(v, reference['activations'][k]) for k, v in sample['activations'].items()):
                            raise ValueError('Selective capture numerical reproduction failed')
                        engine.invariance_test(reference)
                        manifest['numerical_checks_passed'] = True
                    problem = CodingProblem(task_id=task_id, prompt=tasks[task_id]['prompt'], entry_point=tasks[task_id]['entry_point'], test='')
                    code = build_candidate_source(problem, sample['raw_completion'])
                    verification = verifier.verify(task_id, code)
                    if not normalize_completion(sample['raw_completion']).strip():
                        verification.update(passed=False, error_type='empty_extraction')
                    arrays = sample.pop('activations')
                    selected = {k: a[index:index + 1] for k, a in arrays.items() if k in ('0', '8', 'final')}
                    if any(not np.isfinite(a).all() for a in selected.values()):
                        raise ValueError('Nonfinite stored activation')
                    activation = directory / 'activations' / f'{digest(task_id)[:12]}_{attempt:03d}.npz'
                    activation.parent.mkdir(exist_ok=True)
                    temporary = activation.with_suffix('.tmp.npz')
                    np.savez_compressed(temporary, **selected, layers=np.array([selection['layer']]))
                    temporary.replace(activation)
                    record = {**sample, **verification, 'protocol_version': 'excellence-extension-v1', 'run_id': directory.name,
                              'benchmark_version': verifier.dataset['metadata']['version'], 'benchmark_name': 'mbpp',
                              'task_id': task_id, 'attempt_id': attempt, 'seed': seed, 'development_only': phase == 'technical',
                              'model_id': config['model']['id'], 'model_revision': config['model']['revision'],
                              'tokenizer_revision': config['model']['tokenizer_revision'], 'prompt_template_hash': engine.template_hash,
                              'decoding_config': {**config['decoding'], 'temperature': 1.}, 'prompt': tasks[task_id]['prompt'],
                              'extracted_code': code, 'verifier_version': verifier.dataset['metadata']['evalplus_version'],
                              'label_definition': 'evalplus_base_and_plus_pass', 'activation_path': activation.relative_to(directory).as_posix(),
                              'artifact_hashes': {'completion': digest(sample['raw_completion']), 'code': digest(code),
                                                  'tokens': digest(canonical(sample['token_ids'])), 'activations': file_hash(activation)}}
                    validate_bank([record])
                    with (directory / 'trajectories.jsonl').open('a', encoding='utf-8') as stream:
                        stream.write(canonical(record) + '\n')
                        stream.flush()
                        os.fsync(stream.fileno())
                    records.append(record)
                    done.add((task_id, attempt))
                    new_attempts += 1
                    manifest.update(status='running', completed_attempts=len(records), hardware=engine.hardware(), updated_at=now())
                    write_json(manifest_path, manifest)
                    print(f"{phase} {len(records)}/{len(expected)} {task_id} attempt={attempt}", flush=True)
                else:
                    continue
                break
        manifest.update(status='completed' if done == expected else 'partial', completed_attempts=len(records),
                        bank_sha256=file_hash(directory / 'trajectories.jsonl') if records else None)
        manifest['sessions'].append({'ended_at': now(), 'new_attempts': new_attempts, 'seconds': time.monotonic() - started,
                                     'torch': torch.__version__, 'numpy': importlib.metadata.version('numpy'), 'hardware': engine.hardware()})
        write_json(manifest_path, manifest)
        return manifest
    except BaseException as exc:
        try:
            write_json(directory / 'failure.json', {'failed_at': now(), 'type': type(exc).__name__, 'message': str(exc),
                                                    'scope': 'No fabricated label appended; inspect durable JSONL before resuming'})
        except OSError:
            pass
        raise
    finally:
        lock.unlink(missing_ok=True)


def paired_retention_result(counts, *, replications=10000, seed=20261017):
    """Columns: kept-HB correct, kept-B correct, all active correct; task clusters paired."""
    counts = np.asarray(counts, dtype=float)
    if counts.ndim != 2 or counts.shape[1] != 3 or not len(counts) or not np.isfinite(counts).all():
        raise ValueError('Expected finite per-task paired retention counts')
    if (counts < 0).any() or (counts[:, :2] > counts[:, 2:3]).any():
        raise ValueError('Retained counts cannot exceed the common correct denominator')
    rng = np.random.default_rng(seed)
    total = counts[rng.integers(0, len(counts), (replications, len(counts)))].sum(axis=1)
    valid = total[:, 2] > 0
    value = (total[valid, 0] - total[valid, 1]) / total[valid, 2]
    defined = counts[:, 2].sum() > 0
    identified = defined and float((~valid).mean()) <= .01
    interval = np.quantile(value, [.025, .975]).tolist() if len(value) else None
    return {'estimate': float((counts[:, 0].sum() - counts[:, 1].sum()) / counts[:, 2].sum()) if defined else None,
            'ci95': interval, 'tasks': len(counts), 'replications': replications, 'seed': seed,
            'undefined_draws': int((~valid).sum()), 'identified': bool(identified),
            'directional_hypothesis_supported': bool(identified and interval[1] < 0)}


def evaluate(root: Path):
    import joblib
    import pandas as pd
    from slm_steering.research.features import observable_features
    from slm_steering.research.probes import calibrated_scores
    from slm_steering.research.statistics import predictive_metrics, task_auc, paired_task_bootstrap
    from slm_steering.excellence.economics import simulate
    design = approved_design(root)
    assert_preserved(root)
    directory = root / 'runs/excellence/independent_v1/main'
    manifest = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
    records = read_bank(directory / 'trajectories.jsonl')
    expected = {(t, j) for t in design['main_task_ids'] for j in range(20)}
    if manifest['status'] != 'completed' or {(r['task_id'], r['attempt_id']) for r in records} != expected:
        raise ValueError('Independent main sample must be complete; no outcome-dependent replacement')
    if file_hash(directory / 'trajectories.jsonl') != manifest['bank_sha256']:
        raise ValueError('Independent main bank hash differs')
    bundle = joblib.load(root / 'runs/transfer/protocol_v0.2/models.joblib')
    rows = []
    for record in records:
        if record['generated_tokens'] <= 8:
            continue
        path = directory / record['activation_path']
        if file_hash(path) != record['artifact_hashes']['activations']:
            raise ValueError('Independent activation changed')
        with np.load(path, allow_pickle=False) as states:
            hidden = states['8'][0].astype(float)
        obs = observable_features(record, 8)
        values = {}
        for family in ('B', 'HB'):
            x = obs if family == 'B' else np.r_[hidden, obs]
            values[family] = float(calibrated_scores(bundle['models'][(family, 8)], x[None, :])[0])
        rows.append({'task_id': record['task_id'], 'attempt_id': record['attempt_id'],
                     'passed': int(record['passed']), **values})
    frame = pd.DataFrame(rows, columns=['task_id', 'attempt_id', 'passed', 'B', 'HB'])
    counts = []
    for task in design['main_task_ids']:
        part = frame[(frame.task_id == task) & (frame.passed == 1)]
        counts.append([int(part.HB.ge(design['policies']['HB']['threshold']).sum()),
                       int(part.B.ge(design['policies']['B']['threshold']).sum()), len(part)])
    primary = paired_retention_result(counts, replications=design['bootstrap_replications'], seed=design['bootstrap_seed'])
    metrics = []
    for family in ('B', 'HB'):
        if frame.empty:
            continue
        y, p, groups = frame.passed.to_numpy(), frame[family].to_numpy(), frame.task_id.to_numpy()
        auc = list(task_auc(y, p, groups).values())
        metrics.append({'family': family, **predictive_metrics(y, p, groups),
                        'wt_auc_interval': paired_task_bootstrap(auc, replications=10000, seed=20261017) if auc else None})
    policy_rows = []
    for task in design['main_task_ids']:
        stream = [r for r in records if r['task_id'] == task]
        for budget in (1, 5, 10, 20):
            for family in ('B', 'HB', 'verifier', 'fixed', 'oracle'):
                scores = {(r.task_id, r.attempt_id): getattr(r, family) for r in frame[frame.task_id == task].itertuples()} if family in ('B', 'HB') else None
                row = simulate(stream, checkpoint=8, threshold=design['policies'][family]['threshold'] if scores is not None else 0,
                               scores=scores, budget=budget, kind='gate' if family in ('B', 'HB') else family)
                policy_rows.append({**row, 'policy': family, 'budget': budget})
    output = root / 'reports/excellence/independent_v1'
    output.mkdir(exist_ok=True)
    frame.to_csv(output / 'predictions.csv', index=False)
    pd.DataFrame(policy_rows).drop(columns='decisions').to_csv(output / 'policy_per_task.csv', index=False)
    write_json(output / 'results.json', {'created_at': now(), 'status': 'completed_approved_independent_evaluation',
                                        'primary': primary, 'per_task_counts': counts, 'secondary_predictive_metrics': metrics,
                                        'bank_sha256': manifest['bank_sha256'], 'target_fitting': False,
                                        'interpretation': 'New approved endpoint only; does not rewrite original HumanEval or MBPP results'})
    return primary
