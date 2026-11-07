"""Frozen benchmark transfer: all fitting uses source labels only."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from slm_steering.research.policy import replay, select_policy
from slm_steering.research.probes import calibrated_scores, choose_model, load_features
from slm_steering.research.protocol import file_hash, load_config, now, read_bank, write_json
from slm_steering.research.reporting import metric_intervals, summarize_policies
from slm_steering.research.statistics import paired_task_bootstrap, predictive_metrics, task_auc


def validate_bundle(path: Path | None, config_path: Path) -> dict:
    if path is None or not path.exists():
        raise ValueError('Freeze the source-trained transfer bundle before target collection')
    manifest = json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
    if manifest['bundle_sha256'] != file_hash(path) or manifest['config_hash'] != file_hash(config_path):
        raise ValueError('Transfer bundle/config hash differs from frozen manifest')
    if manifest['mode'] != 'frozen_transfer_no_target_fitting':
        raise ValueError('A locally refitted model is not frozen transfer')
    return manifest


def prepare_bundle(bank_dir: Path, probes_dir: Path, config_path: Path, output: Path) -> dict:
    config = load_config(config_path)
    manifest = json.loads((bank_dir / 'manifest.json').read_text(encoding='utf-8'))
    probes = json.loads((probes_dir / 'manifest.json').read_text(encoding='utf-8'))
    if manifest['phase'] != 'definitive' or not manifest['bank_frozen'] or probes['status'] != 'completed':
        raise ValueError('Finish the source bank and grouped main analysis before fitting transfer models')
    if manifest['bank_sha256'] != probes['bank_sha256'] or manifest['bank_sha256'] != file_hash(bank_dir / 'trajectories.jsonl'):
        raise ValueError('Source bank provenance mismatch')
    if manifest['config_hash'] != file_hash(config_path) or probes['config_hash'] != manifest['config_hash']:
        raise ValueError('Source analysis/config provenance mismatch')
    if output.exists() or output.with_suffix('.json').exists():
        raise ValueError('A frozen transfer bundle is immutable; use a new version')
    records = read_bank(bank_dir / 'trajectories.jsonl')
    features = load_features(records, bank_dir, manifest, config)
    models, candidates, selections = {}, {'B': [], 'HB': []}, []
    with threadpool_limits(limits=1):
        for checkpoint in config['policy']['checkpoints']:
            feature = features[str(checkpoint)]
            indices = feature['indices']
            y = np.array([int(records[i]['passed']) for i in indices])
            groups = np.array([records[i]['task_id'] for i in indices])
            obs, hidden = feature['observable'], feature['hidden']
            for family in ('B', 'H', 'HB'):
                variants = [(-1, obs)] if family == 'B' else [(layer, hidden[:, i] if family == 'H' else
                    np.column_stack((hidden[:, i], obs))) for i, layer in enumerate(manifest['layers'])]
                fitted = choose_model(variants, y, groups, config)
                models[(family, checkpoint)] = {key: fitted[key] for key in ('model', 'calibrator', 'layer')}
                selections.append({'family': family, 'checkpoint': checkpoint, 'layer': fitted['layer'],
                                   'c': fitted['c'], 'calibration_audit': fitted['calibration_audit']})
                if family in candidates:
                    values = fitted['calibrator'].predict_proba(fitted['oof'].reshape(-1, 1))[:, 1]
                    lookup = {(records[i]['task_id'], records[i]['attempt_id']): float(p) for i, p in zip(indices, values)}
                    candidates[family].append({'checkpoint': checkpoint, 'probabilities': lookup})
        policies = {family: select_policy(records, options, minimum_recall=config['policy']['minimum_correct_recall'],
                    budget=config['candidate_bank']['attempts_per_task'], tie_tolerance=config['policy']['tie_tolerance'])
                    for family, options in candidates.items()}
    bundle = {'models': models, 'policies': policies, 'layers': manifest['layers'],
              'training_tasks': sorted({r['task_id'] for r in records}),
              'prompt_template_hash': manifest['prompt_template_hash']}
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, output)
    metadata = {'created_at': now(), 'mode': 'frozen_transfer_no_target_fitting',
                'bundle_sha256': file_hash(output), 'config_hash': file_hash(config_path),
                'source_bank_sha256': manifest['bank_sha256'], 'source_benchmark': manifest['benchmark'],
                'training_tasks': bundle['training_tasks'], 'selections': selections, 'policies': policies}
    write_json(output.with_suffix('.json'), metadata)
    return metadata


def score_frozen(bundle: dict, records: list[dict], features: dict) -> tuple[list, dict]:
    """Target labels are copied only for later evaluation, never used in a fit."""
    rows, lookups = [], {}
    for checkpoint in sorted({key[1] for key in bundle['models']}):
        feature = features[str(checkpoint)]
        if not len(feature['indices']):
            for family in ('B', 'H', 'HB'):
                lookups[(family, checkpoint)] = {}
            continue
        scores = {}
        for family in ('B', 'H', 'HB'):
            selected = bundle['models'][(family, checkpoint)]
            obs = feature['observable']
            if family == 'B':
                x = obs
            else:
                hidden = feature['hidden'][:, bundle['layers'].index(selected['layer'])]
                x = hidden if family == 'H' else np.column_stack((hidden, obs))
            scores[family] = calibrated_scores(selected, x)
            lookups[(family, checkpoint)] = {(records[i]['task_id'], records[i]['attempt_id']): float(p)
                                             for i, p in zip(feature['indices'], scores[family])}
        for j, i in enumerate(feature['indices']):
            rows.append({'task_id': records[i]['task_id'], 'attempt_id': records[i]['attempt_id'],
                         'checkpoint': checkpoint, 'passed': int(records[i]['passed']),
                         **{family: float(values[j]) for family, values in scores.items()}})
    return rows, lookups


def evaluate_transfer(bank_dir: Path, bundle_path: Path, config_path: Path, output: Path) -> dict:
    metadata = validate_bundle(bundle_path, config_path)
    config = load_config(config_path)
    manifest = json.loads((bank_dir / 'manifest.json').read_text(encoding='utf-8'))
    if manifest['phase'] != 'transfer' or not manifest['bank_frozen']:
        raise ValueError('A complete frozen target bank is required')
    if manifest['transfer_bundle_sha256'] != metadata['bundle_sha256'] or manifest['bank_sha256'] != file_hash(bank_dir / 'trajectories.jsonl'):
        raise ValueError('Target bank was not collected with this frozen bundle')
    if manifest['config_hash'] != file_hash(config_path):
        raise ValueError('Target bank configuration mismatch')
    records = read_bank(bank_dir / 'trajectories.jsonl')
    bundle = joblib.load(bundle_path)
    if set(bundle['training_tasks']) & {r['task_id'] for r in records}:
        raise ValueError('Source and target identities overlap')
    if bundle['prompt_template_hash'] != manifest['prompt_template_hash']:
        raise ValueError('Target prompt template differs from source training')
    features = load_features(records, bank_dir, manifest, config)
    rows, lookups = score_frozen(bundle, records, features)
    predictions = pd.DataFrame(rows, columns=['task_id', 'attempt_id', 'checkpoint', 'passed', 'B', 'H', 'HB'])
    metrics, deltas = [], []
    opts = {'replications': config['statistics']['bootstrap_replications'], 'seed': config['statistics']['seed']}
    for checkpoint, part in predictions.groupby('checkpoint'):
        y, groups = part['passed'].to_numpy(), part['task_id'].to_numpy()
        for family in ('B', 'H', 'HB'):
            scores = part[family].to_numpy()
            metrics.append({'checkpoint': int(checkpoint), 'family': family, **predictive_metrics(y, scores, groups),
                            'intervals': metric_intervals(y, scores, groups, **opts)})
        b, hb = task_auc(y, part['B'], groups), task_auc(y, part['HB'], groups)
        deltas.append({'checkpoint': int(checkpoint), 'mixed_tasks': len(b), 'delta_wt_auc':
                       paired_task_bootstrap([hb[t] - b[t] for t in b], **opts) if b else None})
    policy_rows = []
    for budget in config['candidate_bank']['budgets']:
        for family, policy in bundle['policies'].items():
            part = replay(records, lookups[(family, policy['checkpoint'])], checkpoint=policy['checkpoint'],
                          threshold=policy['threshold'], budget=budget)
            policy_rows.extend([{**r, 'policy': family} for r in part])
        for family in ('verifier', 'fixed'):
            policy_rows.extend(replay(records, {}, checkpoint=32, threshold=0, budget=budget, policy=family))
    output.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output / 'predictions.csv', index=False)
    result = {'status': 'secondary_frozen_transfer_offline', 'metrics': metrics, 'deltas': deltas,
              'policies': summarize_policies(pd.DataFrame(policy_rows), config), 'bundle_sha256': metadata['bundle_sha256'],
              'bank_sha256': manifest['bank_sha256'], 'target_labels_used_for_fitting': False}
    write_json(output / 'policy_replay.json', policy_rows)
    write_json(output / 'results.json', result)
    from slm_steering.research.visualization import transfer_figures
    transfer_figures(result, output / 'figures')
    return result
