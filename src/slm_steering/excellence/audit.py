from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from slm_steering.research.features import observable_features
from slm_steering.research.freeze import validate_execution_freeze
from slm_steering.research.probes import calibrated_scores, calibrate, fit_model
from slm_steering.research.protocol import attempt_seed, file_hash, load_config, now, read_bank, write_json
from slm_steering.research.statistics import paired_task_bootstrap, predictive_metrics


def initialize(root: Path):
    directory = root / 'reports/excellence'
    directory.mkdir(exist_ok=True)
    path = directory / 'baseline.json'
    if path.exists():
        return json.loads(path.read_text(encoding='utf-8'))
    archive = directory / 'archive_before_extension'
    publication = ['RESULTS.md', 'README.md', 'reports/final_review/evidence.json',
                   'reports/final_review/delivery_manifest.json', 'notebooks/final_research_summary.ipynb',
                   'tcc/build/manuscrito.pdf', 'tcc/referencias.bib']
    publication += [p.relative_to(root).as_posix() for p in (root / 'tcc/chapters').glob('*.typ')]
    protected = ['RESEARCH_PROTOCOL.md', 'configs/protocol_v0.2.yaml',
                 'reports/research/execution_freeze_v0.2/manifest.json']
    for phase in ('pilot', 'definitive', 'transfer'):
        protected += [f'runs/candidate_bank/{phase}_v0.2/{name}' for name in ('manifest.json', 'trajectories.jsonl')]
    protected += ['runs/probes/protocol_v0.2/outer_predictions.csv', 'runs/probes/protocol_v0.2/selection_audit.json',
                  'runs/transfer/protocol_v0.2/models.joblib', 'runs/transfer/protocol_v0.2/models.json',
                  'reports/protocol_v0.2/results.json', 'reports/transfer_v0.2/results.json',
                  'reports/transfer_v0.2/predictions.csv']
    for relative in publication:
        source, dest = root / relative, archive / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)
    status = subprocess.check_output(['git', 'status', '--short'], cwd=root, text=True)
    (directory / 'initial_git_status.txt').write_text(status, encoding='utf-8')
    (directory / 'initial_worktree.patch').write_bytes(subprocess.check_output(['git', 'diff', '--binary'], cwd=root))
    result = {'created_at': now(), 'client_request_date': '2026-10-17',
              'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
              'protected': {p: file_hash(root / p) for p in protected},
              'archived_publication': {p: file_hash(root / p) for p in publication},
              'scope': 'Exploratory extensions authorized; independent data collection and scientific freeze await student/advisor review'}
    write_json(path, result)
    return result


def assert_preserved(root: Path):
    baseline = json.loads((root / 'reports/excellence/baseline.json').read_text(encoding='utf-8'))
    validate_execution_freeze(root, root / 'configs/protocol_v0.2.yaml')
    for path, expected in baseline['protected'].items():
        if file_hash(root / path) != expected:
            raise ValueError(f'Historical scientific artifact changed: {path}')


def checkpoint_data(root: Path, phase='definitive', checkpoint=32) -> dict:
    """Load only the requested stored checkpoint; verify every activation archive used."""
    directory = root / f'runs/candidate_bank/{phase}_v0.2'
    manifest = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
    records = read_bank(directory / 'trajectories.jsonl')
    selected, hidden, observable = [], [], []
    for r in records:
        if r['generated_tokens'] <= checkpoint:
            continue
        path = directory / r['activation_path']
        if file_hash(path) != r['artifact_hashes']['activations']:
            raise ValueError('Activation hash mismatch')
        with np.load(path, allow_pickle=False) as a:
            if a['layers'].tolist() != manifest['layers']:
                raise ValueError('Layer identity mismatch')
            value = a[str(checkpoint)].astype(np.float64)
            if not np.isfinite(value).all():
                raise ValueError('Nonfinite hidden features')
            hidden.append(value)
        selected.append(r)
        observable.append(observable_features(r, checkpoint))
    return {'records': selected, 'hidden': np.asarray(hidden), 'observable': np.asarray(observable),
            'y': np.array([int(r['passed']) for r in selected]),
            'groups': np.array([r['task_id'] for r in selected]), 'layers': manifest['layers'],
            'identities': [(r['task_id'], r['attempt_id']) for r in selected]}


def pair_auc(y, score):
    """Independent tie-aware pair-count implementation for audit (not sklearn AUC)."""
    positive, negative = np.asarray(score)[np.asarray(y) == 1], np.asarray(score)[np.asarray(y) == 0]
    if not len(positive) or not len(negative):
        return None
    return float(((positive[:, None] > negative) + .5 * (positive[:, None] == negative)).mean())


def run(root: Path):
    initialize(root)
    assert_preserved(root)
    load_config(root / 'configs/protocol_v0.2.yaml')
    primary = pd.read_csv(root / 'runs/probes/protocol_v0.2/outer_predictions.csv')
    primary = primary[primary.checkpoint == '32'].copy()
    keys = list(zip(primary.task_id, primary.attempt_id))
    lookup = {key: i for i, key in enumerate(keys)}
    data = checkpoint_data(root)
    primary = primary.iloc[[lookup[k] for k in data['identities']]].reset_index(drop=True)
    if not np.array_equal(primary.passed, data['y']):
        raise ValueError('Prediction labels disagree with bank')
    groups, y = data['groups'], data['y']
    metrics, deltas = {}, []
    for family in ('B', 'H', 'HB'):
        values = [pair_auc(g.passed, g[family]) for _, g in primary.groupby('task_id', sort=True)]
        values = [v for v in values if v is not None]
        metrics[family] = predictive_metrics(y, primary[family].to_numpy(), groups)
        if abs(metrics[family]['within_task_auc'] - np.mean(values)) > 1e-12:
            raise ValueError('Independent pair-count WT-AUC differs')
    for _, group in primary.groupby('task_id', sort=True):
        b, hb = pair_auc(group.passed, group.B), pair_auc(group.passed, group.HB)
        if b is not None:
            deltas.append(hb - b)
    wt = paired_task_bootstrap(deltas, replications=2000, seed=20260921)
    eps = np.finfo(float).eps
    losses = []
    for _, group in primary.groupby('task_id', sort=True):
        labels = group.passed.to_numpy()
        def loss(f):
            p = np.clip(group[f].to_numpy(), eps, 1 - eps)
            return float(-(labels * np.log(p) + (1 - labels) * np.log1p(-p)).mean())
        losses.append(loss('HB') - loss('B'))
    loss_delta = paired_task_bootstrap(losses, replications=2000, seed=20260921)
    published = json.loads((root / 'reports/protocol_v0.2/results.json').read_text(encoding='utf-8'))
    for estimate, key in [(wt, 'delta_hb_minus_b'), (loss_delta, 'delta_log_loss_hb_minus_b')]:
        expected = published['signal']['H2'][key]
        if not np.allclose([estimate['estimate'], *estimate['ci95']], [expected['estimate'], *expected['ci95']], atol=1e-12):
            raise ValueError('Published primary contrast was not reproduced')
    selection = json.loads((root / 'runs/probes/protocol_v0.2/selection_audit.json').read_text(encoding='utf-8'))
    split_checks = 0
    for row in selection:
        if 'train_tasks' not in row:
            continue
        train_tasks, test_tasks = set(row['train_tasks']), set(row['test_tasks'])
        if train_tasks & test_tasks:
            raise ValueError('Outer task leakage')
        for family in ('B', 'H', 'HB'):
            selected = row[family]
            options = selected['selection']
            best = min(options, key=lambda x: x['loss'])
            admissible = [o for o in options if o['loss'] <= best['loss'] + best['standard_error']]
            choice = min(admissible, key=lambda o: (o['c'], o['layer']))
            if (choice['c'], choice['layer']) != (selected['c'], selected['layer']):
                raise ValueError('One-SE selection differs from recorded rule')
            seen = []
            for cross in selected['calibration_audit']:
                fit, score = set(cross['fit_tasks']), set(cross['score_tasks'])
                if fit & score or (fit | score) != train_tasks or score & test_tasks:
                    raise ValueError('Calibration cross-fitting leakage')
                seen.extend(score)
                split_checks += 1
            if len(seen) != len(set(seen)) or set(seen) != train_tasks:
                raise ValueError('Calibration score tasks not an exact partition')
    # Refit selected raw H models and calibration using the recorded inner-training choices.
    # Saved B/HB estimators are independently applied to original activation archives.
    reproduction = []
    for fold in sorted(primary.fold.unique()):
        train, test = primary.fold.to_numpy() != fold, primary.fold.to_numpy() == fold
        audit = next(r for r in selection if r.get('fold') == fold and r.get('checkpoint') == '32' and 'H' in r)
        for family in ('B', 'HB', 'H'):
            if family == 'H':
                s = audit['H']
                x = data['hidden'][:, data['layers'].index(s['layer'])]
                model = fit_model(x[train], y[train], groups[train], s['c'])
                honest = np.full(train.sum(), np.nan)
                training_indices = np.flatnonzero(train)
                for c in s['calibration_audit']:
                    fit = train & np.isin(groups, c['fit_tasks'])
                    score = train & np.isin(groups, c['score_tasks'])
                    local_x = data['hidden'][:, data['layers'].index(c['layer'])]
                    local = fit_model(local_x[fit], y[fit], groups[fit], c['c'])
                    honest[np.isin(training_indices, np.flatnonzero(score))] = local.decision_function(local_x[score])
                fitted = {'model': model, 'calibrator': calibrate(honest, y[train], groups[train])}
            else:
                fitted = joblib.load(root / f'runs/probes/protocol_v0.2/models/fold{fold}_{family}_t32.joblib')
                if set(fitted['training_tasks']) != set(groups[train]):
                    raise ValueError('Saved model training-task metadata differs')
                x = data['observable'] if family == 'B' else np.column_stack((data['hidden'][:, data['layers'].index(fitted['layer'])], data['observable']))
            score = calibrated_scores(fitted, x[test])
            difference = float(np.max(np.abs(score - primary.loc[test, family].to_numpy())))
            if difference > 1e-9:
                raise ValueError(f'Reconstructed scorer differs: fold {fold}, {family}, {difference}')
            reproduction.append({'fold': int(fold), 'predictor': family, 'maximum_absolute_error': difference})
    banks = {}
    for phase in ('pilot', 'definitive', 'transfer'):
        directory = root / f'runs/candidate_bank/{phase}_v0.2'
        manifest = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
        records = read_bank(directory / 'trajectories.jsonl')
        if len(records) != manifest['expected_attempts'] or file_hash(directory / 'trajectories.jsonl') != manifest['bank_sha256']:
            raise ValueError('Incomplete or changed bank')
        for r in records:
            if r['seed'] != attempt_seed(r['protocol_version'], r['benchmark_version'], r['task_id'], r['attempt_id']):
                raise ValueError('Seed identity mismatch')
            if len(r['token_ids']) != r['generated_tokens'] or len(r['token_logprobs']) != r['generated_tokens']:
                raise ValueError('Token accounting differs')
            if not r['passed'] == all(v == 'pass' for v in r['verifier_status'].values()):
                raise ValueError('Functional label differs from official statuses')
        seeds = [r['seed'] for r in records]
        if len(seeds) != len(set(seeds)):
            raise ValueError('Repeated independent candidate seed')
        banks[phase] = {'attempts': len(records), 'tasks': len({r['task_id'] for r in records}),
                        'seed_identities_verified': len(seeds), 'passed': sum(r['passed'] for r in records),
                        'truncated': sum(r['truncated'] for r in records),
                        'generation_seconds': sum(r['generation_time_seconds'] for r in records),
                        'verification_seconds': sum(r['verification_time_seconds'] for r in records)}
    result = {'created_at': now(), 'status': 'initial_scientific_audit_passed_with_limitations', 'banks': banks,
              'risk_set_32': {'attempts': len(y), 'tasks': len(set(groups)), 'mixed_tasks': len(deltas)},
              'metrics': metrics, 'delta_wt_auc': wt, 'delta_log_loss': loss_delta,
              'split_checks': split_checks, 'scorer_reproduction': reproduction,
              'limitations': ['Full hyperparameter searches were audited from saved grids, not independently rerun for all original checkpoints',
                             'Model reproduction covers the primary checkpoint; recorded prefix tests provide numerical invariance evidence',
                             'No historical advisor approval, independent preregistration, or data novelty is inferred']}
    assert_preserved(root)
    write_json(root / 'reports/excellence/reproduction.json', result)
    return result
