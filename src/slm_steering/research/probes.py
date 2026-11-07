from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from slm_steering.research.features import observable_features
from slm_steering.research.policy import replay, select_policy
from slm_steering.research.protocol import file_hash, require_disjoint
from slm_steering.research.statistics import permute_within_tasks, predictive_metrics, task_weights


def make_model(c: float) -> Pipeline:
    return Pipeline([('impute', SimpleImputer(strategy='median', add_indicator=True, keep_empty_features=True)),
                     ('scale', StandardScaler()),
                     ('logistic', LogisticRegression(C=c, solver='lbfgs', max_iter=2000, random_state=0))])


def fit_model(features, y, groups, c):
    if len(np.unique(y)) != 2:
        raise ValueError('Training partition has only one outcome; probe cannot be identified')
    model = make_model(c)
    weights = task_weights(groups)
    with threadpool_limits(limits=1), warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        model.fit(features, y, scale__sample_weight=weights, logistic__sample_weight=weights)
    return model


def calibrate(oof_scores, y, groups):
    calibrator = LogisticRegression(C=1e6, solver='lbfgs', max_iter=2000, random_state=0)
    with threadpool_limits(limits=1), warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        calibrator.fit(np.asarray(oof_scores).reshape(-1, 1), y, sample_weight=task_weights(groups))
    return calibrator


def select_hyperparameters(variants, y, groups, config) -> dict:
    splits = list(GroupKFold(n_splits=config['statistics']['inner_folds']).split(np.zeros(len(y)), y, groups))
    results = []
    for layer, features in variants:
        for c in config['statistics']['regularization_c']:
            oof = np.full(len(y), np.nan)
            losses = []
            for train, validation in splits:
                require_disjoint(groups[train], groups[validation])
                fitted = fit_model(features[train], y[train], groups[train], c)
                oof[validation] = fitted.decision_function(features[validation])
                losses.append(predictive_metrics(y[validation], fitted.predict_proba(features[validation])[:, 1],
                                                 groups[validation])['log_loss'])
            probabilities = 1 / (1 + np.exp(-np.clip(oof, -700, 700)))
            results.append({'layer': layer, 'c': c, 'loss': predictive_metrics(y, probabilities, groups)['log_loss'],
                            'standard_error': float(np.std(losses, ddof=1) / np.sqrt(len(losses))), 'oof': oof,
                            'features': features})
    best = min(results, key=lambda result: result['loss'])
    eligible = [result for result in results if result['loss'] <= best['loss'] + best['standard_error']]
    selected = min(eligible, key=lambda result: (result['c'], result['layer']))
    selected['model'] = fit_model(selected['features'], y, groups, selected['c'])
    selected['selection'] = [{key: result[key] for key in ('layer', 'c', 'loss', 'standard_error')} for result in results]
    return selected


def choose_model(variants, y, groups, config) -> dict:
    """Fit the selected model; cross-fit the entire selection for calibration.

    The calibrator and policy threshold are development fits. Their training
    recall is not a certified guarantee; only the untouched outer fold evaluates
    the complete procedure.
    """
    selected = select_hyperparameters(variants, y, groups, config)
    honest = np.full(len(y), np.nan)
    audit = []
    for train, validation in GroupKFold(n_splits=config['statistics']['inner_folds']).split(
            np.zeros(len(y)), y, groups):
        require_disjoint(groups[train], groups[validation])
        local = select_hyperparameters([(layer, features[train]) for layer, features in variants],
                                       y[train], groups[train], config)
        chosen_features = next(features for layer, features in variants if layer == local['layer'])
        honest[validation] = local['model'].decision_function(chosen_features[validation])
        audit.append({'fit_tasks': sorted(set(groups[train])), 'score_tasks': sorted(set(groups[validation])),
                      'layer': local['layer'], 'c': local['c']})
    selected['oof'] = honest
    selected['calibrator'] = calibrate(honest, y, groups)
    selected['calibration_audit'] = audit
    selected['calibration_scope'] = 'Hyperparameter-selection-cross-fitted raw scores; calibration/threshold fitted on development labels'
    return selected


def calibrated_scores(selected: dict, features) -> np.ndarray:
    raw = selected['model'].decision_function(features)
    return selected['calibrator'].predict_proba(raw.reshape(-1, 1))[:, 1]


def load_features(records: list[dict], bank_dir: Path, manifest: dict, config: dict) -> dict:
    if any(record['protocol_version'] != config['protocol_version'] for record in records):
        raise ValueError('Activation bank and analysis use different protocol versions')
    layers = manifest['layers']
    features = {str(t): {'indices': [], 'hidden': [], 'prompt_hidden': [], 'observable': []}
                for t in config['activations']['checkpoints']}
    prompt_by_task = {}
    for index, record in enumerate(records):
        path = bank_dir / record['activation_path']
        if file_hash(path) != record['artifact_hashes']['activations']:
            raise ValueError('Activation artifact hash mismatch')
        with np.load(path, allow_pickle=False) as artifact:
            if artifact['layers'].tolist() != layers:
                raise ValueError('Activation layer provenance differs from manifest')
            prompt_state = artifact['0'].astype(float)
            if record['task_id'] in prompt_by_task and not np.allclose(
                    prompt_state, prompt_by_task[record['task_id']], atol=config['activations']['invariance_atol'],
                    rtol=config['activations']['invariance_rtol']):
                raise ValueError('Prompt-last state changed across attempts of the same task')
            prompt_by_task.setdefault(record['task_id'], prompt_state)
            prompt_state = prompt_by_task[record['task_id']]
            for checkpoint, feature in features.items():
                offset = record['generated_tokens'] if checkpoint == 'final' else int(checkpoint)
                if checkpoint not in ('0', 'final') and offset >= record['generated_tokens']:
                    continue
                feature['indices'].append(index)
                feature['hidden'].append(prompt_state if checkpoint == '0' else artifact[checkpoint].astype(float))
                feature['prompt_hidden'].append(prompt_state)
                feature['observable'].append(observable_features(record, offset))
    for feature in features.values():
        for key in ('indices', 'hidden', 'prompt_hidden', 'observable'):
            feature[key] = np.asarray(feature[key])
    return features


def grouped_predictions(records: list[dict], features: dict, manifest: dict, config: dict,
                        *, include_policy: bool = True, model_dir: Path | None = None) -> dict:
    if any(record['development_only'] for record in records):
        raise ValueError('Pilot trajectories cannot enter the grouped confirmatory pipeline')
    groups = np.array([record['task_id'] for record in records])
    y = np.array([int(record['passed']) for record in records])
    layers = manifest['layers']
    outer = list(GroupKFold(n_splits=config['statistics']['outer_folds']).split(np.zeros(len(y)), y, groups))
    predictions, selections, policy_results = [], [], []
    for fold, (train, test) in enumerate(outer):
        require_disjoint(groups[train], groups[test], manifest['pilot_task_ids'])
        training_records = [records[index] for index in train]
        test_records = [records[index] for index in test]
        policy_candidates = {'B': [], 'HB': []}
        test_probabilities = {}
        fold_control_seed = config['statistics']['seed'] + fold
        for checkpoint, feature in features.items():
            indices = feature['indices']
            train_mask, test_mask = np.isin(indices, train), np.isin(indices, test)
            train_indices, test_indices = indices[train_mask], indices[test_mask]
            yt, gt = y[train_indices], groups[train_indices]
            observable, hidden = feature['observable'], feature['hidden']
            prompt_hidden = feature['prompt_hidden']
            if not len(test_indices) or len(np.unique(gt)) < config['statistics']['inner_folds']:
                raise ValueError(f'Insufficient grouped risk set at checkpoint {checkpoint}')
            require_disjoint(gt, groups[test_indices], manifest['pilot_task_ids'])
            selected_b = choose_model([(-1, observable[train_mask])], yt, gt, config)
            selected_hb = choose_model([(layer, np.column_stack((hidden[train_mask, index], observable[train_mask])))
                                        for index, layer in enumerate(layers)], yt, gt, config)
            selected_h = choose_model([(layer, hidden[train_mask, index]) for index, layer in enumerate(layers)], yt, gt, config)
            hb_layer, h_layer = layers.index(selected_hb['layer']), layers.index(selected_h['layer'])
            scores = {'B': calibrated_scores(selected_b, observable[test_mask]),
                      'H': calibrated_scores(selected_h, hidden[test_mask, h_layer]),
                      'HB': calibrated_scores(selected_hb, np.column_stack((hidden[test_mask, hb_layer], observable[test_mask])))}
            prior = float(np.average(yt, weights=task_weights(gt)))
            scores['prior'] = np.full(len(test_indices), prior)
            selected_b1 = choose_model([(-1, observable[train_mask, :5])], yt, gt, config)
            selected_b2 = choose_model([(layer, prompt_hidden[train_mask, index])
                                        for index, layer in enumerate(layers)], yt, gt, config)
            scores['B1'] = calibrated_scores(selected_b1, observable[test_mask, :5])
            scores['B2'] = calibrated_scores(selected_b2, prompt_hidden[test_mask, layers.index(selected_b2['layer'])])
            if checkpoint != '0':
                selected_b3 = choose_model([(-1, observable[train_mask, 5:])], yt, gt, config)
                scores['B3'] = calibrated_scores(selected_b3, observable[test_mask, 5:])
            else:
                scores['B3'] = scores['prior']
            # Mandatory controls are fitted using this outer training partition only.
            rng = np.random.default_rng(fold_control_seed)
            direction = rng.normal(size=hidden.shape[-1])
            direction /= np.linalg.norm(direction)
            raw_control = hidden[:, h_layer] @ direction
            random_model = fit_model(raw_control[train_mask, None], yt, gt, selected_h['c'])
            scores['random_direction'] = random_model.predict_proba(raw_control[test_mask, None])[:, 1]
            shuffled = permute_within_tasks(yt, gt, rng)
            shuffled_model = fit_model(hidden[train_mask, h_layer], shuffled, gt, selected_h['c'])
            scores['shuffled_training_labels'] = shuffled_model.predict_proba(hidden[test_mask, h_layer])[:, 1]
            # Prefix length is constant at a fixed t. The EOS indicator is observable
            # at t; full eventual length is NEVER used in an early prediction.
            offsets = [records[index]['generated_tokens'] if checkpoint == 'final' else int(checkpoint) for index in indices]
            stopping = np.array([[offset, int(offset > 0 and records[index]['token_ids'][offset - 1]
                          in records[index].get('eos_token_ids', []))] for index, offset in zip(indices, offsets)])
            length_model = fit_model(stopping[train_mask], yt, gt, selected_b['c'])
            scores['length_stopping'] = length_model.predict_proba(stopping[test_mask])[:, 1]
            # Prompt representations are identical within each task by construction;
            # checkpoint 0 produces the dedicated negative-control curve.
            selection = {'fold': fold, 'checkpoint': checkpoint, 'train_tasks': sorted(set(gt)),
                         'test_tasks': sorted(set(groups[test_indices])), 'selection_data': 'inner_grouped_validation',
                         'B': {key: selected_b[key] for key in ('layer', 'c', 'loss', 'selection', 'calibration_audit', 'calibration_scope')},
                         'H': {key: selected_h[key] for key in ('layer', 'c', 'loss', 'selection', 'calibration_audit', 'calibration_scope')},
                         'HB': {key: selected_hb[key] for key in ('layer', 'c', 'loss', 'selection', 'calibration_audit', 'calibration_scope')}}
            selections.append(selection)
            for position, index in enumerate(test_indices):
                predictions.append({'task_id': records[index]['task_id'], 'attempt_id': records[index]['attempt_id'],
                                    'fold': fold, 'checkpoint': checkpoint, 'passed': int(y[index]),
                                    'generated_tokens': records[index]['generated_tokens'],
                                    'truncated': records[index]['truncated'],
                                    'hidden_linear_score': float(selected_h['model'].decision_function(hidden[test_mask, h_layer])[position]),
                                    **{name: float(values[position]) for name, values in scores.items()},
                                    **{f'observable_{column}': float(value) for column, value in enumerate(observable[test_mask][position])}})
            if checkpoint.isdigit() and int(checkpoint) in config['policy']['checkpoints']:
                for name, selected in (('B', selected_b), ('HB', selected_hb)):
                    oof_calibrated = selected['calibrator'].predict_proba(selected['oof'].reshape(-1, 1))[:, 1]
                    lookup = {(records[index]['task_id'], records[index]['attempt_id']): float(value)
                              for index, value in zip(train_indices, oof_calibrated)}
                    policy_candidates[name].append({'checkpoint': int(checkpoint), 'probabilities': lookup})
                    test_probabilities[(name, int(checkpoint))] = {(records[index]['task_id'], records[index]['attempt_id']):
                        float(value) for index, value in zip(test_indices, scores[name])}
                    if model_dir is not None:
                        import joblib
                        model_dir.mkdir(parents=True, exist_ok=True)
                        joblib.dump({'model': selected['model'], 'calibrator': selected['calibrator'],
                                     'layer': selected['layer'], 'checkpoint': int(checkpoint), 'family': name,
                                     'training_tasks': sorted(set(gt))}, model_dir / f'fold{fold}_{name}_t{checkpoint}.joblib')
            print(f'fold={fold + 1}/{len(outer)} checkpoint={checkpoint} completed', flush=True)
        if include_policy:
            for name in ('B', 'HB'):
                selected = select_policy(training_records, policy_candidates[name],
                    minimum_recall=config['policy']['minimum_correct_recall'],
                    budget=config['candidate_bank']['attempts_per_task'],
                    tie_tolerance=config['policy'].get('tie_tolerance', 1e-9))
                selections.append({'fold': fold, 'policy': name, **selected})
                for budget in config['candidate_bank']['budgets']:
                    rows = replay(test_records, test_probabilities[(name, selected['checkpoint'])],
                                  checkpoint=selected['checkpoint'], threshold=selected['threshold'], budget=budget)
                    policy_results.extend([{**row, 'policy': name, 'fold': fold, 'checkpoint': selected['checkpoint'],
                                            'threshold': selected['threshold']} for row in rows])
            for budget in config['candidate_bank']['budgets']:
                for name in ('verifier', 'fixed'):
                    policy_results.extend([{**row, 'fold': fold} for row in replay(test_records, {}, checkpoint=32,
                        threshold=0, budget=budget, policy=name)])
    return {'predictions': predictions, 'selections': selections, 'policy_results': policy_results}


def run_with_limited_threads(*args, **kwargs):
    with threadpool_limits(limits=1):
        return grouped_predictions(*args, **kwargs)
