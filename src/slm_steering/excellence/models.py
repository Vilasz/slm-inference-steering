"""Interpretable nested grouped ablations and a strictly cross-fitted offset reader."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from slm_steering.excellence.audit import assert_preserved, checkpoint_data
from slm_steering.research.probes import calibrate
from slm_steering.research.protocol import file_hash, now, write_json
from slm_steering.research.statistics import paired_task_bootstrap, predictive_metrics, task_auc, task_weights

ORIGINAL_C = [.001, .01, .1, 1., 10.]
STRONG_C = [.00001, .0001, .001]


def loss_vector(y, p):
    p = np.clip(p, np.finfo(float).eps, 1 - np.finfo(float).eps)
    return -(y * np.log(p) + (1 - y) * np.log1p(-p))


def balanced_loss(y, p, groups):
    return float(np.average(loss_vector(y, p), weights=task_weights(groups)))


@dataclass
class Projection:
    spec: dict
    layers: list

    def fit(self, data, indices):
        self.fit_tasks = sorted(set(data['groups'][indices]))
        weights = task_weights(data['groups'][indices])
        cols = self.spec.get('columns', list(range(10)))
        self.impute = SimpleImputer(strategy='median', add_indicator=True, keep_empty_features=True)
        obs = self.impute.fit_transform(data['observable'][indices][:, cols])
        self.obs_scale = StandardScaler().fit(obs, sample_weight=weights)
        self.hidden_scale, self.pca = None, None
        if self.spec['mode'] != 'observable':
            hidden = data['hidden'][indices, self.layers.index(self.spec['layer'])]
            self.hidden_scale = StandardScaler().fit(hidden, sample_weight=weights)
            if self.spec.get('components'):
                components = self.spec['components']
                if components >= min(hidden.shape):
                    raise ValueError('PCA dimensions exceed the training rank budget')
                self.pca = PCA(n_components=components, svd_solver='randomized', whiten=True, random_state=20261017)
                self.pca.fit(self.hidden_scale.transform(hidden))
        return self

    def transform(self, data, indices):
        cols = self.spec.get('columns', list(range(10)))
        obs = self.obs_scale.transform(self.impute.transform(data['observable'][indices][:, cols]))
        if self.spec['mode'] == 'observable':
            return obs
        hidden = self.hidden_scale.transform(data['hidden'][indices, self.layers.index(self.spec['layer'])])
        if self.pca is not None:
            hidden = self.pca.transform(hidden)
        return hidden if self.spec['mode'] == 'residual' else np.column_stack((hidden, obs))


def logistic_fit(x, y, groups, c):
    if len(np.unique(y)) != 2:
        raise ValueError('Single-class training fold; no favorable replacement split is allowed')
    model = LogisticRegression(C=c, solver='lbfgs', max_iter=3000, random_state=0)
    with warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        model.fit(x, y, sample_weight=task_weights(groups))
    return model


def offset_fit(x, y, offset, groups, c):
    """Weighted logistic likelihood with fixed offset; L2 C follows sklearn's sum-loss convention."""
    weights = task_weights(groups)
    def objective(w):
        score = offset + x @ w
        penalty = np.dot(w, w) / (2 * c)
        value = np.sum(weights * (np.logaddexp(0, score) - y * score)) + penalty
        gradient = x.T @ (weights * (expit(score) - y)) + w / c
        return value / weights.sum(), gradient / weights.sum()
    fitted = minimize(objective, np.zeros(x.shape[1]), jac=True, method='L-BFGS-B',
                      options={'maxiter': 2000, 'gtol': 1e-7, 'ftol': 1e-11})
    if not fitted.success or not np.isfinite(fitted.fun):
        raise ValueError(f'Offset fit failed: {fitted.message}')
    return fitted.x


def grouped_splits(data, indices):
    groups, y = data['groups'][indices], data['y'][indices]
    for train, validation in GroupKFold(n_splits=3).split(np.zeros(len(indices)), y, groups):
        fit, score = indices[train], indices[validation]
        if set(data['groups'][fit]) & set(data['groups'][score]):
            raise ValueError('Task leakage in nested fitting')
        yield fit, score


def select(data, indices, variants, c_grid):
    splits = list(grouped_splits(data, indices))
    positions = {int(index): j for j, index in enumerate(indices)}
    grid = []
    for spec in variants:
        prepared = []
        for fit, validation in splits:
            projection = Projection(spec, data['layers']).fit(data, fit)
            prepared.append((fit, validation, projection.transform(data, fit), projection.transform(data, validation)))
        for c in c_grid:
            probabilities = np.full(len(indices), np.nan)
            losses = []
            for fit, validation, x_fit, x_validation in prepared:
                model = logistic_fit(x_fit, data['y'][fit], data['groups'][fit], c)
                p = model.predict_proba(x_validation)[:, 1]
                probabilities[[positions[int(i)] for i in validation]] = p
                losses.append(balanced_loss(data['y'][validation], p, data['groups'][validation]))
            grid.append({'spec': spec, 'c': c,
                         'loss': balanced_loss(data['y'][indices], probabilities, data['groups'][indices]),
                         'standard_error': float(np.std(losses, ddof=1) / np.sqrt(len(losses)))})
    best = min(grid, key=lambda r: r['loss'])
    candidates = [r for r in grid if r['loss'] <= best['loss'] + best['standard_error']]
    chosen = min(candidates, key=lambda r: (r['c'], r['spec'].get('components', 0), r['spec'].get('layer', -1)))
    projection = Projection(chosen['spec'], data['layers']).fit(data, indices)
    model = logistic_fit(projection.transform(data, indices), data['y'][indices], data['groups'][indices], chosen['c'])
    return {'projection': projection, 'model': model, 'choice': chosen, 'selection_grid': grid}


def raw_predict(fitted, data, indices):
    return fitted['model'].decision_function(fitted['projection'].transform(data, indices))


def fitted_predict(fitted, data, indices):
    score = raw_predict(fitted, data, indices)
    return fitted['calibrator'].predict_proba(score.reshape(-1, 1))[:, 1]


def fit_procedure(data, indices, variants, c_grid):
    selected = select(data, indices, variants, c_grid)
    honest = np.full(len(indices), np.nan)
    positions = {int(i): j for j, i in enumerate(indices)}
    audit = []
    for fit, validation in grouped_splits(data, indices):
        local = select(data, fit, variants, c_grid)
        honest[[positions[int(i)] for i in validation]] = raw_predict(local, data, validation)
        audit.append({'fit_tasks': sorted(set(data['groups'][fit])), 'score_tasks': sorted(set(data['groups'][validation])),
                      'choice': local['choice']})
    if not np.isfinite(honest).all():
        raise ValueError('Incomplete selection-cross-fitted calibration scores')
    selected['calibrator'] = calibrate(honest, data['y'][indices], data['groups'][indices])
    selected['calibration_audit'] = audit
    return selected


class ResidualProcedure:
    """Cross-fit the entire calibrated first-stage B procedure, including its selection."""
    def __init__(self, data):
        self.data, self.base_cache, self.offset_cache = data, {}, {}
        self.base_spec = [{'mode': 'observable', 'columns': list(range(10))}]

    def base(self, indices):
        key = tuple(indices.tolist())
        if key not in self.base_cache:
            self.base_cache[key] = fit_procedure(self.data, indices, self.base_spec, ORIGINAL_C)
        return self.base_cache[key]

    def honest_offsets(self, indices):
        key = tuple(indices.tolist())
        if key not in self.offset_cache:
            p = np.full(len(indices), np.nan)
            positions = {int(i): j for j, i in enumerate(indices)}
            audit = []
            for fit, validation in grouped_splits(self.data, indices):
                # Both model selection and calibration exclude validation labels here.
                first = self.base(fit)
                p[[positions[int(i)] for i in validation]] = fitted_predict(first, self.data, validation)
                audit.append({'first_stage_fit_tasks': sorted(set(self.data['groups'][fit])),
                              'offset_score_tasks': sorted(set(self.data['groups'][validation]))})
            self.offset_cache[key] = (logit(np.clip(p, 1e-12, 1 - 1e-12)), audit)
        return self.offset_cache[key]

    def fit(self, indices, variants, c_grid):
        grid = []
        splits = list(grouped_splits(self.data, indices))
        positions = {int(i): j for j, i in enumerate(indices)}
        for spec in variants:
            prepared = []
            for fit, validation in splits:
                projection = Projection(spec, self.data['layers']).fit(self.data, fit)
                train_offset, _ = self.honest_offsets(fit)
                valid_offset = logit(np.clip(fitted_predict(self.base(fit), self.data, validation), 1e-12, 1 - 1e-12))
                prepared.append((fit, validation, projection.transform(self.data, fit), projection.transform(self.data, validation), train_offset, valid_offset))
            for c in c_grid:
                p = np.full(len(indices), np.nan)
                losses = []
                for fit, validation, x, xv, offset, offsetv in prepared:
                    w = offset_fit(x, self.data['y'][fit], offset, self.data['groups'][fit], c)
                    prediction = expit(offsetv + xv @ w)
                    p[[positions[int(i)] for i in validation]] = prediction
                    losses.append(balanced_loss(self.data['y'][validation], prediction, self.data['groups'][validation]))
                grid.append({'spec': spec, 'c': c, 'loss': balanced_loss(self.data['y'][indices], p, self.data['groups'][indices]),
                             'standard_error': float(np.std(losses, ddof=1) / np.sqrt(3))})
        best = min(grid, key=lambda r: r['loss'])
        chosen = min([r for r in grid if r['loss'] <= best['loss'] + best['standard_error']],
                     key=lambda r: (r['c'], r['spec']['layer']))
        projection = Projection(chosen['spec'], self.data['layers']).fit(self.data, indices)
        offset, audit = self.honest_offsets(indices)
        w = offset_fit(projection.transform(self.data, indices), self.data['y'][indices], offset, self.data['groups'][indices], chosen['c'])
        return {'projection': projection, 'coefficient': w, 'base': self.base(indices), 'choice': chosen,
                'selection_grid': grid, 'offset_audit': audit,
                'probability_link': 'Native logistic offset probabilities; no second Platt fit'}


def residual_predict(fitted, data, indices):
    offset = logit(np.clip(fitted_predict(fitted['base'], data, indices), 1e-12, 1 - 1e-12))
    return expit(offset + fitted['projection'].transform(data, indices) @ fitted['coefficient'])


def specifications(layers):
    columns = {'prompt_only': list(range(5)), 'mean_nll': [5], 'last_nll': [6],
               'mean_entropy': [7], 'max_entropy': [8], 'logit_margin': [9],
               'nll_pair': [5, 6], 'entropy_pair': [7, 8], 'confidence_only': list(range(5, 10)),
               'B_reproduced': list(range(10))}
    result = {name: ([{'mode': 'observable', 'columns': cols}], ORIGINAL_C) for name, cols in columns.items()}
    result['HB_stronger'] = ([{'mode': 'combined', 'layer': layer} for layer in layers], STRONG_C)
    for components in (16, 64):
        result[f'HB_pca{components}'] = ([{'mode': 'combined', 'layer': layer, 'components': components} for layer in layers], [.001, .01, .1, 1.])
    result['H_residual_B'] = ([{'mode': 'residual', 'layer': layer} for layer in layers], STRONG_C)
    return result


def run(root: Path):
    assert_preserved(root)
    report, directory = root / 'reports/excellence/models', root / 'runs/excellence/models_v1'
    report.mkdir(parents=True, exist_ok=True)
    directory.mkdir(parents=True, exist_ok=True)
    data = checkpoint_data(root)
    original = pd.read_csv(root / 'runs/probes/protocol_v0.2/outer_predictions.csv')
    original = original[original.checkpoint == '32'].set_index(['task_id', 'attempt_id']).loc[data['identities']].reset_index()
    fold_ids = original.fold.to_numpy()
    families = specifications(data['layers'])
    contract = {'status': 'post_hoc_exploratory', 'endpoint': 32, 'source_bank_sha256': file_hash(root / 'runs/candidate_bank/definitive_v0.2/trajectories.jsonl'),
                'source_sha256': file_hash(Path(__file__)), 'families': families,
                'outer_folds': 'Exactly the original v0.2 task partition', 'inner_folds': 3,
                'calibration': 'Whole model selection cross-fitted inside outer training',
                'residual': 'Cross-fit entire calibrated first-stage B on each residual training partition; no second Platt fit',
                'pca': 'Training-only weighted scaling followed by unweighted randomized whitened PCA; transforms refitted inside every split',
                'inference': 'Pointwise task bootstrap conditional on fitted scores; exploratory comparisons, no new confirmation'}
    plan = report / 'plan.json'
    serial = json.loads(json.dumps(contract))
    if plan.exists() and json.loads(plan.read_text(encoding='utf-8')) != serial:
        raise ValueError('Exploratory model contract changed; version the output before continuing')
    write_json(plan, serial)
    predictions, audits = [], []
    total = len(families) * 5
    with threadpool_limits(limits=1):
        for fold in range(5):
            train, test = np.flatnonzero(fold_ids != fold), np.flatnonzero(fold_ids == fold)
            residual = ResidualProcedure(data)
            for family, (variants, grid) in families.items():
                stem = directory / f'{family}_fold{fold}'
                metadata = stem.with_suffix('.json')
                if metadata.exists():
                    saved = json.loads(metadata.read_text(encoding='utf-8'))
                    if saved['contract_sha256'] != file_hash(plan) or saved['predictions_sha256'] != file_hash(stem.with_suffix('.csv')):
                        raise ValueError('Completed exploratory unit changed')
                    predictions.extend(pd.read_csv(stem.with_suffix('.csv')).to_dict('records'))
                    audits.append(saved)
                    continue
                start = time.perf_counter()
                fitted = residual.fit(train, variants, grid) if family == 'H_residual_B' else fit_procedure(data, train, variants, grid)
                p = residual_predict(fitted, data, test) if family == 'H_residual_B' else fitted_predict(fitted, data, test)
                if family == 'B_reproduced' and not np.allclose(p, original.loc[test, 'B'], atol=1e-9, rtol=0):
                    raise ValueError('Exploratory baseline does not reproduce original B')
                rows = [{'task_id': data['groups'][i], 'attempt_id': data['identities'][i][1],
                         'fold': fold, 'family': family, 'passed': int(data['y'][i]), 'probability': float(prob)} for i, prob in zip(test, p)]
                joblib.dump(fitted, stem.with_suffix('.joblib'))
                pd.DataFrame(rows).to_csv(stem.with_suffix('.csv'), index=False)
                saved = {'family': family, 'fold': fold, 'seconds': time.perf_counter() - start,
                         'fit_tasks': sorted(set(data['groups'][train])), 'test_tasks': sorted(set(data['groups'][test])),
                         'choice': fitted['choice'], 'selection_grid': fitted['selection_grid'],
                         'cross_fitting': fitted.get('calibration_audit', fitted.get('offset_audit')),
                         'contract_sha256': file_hash(plan), 'predictions_sha256': file_hash(stem.with_suffix('.csv')),
                         'model_sha256': file_hash(stem.with_suffix('.joblib'))}
                write_json(metadata, saved)
                predictions.extend(rows)
                audits.append(saved)
                write_json(report / 'status.json', {'status': 'running', 'completed_units': len(audits), 'expected_units': total,
                                                    'last_family': family, 'fold': fold, 'updated_at': now()})
                print(f"models {len(audits)}/{total}: fold={fold} {family} ({saved['seconds']:.1f}s)", flush=True)
    # Original families enter only as untouched benchmarks, not newly selected winners.
    for family in ('B', 'H', 'HB'):
        predictions.extend([{'task_id': row.task_id, 'attempt_id': row.attempt_id, 'fold': row.fold,
                             'family': family + '_original', 'passed': row.passed, 'probability': getattr(row, family)}
                            for row in original.itertuples()])
    frame = pd.DataFrame(predictions).sort_values(['family', 'task_id', 'attempt_id'])
    frame.to_csv(report / 'predictions.csv', index=False)
    summaries = []
    baseline = original.set_index(['task_id', 'attempt_id'])
    for family, part in frame.groupby('family'):
        part = part.set_index(['task_id', 'attempt_id']).reindex(baseline.index).reset_index()
        y, groups, p = part.passed.to_numpy(), part.task_id.to_numpy(), part.probability.to_numpy()
        auc = task_auc(y, p, groups)
        auc_b = task_auc(y, baseline.B.to_numpy(), groups)
        task_delta = [auc[t] - auc_b[t] for t in auc]
        losses = loss_vector(y, p) - loss_vector(y, baseline.B.to_numpy())
        per_task_loss = pd.DataFrame({'task': groups, 'loss': losses}).groupby('task', sort=True).loss.mean().to_numpy()
        summaries.append({'family': family, **predictive_metrics(y, p, groups),
                          'delta_wt_auc_vs_B': paired_task_bootstrap(task_delta, replications=2000, seed=20260921),
                          'delta_log_loss_vs_B': paired_task_bootstrap(per_task_loss, replications=2000, seed=20260921)})
    assert_preserved(root)
    write_json(report / 'results.json', {'status': 'completed_exploratory_nested_grouped', 'created_at': now(),
                                         'summaries': summaries, 'units': len(audits), 'selection_audits': audits,
                                         'plan_sha256': file_hash(plan), 'predictions_sha256': file_hash(report / 'predictions.csv')})
    write_json(report / 'status.json', {'status': 'completed', 'completed_units': len(audits), 'expected_units': total, 'updated_at': now()})
