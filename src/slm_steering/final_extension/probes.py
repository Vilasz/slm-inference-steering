"""Four matched probe families; every preprocessing/selection step is source-only."""
from __future__ import annotations

import copy
from dataclasses import dataclass
import time

import joblib
import numpy as np
from scipy.special import expit, logit
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from slm_steering.final_extension.common import digest, RUN, REPORT, atomic_json, file_hash, read_json
from slm_steering.research.statistics import task_weights

FAMILIES = ('B_linear', 'HB_linear', 'B_MLP', 'HB_MLP')
SEEDS = (20261104, 20261105, 20261106)
C_GRID = (.001, .01, .1, 1.)
WIDTHS = (16, 32)


def grouped_partition(groups, fraction=.2, namespace='calibration'):
    unique = sorted(set(groups), key=lambda t: (digest(f'final-v1-{namespace}:{t}'), t))
    count = max(1, int(round(len(unique) * fraction)))
    held = set(unique[:count])
    return np.array([t not in held for t in groups]), np.array([t in held for t in groups])


def weighted_loss(y, p, groups):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return float(np.average(-y * np.log(p) - (1 - y) * np.log1p(-p), weights=task_weights(groups)))


@dataclass
class Features:
    use_hidden: bool

    def fit(self, data, indices):
        self.fit_tasks = sorted(set(data['groups'][indices]))
        self.imputer = SimpleImputer(strategy='median', add_indicator=True, keep_empty_features=True)
        b = self.imputer.fit_transform(data['observable'][indices])
        self.bscale = StandardScaler().fit(b, sample_weight=task_weights(data['groups'][indices]))
        if self.use_hidden:
            h = data['hidden'][indices]
            if min(h.shape) <= 64:
                raise ValueError('PCA64 training rank insufficient; do not silently change representation')
            self.hscale = StandardScaler().fit(h, sample_weight=task_weights(data['groups'][indices]))
            self.pca = PCA(n_components=64, svd_solver='randomized', whiten=True, random_state=20261104)
            self.pca.fit(self.hscale.transform(h))
        return self

    def transform(self, data, indices):
        b = self.bscale.transform(self.imputer.transform(data['observable'][indices]))
        if not self.use_hidden:
            return b.astype(np.float32)
        h = self.pca.transform(self.hscale.transform(data['hidden'][indices]))
        return np.column_stack((h, b)).astype(np.float32)


def make_mlp(dimensions, width):
    import torch
    return torch.nn.Sequential(torch.nn.Linear(dimensions, width), torch.nn.ReLU(),
                               torch.nn.Dropout(.1), torch.nn.Linear(width, 1))


def fit_mlp(x, y, groups, *, width, seed, validation=None, epochs=200):
    import torch
    torch.manual_seed(seed)
    net = make_mlp(x.shape[1], width).cpu()
    optimizer = torch.optim.AdamW(net.parameters(), lr=.001, weight_decay=.001)
    xt, yt = torch.tensor(x), torch.tensor(y, dtype=torch.float32)
    weights = torch.tensor(task_weights(groups), dtype=torch.float32)
    best, best_loss, chosen_epoch, stale = None, float('inf'), epochs, 0
    for epoch in range(1, epochs + 1):
        net.train()
        optimizer.zero_grad()
        loss = (torch.nn.functional.binary_cross_entropy_with_logits(net(xt).flatten(), yt, reduction='none') * weights).mean()
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite MLP loss')
        loss.backward()
        optimizer.step()
        if validation is not None:
            xv, yv, gv = validation
            net.eval()
            with torch.inference_mode():
                p = torch.sigmoid(net(torch.tensor(xv)).flatten()).numpy()
            value = weighted_loss(yv, p, gv)
            if value < best_loss - 1e-5:
                best_loss, best, chosen_epoch, stale = value, copy.deepcopy(net.state_dict()), epoch, 0
            else:
                stale += 1
            if stale >= 20:
                break
    if validation is not None:
        net.load_state_dict(best)
    net.eval()
    # NumPy arrays keep joblib artifacts independent of CUDA/pickle tensor storage.
    return {'state': {k: v.detach().numpy() for k, v in net.state_dict().items()},
            'width': width, 'dimensions': x.shape[1], 'epoch': chosen_epoch, 'seed': seed}


def mlp_predict(model, x):
    # Exact evaluation network; dropout is disabled. No torch model deserialization.
    state = model['state']
    hidden = np.maximum(x @ state['0.weight'].T + state['0.bias'], 0)
    return (hidden @ state['3.weight'].T + state['3.bias']).ravel()


def train_raw(data, indices, family, hyperparameter, *, epochs=None):
    features = Features(family.startswith('HB')).fit(data, indices)
    x, y, groups = features.transform(data, indices), data['y'][indices], data['groups'][indices]
    if len(np.unique(y)) != 2:
        raise ValueError('Single-class source training; do not choose a favorable replacement split')
    if family.endswith('linear'):
        model = LogisticRegression(C=hyperparameter, solver='lbfgs', max_iter=3000, random_state=20261104)
        model.fit(x, y, sample_weight=task_weights(groups))
        return {'features': features, 'models': [model], 'family': family}
    if epochs is None:
        raise ValueError('Final MLP refit needs source-selected fixed epochs')
    models = [fit_mlp(x, y, groups, width=hyperparameter, seed=seed, epochs=epochs) for seed in SEEDS]
    return {'features': features, 'models': models, 'family': family}


def raw_scores(fitted, data, indices):
    x = fitted['features'].transform(data, indices)
    if fitted['family'].endswith('linear'):
        return fitted['models'][0].decision_function(x)
    # Equal-weight probability ensemble; all three seeds retained, none selected.
    return logit(np.clip(np.mean([expit(mlp_predict(m, x)) for m in fitted['models']], axis=0), 1e-12, 1 - 1e-12))


def fit_family(data, family):
    groups = data['groups']
    fit_mask, cal_mask = grouped_partition(groups)
    fit_indices, cal_indices = np.flatnonzero(fit_mask), np.flatnonzero(cal_mask)
    candidates = C_GRID if family.endswith('linear') else WIDTHS
    grid, split_audit = [], []
    splits = list(GroupKFold(n_splits=3).split(fit_indices, data['y'][fit_indices], groups[fit_indices]))
    for parameter in candidates:
        losses, epochs, seed_losses = [], [], []
        for fold, (train, validation) in enumerate(splits):
            ti, vi = fit_indices[train], fit_indices[validation]
            if set(groups[ti]) & set(groups[vi]) or set(groups[ti]) & set(groups[cal_indices]):
                raise ValueError('Source task leakage')
            if family.endswith('linear'):
                fitted = train_raw(data, ti, family, parameter)
            else:
                a, e = grouped_partition(groups[ti], namespace=f'early-stop-{fold}')
                ai, ei = ti[a], ti[e]
                # Early stopping validation excludes all inner test and calibration tasks.
                early_features = Features(family.startswith('HB')).fit(data, ai)
                trained = [fit_mlp(early_features.transform(data, ai), data['y'][ai], groups[ai],
                                  width=parameter, seed=seed,
                                  validation=(early_features.transform(data, ei), data['y'][ei], groups[ei])) for seed in SEEDS]
                epoch = int(np.median([m['epoch'] for m in trained]))
                epochs.append(epoch)
                fitted = train_raw(data, ti, family, parameter, epochs=epoch)
                x = fitted['features'].transform(data, vi)
                seed_losses.append([weighted_loss(data['y'][vi], expit(mlp_predict(m, x)), groups[vi])
                                    for m in fitted['models']])
            losses.append(weighted_loss(data['y'][vi], expit(raw_scores(fitted, data, vi)), groups[vi]))
            split_audit.append({'parameter': parameter, 'fold': fold, 'fit_tasks': sorted(set(groups[ti])),
                                'validation_tasks': sorted(set(groups[vi])),
                                'early_stopping_tasks': sorted(set(groups[ei])) if family.endswith('MLP') else []})
        grid.append({'parameter': parameter, 'loss': float(np.mean(losses)), 'fold_losses': losses,
                     'epochs': epochs, 'seed_losses': seed_losses})
    chosen = min(grid, key=lambda r: (r['loss'], r['parameter']))
    epochs = int(np.median(chosen['epochs'])) if chosen['epochs'] else None
    fitted = train_raw(data, fit_indices, family, chosen['parameter'], epochs=epochs)
    calibrator = LogisticRegression(C=1e6, max_iter=2000, random_state=20261104)
    calibrator.fit(raw_scores(fitted, data, cal_indices)[:, None], data['y'][cal_indices],
                   sample_weight=task_weights(groups[cal_indices]))
    if calibrator.coef_[0, 0] <= 0:
        # A decreasing source calibration would reverse the ranking; never hide it.
        raise ValueError('Nonpositive source calibration slope; freeze as infeasible before external outcomes')
    fitted['calibrator'] = calibrator
    probability = predict(fitted, data, cal_indices)
    correct = probability[data['y'][cal_indices] == 1]
    if len(correct) == 0:
        raise ValueError('No correct source calibration candidates')
    # Largest observed threshold keeping >=97% of active source positives.
    ordered = np.sort(correct)
    threshold = float(ordered[int(np.floor(.03 * len(ordered)))])
    fitted['threshold'] = threshold
    fitted['audit'] = {'family': family, 'grid': grid, 'chosen': chosen, 'final_epochs': epochs,
                       'fit_tasks': sorted(set(groups[fit_indices])), 'calibration_tasks': sorted(set(groups[cal_indices])),
                       'split_audit': split_audit, 'calibration_slope': float(calibrator.coef_[0, 0]),
                       'threshold': threshold, 'calibration_positive_candidates': len(correct),
                       'threshold_recall_development_only': float(np.mean(correct >= threshold)),
                       'uncertainty': 'Final external task bootstrap is conditional on these fitted source procedures.'}
    return fitted


def predict(fitted, data, indices):
    return fitted['calibrator'].predict_proba(raw_scores(fitted, data, indices)[:, None])[:, 1]


def train_all(data, name):
    directory = RUN / name / 'probes'
    directory.mkdir(parents=True, exist_ok=True)
    result = {}
    with threadpool_limits(limits=1):
        for family in FAMILIES:
            path = directory / f'{family}.joblib'
            metadata_path = path.with_suffix('.json')
            if path.exists() and metadata_path.exists():
                metadata = read_json(metadata_path)
                if file_hash(path) != metadata['sha256'] or metadata['source_contract'] != data['contract']:
                    raise ValueError('Saved source procedure changed')
                result[family] = joblib.load(path)
                continue
            start = time.perf_counter()
            fitted = fit_family(data, family)
            temp = path.with_suffix('.tmp.joblib')
            joblib.dump(fitted, temp)
            temp.replace(path)
            atomic_json(metadata_path, {'sha256': file_hash(path), 'source_contract': data['contract'],
                                       'seconds': time.perf_counter() - start, **fitted['audit']})
            print(f'{name} {family} fitted in {time.perf_counter()-start:.1f}s', flush=True)
            result[family] = fitted
    atomic_json(REPORT / f'{name}_probe_manifest.json', {f: read_json(directory / f'{f}.json') for f in FAMILIES})
    return result
