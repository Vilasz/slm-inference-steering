"""Training-only geometry, direction stability, and task-count sensitivity."""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits

from slm_steering.excellence.audit import assert_preserved, checkpoint_data
from slm_steering.excellence.models import fit_procedure, fitted_predict
from slm_steering.research.protocol import digest, now, write_json
from slm_steering.research.statistics import paired_task_bootstrap, predictive_metrics, task_auc, task_weights


def centered_correlation(frame, a, b):
    x = frame[a] - frame.groupby('task_id')[a].transform('mean')
    y = frame[b] - frame.groupby('task_id')[b].transform('mean')
    return float(np.corrcoef(x, y)[0, 1]) if x.std() and y.std() else None


def run(root: Path):
    assert_preserved(root)
    output = root / 'reports/excellence/geometry'
    output.mkdir(parents=True, exist_ok=True)
    data = checkpoint_data(root)
    original = pd.read_csv(root / 'runs/probes/protocol_v0.2/outer_predictions.csv')
    original = original[original.checkpoint == '32'].set_index(['task_id', 'attempt_id']).loc[data['identities']].reset_index()
    folds = original.fold.to_numpy()
    spectra, directions, contributions, redundancies, curves = [], [], [], [], []
    with threadpool_limits(limits=1):
        for fold in range(5):
            train, test = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
            for j, layer in enumerate(data['layers']):
                hidden = data['hidden'][train, j]
                # Descriptive PCA uses outer TRAINING representations only. No target/test embedding is fitted.
                pca = PCA(n_components=128, svd_solver='randomized', random_state=20261017).fit(hidden)
                ratios = pca.explained_variance_ratio_
                spectra.append({'fold': fold, 'layer': layer, 'dimension': hidden.shape[1],
                                'training_tasks': len(set(data['groups'][train])), 'training_candidates': len(train),
                                'variance_top16': float(ratios[:16].sum()), 'variance_top64': float(ratios[:64].sum()),
                                'variance_top128': float(ratios.sum())})
            fitted = joblib.load(root / f'runs/probes/protocol_v0.2/models/fold{fold}_HB_t32.joblib')
            direction = fitted['model']['logistic'].coef_[0]
            scale = fitted['model']['scale'].scale_
            dimension = data['hidden'].shape[-1]
            directions.append({'fold': fold, 'layer': fitted['layer'], 'raw': direction[:dimension] / scale[:dimension],
                               'standardized': direction[:dimension]})
            x = np.column_stack((data['hidden'][test, data['layers'].index(fitted['layer'])], data['observable'][test]))
            transformed = fitted['model'][:-1].transform(x)
            hpart, bpart = transformed[:, :dimension] @ direction[:dimension], transformed[:, dimension:] @ direction[dimension:]
            contributions.append({'fold': fold, 'layer': fitted['layer'], 'hidden_coefficient_l2': float(np.linalg.norm(direction[:dimension])),
                                  'observable_coefficient_l2': float(np.linalg.norm(direction[dimension:])),
                                  'held_out_hidden_logit_sd': float(np.std(hpart)), 'held_out_observable_logit_sd': float(np.std(bpart)),
                                  'calibrator_slope': float(fitted['calibrator'].coef_[0, 0])})
            # Predict the learned H+B hidden contribution from B alone, with no test fitting.
            layer_index = data['layers'].index(fitted['layer'])
            xt = np.column_stack((data['hidden'][train, layer_index], data['observable'][train]))
            train_transform = fitted['model'][:-1].transform(xt)
            h_train = train_transform[:, :dimension] @ direction[:dimension]
            ridge = Ridge(alpha=1., fit_intercept=True).fit(train_transform[:, dimension:], h_train,
                                                          sample_weight=task_weights(data['groups'][train]))
            prediction = ridge.predict(transformed[:, dimension:])
            weighted_mean = np.average(hpart, weights=task_weights(data['groups'][test]))
            numerator = np.average((hpart - prediction) ** 2, weights=task_weights(data['groups'][test]))
            denominator = np.average((hpart - weighted_mean) ** 2, weights=task_weights(data['groups'][test]))
            redundancies.append({'fold': fold, 'hidden_contribution_prediction_R2_from_B': float(1 - numerator / denominator),
                                 'scope': 'Fixed alpha=1 descriptive score predictability; not a test of conditional correctness information'})
            # Fixed, disclosed learning-curve algorithms. No subset chooses settings using outer-test labels.
            all_tasks = sorted(set(data['groups'][train]), key=lambda task: digest('excellence-task-curve-v1:' + task))
            for fraction in (.3, .6, 1.):
                tasks = all_tasks[:max(12, int(round(fraction * len(all_tasks))))]
                subset = train[np.isin(data['groups'][train], tasks)]
                for name, spec in [('B_fixed_C01', {'mode': 'observable', 'columns': list(range(10))}),
                                    ('HB_PCA16_fixed_layer13_C01', {'mode': 'combined', 'layer': 13, 'components': 16})]:
                    model = fit_procedure(data, subset, [spec], [.01])
                    p = fitted_predict(model, data, test)
                    metrics = predictive_metrics(data['y'][test], p, data['groups'][test])
                    for i, probability in zip(test, p):
                        curves.append({'fold': fold, 'fraction': fraction, 'training_tasks': len(tasks), 'family': name,
                                       'task_id': data['groups'][i], 'attempt_id': data['identities'][i][1],
                                       'passed': int(data['y'][i]), 'probability': float(probability), **metrics})
            print(f'geometry fold {fold + 1}/5 completed', flush=True)
    pairwise = []
    for i, a in enumerate(directions):
        for b in directions[i + 1:]:
            if a['layer'] != b['layer']:
                continue
            pairwise.append({'fold_a': a['fold'], 'fold_b': b['fold'], 'layer': a['layer'],
                             'cosine_raw_coordinates': float(np.dot(a['raw'], b['raw']) / (np.linalg.norm(a['raw']) * np.linalg.norm(b['raw']))),
                             'scope': 'Same layer only; overlapping training tasks; no independent stability inference'})
    curve_frame = pd.DataFrame(curves)
    curve_frame.to_csv(output / 'learning_curve_predictions.csv', index=False)
    summaries = []
    for (fraction, family), part in curve_frame.groupby(['fraction', 'family']):
        y, p, groups = part.passed.to_numpy(), part.probability.to_numpy(), part.task_id.to_numpy()
        auc = list(task_auc(y, p, groups).values())
        summaries.append({'fraction': float(fraction), 'family': family,
                          'training_task_range': [int(part.training_tasks.min()), int(part.training_tasks.max())],
                          **predictive_metrics(y, p, groups),
                          'within_task_auc_interval': paired_task_bootstrap(auc, replications=2000, seed=20260921)})
    result = {'status': 'completed_exploratory_geometry', 'created_at': now(), 'training_spectra': spectra,
              'same_layer_direction_cosines': pairwise, 'block_contributions': contributions,
              'redundancy_R2': redundancies, 'within_task_score_correlation_B_H': centered_correlation(original, 'B', 'H'),
              'within_task_score_correlation_B_HB': centered_correlation(original, 'B', 'HB'),
              'learning_curves': summaries,
              'learning_curve_scope': 'Fixed C=.01, fixed layer13 and PCA16; deterministic task subsets; not the original selected algorithm or a claim about optimal sample efficiency',
              'limitations': ['PCA variance is a training-only descriptive property, not correctness information',
                             'Direction angles depend on coordinates; only raw-space directions in the same layer compared',
                             'No new causal mechanism or conditional independence is identified']}
    write_json(output / 'results.json', result)
    plot(output, pd.DataFrame(spectra), pd.DataFrame(summaries))
    assert_preserved(root)


def plot(output, spectra, curves):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    grouped = spectra.groupby('layer')[['variance_top16', 'variance_top64', 'variance_top128']].mean()
    grouped.plot.bar(ax=axes[0], ylim=(0, 1), ylabel='Fração da variância (treino externo)')
    axes[0].set_xlabel('Camada')
    for family, part in curves.groupby('family'):
        part = part.sort_values('fraction')
        axes[1].plot(part.fraction, part.within_task_auc, 'o-', label=family)
    axes[1].set(xlabel='Fração de tarefas de treino externo', ylabel='WT-AUC externa', ylim=(.45, .75))
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    for ext in ('png', 'svg', 'pdf'):
        fig.savefig(output / f'geometry_learning.{ext}', dpi=180)
    plt.close(fig)
