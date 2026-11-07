from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

from slm_steering.research.protocol import write_json
from slm_steering.research.statistics import (fold_association_test, paired_task_bootstrap, predictive_metrics,
                                            task_auc, task_weights)

SCORE_COLUMNS = ['prior', 'B1', 'B2', 'B3', 'B', 'H', 'HB', 'random_direction', 'shuffled_training_labels', 'length_stopping']


def metric_intervals(y, scores, groups, *, replications, seed):
    tasks, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
    rng = np.random.default_rng(seed)
    values = {key: [] for key in ('roc_auc', 'pr_auc', 'log_loss', 'brier')}
    for _ in range(replications):
        multiplicities = np.bincount(rng.integers(0, len(tasks), len(tasks)), minlength=len(tasks))
        weights = multiplicities[inverse] / counts[inverse]
        selected = weights > 0
        ys, ps, ws = y[selected], scores[selected], weights[selected]
        values['log_loss'].append(log_loss(ys, ps, labels=[0, 1], sample_weight=ws))
        values['brier'].append(brier_score_loss(ys, ps, sample_weight=ws))
        if len(np.unique(ys)) == 2:
            values['roc_auc'].append(roc_auc_score(ys, ps, sample_weight=ws))
        if np.any(ys):
            values['pr_auc'].append(average_precision_score(ys, ps, sample_weight=ws))
    return {key: np.quantile(value, [0.025, 0.975]).tolist() if value else None for key, value in values.items()}


def fixed_effects(primary: pd.DataFrame) -> dict:
    try:
        import statsmodels.api as sm
        from statsmodels.tools.sm_exceptions import PerfectSeparationWarning
        mixed = primary.groupby('task_id')['passed'].nunique()
        frame = primary[primary['task_id'].isin(mixed[mixed == 2].index)].copy()
        if frame.empty:
            return {'status': 'not_identified', 'reason': 'No mixed tasks'}
        columns = [name for name in frame if name.startswith('observable_')]
        observables = frame[columns].copy()
        for name in columns:
            if observables[name].isna().any():
                observables[name + '_missing'] = observables[name].isna().astype(float)
        columns = list(observables.columns)
        observables = observables.fillna(observables.median()).fillna(0)
        # Task-constant columns are absorbed by task intercepts, so they cannot
        # identify a separate coefficient in this complementary regression.
        varying = [name for name in columns if (observables.groupby(frame['task_id'])[name].var().fillna(0) > 1e-12).any()]
        observed = observables[varying]
        observed = (observed - observed.mean()) / observed.std().replace(0, 1)
        design = pd.concat([pd.get_dummies(frame['task_id'], prefix='task', dtype=float), observed,
                            frame[['hidden_linear_score']]], axis=1).astype(float)
        with warnings.catch_warnings():
            warnings.simplefilter('error', PerfectSeparationWarning)
            fitted = sm.GLM(frame['passed'], design, family=sm.families.Binomial()).fit(
                maxiter=200, cov_type='cluster', cov_kwds={'groups': frame['task_id']})
        if not fitted.converged:
            raise ValueError('Task fixed-effects GLM did not converge')
        if not np.isfinite(fitted.params['hidden_linear_score']) or not np.isfinite(fitted.bse['hidden_linear_score']):
            raise ValueError('Task fixed-effects coefficient/clustered variance is not finite')
        return {'status': 'estimated', 'beta_z': float(fitted.params['hidden_linear_score']),
                'clustered_standard_error': float(fitted.bse['hidden_linear_score']),
                'ci95': fitted.conf_int().loc['hidden_linear_score'].tolist(), 'tasks': frame['task_id'].nunique(),
                'trajectories': len(frame), 'varying_observables': varying,
                'interpretation': 'Complementary inference on OOF scores; not an out-of-sample fitted predictor'}
    except (ValueError, np.linalg.LinAlgError, ImportError, Warning) as exc:
        return {'status': 'not_identified', 'reason': str(exc)}


def summarize_signal(predictions: pd.DataFrame, config: dict) -> dict:
    seed = config['statistics']['seed']
    bootstrap = config['statistics']['bootstrap_replications']
    permutations = config['statistics']['permutation_replications']
    metrics, risks, composition = [], [], []
    for checkpoint, frame in predictions.groupby('checkpoint', sort=False):
        y = frame['passed'].to_numpy()
        groups = frame['task_id'].to_numpy()
        task_outcomes = frame.groupby('task_id')['passed'].agg(['min', 'max'])
        mixed = task_outcomes['min'] != task_outcomes['max']
        risks.append({'checkpoint': str(checkpoint), 'trajectories': len(frame), 'tasks': len(task_outcomes),
                      'mixed_tasks': int(mixed.sum()), 'always_correct': int((task_outcomes['min'] == 1).sum()),
                      'always_wrong': int((task_outcomes['max'] == 0).sum()),
                      'pass_rate': float(y.mean()), 'truncation_rate': float(frame['truncated'].mean())})
        composition.extend([{'checkpoint': str(checkpoint), 'task_id': task,
                             'category': 'mixed' if row['min'] != row['max'] else 'always_correct' if row['min'] else 'always_wrong'}
                            for task, row in task_outcomes.iterrows()])
        for name in SCORE_COLUMNS:
            scores = frame[name].to_numpy()
            result = predictive_metrics(y, scores, groups)
            intervals = metric_intervals(y, scores, groups, replications=bootstrap, seed=seed)
            auc_values = task_auc(y, scores, groups)
            auc_ci = paired_task_bootstrap(list(auc_values.values()), replications=bootstrap, seed=seed)['ci95'] if auc_values else None
            metrics.append({'checkpoint': str(checkpoint), 'predictor': name, **result, 'within_task_auc_ci95': auc_ci,
                            'intervals': intervals})
        print(f'checkpoint={checkpoint}: clustered metrics completed', flush=True)
    primary = predictions[predictions['checkpoint'].astype(str) == str(config['activations']['primary_checkpoint'])]
    if primary.empty:
        raise ValueError('Missing predeclared 32-token primary endpoint')
    y, groups = primary['passed'].to_numpy(), primary['task_id'].to_numpy()
    baseline = task_auc(y, primary['B'], groups)
    combined = task_auc(y, primary['HB'], groups)
    hidden = task_auc(y, primary['H'], groups)
    bootstrap_result = paired_task_bootstrap([combined[t] - baseline[t] for t in baseline],
                                            replications=bootstrap, seed=seed) if baseline else None
    hidden_delta = paired_task_bootstrap([hidden[t] - baseline[t] for t in baseline],
                                        replications=bootstrap, seed=seed) if baseline else None
    permutation = fold_association_test(y, primary['HB'], groups, primary['fold'], replications=permutations,
                                       seed=seed, n_folds=config['statistics']['outer_folds'])
    metric_b = predictive_metrics(y, primary['B'], groups)
    metric_hb = predictive_metrics(y, primary['HB'], groups)
    loss_differences = [log_loss(part['passed'], part['HB'], labels=[0, 1]) -
                        log_loss(part['passed'], part['B'], labels=[0, 1]) for _, part in primary.groupby('task_id')]
    loss_delta = paired_task_bootstrap(loss_differences, replications=bootstrap, seed=seed)
    supported = bool(bootstrap_result and bootstrap_result['estimate'] > 0 and bootstrap_result['ci95'][0] > 0 and
                     metric_hb['log_loss'] < metric_b['log_loss'])
    return {'metrics': metrics, 'risk_sets': risks, 'task_composition': composition,
            'H2': {'supported': supported, 'checkpoint': 32, 'delta_hb_minus_b': bootstrap_result,
                   'descriptive_delta_h_minus_b': hidden_delta, 'permutation': permutation,
                   'delta_log_loss_hb_minus_b': loss_delta,
                   'probabilistic_gain_uncertain': loss_delta['ci95'][1] >= 0,
                   'observable_log_loss': metric_b['log_loss'], 'combined_log_loss': metric_hb['log_loss'],
                   'conditional_population': 'Held-out tasks with both correct and incorrect eligible attempts'},
            'fixed_effects': fixed_effects(primary),
            'interval_scope': 'Task bootstrap conditional on fitted OOF scorers; overlapping training folds are not refitted'}


def summarize_policies(frame: pd.DataFrame, config: dict, *, online_validated: bool = False) -> dict:
    if frame.empty:
        return {'status': 'not_run'}
    summaries, comparisons = [], []
    for (budget, policy), rows in frame.groupby(['budget', 'policy']):
        summaries.append({'budget': int(budget), 'policy': policy, 'tasks': len(rows),
                          'resolution': float(rows['resolved'].mean()),
                          'completion_tokens_per_task': float(rows['completion_tokens'].mean()),
                          'attempts_started_per_task': float(rows['attempts_started'].mean()),
                          'fully_generated_attempts_per_task': float(rows['fully_generated_attempts'].mean()),
                          'correct_false_abort_rate': float(rows['correct_false_aborts'].sum() /
                             rows['correct_attempts_started'].sum()) if rows['correct_attempts_started'].sum() else None,
                          'active_correct_false_abort_rate': float(rows['correct_false_aborts'].sum() /
                             rows['active_correct_attempts_started'].sum()) if rows['active_correct_attempts_started'].sum() else None,
                          'correct_attempts_started': int(rows['correct_attempts_started'].sum()),
                          'active_correct_attempts_started': int(rows['active_correct_attempts_started'].sum())})
    for budget in config['candidate_bank']['budgets']:
        latent = frame[(frame['budget'] == budget) & (frame['policy'] == 'HB')].set_index('task_id')
        for name in ('verifier', 'B'):
            baseline = frame[(frame['budget'] == budget) & (frame['policy'] == name)].set_index('task_id').reindex(latent.index)
            if baseline['completion_tokens'].isna().any():
                raise ValueError('Policy comparison requires the same task set')
            comparisons.append({'budget': budget, 'comparison': f'HB minus {name}',
                'resolution_delta': paired_task_bootstrap((latent['resolved'] - baseline['resolved']).to_numpy(),
                    replications=config['statistics']['bootstrap_replications'], seed=config['statistics']['seed']),
                'token_delta': paired_task_bootstrap((latent['completion_tokens'] - baseline['completion_tokens']).to_numpy(),
                    replications=config['statistics']['bootstrap_replications'], seed=config['statistics']['seed'])})
    primary = next(c for c in comparisons if c['budget'] == 20 and c['comparison'] == 'HB minus verifier')
    statistical_support = primary['resolution_delta']['ci95'][0] > -config['policy']['noninferiority_margin'] and primary['token_delta']['ci95'][1] < 0
    observable = next(c for c in comparisons if c['budget'] == 20 and c['comparison'] == 'HB minus B')
    incremental_support = (observable['resolution_delta']['ci95'][0] > -config['policy']['noninferiority_margin']
                           and observable['token_delta']['ci95'][1] < 0)
    return {'summaries': summaries, 'comparisons': comparisons,
            'H3': {'offline_statistical_support': statistical_support, 'online_replay_validated': online_validated,
                   'incremental_over_observables': incremental_support,
                   'system_result_finalized': bool(statistical_support and incremental_support and online_validated),
                   'primary_budget': 20},
            'wall_clock_savings': 'Not inferred from token replay; require online timing',
            'accounting': 'Completion tokens only; prompt-prefill tokens and verification times are separate'}


def generate_figures(predictions: pd.DataFrame, signal: dict, policies: dict, output: Path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figures = output / 'figures'
    figures.mkdir(parents=True, exist_ok=True)
    data = pd.DataFrame(signal['metrics'])
    numeric = data[data['checkpoint'] != 'final'].copy()
    numeric['checkpoint'] = numeric['checkpoint'].astype(int)
    for metric, filename in [('roc_auc', 'temporal_prediction.png'), ('within_task_auc', 'within_task_discrimination.png')]:
        fig, axis = plt.subplots(figsize=(7, 4))
        for name in ['B', 'H', 'HB', 'B2']:
            rows = numeric[numeric['predictor'] == name].sort_values('checkpoint')
            axis.plot(rows['checkpoint'], rows[metric], marker='o', label='H+B' if name == 'HB' else name)
            limits = rows['within_task_auc_ci95'].tolist() if metric == 'within_task_auc' else [row['roc_auc'] for row in rows['intervals']]
            if all(limit is not None for limit in limits):
                limits = np.asarray(limits)
                axis.fill_between(rows['checkpoint'], limits[:, 0], limits[:, 1], alpha=0.12)
        axis.axhline(0.5, color='grey', linestyle='--', linewidth=1)
        axis.axvline(32, color='grey', alpha=0.3)
        axis.set(xlabel='Completion tokens', ylabel='WT-AUC' if metric == 'within_task_auc' else 'ROC-AUC', ylim=(0, 1))
        axis.legend()
        fig.tight_layout()
        fig.savefig(figures / filename, dpi=180)
        plt.close(fig)
    primary = predictions[predictions['checkpoint'].astype(str) == '32']
    primary = primary.copy()
    primary['weight'] = task_weights(primary['task_id'].to_numpy())
    fig, axis = plt.subplots(figsize=(5, 4))
    for name in ['B', 'H', 'HB']:
        bins = pd.cut(primary[name], bins=np.linspace(0, 1, 11), include_lowest=True)
        grouped = primary.groupby(bins, observed=True)
        probability = [np.average(part[name], weights=part['weight']) for _, part in grouped]
        frequency = [np.average(part['passed'], weights=part['weight']) for _, part in grouped]
        axis.plot(probability, frequency, marker='o', label='H+B' if name == 'HB' else name)
    axis.plot([0, 1], [0, 1], color='grey', linestyle='--')
    axis.set(xlabel='OOF calibrated probability', ylabel='Observed correctness', xlim=(0, 1), ylim=(0, 1))
    axis.legend()
    fig.tight_layout()
    fig.savefig(figures / 'calibration_32.png', dpi=180)
    plt.close(fig)
    if policies.get('summaries'):
        fig, axis = plt.subplots(figsize=(6, 4))
        summary = pd.DataFrame(policies['summaries'])
        for name, rows in summary.groupby('policy'):
            axis.plot(rows['completion_tokens_per_task'], rows['resolution'], marker='o', label=name)
        axis.set(xlabel='Generated completion tokens per task', ylabel='Task resolution', ylim=(0, 1))
        axis.legend()
        fig.tight_layout()
        fig.savefig(figures / 'compute_resolution.png', dpi=180)
        plt.close(fig)


def write_artifacts(predictions: pd.DataFrame, policy_frame: pd.DataFrame, config: dict, output: Path,
                    *, online_validated=False) -> dict:
    signal = summarize_signal(predictions, config)
    policies = summarize_policies(policy_frame, config, online_validated=online_validated)
    if 'H3' in policies:
        policies['H3']['h2_supported'] = signal['H2']['supported']
        policies['H3']['operational_benefit_claim_supported'] = bool(
            signal['H2']['supported'] and policies['H3']['system_result_finalized'])
        if not signal['H2']['supported']:
            policies['H3']['interpretation'] = 'Predeclared diagnostic only; no operational benefit claim when H2 is unsupported'
    tables = output / 'tables'
    tables.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(signal['metrics']).to_csv(tables / 'predictive_performance.csv', index=False)
    pd.DataFrame(signal['risk_sets']).to_csv(tables / 'risk_sets.csv', index=False)
    pd.DataFrame(signal['task_composition']).to_csv(tables / 'task_composition.csv', index=False)
    pd.DataFrame(policies.get('summaries', [])).to_csv(tables / 'adaptive_policy.csv', index=False)
    result = {'signal': signal, 'policies': policies, 'status': 'confirmatory_offline_analysis',
              'robustness': 'Run MBPP+ only after the main pipeline has passed its gates'}
    write_json(output / 'results.json', result)
    generate_figures(predictions, signal, policies, output)
    from slm_steering.research.visualization import additional_figures
    additional_figures(predictions, signal, policies, output / 'figures')
    return result
