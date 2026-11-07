"""Paired task inference and replay after the complete frozen external sample."""
from __future__ import annotations

import time
import joblib
import numpy as np
import pandas as pd

from slm_steering.final_extension.common import REPORT, RUN, atomic_json, read_json, validate_freeze
from slm_steering.final_extension.collection import records, load_features
from slm_steering.final_extension.probes import FAMILIES, predict
from slm_steering.research.statistics import task_auc, predictive_metrics, paired_task_bootstrap, task_weights
from slm_steering.excellence.economics import simulate


def paired_auc_contrast(frame, left, right, *, replications=10000, seed=20261104):
    tables = {}
    for family in (left, right):
        part = frame[frame.family == family].sort_values(['task_id', 'attempt_id'])
        if part.duplicated(['task_id', 'attempt_id']).any():
            raise ValueError('Duplicate scored identity')
        tables[family] = part
    a, b = tables[left], tables[right]
    if not a[['task_id', 'attempt_id', 'passed']].reset_index(drop=True).equals(
            b[['task_id', 'attempt_id', 'passed']].reset_index(drop=True)):
        raise ValueError('Paired AUC requires identical candidates and labels')
    av, bv = (task_auc(t.passed, t.probability, t.task_id) for t in (a, b))
    if not av:
        return {'estimate': None, 'ci95': None, 'tasks': 0, 'interpretation': 'No mixed active tasks'}
    if set(av) != set(bv):
        raise ValueError('Mixed task denominator differs')
    delta = {t: av[t] - bv[t] for t in av}
    result = paired_task_bootstrap(list(delta.values()), replications=replications, seed=seed)
    low, high = result['ci95']
    result.update(per_task=delta, interpretation='positive' if low > 0 else 'deterioration' if high < 0 else 'inconclusive',
                  scope='Task bootstrap conditional on frozen fitted source probes and observed mixed risk set')
    return result


def evaluate():
    validate_freeze()
    from slm_steering.final_extension.bank_audit import audit
    audit('external')
    design = read_json(REPORT / 'final_design.json')
    expected = {(t, j) for t in design['external_ids'] for j in range(20)}
    predictions, metrics, contrasts, replay_rows, timing = [], [], {}, [], []
    for name in ('deepseek', 'qwen'):
        directory = RUN / name / 'external'
        manifest = read_json(directory / 'manifest.json')
        bank = records(directory)
        if manifest['status'] != 'completed' or {(r['task_id'], r['attempt_id']) for r in bank} != expected:
            raise ValueError('External bank incomplete; no inferential report from partial outcomes')
        data = load_features(name, 'external')
        fitted = {f: joblib.load(RUN / name / 'probes' / f'{f}.joblib') for f in FAMILIES}
        score_maps = {}
        for family, procedure in fitted.items():
            start = time.perf_counter()
            p = predict(procedure, data, np.arange(len(data['y']))) if len(data['y']) else np.array([])
            timing.append({'model': name, 'family': family, 'seconds': time.perf_counter() - start,
                           'candidates': len(p), 'scope': 'Batched CPU scoring including preprocessing, excludes model generation'})
            rows = [{'model': name, 'family': family, 'task_id': r['task_id'], 'attempt_id': r['attempt_id'],
                     'passed': int(r['passed']), 'probability': float(value)} for r, value in zip(data['records'], p)]
            predictions.extend(rows)
            metrics.append({'model': name, 'family': family, 'active_candidates': len(p),
                            'total_generated_candidates': len(bank), 'total_tasks': len(design['external_ids']),
                            'active_tasks': len(set(data['groups'])),
                            'mixed_active_tasks': len(task_auc(data['y'], p, data['groups'])),
                            'zero_active_tasks': len(design['external_ids']) - len(set(data['groups'])),
                            **(predictive_metrics(data['y'], p, data['groups']) if len(p) else
                               {'roc_auc': None, 'pr_auc': None, 'log_loss': None, 'brier': None, 'within_task_auc': None})})
            score_maps[family] = {(r['task_id'], r['attempt_id']): float(value) for r, value in zip(data['records'], p)}
        frame = pd.DataFrame([r for r in predictions if r['model'] == name],
                             columns=['model', 'family', 'task_id', 'attempt_id', 'passed', 'probability'])
        contrasts[name] = {kind: paired_auc_contrast(frame, f'HB_{kind}', f'B_{kind}') for kind in ('linear', 'MLP')}
        for kind in ('linear', 'MLP'):
            a = frame[frame.family == f'HB_{kind}'].sort_values(['task_id', 'attempt_id'])
            b = frame[frame.family == f'B_{kind}'].sort_values(['task_id', 'attempt_id'])
            if a.empty:
                contrasts[name][kind].update(delta_log_loss=None, delta_brier=None)
                continue
            y = a.passed.to_numpy()
            pa, pb = (np.clip(t.probability.to_numpy(), 1e-12, 1-1e-12) for t in (a, b))
            def loss(p):
                return -y*np.log(p) - (1-y)*np.log1p(-p)
            task_losses = pd.DataFrame({'task_id': a.task_id.to_numpy(), 'log_loss': loss(pa)-loss(pb),
                                        'brier': (pa-y)**2 - (pb-y)**2}).groupby('task_id').mean()
            for metric in ('log_loss', 'brier'):
                contrasts[name][kind][f'delta_{metric}'] = paired_task_bootstrap(
                    task_losses[metric].to_numpy(), replications=10000, seed=20261104)
        d1, d2 = contrasts[name]['MLP'].get('per_task', {}), contrasts[name]['linear'].get('per_task', {})
        contrasts[name]['Gamma'] = paired_task_bootstrap([d1[t] - d2[t] for t in d1], replications=10000, seed=20261104) if d1 else None
        for budget in (1, 5, 10, 20):
            for task in design['external_ids']:
                stream = [r for r in bank if r['task_id'] == task]
                for policy in ('fixed', 'verifier', 'oracle', *FAMILIES):
                    kind = 'gate' if policy in FAMILIES else policy
                    row = simulate(stream, checkpoint=32, budget=budget, kind=kind,
                                   threshold=fitted[policy]['threshold'] if kind == 'gate' else 0,
                                   scores=score_maps.get(policy))
                    replay_rows.append({'model': name, 'policy': policy, 'budget': budget,
                                        **{k: v for k, v in row.items() if k != 'decisions'}})
    frame = pd.DataFrame(predictions, columns=['model', 'family', 'task_id', 'attempt_id', 'passed', 'probability'])
    frame.to_csv(REPORT / 'external_predictions.csv', index=False)
    replay = pd.DataFrame(replay_rows)
    replay.to_csv(REPORT / 'replay_per_task.csv', index=False)
    summaries = []
    def ratio_interval(numerator, denominator):
        numerator, denominator = np.asarray(numerator, float), np.asarray(denominator, float)
        rng = np.random.default_rng(20261104)
        index = rng.integers(0, len(numerator), (10000, len(numerator)))
        top, bottom = numerator[index].sum(1), denominator[index].sum(1)
        valid = bottom > 0
        estimate = float(numerator.sum()/denominator.sum()) if denominator.sum() else None
        return {'estimate': estimate, 'ci95': np.quantile(top[valid]/bottom[valid], [.025, .975]).tolist() if valid.any() else None,
                'undefined_draws': int((~valid).sum()), 'unit': 'paired task ratio', 'replications': 10000}
    for (name, budget, policy), part in replay.groupby(['model', 'budget', 'policy']):
        part = part.sort_values('task_id')
        baseline = replay[(replay.model == name) & (replay.budget == budget) & (replay.policy == 'verifier')].sort_values('task_id')
        active = int(part.active_correct_in_budget.sum())
        summaries.append({'model': name, 'budget': int(budget), 'policy': policy,
                          'resolved_tasks': int(part.resolved.sum()), 'mean_completion_tokens': float(part.tokens.mean()),
                          'aggregate_tokens_per_resolved_task': float(part.tokens.sum() / part.resolved.sum()) if part.resolved.sum() else None,
                          'false_aborts_started': int(part.false_aborts.sum()),
                          'retention_all_active_correct': float(part.retained_correct_in_budget.sum() / active) if active else None,
                          'retention_interval': ratio_interval(part.retained_correct_in_budget, part.active_correct_in_budget),
                          'tokens_per_resolved_interval': ratio_interval(part.tokens, part.resolved),
                          'lost_tasks': int(((baseline.resolved.to_numpy() == 1) & (part.resolved.to_numpy() == 0)).sum()),
                          'delta_mean_tokens_vs_verifier': paired_task_bootstrap((part.tokens.to_numpy() - baseline.tokens.to_numpy()), replications=10000, seed=20261104),
                          'delta_coverage_vs_verifier': paired_task_bootstrap((part.resolved.to_numpy() - baseline.resolved.to_numpy()), replications=10000, seed=20261104)})
    calibration = []
    for (name, family), part in frame.groupby(['model', 'family']):
        bins = np.minimum((part.probability.to_numpy() * 10).astype(int), 9)
        weights = task_weights(part.task_id)
        for b in range(10):
            mask = bins == b
            calibration.append({'model': name, 'family': family, 'bin': b, 'count': int(mask.sum()),
                                'mean_probability': float(np.average(part.probability[mask], weights=weights[mask])) if mask.any() else None,
                                'fraction_correct': float(np.average(part.passed[mask], weights=weights[mask])) if mask.any() else None})
    pd.DataFrame(calibration).to_csv(REPORT / 'calibration.csv', index=False)
    # Model comparisons use the same resampled task; restrict AUC contrasts to
    # the common mixed population and report that change of estimand explicitly.
    common = sorted(set(contrasts['deepseek']['MLP'].get('per_task', {})) & set(contrasts['qwen']['MLP'].get('per_task', {})))
    model_comparison = paired_task_bootstrap([contrasts['deepseek']['MLP']['per_task'][t] - contrasts['qwen']['MLP']['per_task'][t]
                                             for t in common], replications=10000, seed=20261104) if common else None
    result = {'status': 'completed', 'primary': contrasts['deepseek']['MLP'], 'contrasts': contrasts,
              'metrics': metrics, 'operational': summaries, 'scoring_timing': timing,
              'secondary_cross_model_common_mixed': model_comparison,
              'secondary_scope': 'Descriptive pointwise 95% intervals; no multiplicity-adjusted confirmatory claims',
              'limitations': ['Conditional fitted-probe uncertainty', 'Tokenizer-specific t=32 risk sets',
                              'Token savings are not latency savings', 'Pretraining contamination not ruled out']}
    atomic_json(REPORT / 'results.json', result)
    return result
