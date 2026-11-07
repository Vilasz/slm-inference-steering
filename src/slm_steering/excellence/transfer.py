"""Explain frozen transfer descriptively, without selecting thresholds from target labels."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from slm_steering.excellence.audit import assert_preserved, checkpoint_data
from slm_steering.excellence.economics import simulate
from slm_steering.research.probes import fit_model
from slm_steering.research.protocol import now, read_bank, write_json
from slm_steering.research.statistics import paired_task_bootstrap, predictive_metrics, task_auc


def retention_interval(frame, probability_column, threshold, *, replications=2000, seed=20260921):
    rows = []
    for _, task in frame.groupby('task_id', sort=True):
        positive = task.passed.eq(1)
        rows.append([int((positive & task[probability_column].ge(threshold)).sum()), int(positive.sum())])
    counts = np.array(rows)
    rng = np.random.default_rng(seed)
    totals = counts[rng.integers(0, len(counts), (replications, len(counts)))].sum(axis=1)
    defined = totals[:, 1] > 0
    estimates = totals[defined, 0] / totals[defined, 1]
    return {'estimate': float(counts[:, 0].sum() / counts[:, 1].sum()) if counts[:, 1].sum() else None,
            'ci95': np.quantile(estimates, [.025, .975]).tolist() if len(estimates) else None,
            'correct': int(counts[:, 1].sum()), 'retained': int(counts[:, 0].sum()),
            'undefined_draws': int((~defined).sum())}


def source_development(root: Path, metadata, bundle):
    """Reconstitute the raw cross-fitted scores used for SOURCE gate development, not OOS performance."""
    rows = []
    with threadpool_limits(limits=1):
        for checkpoint in (8, 16, 32, 64):
            data = checkpoint_data(root, checkpoint=checkpoint)
            for family in ('B', 'HB'):
                selection = next(r for r in metadata['selections'] if r['family'] == family and r['checkpoint'] == checkpoint)
                raw = np.full(len(data['y']), np.nan)
                for part in selection['calibration_audit']:
                    train = np.isin(data['groups'], part['fit_tasks'])
                    score = np.isin(data['groups'], part['score_tasks'])
                    if set(part['fit_tasks']) & set(part['score_tasks']):
                        raise ValueError('Source-development partition leakage')
                    x = data['observable'] if family == 'B' else np.column_stack((data['hidden'][:, data['layers'].index(part['layer'])], data['observable']))
                    model = fit_model(x[train], data['y'][train], data['groups'][train], part['c'])
                    raw[score] = model.decision_function(x[score])
                if not np.isfinite(raw).all():
                    raise ValueError('Source-development cross-fit is incomplete')
                p = bundle['models'][(family, checkpoint)]['calibrator'].predict_proba(raw.reshape(-1, 1))[:, 1]
                for identity, label, probability in zip(data['identities'], data['y'], p):
                    rows.append({'task_id': identity[0], 'attempt_id': identity[1], 'checkpoint': checkpoint,
                                 'passed': int(label), 'family': family, 'probability': float(probability)})
                print(f'transfer source-development reconstructed: {family} t={checkpoint}', flush=True)
    return pd.DataFrame(rows)


def run(root: Path):
    assert_preserved(root)
    output = root / 'reports/excellence/transfer'
    output.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((root / 'runs/transfer/protocol_v0.2/models.json').read_text(encoding='utf-8'))
    bundle = joblib.load(root / 'runs/transfer/protocol_v0.2/models.joblib')
    source = pd.read_csv(root / 'runs/probes/protocol_v0.2/outer_predictions.csv')
    target = pd.read_csv(root / 'reports/transfer_v0.2/predictions.csv')
    source = source[source.checkpoint.isin(['8', '16', '32', '64'])].copy()
    source['checkpoint'] = source.checkpoint.astype(int)
    development = source_development(root, metadata, bundle)
    development.to_csv(output / 'source_development_predictions.csv', index=False)
    long = []
    for dataset, frame in [('source_OOF_external', source), ('target_frozen_external', target)]:
        for family in ('B', 'HB'):
            long.append(frame[['task_id', 'attempt_id', 'checkpoint', 'passed', family]].rename(columns={family: 'probability'}).assign(family=family, dataset=dataset))
    long.append(development.assign(dataset='source_development_not_external'))
    frame = pd.concat(long, ignore_index=True)
    frame.to_csv(output / 'score_distributions.csv', index=False)
    summaries, curves, task_rows, losses = [], [], [], []
    for (dataset, family, checkpoint), part in frame.groupby(['dataset', 'family', 'checkpoint']):
        p = metadata['policies'][family]
        threshold = p['threshold']
        original_gate = checkpoint == p['checkpoint']
        y, probability, groups = part.passed.to_numpy(), part.probability.to_numpy(), part.task_id.to_numpy()
        auc = list(task_auc(y, probability, groups).values())
        quantiles = {str(label): np.quantile(probability[y == label], [.1, .25, .5, .75, .9]).tolist()
                     if np.any(y == label) else None for label in (0, 1)}
        recall = retention_interval(part, 'probability', threshold)
        if dataset == 'source_development_not_external' and original_gate:
            if abs(recall['estimate'] - p['validation_correct_recall']) > 1e-12:
                raise ValueError('Source-development retention reconstruction differs from frozen gate metadata')
        summaries.append({'dataset': dataset, 'family': family, 'checkpoint': int(checkpoint),
                          'attempts': len(part), 'tasks': part.task_id.nunique(), 'candidate_prevalence': float(y.mean()),
                          'metrics': predictive_metrics(y, probability, groups),
                          'within_task_auc_interval': paired_task_bootstrap(auc, replications=2000, seed=20260921) if auc else None,
                          'score_quantiles_given_label': quantiles, 'original_policy_checkpoint': bool(original_gate),
                          'frozen_threshold': threshold, 'retention_at_frozen_numeric_threshold': recall,
                          'near_threshold_attempts': int((np.abs(probability - threshold) <= .02).sum()),
                          'near_threshold_correct': int(((np.abs(probability - threshold) <= .02) & (y == 1)).sum())})
        for q in sorted(set(np.linspace(0, 1, 101).tolist() + [threshold])):
            positive = y == 1
            curves.append({'dataset': dataset, 'family': family, 'checkpoint': int(checkpoint), 'threshold': q,
                           'retention_correct': float((probability[positive] >= q).mean()) if positive.any() else None,
                           'fraction_kept': float((probability >= q).mean()), 'frozen_gate_at_this_checkpoint': original_gate and q == threshold})
        if original_gate:
            for task, task_part in part.groupby('task_id'):
                correct = task_part.passed.eq(1)
                wrong_abort = correct & task_part.probability.lt(threshold)
                task_rows.append({'dataset': dataset, 'family': family, 'checkpoint': int(checkpoint),
                                  'task_id': task, 'active_correct': int(correct.sum()),
                                  'false_aborts_all_correct': int(wrong_abort.sum()),
                                  'kept_correct': int((correct & ~wrong_abort).sum())})
    for dataset, phase, wide in [('HumanEval+', 'definitive', source), ('MBPP+', 'transfer', target)]:
        records = read_bank(root / f'runs/candidate_bank/{phase}_v0.2/trajectories.jsonl')
        if phase == 'transfer':
            by_task = {}
            for r in records:
                by_task.setdefault(r['task_id'], []).append(r)
            for family in ('B', 'HB'):
                p = metadata['policies'][family]
                scores = {(r.task_id, r.attempt_id): getattr(r, family) for r in wide[wide.checkpoint == p['checkpoint']].itertuples()}
                for task, stream in by_task.items():
                    verifier = simulate(stream, checkpoint=p['checkpoint'], kind='verifier')
                    gate = simulate(stream, checkpoint=p['checkpoint'], kind='gate', threshold=p['threshold'], scores=scores)
                    losses.append({'task_id': task, 'family': family, 'verifier_resolved': verifier['resolved'],
                                   'gate_resolved': gate['resolved'], 'lost_task': bool(verifier['resolved'] and not gate['resolved']),
                                   'verifier_started': verifier['started'], 'gate_started': gate['started'],
                                   'started_correct': gate['correct_started'], 'started_active_correct': gate['active_correct_started'],
                                   'started_correct_false_aborts': gate['false_aborts']})
    composition = []
    for phase in ('definitive', 'transfer'):
        records = read_bank(root / f'runs/candidate_bank/{phase}_v0.2/trajectories.jsonl')
        length = np.array([r['generated_tokens'] for r in records])
        for checkpoint in (0, 8, 16, 32, 64):
            active = [r for r in records if r['generated_tokens'] > checkpoint]
            composition.append({'phase': phase, 'checkpoint': checkpoint, 'attempts': len(active),
                                'tasks': len({r['task_id'] for r in active}),
                                'prevalence': float(np.mean([r['passed'] for r in active])),
                                'length_quantiles_all': np.quantile(length, [.1, .5, .9]).tolist(), 'mean_length_all': float(length.mean())})
    tasks = pd.DataFrame(task_rows)
    concentration = []
    for (dataset, family), part in tasks.groupby(['dataset', 'family']):
        total = part.false_aborts_all_correct.sum()
        concentration.append({'dataset': dataset, 'family': family,
                              'false_aborts_all_correct': int(total),
                              'tasks_with_false_aborts': int(part.false_aborts_all_correct.gt(0).sum()),
                              'share_in_top_five_tasks': float(part.false_aborts_all_correct.nlargest(5).sum() / total) if total else None})
    tasks.to_csv(output / 'false_aborts_by_task.csv', index=False)
    pd.DataFrame(losses).to_csv(output / 'target_task_coverage.csv', index=False)
    pd.DataFrame(curves).to_csv(output / 'retention_threshold_curves.csv', index=False)
    result = {'status': 'completed_frozen_transfer_diagnostics', 'created_at': now(), 'summaries': summaries,
              'population_composition': composition, 'false_abort_concentration': concentration,
              'target_coverage': losses, 'policies': {f: {k:v for k,v in p.items() if k != 'candidates'} for f,p in metadata['policies'].items()},
              'source_development_scope': 'Raw scores cross-fit whole selection, but source calibrator and threshold fitted on these same development labels; not external performance',
              'limitations': ['No target fitting, recalibration or threshold selection',
                             'Source OOF and target full-source-fitted scorers are different procedures',
                             'Curves using target outcomes are retrospective diagnostics, not deployable target-optimized gates',
                             'Length and score shifts do not identify a unique causal explanation',
                             'All-candidate retention and initiated-candidate false-abort rates have different denominators']}
    write_json(output / 'results.json', result)
    plot(output, frame, pd.DataFrame(curves), result['policies'])
    assert_preserved(root)


def plot(output, frame, curves, policies):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = {'source_OOF_external': '#3274A1', 'source_development_not_external': '#888888', 'target_frozen_external': '#C44E52'}
    labels = {'source_OOF_external': 'Origem: teste OOF', 'source_development_not_external': 'Origem: desenvolvimento',
              'target_frozen_external': 'Alvo: modelo congelado'}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, family in zip(axes, ('B', 'HB')):
        policy = policies[family]
        for dataset, group in curves[(curves.family == family) & (curves.checkpoint == policy['checkpoint'])].groupby('dataset'):
            ax.plot(group.threshold, group.retention_correct, label=labels[dataset], color=colors[dataset])
        ax.axvline(policy['threshold'], color='black', ls='--', lw=.8)
        ax.axhline(.97, color='gray', ls=':', lw=.8)
        ax.set(xlabel='Limiar', ylabel='Retenção dos corretos ativos', ylim=(0, 1.02), title=f"{family}, gate em {policy['checkpoint']} tokens")
        ax.legend(fontsize=6)
    fig.tight_layout()
    for ext in ('png', 'svg', 'pdf'):
        fig.savefig(output / f'retention_curves.{ext}', dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.7))
    policy = policies['HB']
    for ax, (dataset, color) in zip(axes, colors.items()):
        part = frame[(frame.dataset == dataset) & (frame.family == 'HB') & (frame.checkpoint == policy['checkpoint'])]
        for label, style in [(0, '--'), (1, '-')]:
            values = part.loc[part.passed == label, 'probability']
            ax.hist(values, bins=np.linspace(0, 1, 21), density=True, histtype='step', linestyle=style,
                    label='correto' if label else 'incorreto', color=color)
        ax.axvline(policy['threshold'], color='black', lw=.8)
        ax.set(title=labels[dataset], xlabel='Probabilidade prevista H+B', ylabel='Densidade por candidato')
        ax.legend(fontsize=8)
    fig.tight_layout()
    for ext in ('png', 'svg', 'pdf'):
        fig.savefig(output / f'conditional_scores.{ext}', dpi=180)
    plt.close(fig)
