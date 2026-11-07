"""Publication figures from held-out predictions, without model selection."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def additional_figures(predictions: pd.DataFrame, signal: dict, policies: dict, directory: Path) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve, roc_curve
    from slm_steering.research.statistics import task_auc, task_weights

    directory.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})

    def save(fig, name):
        fig.tight_layout()
        for suffix in ('png', 'svg', 'pdf'):
            fig.savefig(directory / f'{name}.{suffix}', dpi=220, bbox_inches='tight')
        plt.close(fig)

    risks = pd.DataFrame(signal['risk_sets'])
    numeric = risks[risks['checkpoint'] != 'final'].copy()
    numeric['checkpoint'] = numeric['checkpoint'].astype(int)
    numeric = numeric.sort_values('checkpoint')
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    axes[0].plot(numeric['checkpoint'], numeric['trajectories'], 'o-', color='#245c80')
    axes[0].set(xlabel='Tokens gerados no checkpoint', ylabel='Tentativas elegíveis', title='População ativa: T > t')
    axes[1].stackplot(numeric['checkpoint'], numeric['mixed_tasks'], numeric['always_correct'], numeric['always_wrong'],
                      labels=['Mistas', 'Somente corretas', 'Somente incorretas'],
                      colors=['#245c80', '#147d78', '#9c6767'], alpha=.85)
    axes[1].set(xlabel='Tokens gerados no checkpoint', ylabel='Tarefas', title='Composição da população ativa')
    axes[1].legend(frameon=False, fontsize=8)
    save(fig, 'risk_set_attrition')

    primary = predictions[predictions['checkpoint'].astype(str) == '32']
    metrics = {row['predictor']: row for row in signal['metrics'] if row['checkpoint'] == '32'}
    names = ['prior', 'B1', 'B2', 'B3', 'B', 'H', 'HB', 'random_direction', 'shuffled_training_labels']
    labels = ['Prior', 'B1', 'B2', 'B3', 'B', 'H', 'H+B', 'Direção aleatória', 'Rótulos permutados']
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for axis, metric, caption in zip(axes, ['within_task_auc', 'log_loss'], ['WT-AUC em 32 tokens', 'Log loss em 32 tokens']):
        for index, name in enumerate(names):
            row = metrics[name]
            ci = row['within_task_auc_ci95'] if metric == 'within_task_auc' else row['intervals']['log_loss']
            estimate = row[metric]
            if ci is not None:
                axis.plot(ci, [index, index], color='#245c80')
            axis.plot(estimate, index, 'o', color='#147d78' if name == 'HB' else '#245c80')
        axis.set(yticks=range(len(names)), yticklabels=labels, xlabel=caption)
        axis.invert_yaxis()
        if metric == 'within_task_auc':
            axis.axvline(.5, color='grey', linestyle='--', linewidth=.8)
    save(fig, 'primary_predictor_comparison')

    baseline = task_auc(primary['passed'], primary['B'], primary['task_id'])
    combined = task_auc(primary['passed'], primary['HB'], primary['task_id'])
    deltas = sorted((combined[task] - baseline[task], task) for task in baseline)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(range(len(deltas)), [value for value, _ in deltas], color='#245c80', width=1)
    axes[0].axhline(0, color='grey', linewidth=.8)
    axes[0].set(xlabel='Tarefas mistas, ordenadas pelo efeito', ylabel='ΔWT-AUC (H+B − B)', title='Heterogeneidade do ganho')
    weights = task_weights(primary['task_id'].to_numpy())
    for name, color in [('B', '#8c9298'), ('H', '#245c80'), ('HB', '#147d78')]:
        for outcome, style in [(0, '--'), (1, '-')]:
            selected = primary['passed'].to_numpy() == outcome
            axes[1].hist(primary.loc[selected, name], bins=np.linspace(0, 1, 21), weights=weights[selected],
                         density=True, histtype='step', linestyle=style, color=color,
                         label=f'{"H+B" if name == "HB" else name}: {"correta" if outcome else "incorreta"}')
    axes[1].set(xlabel='Probabilidade OOF calibrada', ylabel='Densidade ponderada por tarefa', title='Separação das probabilidades')
    axes[1].legend(frameon=False, fontsize=7)
    save(fig, 'primary_task_deltas')

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for name in ['B', 'H', 'HB']:
        fpr, tpr, _ = roc_curve(primary['passed'], primary[name], sample_weight=weights)
        precision, recall, _ = precision_recall_curve(primary['passed'], primary[name], sample_weight=weights)
        axes[0].plot(fpr, tpr, label='H+B' if name == 'HB' else name)
        axes[1].plot(recall, precision, label='H+B' if name == 'HB' else name)
    axes[0].plot([0, 1], [0, 1], '--', color='grey', linewidth=.8)
    axes[1].axhline(np.average(primary['passed'], weights=weights), linestyle='--', color='grey', linewidth=.8)
    axes[0].set(xlabel='Taxa de falsos positivos', ylabel='Sensibilidade', title='ROC agregada em 32 tokens', xlim=(0, 1), ylim=(0, 1))
    axes[1].set(xlabel='Recall', ylabel='Precisão', title='Precisão–recall em 32 tokens', xlim=(0, 1), ylim=(0, 1))
    for axis in axes:
        axis.legend(frameon=False)
    save(fig, 'primary_roc_pr')

    if policies.get('comparisons'):
        comparisons = policies['comparisons']
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.8))
        for axis, key, xlabel in zip(axes, ['resolution_delta', 'token_delta'],
                                     ['Δresolução por tarefa', 'Δtokens por tarefa']):
            for index, row in enumerate(comparisons):
                result = row[key]
                axis.plot(result['ci95'], [index, index], color='#245c80')
                axis.plot(result['estimate'], index, 'o', color='#245c80')
            axis.axvline(0, color='grey', linewidth=.8)
            if key == 'resolution_delta':
                axis.axvline(-.03, linestyle='--', color='#9c6767', linewidth=.8)
            axis.set(yticks=range(len(comparisons)),
                     yticklabels=[f"N={r['budget']}: {r['comparison'].replace('HB', 'H+B')}" for r in comparisons], xlabel=xlabel)
            axis.invert_yaxis()
        save(fig, 'paired_policy_effects')
        summary = pd.DataFrame(policies['summaries'])
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
        for name, rows in summary.groupby('policy'):
            rows = rows.sort_values('budget')
            axes[0].plot(rows['budget'], rows['completion_tokens_per_task'], 'o-', label=name)
            axes[1].plot(rows['budget'], rows['active_correct_false_abort_rate'].fillna(0), 'o-', label=name)
        axes[0].set(xlabel='Orçamento máximo N', ylabel='Tokens de continuação por tarefa', title='Custo com parada por sucesso')
        axes[1].set(xlabel='Orçamento máximo N', ylabel='Aborto falso / corretas ativas iniciadas', title='Risco da interrupção')
        for axis in axes:
            axis.legend(frameon=False)
        save(fig, 'policy_cost_and_false_aborts')


def transfer_figures(result: dict, directory: Path) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    directory.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    metrics = pd.DataFrame(result['metrics'])
    if not metrics.empty:
        for family, rows in metrics.groupby('family'):
            rows = rows.sort_values('checkpoint')
            axes[0].plot(rows['checkpoint'], rows['within_task_auc'], 'o-', label=family.replace('HB', 'H+B'))
    axes[0].axhline(.5, linestyle='--', color='grey', linewidth=.8)
    axes[0].set(xlabel='Tokens gerados no checkpoint', ylabel='WT-AUC', title='MBPP+: sistema congelado', ylim=(0, 1))
    if not metrics.empty:
        axes[0].legend(frameon=False)
    for row in result['deltas']:
        delta = row['delta_wt_auc']
        if delta is None:
            continue
        axes[1].plot([row['checkpoint'], row['checkpoint']], delta['ci95'], color='#245c80')
        axes[1].plot(row['checkpoint'], delta['estimate'], 'o', color='#245c80')
    axes[1].axhline(0, color='grey', linewidth=.8)
    axes[1].set(xlabel='Tokens gerados no checkpoint', ylabel='ΔWT-AUC (H+B − B)', title='Incremento externo com IC 95%')
    fig.tight_layout()
    for suffix in ('png', 'svg', 'pdf'):
        fig.savefig(directory / f'frozen_transfer_signal.{suffix}', dpi=220, bbox_inches='tight')
    plt.close(fig)
