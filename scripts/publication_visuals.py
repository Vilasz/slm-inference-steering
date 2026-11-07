"""Descriptive publication plots; never selects generation or fitted predictors."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def render(root: Path, output: Path, evidence: dict):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from slm_steering.research.statistics import task_auc

    figures = output / 'figures'
    figures.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})

    def save(fig, name):
        fig.tight_layout()
        for suffix in ('png', 'svg', 'pdf'):
            fig.savefig(figures / f'{name}.{suffix}', dpi=220, bbox_inches='tight')
        plt.close(fig)

    pilot_source = next((root / relative for relative in evidence['inputs']
                         if relative.endswith('pilot_v0.2/trajectories.jsonl')), None)
    if pilot_source:
        records = [json.loads(line) for line in pilot_source.read_text(encoding='utf-8').splitlines() if line]
        frame = pd.DataFrame(records)
        manifest = json.loads(pilot_source.with_name('manifest.json').read_text(encoding='utf-8'))
        task_ids = manifest['pilot_task_ids']
        temperatures = manifest['config']['pilot']['temperatures']
        grid = np.full((len(task_ids), len(temperatures)), np.nan)
        counts = {}
        for i, task in enumerate(task_ids):
            for j, temperature in enumerate(temperatures):
                rows = [r for r in records if r['task_id'] == task and r['decoding_config']['temperature'] == temperature]
                if rows:
                    successes = sum(r['passed'] for r in rows)
                    grid[i, j] = successes / len(rows)
                    counts[(i, j)] = f'{successes}/{len(rows)}'
        fig, axis = plt.subplots(figsize=(5.8, 9))
        cmap = plt.get_cmap('RdYlGn').copy()
        cmap.set_bad('#e0e4e8')
        image = axis.imshow(grid, aspect='auto', vmin=0, vmax=1, cmap=cmap)
        for (i, j), text in counts.items():
            axis.text(j, i, text, ha='center', va='center', fontsize=8, color='#172b3a')
        axis.set(xticks=range(len(temperatures)), xticklabels=temperatures,
                 yticks=range(len(task_ids)), yticklabels=task_ids, xlabel='Temperatura',
                 title='Piloto: acertos / tentativas observadas por tarefa')
        fig.colorbar(image, ax=axis, label='Proporção observada de acertos', shrink=.65)
        save(fig, 'pilot_task_temperature_grid')
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
        for temperature in temperatures:
            lengths = [r['generated_tokens'] for r in records if r['decoding_config']['temperature'] == temperature]
            axes[0].hist(lengths, bins=np.linspace(0, 512, 33), histtype='step', label=f'T={temperature}', linewidth=1.5)
        axes[0].set(xlabel='Tokens gerados até término', ylabel='Trajetórias', title='Comprimentos no piloto')
        axes[0].legend(frameon=False)
        statuses = [sum(r['passed'] for r in records),
                    sum(r['verifier_status']['base'] == 'pass' and r['verifier_status']['plus'] != 'pass' for r in records),
                    sum(r['verifier_status']['base'] != 'pass' for r in records)]
        bars = axes[1].barh(['Base e plus passam', 'Base passa, plus falha', 'Base falha'], statuses,
                            color=['#147d78', '#b99551', '#9c6767'])
        axes[1].bar_label(bars, padding=4)
        axes[1].set(xlabel='Trajetórias observadas', title='Regra funcional do EvalPlus', xlim=(0, max(statuses) * 1.15 + 1))
        save(fig, 'pilot_lengths_and_verifier')
        frame[['task_id', 'attempt_id', 'generated_tokens', 'passed', 'truncated']].to_csv(
            output / 'tables/pilot_descriptive_trajectories.csv', index=False)

    late = pd.read_csv(output / 'legacy_outer_predictions.csv')
    baseline = task_auc(late['passed'], late['B_late'], late['task_id'])
    combined = task_auc(late['passed'], late['HB_late'], late['task_id'])
    differences = sorted((combined[task] - baseline[task], task) for task in baseline)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    positions = np.arange(len(differences))
    axes[0].bar(positions, [value for value, _ in differences], color='#245c80')
    axes[0].axhline(0, color='grey', linewidth=.8)
    axes[0].set(xticks=positions, xticklabels=[task.split('/')[-1] for _, task in differences],
                xlabel='Tarefa HumanEval (ID)', ylabel='ΔWT-AUC (H+B tardio − B tardio)',
                title='Heterogeneidade entre tarefas mistas')
    axes[0].tick_params(axis='x', labelsize=8)
    axes[1].scatter([baseline[task] for task in baseline], [combined[task] for task in baseline],
                     color='#147d78', alpha=.75)
    axes[1].plot([0, 1], [0, 1], '--', color='grey', linewidth=.8)
    axes[1].set(xlabel='WT-AUC: B tardio', ylabel='WT-AUC: H+B tardio',
                xlim=(-.04, 1.04), ylim=(-.04, 1.04), title='Cada ponto representa uma tarefa')
    fig.suptitle('Reanálise histórica exploratória • estados finais • nenhuma seleção do experimento principal', fontsize=10)
    save(fig, 'legacy_incremental_by_task')
