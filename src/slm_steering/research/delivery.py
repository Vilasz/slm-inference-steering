"""Auditable presentation of observed evidence, without promoting pilot claims."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from slm_steering.research.probes import select_hyperparameters
from slm_steering.research.protocol import file_hash, now, write_json
from slm_steering.research.statistics import paired_task_bootstrap, predictive_metrics, task_auc


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def legacy_reanalysis(root: Path, output: Path) -> dict:
    """Retrospective late-state analysis; no missing early features are invented."""
    source = root / 'runs/phase3/qwen_phase3_probe'
    metadata = pd.DataFrame(read_jsonl(source / 'metadata.jsonl'))
    manifest = json.loads((source / 'manifest.json').read_text(encoding='utf-8'))
    if manifest['token_position'] != 'completion_last':
        raise ValueError('This historical analysis is defined for completion-last states')
    with np.load(source / 'activations.npz', allow_pickle=False) as arrays:
        hidden = arrays['activations'].astype(float)
        y, layers = arrays['labels'].astype(int), arrays['layers'].tolist()
    if not np.array_equal(y, metadata['passed'].astype(int)):
        raise ValueError('Legacy metadata labels disagree with activation labels')
    groups = metadata['task_id'].to_numpy()
    # Explicitly LATE observable baseline, different from prefix confidence B.
    obs = np.log1p(metadata[['prompt_tokens', 'generated_tokens']].to_numpy(float))
    configuration = {'statistics': {'inner_folds': 3, 'regularization_c': [0.001, 0.01, 0.1, 1, 10]}}
    rows, selections = [], []
    with threadpool_limits(limits=1):
        for fold, (train, test) in enumerate(GroupKFold(5).split(obs, y, groups)):
            families = {'B_late': [(-1, obs)],
                        'H_final': [(layer, hidden[:, index]) for index, layer in enumerate(layers)],
                        'HB_late': [(layer, np.column_stack((hidden[:, index], obs))) for index, layer in enumerate(layers)]}
            scores = {}
            for family, variants in families.items():
                fitted = select_hyperparameters([(layer, x[train]) for layer, x in variants], y[train], groups[train], configuration)
                x = next(x for layer, x in variants if layer == fitted['layer'])
                scores[family] = fitted['model'].predict_proba(x[test])[:, 1]
                selections.append({'fold': fold, 'family': family, 'layer': fitted['layer'], 'C': fitted['c'],
                                   'train_tasks': sorted(set(groups[train])), 'test_tasks': sorted(set(groups[test]))})
            for index, source_index in enumerate(test):
                rows.append({'task_id': groups[source_index], 'attempt': int(metadata.iloc[source_index]['attempt']),
                             'passed': int(y[source_index]), 'fold': fold,
                             **{family: float(values[index]) for family, values in scores.items()}})
            print(f'Historical final-state analysis: fold {fold + 1}/5', flush=True)
    frame = pd.DataFrame(rows)
    opts = {'replications': 2000, 'seed': 20261012}
    metrics = []
    for family in families:
        task_values = task_auc(frame['passed'], frame[family], frame['task_id'])
        metrics.append({'family': family, **predictive_metrics(frame['passed'], frame[family], frame['task_id']),
                        'wt_auc_interval': paired_task_bootstrap(list(task_values.values()), **opts),
                        'mixed_tasks': len(task_values)})
    b = task_auc(frame['passed'], frame['B_late'], frame['task_id'])
    hb = task_auc(frame['passed'], frame['HB_late'], frame['task_id'])
    loss_deltas = [log_loss(part['passed'], part['HB_late'], labels=[0, 1]) -
                   log_loss(part['passed'], part['B_late'], labels=[0, 1]) for _, part in frame.groupby('task_id')]
    result = {'status': 'retrospective_exploratory_late_state', 'created_at': now(), 'tasks': len(set(groups)),
              'attempts': len(y), 'correct': int(y.sum()), 'mixed_tasks': len(b),
              'position': 'completion_last', 'baseline': 'log1p(prompt tokens), log1p(final completion tokens)',
              'labels': 'Historical local HumanEval verifier, not EvalPlus HumanEval+', 'calibration': 'none',
              'metrics': metrics, 'delta_wt_auc': paired_task_bootstrap([hb[t] - b[t] for t in b], **opts),
              'delta_log_loss': paired_task_bootstrap(loss_deltas, **opts), 'selection': selections,
              'limitations': ['Retrospective design after legacy results were seen',
                              'Only final states available; no inference about 32-token prediction',
                              'Historical retokenization/extraction lacks the new causal and token-ID validation',
                              'Conditional bootstrap of fixed OOF scores, not full procedure retraining',
                              'Late baseline lacks token confidence features; not the main B comparison'],
              'inputs': {str(path.relative_to(root)).replace('\\', '/'): file_hash(path)
                         for path in (source / 'metadata.jsonl', source / 'manifest.json', source / 'activations.npz')}}
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / 'legacy_outer_predictions.csv', index=False)
    write_json(output / 'legacy_reanalysis.json', result)
    return result


def build_evidence(root: Path, output: Path, *, as_of: str, reanalyze: bool = False,
                   allow_running_snapshot: bool = False) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    tables, figures = output / 'tables', output / 'figures'
    tables.mkdir(exist_ok=True)
    figures.mkdir(exist_ok=True)
    source_path = root / 'runs/phase2/qwen15b_temp08_n5.jsonl'
    source = read_jsonl(source_path)
    curves, composition = [], []
    for task in source:
        attempts = task['attempts']
        composition.append({'task_id': task['task_id'], 'attempts': len(attempts),
                            'correct': sum(a['passed'] for a in attempts),
                            'category': 'mixed' if 0 < sum(a['passed'] for a in attempts) < len(attempts)
                                        else 'all_correct' if all(a['passed'] for a in attempts) else 'all_failed'})
        for budget in (1, 2, 3, 4, 5):
            candidates = attempts[:budget]
            if len(candidates) != budget:
                raise ValueError('Incomplete historical stream')
            resolved, stop_tokens = False, 0
            for candidate in candidates:
                stop_tokens += candidate['generated_tokens']
                if candidate['passed']:
                    resolved = True
                    break
            curves.append({'task_id': task['task_id'], 'budget': budget, 'resolved': int(resolved),
                           'fixed_tokens': sum(a['generated_tokens'] for a in candidates),
                           'verifier_tokens': stop_tokens})
    task_curves = pd.DataFrame(curves)
    task_curves.to_csv(tables / 'legacy_task_curves.csv', index=False)
    pd.DataFrame(composition).to_csv(tables / 'legacy_task_composition.csv', index=False)
    opts = {'replications': 2000, 'seed': 20261012}
    summaries = []
    for budget, part in task_curves.groupby('budget'):
        summaries.append({'budget': int(budget), 'tasks': len(part), 'solved': int(part['resolved'].sum()),
                          'resolution': float(part['resolved'].mean()),
                          'resolution_ci95': paired_task_bootstrap(part['resolved'], **opts)['ci95'],
                          'fixed_tokens': int(part['fixed_tokens'].sum()), 'verifier_tokens': int(part['verifier_tokens'].sum()),
                          'token_saving_fraction': float(1 - part['verifier_tokens'].sum() / part['fixed_tokens'].sum()),
                          'paired_token_delta_per_task': paired_task_bootstrap(part['verifier_tokens'] - part['fixed_tokens'], **opts)})
    pd.DataFrame(summaries).to_csv(tables / 'legacy_budget_summary.csv', index=False)
    legacy_path = output / 'legacy_reanalysis.json'
    legacy = legacy_reanalysis(root, output) if reanalyze or not legacy_path.exists() else json.loads(legacy_path.read_text(encoding='utf-8'))
    for relative, expected in legacy['inputs'].items():
        if file_hash(root / relative) != expected:
            raise ValueError('Historical input changed; rerun with --reanalyze-legacy')
    pilots, pilot_temperatures = [], []
    inputs = {source_path.relative_to(root).as_posix(): file_hash(source_path), **legacy['inputs']}
    for version in ('0.1', '0.2'):
        bank = root / f'runs/candidate_bank/pilot_v{version}'
        if not (bank / 'manifest.json').exists():
            continue
        manifest = json.loads((bank / 'manifest.json').read_text(encoding='utf-8'))
        if manifest['status'] == 'running' and not allow_running_snapshot:
            raise ValueError('Wait for the pilot session to close before publishing a snapshot')
        source_bank, source_manifest = bank / 'trajectories.jsonl', bank / 'manifest.json'
        if manifest['status'] == 'running':
            snapshot = output / 'snapshots' / f'pilot_v{version}'
            snapshot.mkdir(parents=True, exist_ok=True)
            raw = source_bank.read_bytes()
            complete_prefix = raw[:raw.rfind(b'\n') + 1]
            source_bank = snapshot / 'trajectories.jsonl'
            source_manifest = snapshot / 'manifest.json'
            source_bank.write_bytes(complete_prefix)
            write_json(source_manifest, manifest)
        records = read_jsonl(source_bank)
        inputs[source_bank.relative_to(root).as_posix()] = file_hash(source_bank)
        inputs[source_manifest.relative_to(root).as_posix()] = file_hash(source_manifest)
        generation = sum(r['generation_time_seconds'] for r in records)
        peaks = [manifest.get('hardware', {}).get('peak_reserved_mib', 0)]
        peaks.extend(s.get('hardware', {}).get('peak_reserved_mib', 0) for s in manifest.get('sessions', []))
        measured = root / 'reports/research/hardware_measurements.json'
        if version == '0.1' and measured.exists():
            peaks.extend(row['peak_reserved_mib'] for row in json.loads(measured.read_text(encoding='utf-8')))
            inputs[measured.relative_to(root).as_posix()] = file_hash(measured)
        pilots.append({'version': version, 'status': manifest['status'], 'attempts': len(records),
                       'planned': manifest['expected_attempts'], 'tasks': len({r['task_id'] for r in records}),
                       'completion_tokens': sum(r['generated_tokens'] for r in records),
                       'generation_seconds': generation, 'peak_reserved_mib': max(peaks),
                       'temperature_selected': manifest.get('selected_temperature'),
                       'causal_validation': manifest.get('causal_invariance', {}).get('passed', False),
                       'seed_validation': manifest.get('seed_reproduction_passed', False),
                       'label_definition': 'functional EvalPlus pass' if version == '0.2' else 'EvalPlus pass AND not truncated'})
        for temperature in manifest['config']['pilot']['temperatures']:
            part = [r for r in records if r['decoding_config']['temperature'] == temperature]
            pilot_temperatures.append({'version': version, 'temperature': temperature, 'attempts': len(part),
                                      'correct': sum(r['passed'] for r in part), 'truncated': sum(r['truncated'] for r in part),
                                      'planned': manifest['config']['pilot']['tasks'] * manifest['config']['pilot']['attempts_per_task'],
                                      'mean_tokens': float(np.mean([r['generated_tokens'] for r in part])) if part else None})
    pd.DataFrame(pilots).to_csv(tables / 'pilot_status.csv', index=False)
    pd.DataFrame(pilot_temperatures).to_csv(tables / 'pilot_temperatures.csv', index=False)
    exposure_path = root / 'reports/research/legacy_task_presence.json'
    exposure = json.loads(exposure_path.read_text(encoding='utf-8'))
    inputs[exposure_path.relative_to(root).as_posix()] = file_hash(exposure_path)
    result = {'as_of': as_of, 'built_at': now(), 'status': 'observed_exploratory_and_technical_evidence',
              'confirmatory_results_available': False, 'legacy_sampling': summaries, 'legacy_reanalysis': legacy,
              'pilot': pilots, 'pilot_temperatures': pilot_temperatures, 'legacy_task_composition': composition,
              'exposure': {key: exposure[key] for key in ('legacy_tasks', 'current_pilot_overlap', 'current_reserved_overlap')},
              'inputs': inputs, 'primary_questions': {'Q1_early': 'not_estimated', 'Q2_32_tokens': 'not_estimated',
                                                     'Q3_early_abort': 'not_estimated'}}
    main_path = root / 'reports/protocol_v0.2/results.json'
    if main_path.exists():
        main = json.loads(main_path.read_text(encoding='utf-8'))
        bank_path = root / 'runs/candidate_bank/definitive_v0.2/trajectories.jsonl'
        predictions_path = root / 'runs/probes/protocol_v0.2/outer_predictions.csv'
        if (main['provenance']['bank_sha256'] != file_hash(bank_path)
                or main['provenance']['predictions_sha256'] != file_hash(predictions_path)):
            raise ValueError('Main report differs from frozen bank/predictions')
        for path in (main_path, bank_path, predictions_path):
            inputs[path.relative_to(root).as_posix()] = file_hash(path)
        result.update(status='fixed_design_main_analysis', confirmatory_results_available=True, main=main)
        result['primary_questions'] = {'Q1_early': 'estimated_on_held_out_tasks',
                                      'Q2_32_tokens': 'supported' if main['signal']['H2']['supported'] else 'not_supported',
                                      'Q3_early_abort': 'supported' if main['policies']['H3']['operational_benefit_claim_supported'] else 'not_supported'}
        transfer_path = root / 'reports/transfer_v0.2/results.json'
        if transfer_path.exists():
            result['transfer'] = json.loads(transfer_path.read_text(encoding='utf-8'))
            inputs[transfer_path.relative_to(root).as_posix()] = file_hash(transfer_path)
        online_path = root / 'runs/probes/protocol_v0.2/online_validation.json'
        if online_path.exists():
            result['online'] = json.loads(online_path.read_text(encoding='utf-8'))
            inputs[online_path.relative_to(root).as_posix()] = file_hash(online_path)
        authorization = root / 'reports/research/execution_freeze_v0.2/manifest.json'
        if authorization.exists():
            result['execution_authorization'] = json.loads(authorization.read_text(encoding='utf-8'))
            inputs[authorization.relative_to(root).as_posix()] = file_hash(authorization)
    write_json(output / 'evidence.json', result)
    plot_evidence(result, figures)
    return result


def plot_evidence(evidence: dict, output: Path) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.titleweight': 'bold', 'savefig.dpi': 220})
    def save(fig, name):
        fig.tight_layout()
        for extension in ('png', 'svg'):
            fig.savefig(output / f'{name}.{extension}', bbox_inches='tight')
        plt.close(fig)
    data = pd.DataFrame(evidence['legacy_sampling'])
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    intervals = np.array(data['resolution_ci95'].tolist())
    axes[0].plot(data['budget'], data['resolution'], 'o-', color='#245c80')
    axes[0].fill_between(data['budget'], intervals[:, 0], intervals[:, 1], color='#245c80', alpha=.15)
    axes[0].set(xlabel='Máximo de tentativas', ylabel='Proporção de tarefas resolvidas', ylim=(0, 1),
                title='Cobertura no banco histórico')
    axes[0].set_xticks(data['budget'])
    for column, label, color in [('fixed_tokens', 'Gerar todos', '#8c9298'), ('verifier_tokens', 'Parar após verificação', '#147d78')]:
        axes[1].plot(data['budget'], data[column] / data['tasks'], 'o-', label=label, color=color)
    axes[1].set(xlabel='Máximo de tentativas', ylabel='Tokens de continuação por tarefa', title='Mesmo banco de candidatos')
    axes[1].set_xticks(data['budget'])
    axes[1].legend(frameon=False)
    save(fig, 'legacy_sampling')
    values = evidence['legacy_reanalysis']['metrics']
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, key, title in zip(axes, ['roc_auc', 'within_task_auc'], ['AUC agregada', 'AUC dentro da tarefa']):
        for i, row in enumerate(values):
            ax.bar(i, row[key], color=['#8c9298', '#245c80', '#147d78'][i], width=.6)
            if key == 'within_task_auc':
                low, high = row['wt_auc_interval']['ci95']
                ax.errorbar(i, row[key], yerr=[[row[key]-low], [high-row[key]]], fmt='none', color='#172b3a', capsize=4)
        ax.axhline(.5, color='#888', linestyle='--', linewidth=.8)
        ax.set(xticks=range(3), xticklabels=['B tardio', 'H final', 'H+B tardio'], ylim=(0, 1), title=title)
    fig.suptitle('Reanálise exploratória • estado final • validação agrupada por tarefa', fontsize=11)
    save(fig, 'legacy_late_probes')
    categories = pd.Series([r['category'] for r in evidence['legacy_task_composition']]).value_counts()
    names = [('all_failed', 'Nenhum acerto'), ('mixed', 'Resultados mistos'), ('all_correct', 'Todos corretos')]
    fig, axis = plt.subplots(figsize=(6.5, 3.5))
    counts = [int(categories.get(key, 0)) for key, _ in names]
    bars = axis.barh([label for _, label in names], counts, color=['#9c6767', '#245c80', '#147d78'])
    axis.bar_label(bars, padding=5)
    axis.set(xlabel='Tarefas (5 tentativas observadas por tarefa)', xlim=(0, max(counts) + 3),
             title='Heterogeneidade no banco histórico')
    save(fig, 'legacy_task_composition')
    pilot = pd.DataFrame(evidence.get('pilot_temperatures', []))
    if not pilot.empty:
        pilot = pilot[pilot['version'] == '0.2'].sort_values('temperature')
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
        x = np.arange(len(pilot))
        incorrect = pilot['attempts'] - pilot['correct']
        pending = pilot['planned'] - pilot['attempts']
        axes[0].bar(x, pilot['correct'], label='Corretas observadas', color='#147d78')
        axes[0].bar(x, incorrect, bottom=pilot['correct'], label='Incorretas observadas', color='#9c6767')
        axes[0].bar(x, pending, bottom=pilot['attempts'], label='Ainda não geradas', color='#e0e4e8')
        axes[0].set(xticks=x, xticklabels=pilot['temperature'], xlabel='Temperatura', ylabel='Trajetórias', title='Cobertura do desenho do piloto')
        axes[0].legend(frameon=False, fontsize=8)
        axes[1].bar(x, pilot['mean_tokens'], color='#245c80')
        axes[1].set(xticks=x, xticklabels=pilot['temperature'], xlabel='Temperatura', ylabel='Tokens médios observados', title='Comprimento das gerações disponíveis')
        for index, row in enumerate(pilot.to_dict('records')):
            axes[1].text(index, row['mean_tokens'], f"n={row['attempts']}", ha='center', va='bottom', fontsize=8)
        fig.suptitle('Piloto de desenvolvimento • snapshot descritivo • sem seleção parcial', fontsize=11)
        save(fig, 'pilot_sampling_progress')
