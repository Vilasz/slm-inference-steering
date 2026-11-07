"""Audit and describe the completed, prospectively approved independent endpoint."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from slm_steering.excellence.approvals import approved_design
from slm_steering.excellence.audit import assert_preserved
from slm_steering.excellence.economics import paired_ratio_saving
from slm_steering.research.features import observable_features
from slm_steering.research.probes import calibrated_scores
from slm_steering.research.protocol import attempt_seed, file_hash, now, read_bank, validate_bank, write_json
from slm_steering.research.statistics import paired_task_bootstrap, task_weights


def recall_intervals(counts, *, replications, seed):
    """Ratios of candidate counts, resampling entire paired task clusters."""
    counts = np.asarray(counts, dtype=float)
    if counts.ndim != 2 or counts.shape[1] != 3 or not len(counts):
        raise ValueError('Expected paired per-task HB/B/active-correct counts')
    if not np.isfinite(counts).all() or (counts < 0).any() or (counts[:, :2] > counts[:, 2:3]).any():
        raise ValueError('Invalid retention counts')
    rng = np.random.default_rng(seed)
    totals = counts[rng.integers(0, len(counts), (replications, len(counts)))].sum(axis=1)
    valid = totals[:, 2] > 0
    original = counts.sum(axis=0)
    rows = []
    for column, family in enumerate(('HB', 'B')):
        draws = totals[valid, column] / totals[valid, 2]
        rows.append({'family': family, 'retained_correct': int(original[column]),
                     'active_correct': int(original[2]),
                     'recall': float(original[column] / original[2]) if original[2] else None,
                     'ci95': np.quantile(draws, [.025, .975]).tolist() if len(draws) else None,
                     'undefined_draws': int((~valid).sum())})
    return rows


def calibration_rows(frame, family):
    weights = task_weights(frame.task_id.to_numpy())
    scores = frame[family].to_numpy()
    labels = frame.passed.to_numpy()
    bins = np.minimum((scores * 10).astype(int), 9)
    rows, ece = [], 0.
    for index in range(10):
        mask = bins == index
        if not mask.any():
            rows.append({'family': family, 'bin': index, 'candidates': 0, 'weight': 0.,
                         'mean_score': None, 'accuracy': None})
            continue
        weight = float(weights[mask].sum() / weights.sum())
        score, accuracy = float(np.average(scores[mask], weights=weights[mask])), float(np.average(labels[mask], weights=weights[mask]))
        ece += weight * abs(score - accuracy)
        rows.append({'family': family, 'bin': index, 'candidates': int(mask.sum()), 'weight': weight,
                     'mean_score': score, 'accuracy': accuracy})
    return rows, ece


def run(root: Path):
    design = approved_design(root)
    assert_preserved(root)
    output = root / 'reports/excellence/independent_v1'
    endpoint = json.loads((output / 'results.json').read_text(encoding='utf-8'))
    if endpoint['status'] != 'completed_approved_independent_evaluation':
        raise ValueError('Independent endpoint is not complete')
    freeze = json.loads((output / 'freeze/manifest.json').read_text(encoding='utf-8'))
    for relative, expected in freeze['inputs'].items():
        if file_hash(root / relative) != expected:
            raise ValueError(f'Approved scientific input changed: {relative}')
    audit, main_records = [], []
    for phase, key in [('technical', 'technical_pilot_task_ids'), ('main', 'main_task_ids')]:
        directory = root / f'runs/excellence/independent_v1/{phase}'
        manifest = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
        records = read_bank(directory / 'trajectories.jsonl')
        validate_bank(records)
        expected = {(task, j) for task in design[key] for j in range(20)}
        if manifest['status'] != 'completed' or not manifest['numerical_checks_passed']:
            raise ValueError('Incomplete bank or failed numerical controls')
        if {(v['task_id'], v['attempt_id']) for v in records} != expected or len(records) != len(expected):
            raise ValueError('Sample identities differ from the approved reservation')
        if file_hash(directory / 'trajectories.jsonl') != manifest['bank_sha256']:
            raise ValueError('Completed bank hash differs')
        for record in records:
            if record['seed'] != attempt_seed('excellence-extension-v1', record['benchmark_version'], record['task_id'], record['attempt_id']):
                raise ValueError('Seed does not match approved namespace and identity')
            path = directory / record['activation_path']
            if file_hash(path) != record['artifact_hashes']['activations']:
                raise ValueError('Activation hash differs')
            with np.load(path, allow_pickle=False) as states:
                for name in states.files:
                    if not np.isfinite(states[name]).all():
                        raise ValueError('Nonfinite stored state')
        audit.append({'phase': phase, 'tasks': len(design[key]), 'attempts': len(records),
                      'bank_sha256': manifest['bank_sha256'], 'numerical_checks_passed': True,
                      'sessions': len(manifest['sessions']),
                      'session_seconds': sum(v['seconds'] for v in manifest['sessions'])})
        if phase == 'main':
            main_records = records
    if set(design['main_task_ids']) & set(design['technical_pilot_task_ids']):
        raise ValueError('Technical and main reservations overlap')
    exposure = json.loads((root / 'reports/excellence/target_exposure_audit.json').read_text(encoding='utf-8'))
    if set(design['main_task_ids'] + design['technical_pilot_task_ids']) & set(exposure['observed_task_ids']):
        raise ValueError('Approved reservation overlaps previously observed tasks')
    frame = pd.read_csv(output / 'predictions.csv', float_precision='round_trip')
    active = [v for v in main_records if v['generated_tokens'] > 8]
    by_identity = frame.set_index(['task_id', 'attempt_id'])
    if by_identity.index.has_duplicates or set(by_identity.index) != {(v['task_id'], v['attempt_id']) for v in active}:
        raise ValueError('Predictions do not cover the strict active population exactly')
    bundle = joblib.load(root / 'runs/transfer/protocol_v0.2/models.joblib')
    observable, hidden, original = [], [], []
    for record in active:
        with np.load(root / 'runs/excellence/independent_v1/main' / record['activation_path'], allow_pickle=False) as states:
            hidden.append(states['8'][0].astype(float))
        observable.append(observable_features(record, 8))
        row = by_identity.loc[(record['task_id'], record['attempt_id'])]
        if int(row.passed) != int(record['passed']):
            raise ValueError('Prediction labels differ from the official records')
        original.append([row.B, row.HB])
    maximum_error = 0.
    if active:
        obs, h = np.asarray(observable), np.asarray(hidden)
        reproduced = np.column_stack([calibrated_scores(bundle['models'][(family, 8)], x)
                                     for family, x in [('B', obs), ('HB', np.column_stack([h, obs]))]])
        maximum_error = float(np.max(np.abs(reproduced - np.asarray(original))))
        if maximum_error > 1e-12:
            raise ValueError('Independent scoring reproduction differs')
    counts = np.asarray(endpoint['per_task_counts'])
    fresh_counts = []
    for task in design['main_task_ids']:
        part = frame[(frame.task_id == task) & (frame.passed == 1)]
        fresh_counts.append([int(part.HB.ge(design['policies']['HB']['threshold']).sum()),
                             int(part.B.ge(design['policies']['B']['threshold']).sum()), len(part)])
    if not np.array_equal(counts, fresh_counts):
        raise ValueError('Primary per-task sufficient counts differ')
    recall = recall_intervals(counts, replications=design['bootstrap_replications'], seed=design['bootstrap_seed'])
    primary = endpoint['primary']
    denominator = int(counts[:, 2].sum())
    direct = float((counts[:, 0].sum() - counts[:, 1].sum()) / denominator) if denominator else None
    if direct != primary['estimate']:
        raise ValueError('Primary ratio estimate differs')
    policies = pd.read_csv(output / 'policy_per_task.csv')
    summaries, effects = [], []
    for budget in (1, 5, 10, 20):
        group = policies[policies.budget == budget]
        verifier = group[group.policy == 'verifier'].set_index('task_id').sort_index()
        baseline = group[group.policy == 'B'].set_index('task_id').sort_index()
        for policy in ('fixed', 'verifier', 'B', 'HB', 'oracle'):
            part = group[group.policy == policy].set_index('task_id').reindex(verifier.index)
            if len(part) != len(design['main_task_ids']) or part.tokens.isna().any():
                raise ValueError('Incomplete policy replay table')
            summaries.append({'policy': policy, 'budget': budget, 'resolution': float(part.resolved.mean()),
                              'resolved_tasks': int(part.resolved.sum()), 'mean_tokens': float(part.tokens.mean()),
                              'mean_attempts_started': float(part.started.mean()),
                              'correct_false_aborts_started': int(part.false_aborts.sum()),
                              'active_correct_started': int(part.active_correct_started.sum()),
                              'saving_vs_verifier': paired_ratio_saving(part.tokens, verifier.tokens,
                                  replications=design['bootstrap_replications'], seed=design['bootstrap_seed'])})
            if policy == 'HB':
                effects.append({'budget': budget,
                    'resolution_delta_vs_B': paired_task_bootstrap((part.resolved - baseline.resolved).to_numpy(),
                        replications=design['bootstrap_replications'], seed=design['bootstrap_seed']),
                    'token_delta_vs_B': paired_task_bootstrap((part.tokens - baseline.tokens).to_numpy(),
                        replications=design['bootstrap_replications'], seed=design['bootstrap_seed'])})
    calibration, metric_rows = [], []
    for metric in endpoint['secondary_predictive_metrics']:
        rows, ece = calibration_rows(frame, metric['family'])
        calibration.extend(rows)
        metric_rows.append({**metric, 'ece_10_bins_task_weighted': ece})
    pd.DataFrame(calibration).to_csv(output / 'calibration.csv', index=False)
    task_counts = pd.DataFrame(counts, columns=['retained_HB', 'retained_B', 'active_correct'])
    task_counts.insert(0, 'task_id', design['main_task_ids'])
    task_counts.to_csv(output / 'retention_per_task.csv', index=False)
    result = {'created_at': now(), 'status': 'completed_approved_independent_analysis',
              'primary': primary, 'recall': recall, 'main_tasks': len(design['main_task_ids']),
              'main_attempts': len(main_records), 'technical_tasks': len(design['technical_pilot_task_ids']),
              'active_attempts_t8': len(frame), 'active_correct_t8': denominator,
              'mixed_tasks_t8': int((frame.groupby('task_id').passed.nunique() == 2).sum()),
              'secondary_predictive_metrics': metric_rows, 'policy_summaries': summaries,
              'secondary_HB_vs_B_effects': effects, 'audit': audit,
              'rescoring_max_probability_error': maximum_error,
              'protocol_sha256': file_hash(root / 'docs/protocols/EXCELLENCE_EXTENSION_v1.md'),
              'freeze_manifest_sha256': file_hash(output / 'freeze/manifest.json'),
              'approval_source': 'Explicit student approval; advisor review reported by student',
              'interpretation': 'Sole confirmatory endpoint is paired candidate retention at t8; all other comparisons are descriptive',
              'limitations': ['One fixed model and one additional function-level benchmark sample',
                             'New locally unexposed tasks do not certify absence of pretraining contamination',
                             'Task bootstrap assumes exchangeable tasks; deterministic hash reservation is not a census',
                             'No target fitting, threshold revision, outcome-dependent replacement or significance stopping',
                             'Independent retention confirmation does not confirm original incremental-value or operational-benefit hypotheses',
                             'Token replay does not establish wall-clock policy savings']}
    plot(output, result, calibration)
    paths = ['docs/protocols/EXCELLENCE_EXTENSION_v1.approval.json', 'docs/protocols/EXCELLENCE_EXTENSION_v1.md',
             'reports/excellence/proposal_design.json', 'reports/excellence/independent_v1/freeze/manifest.json',
             'reports/excellence/independent_v1/results.json', 'reports/excellence/independent_v1/predictions.csv',
             'reports/excellence/independent_v1/policy_per_task.csv', 'reports/excellence/independent_v1/retention_per_task.csv',
             'reports/excellence/independent_v1/calibration.csv', 'src/slm_steering/excellence/independent_report.py']
    paths += [f'runs/excellence/independent_v1/{phase}/{name}' for phase in ('technical', 'main')
              for name in ('manifest.json', 'trajectories.jsonl')]
    paths += [p.relative_to(root).as_posix() for p in output.glob('independent_*.?*') if p.suffix in ('.png', '.svg', '.pdf')]
    result['inputs'] = {p: file_hash(root / p) for p in paths}
    write_json(output / 'analysis.json', result)
    write_readme(output, result)
    print(json.dumps({'primary': primary, 'audited_attempts': sum(v['attempts'] for v in audit),
                      'rescoring_error': maximum_error}), flush=True)
    return result


def plot(output, result, calibration):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.7))
    for j, row in enumerate(result['recall']):
        if row['recall'] is not None:
            axes[0].plot([j, j], row['ci95'], color='#3274A1')
            axes[0].scatter(j, row['recall'], color='#3274A1')
    axes[0].axhline(.97, color='gray', ls='--', lw=.8, label='Alvo empírico na origem')
    axes[0].set(xticks=[0, 1], xticklabels=['H+B', 'B'], ylim=(0, 1.02), ylabel='Corretos ativos mantidos', title='Nova amostra MBPP+, t=8')
    axes[0].legend(fontsize=8)
    p = result['primary']
    if p['estimate'] is not None:
        axes[1].plot(p['ci95'], [0, 0], color='#C44E52')
        axes[1].scatter(p['estimate'], 0, color='#C44E52')
    axes[1].axvline(0, color='gray', lw=.8)
    axes[1].set(yticks=[], xlabel='Retenção H+B menos B; IC95%', title='Único contraste confirmatório')
    fig.tight_layout()
    for ext in ('png', 'svg', 'pdf'):
        fig.savefig(output / f'independent_retention.{ext}', dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for policy in ('fixed', 'verifier', 'B', 'HB', 'oracle'):
        rows = [v for v in result['policy_summaries'] if v['policy'] == policy]
        axes[0].plot([v['mean_tokens'] for v in rows], [v['resolution'] for v in rows], marker='o', label=policy)
    axes[0].set(xlabel='Tokens médios por tarefa', ylabel='Fração de tarefas resolvidas', title='Replay; N=1,5,10,20')
    axes[0].legend(fontsize=8)
    for family in ('B', 'HB'):
        rows = [v for v in calibration if v['family'] == family and v['candidates']]
        axes[1].plot([v['mean_score'] for v in rows], [v['accuracy'] for v in rows], marker='o', label=family)
    axes[1].plot([0, 1], [0, 1], color='gray', ls='--', lw=.8)
    axes[1].set(xlabel='Probabilidade média prevista', ylabel='Acerto observado ponderado por tarefa', title='Calibração; dez bins fixos')
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    for ext in ('png', 'svg', 'pdf'):
        fig.savefig(output / f'independent_secondary.{ext}', dpi=180)
    plt.close(fig)


def write_readme(output, result):
    p = result['primary']
    lines = ['# Avaliação independente aprovada', '',
             f"Amostra principal: {result['main_tasks']} tarefas e {result['main_attempts']} tentativas; piloto técnico disjunto: {result['technical_tasks']} tarefas.",
             'Modelo, gates e amostra congelados antes da primeira trajetória nova. Aprovação explícita do estudante; revisão do orientador informada por ele.', '',
             '## Único endpoint confirmatório', '',
             f"Diferença de retenções H+B−B em t=8: {p['estimate']}; IC95%: {p['ci95']}.",
             f"Hipótese direcional sustentada pelo critério pré-especificado: {p['directional_hypothesis_supported']}. Endpoint identificável: {p['identified']}.",
             f"Bootstrap: {p['replications']} reamostragens por tarefa; seed {p['seed']}; {p['undefined_draws']} replicações sem denominador positivo.", '',
             '| Gate | Corretos retidos/ativos | Retenção | IC95% |', '|---|---:|---:|---|']
    for row in result['recall']:
        lines.append(f"| {row['family']} | {row['retained_correct']}/{row['active_correct']} | {row['recall']} | {row['ci95']} |")
    lines += ['', 'Todas as tarefas principais entram nas contagens, inclusive aquelas sem acertos. O denominador é comum aos gates.',
              'Os intervalos das retenções absolutas e dos resultados abaixo são descritivos. A decisão acima não substitui Q2/H3 originais.', '',
              '## Replay descritivo, N=20', '', '| Política | Tarefas resolvidas | Tokens médios | Falsos abortos iniciados |', '|---|---:|---:|---:|']
    for row in result['policy_summaries']:
        if row['budget'] == 20:
            lines.append(f"| {row['policy']} | {row['resolved_tasks']}/{result['main_tasks']} | {row['mean_tokens']:.3f} | {row['correct_false_aborts_started']} |")
    lines += ['', 'Dados: [analysis.json](analysis.json), [predições](predictions.csv), [contagens por tarefa](retention_per_task.csv),',
              '[replay completo](policy_per_task.csv), [calibração](calibration.csv) e [congelamento](freeze/manifest.json).', '',
              'A auditoria verificou todas as identidades, seeds e ativações das duas fases e reaplicou os scorers congelados.',
              f"Erro máximo de reprodução das probabilidades: {result['rescoring_max_probability_error']:.3g}.", '',
              'Limites: um modelo e uma reserva MBPP+; exposição de pré-treino não certificada; bootstrap pressupõe tarefas intercambiáveis;',
              'não há ajuste no alvo, atribuição causal, equivalência pelo cruzamento de zero ou inferência de latência a partir de tokens.', '']
    (output / 'README.md').write_text('\n'.join(lines), encoding='utf-8')


def render_typst(result, number, interval, table):
    def ci(values):
        return interval(values) if values is not None else 'não identificável'
    p = result['primary']
    finding = ('O critério direcional foi satisfeito: o limite superior do intervalo é negativo.'
               if p['directional_hypothesis_supported'] else
               'O critério direcional não foi satisfeito; isso não demonstra equivalência entre os gates.')
    if not p['identified']:
        finding = 'O endpoint não é identificável no orçamento segundo a regra pré-especificada; não se declara confirmação.'
    text = '''
== Validação independente aprovada

Depois dos diagnósticos, o estudante aprovou explicitamente a proposta e informou
que o orientador já a havia revisado. Esse relato, sem assinatura criada em nome
do orientador, foi vinculado aos hashes do protocolo, desenho e implementação.
Um snapshot prospectivo de 46 arquivos precede a primeira trajetória nova.
O cabeçalho da proposta registra seu estado histórico de elaboração; o arquivo
de aprovação e o manifesto de congelamento registram a autorização subsequente.
Não se reivindica pré-registro externo nem se modifica a autorização histórica v0.2.

A hipótese única compara retenção de candidatos corretos por H+B e B no mesmo
checkpoint de oito tokens. Os limiares de origem são 0,26953905544150975 e
0,3166734153299228, respectivamente, sem ajuste nos novos dados. A reserva por
hash exclui tarefas com resultados locais anteriores e o controle Mbpp/2.
Vinte tarefas técnicas disjuntas verificam a implementação; suas 400 tentativas
não entram na inferência principal. A amostra principal contém 120 novas tarefas
MBPP+ e 2.400 tentativas completas, mesmo após sucessos. A população é T>8;
o denominador comum contém todos os candidatos corretos ativos, inclusive os
que não seriam iniciados no replay. Não se usa a média das retenções por tarefa.

O estimando é a diferença das razões das contagens esperadas por tarefa:
$ theta = r_(H+B)-r_B. $
Reamostram-se tarefas inteiras, mantendo o pareamento, 10.000 vezes com seed
20261017. O único critério de suporte é limite superior do IC bilateral de
95% estritamente negativo. Mais de 1% de replicações sem denominador positivo
tornaria o endpoint não identificável. Não houve troca de tarefa, seleção de
limiar, expansão por resultado ou parada por significância.

'''
    text += (f"Em T>8 restam {result['active_attempts_t8']} candidatos, "
             f"{result['mixed_tasks_t8']} tarefas mistas e {result['active_correct_t8']} corretos ativos. "
             f"A estimativa de θ é {number(p['estimate'], 4)}, IC95% {ci(p['ci95'])}. "
             f"Houve {p['undefined_draws']} replicações indefinidas. {finding}\n\n")
    text += table(['Gate', 'Corretos retidos/ativos', 'Retenção', 'IC95%'],
                  [[r['family'], f"{r['retained_correct']}/{r['active_correct']}",
                    number(r['recall'], 4), ci(r['ci95'])] for r in result['recall']],
                  'tab-independent-recall', 'Retenções na nova amostra. Seus intervalos absolutos são descritivos; a confirmação usa somente o contraste pareado.')
    text += '''
#figure(image("../../reports/excellence/independent_v1/independent_retention.svg", width: 100%),
 caption: [Avaliação independente no checkpoint comum. A linha em 0,97 é o alvo empírico de desenvolvimento, sem garantia de transporte.]) <fig-independent-recall>

=== Resultados secundários e limites da confirmação

'''
    text += table(['Preditor', 'AUC', 'WT-AUC', 'Log loss', 'Brier'],
                  [[r['family'], number(r['roc_auc'], 4), number(r['within_task_auc'], 4),
                    number(r['log_loss'], 4), number(r['brier'], 4)] for r in result['secondary_predictive_metrics']],
                  'tab-independent-metrics', 'Métricas secundárias na população ativa. Discriminação não equivale a calibração ou utilidade operacional.')
    text += table(['Política', 'Tarefas resolvidas', 'Tokens médios', 'Falsos abortos iniciados'],
                  [[r['policy'], f"{r['resolved_tasks']}/{result['main_tasks']}",
                    number(r['mean_tokens'], 2), str(r['correct_false_aborts_started'])]
                   for r in result['policy_summaries'] if r['budget'] == 20],
                  'tab-independent-replay', 'Replay secundário N=20; apenas tentativas iniciadas geram custo. Oracle conhece os rótulos e preserva todos os corretos.')
    text += '''
#figure(image("../../reports/excellence/independent_v1/independent_secondary.svg", width: 100%),
 caption: [Custo e resolução nos quatro orçamentos e calibração com dez bins fixos. Resultados secundários descritivos, sem seleção de política no alvo.]) <fig-independent-secondary>

Os CSVs publicam todos os orçamentos N=1,5,10,20, custos, perdas de cobertura,
retenções, calibração e denominadores dos abortos iniciados. A auditoria reabriu
as ativações de ambas as fases, conferiu todas as identidades e seeds e reaplicou
os scorers congelados. A hipótese nova versa sobre transporte relativo de decisões;
seu suporte não confirma ganho incremental de representações, os critérios Q2/H3
originais, uma causa de domain shift ou superioridade universal de um método.

São novas tarefas sem resultados locais anteriores, não tarefas certificadamente
ausentes do pré-treinamento. O bootstrap pressupõe intercambiabilidade entre
tarefas da população; a reserva determinística por hash não é um censo nem
valida outros modelos ou benchmarks. Os intervalos secundários são pontuais.
O replay mede tokens e resolução, sem demonstrar economia de latência, energia
ou FLOPs. Nenhum resultado retrospectivo substituiu o critério aprovado.
'''
    return text
