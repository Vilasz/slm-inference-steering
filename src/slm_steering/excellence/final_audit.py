"""Independent arithmetic reconstruction of the final tables, without modifying history."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from slm_steering.excellence import audit as original_auditor
from slm_steering.excellence.audit import assert_preserved
from slm_steering.research.protocol import file_hash, now, read_bank, write_json


def close(value, expected, label):
    if value is None or expected is None:
        if value != expected:
            raise ValueError(f'Undefined metric mismatch: {label}')
    elif not np.isclose(value, expected, rtol=0, atol=2e-12):
        raise ValueError(f'Numerical divergence {label}: {value} versus {expected}')


def manual_metrics(labels, scores, groups):
    y, p, g = np.asarray(labels), np.asarray(scores), np.asarray(groups)
    p = np.clip(p.astype(float), np.finfo(float).eps, 1 - np.finfo(float).eps)
    weights = np.empty(len(y), dtype=float)
    auc, losses, briars = {}, {}, {}
    for task in sorted(set(g)):
        selected = g == task
        weights[selected] = 1 / selected.sum()
        yy, pp = y[selected], p[selected]
        losses[task] = float(-(yy * np.log(pp) + (1 - yy) * np.log1p(-pp)).mean())
        briars[task] = float(np.square(pp - yy).mean())
        if yy.min() != yy.max():
            differences = pp[yy == 1, None] - pp[None, yy == 0]
            auc[task] = float(((differences > 0) + .5 * (differences == 0)).mean())
    positive, negative = y == 1, y == 0
    pooled = None
    if positive.any() and negative.any():
        differences = p[positive, None] - p[None, negative]
        pair_weights = weights[positive, None] * weights[None, negative]
        pooled = float(np.sum(pair_weights * ((differences > 0) + .5 * (differences == 0))) / pair_weights.sum())
    return {'roc_auc': pooled, 'within_task_auc': float(np.mean(list(auc.values()))) if auc else None,
            'log_loss': float(np.mean(list(losses.values()))), 'brier': float(np.mean(list(briars.values()))),
            'task_auc': auc, 'task_losses': losses}


def bootstrap_means(values, replications=2000, seed=20260921):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), (replications, len(values)))].mean(axis=1)
    return {'estimate': float(values.mean()), 'ci95': np.quantile(means, [.025, .975]).tolist()}


def check_effect(actual, expected, label):
    close(actual['estimate'], expected['estimate'], label)
    for a, b in zip(actual['ci95'], expected['ci95']):
        close(a, b, label + ' CI')


def reconstruct_independent_endpoint(root):
    """Recount the new endpoint and every replay row with a separate direct loop."""
    directory = root / 'reports/excellence/independent_v1'
    if not (directory / 'analysis.json').exists():
        return None
    design = json.loads((root / 'reports/excellence/proposal_design.json').read_text(encoding='utf-8'))
    endpoint = json.loads((directory / 'results.json').read_text(encoding='utf-8'))
    records = read_bank(root / 'runs/excellence/independent_v1/main/trajectories.jsonl')
    frame = pd.read_csv(directory / 'predictions.csv', float_precision='round_trip')
    replay = pd.read_csv(directory / 'policy_per_task.csv')
    lookup = {(r.task_id, r.attempt_id): {'B': r.B, 'HB': r.HB} for r in frame.itertuples()}
    counts, streams = [], {}
    for task in design['main_task_ids']:
        stream = sorted((r for r in records if r['task_id'] == task), key=lambda r: r['attempt_id'])
        streams[task] = stream
        active = [r for r in stream if r['passed'] and r['generated_tokens'] > 8]
        counts.append([sum(lookup[(task, r['attempt_id'])][family] >= design['policies'][family]['threshold']
                           for r in active) for family in ('HB', 'B')] + [len(active)])
    counts = np.asarray(counts, dtype=int)
    if not np.array_equal(counts, endpoint['per_task_counts']):
        raise ValueError('Independent endpoint sufficient counts differ in direct reconstruction')
    total = counts.sum(axis=0)
    rng = np.random.default_rng(design['bootstrap_seed'])
    totals = counts[rng.integers(0, len(counts), (design['bootstrap_replications'], len(counts)))].sum(axis=1)
    valid = totals[:, 2] > 0
    estimate = float((total[0] - total[1]) / total[2]) if total[2] else None
    interval = np.quantile((totals[valid, 0] - totals[valid, 1]) / totals[valid, 2], [.025, .975]).tolist() if valid.any() else None
    close(estimate, endpoint['primary']['estimate'], 'new primary ratio')
    if interval is None:
        if endpoint['primary']['ci95'] is not None:
            raise ValueError('Undefined new primary interval differs')
    else:
        for a, b in zip(interval, endpoint['primary']['ci95']):
            close(a, b, 'new primary bootstrap')
    expected = {(task, policy, budget) for task in streams
                for policy in ('fixed', 'verifier', 'B', 'HB', 'oracle') for budget in (1, 5, 10, 20)}
    if len(replay) != len(expected) or {(r.task_id, r.policy, r.budget) for r in replay.itertuples()} != expected:
        raise ValueError('Independent replay has missing or duplicate rows')
    columns = ('tokens', 'resolved', 'started', 'correct_started', 'active_correct_started',
               'false_aborts', 'active_correct_in_budget', 'retained_correct_in_budget')
    for row in replay.itertuples():
        actual = dict.fromkeys(columns, 0)
        for record in streams[row.task_id][:row.budget]:
            active = record['generated_tokens'] > 8
            abort = active and not record['passed'] if row.policy == 'oracle' else False
            if row.policy in ('B', 'HB') and active:
                abort = lookup[(row.task_id, record['attempt_id'])][row.policy] < design['policies'][row.policy]['threshold']
            correct_active = int(active and record['passed'])
            actual['active_correct_in_budget'] += correct_active
            actual['retained_correct_in_budget'] += int(correct_active and not abort)
            # Budget retention counts include candidates that are never started.
            if actual['resolved'] and row.policy != 'fixed':
                continue
            actual['tokens'] += 8 if abort else record['generated_tokens']
            actual['started'] += 1
            actual['correct_started'] += int(record['passed'])
            actual['active_correct_started'] += correct_active
            actual['false_aborts'] += int(record['passed'] and abort)
            actual['resolved'] |= int(record['passed'] and not abort)
        if any(actual[column] != getattr(row, column) for column in columns):
            raise ValueError(f'Independent direct replay differs: {row.task_id} {row.policy} N={row.budget}')
    return {'estimate': estimate, 'ci95': interval, 'counts_HB_B_active': total.tolist(),
            'replay_rows_verified': len(replay), 'undefined_bootstrap_draws': int((~valid).sum()),
            'method': 'Direct candidate-count and replay loops; independent NumPy task bootstrap; no simulate call'}


def run(root: Path):
    assert_preserved(root)
    output = root / 'reports/excellence'
    def load(relative):
        return json.loads((root / relative).read_text(encoding='utf-8'))
    before = file_hash(output / 'reproduction.json')
    def save_copy(path, value):
        if path != output / 'reproduction.json':
            raise ValueError('Unexpected output from the historical reproduction audit')
        write_json(output / 'final_primary_reconstruction.json', value)
    with threadpool_limits(limits=1), patch.object(original_auditor, 'write_json', side_effect=save_copy):
        original_auditor.run(root)
    if file_hash(output / 'reproduction.json') != before:
        raise ValueError('Historical reproduction receipt was overwritten')
    original = load('reports/protocol_v0.2/results.json')
    source = pd.read_csv(root / 'runs/probes/protocol_v0.2/outer_predictions.csv', dtype={'checkpoint': str})
    primary = source[source.checkpoint == '32']
    tables = []
    for expected in original['signal']['metrics']:
        if expected['checkpoint'] != '32':
            continue
        family = expected['predictor']
        values = manual_metrics(primary.passed, primary[family], primary.task_id)
        for key in ('roc_auc', 'within_task_auc', 'log_loss', 'brier'):
            close(values[key], expected[key], 'primary ' + family + ' ' + key)
        tables.append({'table': 'primary_predictors', 'family': family,
                       **{k: v for k, v in values.items() if not k.startswith('task_')}})
    baseline = manual_metrics(primary.passed, primary.B, primary.task_id)
    combined = manual_metrics(primary.passed, primary.HB, primary.task_id)
    effect_wt = bootstrap_means([combined['task_auc'][task] - baseline['task_auc'][task] for task in baseline['task_auc']])
    effect_loss = bootstrap_means([combined['task_losses'][task] - baseline['task_losses'][task] for task in baseline['task_losses']])
    check_effect(effect_wt, original['signal']['H2']['delta_hb_minus_b'], 'original primary WT contrast')
    check_effect(effect_loss, original['signal']['H2']['delta_log_loss_hb_minus_b'], 'original primary loss contrast')
    models = load('reports/excellence/models/results.json')
    model_predictions = pd.read_csv(output / 'models/predictions.csv')
    for expected in models['summaries']:
        part = model_predictions[model_predictions.family == expected['family']]
        values = manual_metrics(part.passed, part.probability, part.task_id)
        for key in ('roc_auc', 'within_task_auc', 'log_loss', 'brier'):
            close(values[key], expected[key], 'ablation ' + expected['family'] + ' ' + key)
        wt = bootstrap_means([values['task_auc'][t] - baseline['task_auc'][t] for t in values['task_auc']])
        loss = bootstrap_means([values['task_losses'][t] - baseline['task_losses'][t] for t in values['task_losses']])
        check_effect(wt, expected['delta_wt_auc_vs_B'], 'ablation WT ' + expected['family'])
        check_effect(loss, expected['delta_log_loss_vs_B'], 'ablation loss ' + expected['family'])
        tables.append({'table': 'exploratory_readouts', 'family': expected['family'],
                       **{k: v for k, v in values.items() if not k.startswith('task_')},
                       'delta_wt_vs_B': wt, 'delta_loss_vs_B': loss})
    target_default = pd.read_csv(root / 'reports/transfer_v0.2/predictions.csv', dtype={'checkpoint': str})
    target = pd.read_csv(root / 'reports/transfer_v0.2/predictions.csv', dtype={'checkpoint': str}, float_precision='round_trip')
    transfer = load('reports/transfer_v0.2/results.json')
    for expected in transfer['metrics']:
        part = target[target.checkpoint == str(expected['checkpoint'])]
        values = manual_metrics(part.passed, part[expected['family']], part.task_id)
        for key in ('roc_auc', 'within_task_auc', 'log_loss', 'brier'):
            close(values[key], expected[key], f"transfer {expected['checkpoint']} {expected['family']} {key}")
        tables.append({'table': 'frozen_transfer', 'checkpoint': expected['checkpoint'], 'family': expected['family'],
                       **{k: v for k, v in values.items() if not k.startswith('task_')}})
    selections = load('runs/probes/protocol_v0.2/selection_audit.json')
    source_policies = [r for r in selections if 'policy' in r]
    target_metadata = load('runs/transfer/protocol_v0.2/models.json')
    target_policies = [dict(v, policy=k) for k, v in target_metadata['policies'].items()]
    supplemental = load('reports/final_review/supplemental/results.json')
    retention_rows = []
    for dataset, frame, policies in [('HumanEval+ OOF', source, source_policies), ('MBPP+ frozen', target, target_policies)]:
        saved = supplemental['correct_retention'][dataset]
        for expected in saved['aggregate']:
            counts = []
            for policy in policies:
                if policy['policy'] != expected['policy']:
                    continue
                part = frame[frame.checkpoint == str(policy['checkpoint'])]
                if 'fold' in policy:
                    part = part[part.fold == policy['fold']]
                for task, group in part.groupby('task_id', sort=True):
                    correct = group.passed.eq(1)
                    counts.append((task, int((correct & group[policy['policy']].ge(policy['threshold'])).sum()), int(correct.sum())))
            counts.sort()
            values = np.array([[r[1], r[2]] for r in counts])
            rng = np.random.default_rng(20260921)
            draws = values[rng.integers(0, len(values), (2000, len(values)))].sum(axis=1)
            valid = draws[:, 1] > 0
            ci = np.quantile(draws[valid, 0] / draws[valid, 1], [.025, .975]).tolist()
            actual = {'retained_correct': int(values[:, 0].sum()), 'active_correct': int(values[:, 1].sum()),
                      'recall': float(values[:, 0].sum() / values[:, 1].sum()), 'ci95': ci}
            for key in ('retained_correct', 'active_correct', 'recall'):
                close(actual[key], expected[key], 'correct retention ' + dataset + ' ' + key)
            for a, b in zip(ci, expected['ci95']):
                close(a, b, 'retention CI')
            retention_rows.append({'dataset': dataset, 'policy': expected['policy'], **actual})
    replay = pd.read_csv(output / 'economics/per_task.csv')
    original_replay = pd.read_csv(output / 'economics/original_selected_policies.csv')
    replay_checks, label_checks = 0, 0
    for dataset, phase, predictions in [('HumanEval+', 'definitive', source), ('MBPP+', 'transfer', target)]:
        records = read_bank(root / f'runs/candidate_bank/{phase}_v0.2/trajectories.jsonl')
        streams = {}
        for record in records:
            streams.setdefault(record['task_id'], []).append(record)
            official = all(v == 'pass' for v in record['verifier_status'].values())
            if bool(record['passed']) != official and record.get('error_type') != 'empty_extraction':
                raise ValueError('Functional label differs from the saved official statuses')
            label_checks += 1
        scores = {(int(r.checkpoint), r.task_id, r.attempt_id, family): getattr(r, family)
                  for r in predictions[predictions.checkpoint.isin(['8', '16', '32', '64'])].itertuples()
                  for family in ('B', 'HB')}
        for frame in (replay, original_replay):
            for row in frame[frame.dataset == dataset].itertuples():
                stream = sorted(streams[row.task_id], key=lambda x: x['attempt_id'])[:row.budget]
                tokens, resolved, started, false_aborts = 0, False, 0, 0
                for record in stream:
                    active = record['generated_tokens'] > row.checkpoint
                    abort = active and (not record['passed']) if row.policy == 'oracle' else False
                    if row.policy in ('B', 'HB'):
                        abort = active and scores[(row.checkpoint, row.task_id, record['attempt_id'], row.policy)] < row.threshold
                    tokens += row.checkpoint if abort else record['generated_tokens']
                    false_aborts += bool(record['passed']) and abort
                    started += 1
                    resolved = resolved or (bool(record['passed']) and not abort)
                    if resolved and row.policy != 'fixed':
                        break
                if (tokens, int(resolved), started, false_aborts) != (row.tokens, row.resolved, row.started, row.false_aborts):
                    raise ValueError(f'Replay mismatch: {dataset} {row.task_id} {row.policy} N={row.budget}')
                replay_checks += 1
    economic = load('reports/excellence/economics/results.json')
    oracle_rows = []
    for row in economic['summaries']:
        part = replay[(replay.dataset == row['dataset']) & (replay.checkpoint == row['checkpoint']) &
                      (replay.budget == row['budget']) & (replay.policy == row['policy'])].set_index('task_id').sort_index()
        verifier = replay[(replay.dataset == row['dataset']) & (replay.checkpoint == row['checkpoint']) &
                          (replay.budget == row['budget']) & (replay.policy == 'verifier')].set_index('task_id').reindex(part.index)
        close(float(part.tokens.mean()), row['tokens_per_task'], 'economic mean tokens')
        close(float(part.resolved.mean()), row['resolution'], 'economic resolution')
        rng = np.random.default_rng(20260921)
        index = rng.integers(0, len(part), (2000, len(part)))
        saving = 1 - part.tokens.to_numpy()[index].sum(axis=1) / verifier.tokens.to_numpy()[index].sum(axis=1)
        close(float(1 - part.tokens.sum() / verifier.tokens.sum()), row['saving_vs_verifier']['estimate'], 'economic saving')
        for a, b in zip(np.quantile(saving, [.025, .975]), row['saving_vs_verifier']['ci95']):
            close(a, b, 'economic saving CI')
        if row['policy'] == 'oracle' and row['budget'] == 20:
            if not np.array_equal(part.resolved, verifier.resolved):
                raise ValueError('Oracle lost verifier coverage')
            oracle_rows.append(row)
    inputs = ['runs/probes/protocol_v0.2/outer_predictions.csv', 'reports/protocol_v0.2/results.json',
              'reports/transfer_v0.2/predictions.csv', 'reports/transfer_v0.2/results.json',
              'reports/excellence/models/results.json', 'reports/excellence/models/predictions.csv',
              'reports/excellence/economics/per_task.csv', 'reports/excellence/economics/original_selected_policies.csv',
              'reports/excellence/economics/results.json', 'reports/excellence/final_primary_reconstruction.json',
              'src/slm_steering/excellence/final_audit.py']
    result = {'created_at': now(), 'status': 'completed_independent_arithmetic_reconstruction',
              'primary_wt_effect': effect_wt, 'primary_loss_effect': effect_loss,
              'table_rows_verified': tables, 'selected_policy_retention': retention_rows,
              'independent_replay_rows_verified': replay_checks, 'functional_status_records_verified': label_checks,
              'oracle_table_N20': oracle_rows, 'historical_files_unchanged': True,
              'approved_extension_reconstruction': reconstruct_independent_endpoint(root),
              'transfer_csv_precision': {'reader': 'round_trip, matching full-precision scores used to write the frozen transfer metrics',
                  'max_probability_change_vs_default_parser': max(float(np.max(np.abs(target[f] - target_default[f]))) for f in ('B', 'H', 'HB'))},
              'scope': 'Pair-count AUC, task-mean proper losses, vectorized paired bootstrap and direct loop replay; no original simulate call',
              'inputs': {p: file_hash(root / p) for p in inputs}}
    reconstructions = {}
    for name, original, reconstructed in [
        ('main_statistics', 'reports/protocol_v0.2/results.json', 'reports/excellence/final_main_reconstruction/results.json'),
        ('transfer_statistics', 'reports/transfer_v0.2/results.json', 'reports/excellence/final_transfer_reconstruction/results.json'),
        ('transfer_policy_replay', 'reports/transfer_v0.2/policy_replay.json', 'reports/excellence/final_transfer_reconstruction/policy_replay.json')]:
        if (root / reconstructed).exists():
            if load(original) != load(reconstructed):
                raise ValueError(f'Full frozen statistical rerun differs: {name}')
            reconstructions[name] = {'exact_json_equality': True, 'original_sha256': file_hash(root / original),
                                     'rerun_sha256': file_hash(root / reconstructed), 'rerun': reconstructed}
    result['full_statistical_reruns'] = reconstructions
    if result['approved_extension_reconstruction']:
        for relative in ('reports/excellence/proposal_design.json', 'reports/excellence/independent_v1/results.json',
                         'reports/excellence/independent_v1/predictions.csv', 'reports/excellence/independent_v1/policy_per_task.csv',
                         'runs/excellence/independent_v1/main/trajectories.jsonl'):
            result['inputs'][relative] = file_hash(root / relative)
    write_json(output / 'final_table_reconstruction.json', result)
    write_report(root, result)
    assert_preserved(root)
    print(json.dumps({'table_rows': len(tables), 'replay_rows': replay_checks, 'functional_records': label_checks,
                      'primary_wt': effect_wt, 'primary_loss': effect_loss}), flush=True)


def write_report(root, result):
    lines = ['# Auditoria científica final', '', f"Verificação: {result['created_at']}.", '',
             '## Verificações concluídas', '',
             'Os valores abaixo foram recalculados dos CSVs e bancos, sem copiar os números do texto. A comparação usou tolerância absoluta de 2×10⁻¹².',
             'AUC foi recontada por pares, incluindo pesos de tarefa na AUC agregada; perdas são médias por tarefa.', '',
             f"ΔWT-AUC principal: {result['primary_wt_effect']}; Δlog loss: {result['primary_loss_effect']}.",
             f"Foram conferidas {len(result['table_rows_verified'])} linhas de preditores e {result['independent_replay_rows_verified']} linhas de replay, incluindo todas as parametrizações do oracle.",
             f"{result['functional_status_records_verified']} rótulos dos bancos principal e alvo conferem com os status oficiais salvos; o piloto também integra a reconstrução de seeds e scorer arquivada.", '',
             '| Avaliação | Política | Corretos retidos/ativos | Retenção e IC95% |', '|---|---|---:|---|']
    for row in result['selected_policy_retention']:
        lines.append(f"| {row['dataset']} | {row['policy']} | {row['retained_correct']}/{row['active_correct']} | {row['recall']:.6f} {row['ci95']} |")
    lines += ['', 'Tabela de preditores principal, tabela de retenção, ablações/leitura e oracle: todas reconstruídas. A numeração pode mudar após integrar a extensão; os identificadores semânticos dos artefatos permanecem.',
              'Os critérios Q2/Q3 são os originais. Q2 exige IC de incremento WT-AUC inteiramente positivo e redução pontual de perda; não exige IC da perda inteiramente negativo. O resultado negativo não mudou.',
              'O replay direto confirmou custos, ordem, orçamento, primeira solução, tentativas iniciadas e falsos abortos, sem chamar a implementação de simulate.',
              'Os casos manuais adicionais cobrem nenhum sucesso, sucesso inicial/final, término exatamente no checkpoint, aborto de corretos, informação futura e enumeração da fórmula de custo.', '',
              'Precisão de serialização: o parser padrão do CSV altera alguns empates de H em t=8 no alvo (até 8,33×10⁻¹⁷ nas probabilidades, com diferença de 0,002193 na WT-AUC). A leitura round_trip recupera todas as métricas da avaliação original em precisão completa. O leitor complementar foi corrigido; não houve retreino, alteração de rótulo ou redefinição do endpoint. A reexecução integral da transferência em pasta separada reproduziu exatamente métricas, deltas e políticas.', '',
              '## Controles e alcance', '',
              'A reconstrução de scorers e a auditoria das partições foram reexecutadas para arquivo novo, preservando o recibo anterior byte a byte: [final_primary_reconstruction.json](final_primary_reconstruction.json).',
              'As 270 partições de calibração e as escolhas one-SE foram verificadas. O teste de residual exclui o rótulo pontuado da seleção e calibração do primeiro estágio; a seleção residual repete essa separação internamente.',
              'PCA é ajustado apenas em treinos. A reprodução não reexecuta toda a busca histórica em todos os checkpoints; isso é distinto da reconstrução dos scorers principais e das duas execuções já arquivadas das 70 unidades exploratórias.',
              f"As rotinas congeladas de estatística foram também reexecutadas em pastas separadas, incluindo permutações e bootstrap: {result['full_statistical_reruns']}. Não se confundem com novo ajuste de todas as famílias em todos os checkpoints.",
              'Os três bancos antigos tiveram todos os arquivos de ativação reabertos e seus hashes/finitude verificados na auditoria anterior, preservada. A nova avaliação reabre suas próprias ativações ao concluir.', '',
              '## Problemas reais corrigidos', '',
              'Corrigidos o critério de perda na tabela complementar, a perda de entradas na regeneração da bibliografia, a alegação não demonstrada de sandbox completo, o teste legado que contradizia o padrão de normalização e o estado desatualizado da seção 1.4.',
              'As correções de apresentação não reescreveram protocolos, dados, seeds, fontes congelados ou decisões originais.', '',
              '## Avaliação independente e verificações posteriores', '',
              'Aprovação explícita e revisão do orientador informada pelo estudante já foram registradas, antes da primeira trajetória. Não houve nova solicitação da mesma aprovação.',
              'A auditoria de IDs, prompts, estrutura canônica e hashes dos testes está em [reserved_task_audit.json](reserved_task_audit.json). Foi feita após o início sob aprovação anterior, sem usar rótulos, excluir tarefas ou alterar o endpoint.',
              'Os helpers de parsing e dataset não eram entradas separadas no snapshot de 46 arquivos. Seus bytes anteriores à aprovação foram reconstruídos do Git HEAD e patch preservados antes da coleta, e coincidem com os fontes atuais: [parser_provenance_reconstruction.json](parser_provenance_reconstruction.json). Essa recuperação documenta a omissão; não cria retroativamente uma entrada no congelamento.',
              'O estado e os resultados completos estão em [independent_v1/](independent_v1/README.md). Dados incompletos não foram usados para calcular o endpoint.', '',
              '## Incertezas e não verificável', '',
              'Os intervalos antigos condicionam nos scorers ajustados; não incluem toda a variabilidade de retreino. Há exposição histórica a 26 tarefas. Ausência de significância não demonstra equivalência.',
              'A ausência de exposição no pré-treinamento e de toda semelhança semântica não pode ser certificada por estes arquivos. Aprovação nos testes não prova corretude universal.',
              'Não há assinatura do orientador nem aprovação formal da banca criada pelo agente. A revisão da extensão foi informada explicitamente pelo estudante e está descrita com esse alcance.',
              'Tempos técnicos e replay em tokens não demonstram economia de energia/FLOPs; a medição operacional adicional tem a escala efetivamente executada registrada em seus próprios artefatos.', '',
              'Dados numéricos, comandos e hashes: [final_table_reconstruction.json](final_table_reconstruction.json).', '']
    independent_path = root / 'reports/excellence/independent_v1/analysis.json'
    if independent_path.exists():
        independent = json.loads(independent_path.read_text(encoding='utf-8'))
        lines += ['## Resultado independente concluído', '',
                  f"{independent['main_tasks']} tarefas principais e {independent['main_attempts']} tentativas; piloto disjunto de {independent['technical_tasks']} tarefas.",
                  f"Endpoint primário: {independent['primary']}.",
                  f"Reconstrução aritmética separada do endpoint e de todas as linhas do replay novo: {result['approved_extension_reconstruction']}.",
                  'O endpoint novo não substitui Q2/H3 originais. Influência e agrupamento textual são sensibilidades posteriores, sem mudança da amostra ou novo critério de sucesso.', '']
    (root / 'reports/excellence/final_scientific_audit.md').write_text('\n'.join(lines), encoding='utf-8')
