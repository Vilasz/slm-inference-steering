"""Reconstruct the final statistics on CPU from saved scores and candidate metadata.

No model loading, generation, training, label execution, or historical writes.
The independent AUC and replay implementations deliberately do not call the
original evaluation/replay functions. Frozen code is used only as a cross-check
for prospective power simulation and the synthetic decision solver.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
OUT = ROOT / 'reports/finalization'
SEED = 20261104
FAMILIES = ('B_linear', 'HB_linear', 'B_MLP', 'HB_MLP')


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def read(path):
    return json.loads((ROOT / path).read_text(encoding='utf-8'))


def close(observed, expected, label, atol=1e-11):
    if expected is None:
        if observed is not None:
            raise ValueError(f'{label}: expected undefined, observed {observed}')
    elif not np.allclose(observed, expected, rtol=0, atol=atol):
        raise ValueError(f'{label}: {observed} != {expected}')


def paired_interval(values, *, replications=10000, seed=SEED):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError('One finite paired value per task required')
    indices = np.random.default_rng(seed).integers(0, len(values), (replications, len(values)))
    return {'estimate': float(values.mean()),
            'ci95': np.quantile(values[indices].mean(axis=1), [.025, .975]).tolist(),
            'tasks': len(values)}


def verify_effect(observed, expected, label):
    close(observed['estimate'], expected['estimate'], label + ' estimate')
    close(observed['ci95'], expected['ci95'], label + ' CI')
    if observed['tasks'] != expected['tasks']:
        raise ValueError(label + ': task denominator differs')


def pair_auc(frame):
    """Uniform average of positive-negative comparisons, with half credit for ties."""
    result, ties = {}, 0
    for task, part in frame.groupby('task_id', sort=True):
        pos = part.loc[part.passed == 1, 'probability'].to_numpy()
        neg = part.loc[part.passed == 0, 'probability'].to_numpy()
        if not len(pos) or not len(neg):
            continue
        equal = pos[:, None] == neg[None, :]
        result[task] = float(((pos[:, None] > neg[None, :]) + .5 * equal).mean())
        ties += int(equal.sum())
    return result, ties


def ratio_interval(top, bottom):
    top, bottom = np.asarray(top, float), np.asarray(bottom, float)
    index = np.random.default_rng(SEED).integers(0, len(top), (10000, len(top)))
    den = bottom[index].sum(axis=1)
    valid = den > 0
    return {'estimate': float(top.sum() / bottom.sum()) if bottom.sum() else None,
            'ci95': np.quantile(top[index].sum(axis=1)[valid] / den[valid], [.025, .975]).tolist() if valid.any() else None,
            'undefined_draws': int((~valid).sum())}


def candidates():
    """Use original local records, or the documented compact export in a clone."""
    target = OUT / 'reproduction/external_candidates.csv'
    catalog = OUT / 'reproduction/candidate_record_hashes.json'
    if (ROOT / 'runs/final_extension/v1/deepseek/external/records').exists():
        rows, hashes = [], {}
        for model in ('deepseek', 'qwen'):
            directory = ROOT / f'runs/final_extension/v1/{model}/external/records'
            for path in sorted(directory.glob('*.json')):
                record = json.loads(path.read_text(encoding='utf-8'))
                hashes[path.relative_to(ROOT).as_posix()] = sha(path)
                rows.append({'model': model, 'task_id': record['task_id'], 'attempt_id': record['attempt_id'],
                             'generated_tokens': record['generated_tokens'], 'passed': int(record['passed'])})
        bank = pd.DataFrame(rows).sort_values(['model', 'task_id', 'attempt_id'])
        target.parent.mkdir(parents=True, exist_ok=True)
        bank.to_csv(target, index=False)
        catalog.write_text(json.dumps({'scope': 'Hashes of unchanged original records; compact CSV omits code and activations',
                                       'files': hashes, 'export_sha256': sha(target)}, indent=2) + '\n', encoding='utf-8')
        return bank, 'Original per-candidate JSON read; compact metadata exported'
    metadata = json.loads(catalog.read_text(encoding='utf-8'))
    if sha(target) != metadata['export_sha256']:
        raise ValueError('Compact candidate export checksum differs')
    return pd.read_csv(target), 'Compact metadata CSV; original-record checksums retained, originals not bundled'


def replay(bank, predictions, design):
    thresholds = {model: {family: read(f'reports/final_extension/{model}_probe_manifest.json')[family]['threshold']
                          for family in FAMILIES} for model in ('deepseek', 'qwen')}
    scores = {(r.model, r.family, r.task_id, r.attempt_id): r.probability for r in predictions.itertuples()}
    rows = []
    for model in ('deepseek', 'qwen'):
        grouped = {task: part.sort_values('attempt_id') for task, part in bank[bank.model == model].groupby('task_id')}
        for budget, policy, task in itertools.product((1, 5, 10, 20), ('fixed', 'verifier', 'oracle', *FAMILIES), design['external_ids']):
            stream = grouped[task].iloc[:budget]
            active_success = stream[(stream.passed == 1) & (stream.generated_tokens > 32)]
            retained = len(active_success) if policy not in FAMILIES else sum(
                scores[model, policy, task, r.attempt_id] >= thresholds[model][policy] for r in active_success.itertuples())
            row = {'model': model, 'policy': policy, 'budget': budget, 'task_id': task,
                   'tokens': 0, 'resolved': 0, 'started': 0, 'correct_started': 0,
                   'active_correct_started': 0, 'false_aborts': 0, 'aborts_started': 0,
                   'active_correct_in_budget': len(active_success), 'retained_correct_in_budget': retained}
            for r in stream.itertuples():
                active = r.generated_tokens > 32
                abort = active and ((policy == 'oracle' and r.passed == 0) or
                                     (policy in FAMILIES and scores[model, policy, task, r.attempt_id] < thresholds[model][policy]))
                row['tokens'] += 32 if abort else r.generated_tokens
                row['started'] += 1
                row['correct_started'] += int(r.passed == 1)
                row['active_correct_started'] += int(active and r.passed == 1)
                row['false_aborts'] += int(abort and r.passed == 1)
                row['aborts_started'] += int(abort)
                row['resolved'] = int(row['resolved'] or (r.passed == 1 and not abort))
                if row['resolved'] and policy != 'fixed':
                    break
            rows.append(row)
    frame = pd.DataFrame(rows).sort_values(['model', 'budget', 'policy', 'task_id']).reset_index(drop=True)
    saved = pd.read_csv(ROOT / 'reports/final_extension/replay_per_task.csv').sort_values(
        ['model', 'budget', 'policy', 'task_id']).reset_index(drop=True)
    pd.testing.assert_frame_equal(frame[saved.columns], saved, check_dtype=False)
    frame.to_csv(OUT / 'reproduction/replay_recomputed.csv', index=False)
    return frame


def verify_power():
    from slm_steering.final_extension.power import simulate
    historical = pd.read_csv(ROOT / 'runs/probes/protocol_v0.2/outer_predictions.csv')
    historical = historical[historical.checkpoint.astype(str) == '32']
    effects = {}
    for family in ('B', 'HB'):
        effects[family] = pair_auc(historical.rename(columns={family: 'probability'}))[0]
    values = np.array([effects['HB'][t] - effects['B'][t] for t in sorted(effects['B'])])
    expected = read('reports/final_extension/power_results.json')
    rho = len(values) / historical.task_id.nunique()
    close(values.std(ddof=1), expected['paired_delta_sd'], 'Historical effect SD')
    close(rho, expected['mixed_fraction'], 'Historical mixed fraction')
    saved = pd.read_csv(ROOT / 'reports/final_extension/power_simulation.csv')
    sizes = sorted(saved.tasks.unique())
    frames = []
    for mixed_fraction in (rho, .4, .55, .85):
        part = simulate(values, mixed_fraction, sizes)
        part['mixed_fraction_assumed'] = mixed_fraction
        frames.append(part)
    reproduced = pd.concat(frames, ignore_index=True)
    pd.testing.assert_frame_equal(reproduced[saved.columns], saved, check_exact=False, rtol=0, atol=1e-11)
    # Closed-form sample size and an independent reimplementation of each RNG cell.
    analytic = {str(effect): int(np.ceil(((norm.ppf(.975) + norm.ppf(.8)) * values.std(ddof=1) / effect)**2 / rho))
                for effect in (.02, .03, .05, .075)}
    if analytic != expected['normal_approx_tasks_for_80pct_power']:
        raise ValueError('Analytic planning size differs')
    rng = np.random.default_rng(SEED)
    independent = []
    residual = values - values.mean()
    for effect in (0., .02, .03, .05, .075):
        scale = min(1., (1-effect)/max(residual.max(), 1e-12), (1+effect)/max(-residual.min(), 1e-12))
        population = effect + scale * residual
        for n in sizes:
            mixed = rng.binomial(n, rho, size=4000)
            draw = population[rng.integers(len(values), size=(4000, n))]
            observed = np.array([row[:m].mean() if m else 0 for row, m in zip(draw, mixed)])
            variances = np.array([row[:m].var(ddof=1) if m > 1 else 0 for row, m in zip(draw, mixed)])
            se = np.sqrt(variances / np.maximum(mixed, 1))
            for sd in (0., .01, .02):
                estimate = observed + rng.normal(0, sd, 4000)
                half = norm.ppf(.975) * np.sqrt(se**2 + sd**2)
                independent.append({'effect': effect, 'tasks': n, 'training_sd_assumed': sd,
                                    'positive_detection': float(np.mean((mixed >= 3) & (estimate > half))),
                                    'mean_interval_width': float((2 * half[mixed >= 3]).mean())})
    independent = pd.DataFrame(independent)
    reference = reproduced[reproduced.mixed_fraction_assumed == rho].reset_index(drop=True)
    for column in ('positive_detection', 'mean_interval_width'):
        close(independent[column], reference[column], 'Independent simulation ' + column)
    reproduced.to_csv(OUT / 'reproduction/power_recomputed.csv', index=False)
    null = reference[(reference.effect == 0) & (reference.training_sd_assumed == 0) & (reference.tasks == 100)].iloc[0]
    return {'cells_reproduced': len(reproduced), 'independent_reference_cells': len(independent),
            'analytic_tasks_for_80pct': analytic, 'planning_rows': expected['planning_rows'],
            'null_rejection_at_100': float(null.power_two_sided),
            'max_monte_carlo_standard_error': float(np.sqrt(.25 / 4000))}


def verify_bellman():
    from slm_steering.final_extension.decision import finite_budget_policy, simplified_continue
    cases = 0
    for p, c, a, penalty, k, b in itertools.product((0., .3, 1.), (1, 3), (1, 2), (0, 5), (0, 1, 2), (0, 2, 5)):
        possible = [(a+c) * sum((1-p)**j for j in range(n)) + penalty * (1-p)**n
                    for n in range(min(k, b // (a+c)) + 1)]
        close(finite_budget_policy(p, c, a, penalty, k, b)['value'], min(possible), 'Synthetic Bellman')
        cases += 1
    for p, v, c in itertools.product((0., .2, 1.), (0., 3., 10.), (0., 1., 5.)):
        if simplified_continue(p, v, c) != (c+(1-p)*v <= v+1e-12):
            raise ValueError('Simplified threshold differs from expected cost comparison')
    return {'finite_cases': cases, 'threshold_cases': 27,
            'scope': 'Known iid probability, integer costs; not an empirical optimal-policy claim'}


def verify_historical():
    original = read('reports/protocol_v0.2/results.json')['signal']['H2']
    scores = pd.read_csv(ROOT / 'runs/probes/protocol_v0.2/outer_predictions.csv')
    scores = scores[scores.checkpoint.astype(str) == '32']
    auc = {family: pair_auc(scores.rename(columns={family: 'probability'}))[0] for family in ('B', 'HB')}
    delta = [auc['HB'][t] - auc['B'][t] for t in sorted(auc['B'])]
    result = {'original_auc': paired_interval(delta, replications=2000, seed=20260921)}
    verify_effect(result['original_auc'], original['delta_hb_minus_b'], 'Original H2 AUC')
    y = scores.passed.to_numpy()
    losses = {}
    for family in ('B', 'HB'):
        p = np.clip(scores[family].to_numpy(), 1e-12, 1-1e-12)
        losses[family] = -y*np.log(p)-(1-y)*np.log1p(-p)
    per_task = pd.DataFrame({'task_id': scores.task_id, 'delta': losses['HB']-losses['B']}).groupby('task_id').delta.mean()
    result['original_log_loss'] = paired_interval(per_task, replications=2000, seed=20260921)
    verify_effect(result['original_log_loss'], original['delta_log_loss_hb_minus_b'], 'Original H2 log loss')
    independent = read('reports/excellence/independent_v1/results.json')
    counts = pd.read_csv(ROOT / 'reports/excellence/independent_v1/retention_per_task.csv')
    array = counts[['retained_HB', 'retained_B', 'active_correct']].to_numpy()
    if not np.array_equal(array, independent['per_task_counts']):
        raise ValueError('Independent gate counts differ')
    indices = np.random.default_rng(20261017).integers(0, len(array), (10000, len(array)))
    sums = array[indices].sum(axis=1)
    valid = sums[:, 2] > 0
    result['independent_retention'] = {'estimate': float((array[:,0].sum()-array[:,1].sum())/array[:,2].sum()),
        'ci95': np.quantile((sums[valid,0]-sums[valid,1])/sums[valid,2], [.025,.975]).tolist(),
        'tasks': len(array), 'undefined_draws': int((~valid).sum())}
    verify_effect(result['independent_retention'], independent['primary'], 'Independent gate retention')
    if result['independent_retention']['undefined_draws'] != independent['primary']['undefined_draws']:
        raise ValueError('Independent retention undefined resamples differ')
    return result


def write_report(result):
    def number(value, digits=4):
        return 'indefinido' if value is None else f'{value:.{digits}f}'.replace('-', '−').replace('.', ',')
    def interval(row):
        return '[' + '; '.join(number(v) for v in row['ci95']) + ']'
    lines = ['# Auditoria estatística final', '',
             f"Execução: {result['audited_at_utc']}. Duração: {result['elapsed_seconds']:.1f} segundos.",
             'Reconstrução em CPU, sem carregar geradores, ajustar probes ou executar código gerado.', '',
             '## Insumos e integridade', '',
             '`external_predictions.csv`, `replay_per_task.csv`, manifests dos probes, desenho externo,',
             'simulação prospectiva e metadados de candidatos já existentes. Os hashes estão em',
             '`statistical_verification.json`. Os registros originais permanecem intactos.',
             'O CSV compacto `reproduction/external_candidates.csv` conserva somente modelo, tarefa,',
             'tentativa, comprimento e label; permite a mesma reconstrução em uma cópia pública.', '',
             f"Foram conferidos {result['protected_files_verified']} arquivos científicos e históricos protegidos.", '',
             '## AUC, pareamento e população', '',
             'A implementação independente conta pares positivo-negativo dentro de cada tarefa,',
             'atribui meio ponto aos empates e exclui homogêneas somente da WT-AUC. A média dá o',
             'mesmo peso a cada tarefa mista. Cada família possui exatamente os mesmos candidatos',
             'e labels de sua contraparte; foram verificados IDs, duplicações e elegibilidade T>32.', '',
             '| Modelo | Geradas | Ativas | Tarefas ativas | Mistas | Sem tentativa ativa |',
             '|---|---:|---:|---:|---:|---:|']
    for model, row in result['populations'].items():
        lines.append(f"| {model} | {row['generated']} | {row['active']} | {row['active_tasks']} | {row['mixed_tasks']} | {row['zero_active_tasks']} |")
    historical = result['historical']
    lines += ['', 'Também foram reconstruídos os dois contrastes H2 originais (2.000 réplicas,',
              'seed 20260921) e a diferença de retenção independente (10.000 réplicas, seed',
              '20261017), respeitando a ordem e os denominadores dos registros históricos:', '',
              f"- Qwen original: ΔWT-AUC {number(historical['original_auc']['estimate'])}, IC95% {interval(historical['original_auc'])}.",
              f"- Qwen original: Δlog loss {number(historical['original_log_loss']['estimate'])}, IC95% {interval(historical['original_log_loss'])}.",
              f"- Gates anteriores: Δretenção {number(historical['independent_retention']['estimate'])}, IC95% {interval(historical['independent_retention'])}, 120 tarefas.", '',
              'Os scores apresentam 28 pares positivo-negativo empatados por família no DeepSeek',
              'e 13 no Qwen; a convenção foi também verificada em um exemplo com AUC exatamente 0,5.', '',
              '| Modelo | Classe | ΔWT-AUC H+B−B | IC95% | Estatuto |', '|---|---|---:|---|---|']
    for model in ('deepseek', 'qwen'):
        for family in ('linear', 'MLP'):
            row = result['contrasts'][model][family]
            status = 'Primário' if (model, family) == ('deepseek', 'MLP') else 'Secundário'
            lines.append(f"| {model} | {family} | {number(row['estimate'])} | {interval(row)} | {status} |")
    lines += ['', 'Os quatro pontos e IC reproduzem os resultados salvos com tolerância absoluta',
              'de 1e−11. Foram reamostradas tarefas inteiras, com 10.000 réplicas, seed 20261104',
              'e percentis 2,5% e 97,5%, após formar as diferenças pareadas. Não se reamostram',
              'tentativas ou pares como unidades independentes.', '', '## Interação Γ', '']
    for model in ('deepseek', 'qwen'):
        row = result['contrasts'][model]['Gamma']
        lines.append(f"- {model}: Γ = {number(row['estimate'])}, IC95% {interval(row)}, {row['tasks']} tarefas mistas.")
    lines += ['', 'Cada tarefa contribui com (AUC_HB_MLP−AUC_B_MLP)−(AUC_HB_linear−AUC_B_linear).',
              'O IC reamostra esse vetor conjunto; não subtrai limites de dois IC individuais.',
              'No Qwen, a interação positiva reduz uma desvantagem linear; o incremento absoluto',
              'MLP permanece negativo e inconclusivo. Γ é secundário, com IC pontual sem ajuste',
              'de multiplicidade. A comparação entre modelos foi igualmente reproduzida nas',
              '28 tarefas mistas comuns, preservando o cluster compartilhado.', '',
              '## Probabilidades e calibração', '',
              'Foram reproduzidas as oito linhas de AUC agregada, PR-AUC, WT-AUC, log loss e Brier,',
              'os contrastes pareados de perdas e os 80 bins descritivos de calibração. As perdas',
              'usam 100 tarefas ativas no DeepSeek e 81 no Qwen, incluindo homogêneas, com peso',
              'total igual por tarefa. O DeepSeek favorece H+B nas perdas secundárias, enquanto',
              'Qwen/MLP o desfavorece; isso não substitui o endpoint primário de ranking.', '',
              '## Poder prospectivo', '']
    if result['power'] is None:
        lines += ['A reprodução do poder foi explicitamente omitida por `--skip-power`.', '']
    else:
        power = result['power']
        lines += ['As 540 células salvas foram reproduzidas; 135 do cenário histórico também foram',
                  'conferidas por uma implementação independente do cálculo. A simulação usa 4.000',
                  'bancos por célula, seed 20261104, efeitos históricos centrados e deslocados,',
                  'contagem mista binomial e IC normais de planejamento. Não usa o efeito externo',
                  'observado nem substitui o bootstrap percentil final.', '',
                  '| Efeito fixado | Detecção positiva, 100 tarefas | Largura média do IC |', '|---:|---:|---:|']
        for row in power['planning_rows']:
            lines.append(f"| {number(row['effect'], 3)} | {number(row['positive_detection']*100, 2)}% | {number(row['mean_interval_width'], 3)} |")
        lines += ['', f"No cenário nulo, a rejeição bilateral em 100 tarefas foi {number(power['null_rejection_at_100']*100, 3)}%.",
                  'O erro Monte Carlo máximo de um desvio-padrão é aproximadamente 0,79 ponto',
                  'percentual. A aproximação para 80% de poder exige 1.998/888/320/143 tarefas',
                  'para efeitos 0,02/0,03/0,05/0,075. Esses números dependem da heterogeneidade',
                  'Qwen/HumanEval e da fração mista 92/132; não garantem transporte ao DeepSeek.',
                  'Os cenários de SD de treinamento são sensibilidades, não correções formais',
                  'dos IC condicionais. Pouco poder não demonstra equivalência nem inexistência',
                  'de sinal; o IC primário ainda admite ganho próximo de 0,05 e perda próxima de 0,06.', '']
    lines += ['## Replay operacional', '',
              'Foram reconstruídas independentemente todas as 5.600 linhas tarefa/política/orçamento',
              'e as 56 sínteses, incluindo IC pareados de custo/cobertura e IC das razões de',
              'retenção e tokens por tarefa resolvida. A ordem é attempt_id, a condição é T>32,',
              'e o aborto ocorre se score < limiar congelado. N=1,5,10,20 usa a mesma regra.', '',
              'Resumo em N=20; custo agregado por resolvida inclui os tokens das não resolvidas.', '',
              '| Modelo | Política | Resolvidas | Tokens/tarefa | Tokens/resolvida | Retenção ativa | Abortos/iniciadas | Corretos abortados | Perdidas |',
              '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in result['operational']:
        if row['budget'] == 20:
            retention = number(row['retention_all_active_correct']*100, 2) + '%' if row['retention_all_active_correct'] is not None else 'indefinido'
            lines.append(f"| {row['model']} | {row['policy']} | {row['resolved_tasks']} | {number(row['mean_completion_tokens'], 2)} | {number(row['aggregate_tokens_per_resolved_task'], 2)} | {retention} | {row['aborts_started']}/{row['attempts_started']} | {row['false_aborts_started']} | {row['lost_tasks']} |")
    lines += ['', 'Retenção abrange todos os corretos ativos potenciais no orçamento; falsos abortos',
              'contam apenas candidatos iniciados. Não são complementares. Alguns gates DeepSeek',
              'não interrompem nenhuma tentativa iniciada; custos iguais ao verificador representam',
              'ausência de economia adicional. A economia de H+B/MLP nesse modelo é pequena.',
              'Nenhuma tarefa perdida no banco produz IC empírico degenerado de cobertura; isso',
              'não certifica ausência de perdas raras futuras. Tokens excluem prompt, captura,',
              'score, verificação, energia e preço. Não se reproduziu aceleração online.', '',
              '## Validação matemática e limites da reprodução', '',
              f"Bellman conferido em {result['bellman']['finite_cases']} casos finitos e 27 comparações de limiar.",
              'A enumeração usa p conhecido, custos inteiros e tentativas iid. Ela não demonstra',
              'otimalidade dos gates empíricos. Geração, treinamento, seleção e labels funcionais',
              'não foram reexecutados; o escopo desta auditoria é reconstruir os cálculos a partir',
              'das evidências existentes, preservando seu histórico. A incerteza completa do',
              'treinamento e a ausência de contaminação no pré-treinamento permanecem abertas.', '',
              'Reprodução: `python scripts/finalization_audit.py`.', '']
    (OUT / 'STATISTICAL_AUDIT.md').write_text('\n'.join(lines), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--skip-power', action='store_true', help='Reuse no power result; explicitly skip reproduction')
    args = parser.parse_args()
    started = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'reproduction').mkdir(exist_ok=True)
    from slm_steering.final_extension.common import validate_freeze
    expected_freeze = (ROOT / 'reports/final_extension/freeze.sha256').read_text(encoding='utf-8').strip()
    if sha(ROOT / 'reports/final_extension/freeze.json') != expected_freeze:
        raise ValueError('Freeze manifest checksum differs')
    validate_freeze()
    protected = {**read('reports/final_extension/freeze.json')['files'],
                 **read('reports/final_extension/historical_hashes.json')['files']}
    inputs = ['reports/final_extension/' + name for name in ('external_predictions.csv', 'results.json',
              'replay_per_task.csv', 'calibration.csv', 'final_design.json', 'power_results.json', 'power_simulation.csv',
              'deepseek_probe_manifest.json', 'qwen_probe_manifest.json')]
    input_hashes = {p: sha(ROOT / p) for p in inputs}
    historical = verify_historical()
    final = read('reports/final_extension/results.json')
    design = read('reports/final_extension/final_design.json')
    predictions = pd.read_csv(ROOT / 'reports/final_extension/external_predictions.csv')
    if predictions.duplicated(['model', 'family', 'task_id', 'attempt_id']).any():
        raise ValueError('Duplicate scored candidate')
    if not np.isfinite(predictions.probability).all() or not predictions.probability.between(0, 1).all():
        raise ValueError('Invalid probability')
    bank, bank_scope = candidates()
    if bank.duplicated(['model', 'task_id', 'attempt_id']).any():
        raise ValueError('Duplicate candidate metadata')
    actual_metrics = {(r['model'], r['family']): r for r in final['metrics']}
    all_auc, contrasts, populations, tie_counts = {}, {}, {}, {}
    loss_deltas, per_task_rows = {}, []
    calibration = []
    for model in ('deepseek', 'qwen'):
        part = bank[bank.model == model]
        expected_ids = {(task, j) for task in design['external_ids'] for j in range(20)}
        if set(zip(part.task_id, part.attempt_id)) != expected_ids or len(part) != 2000:
            raise ValueError('Incomplete final cohort')
        active = part[part.generated_tokens > 32].sort_values(['task_id', 'attempt_id'])
        all_auc[model], tie_counts[model], loss_deltas[model] = {}, {}, {}
        for family in FAMILIES:
            scored = predictions[(predictions.model == model) & (predictions.family == family)].sort_values(['task_id', 'attempt_id'])
            pd.testing.assert_frame_equal(scored[['task_id', 'attempt_id', 'passed']].reset_index(drop=True),
                                          active[['task_id', 'attempt_id', 'passed']].reset_index(drop=True), check_dtype=False)
            aucs, ties = pair_auc(scored)
            all_auc[model][family] = aucs
            tie_counts[model][family] = ties
            y, p = scored.passed.to_numpy(), scored.probability.to_numpy()
            clipped = np.clip(p, 1e-12, 1-1e-12)
            n_tasks = scored.task_id.nunique()
            weight = 1 / (n_tasks * scored.groupby('task_id').task_id.transform('size').to_numpy())
            loss = -y*np.log(clipped) - (1-y)*np.log1p(-clipped)
            computed = {'within_task_auc': np.mean(list(aucs.values())), 'log_loss': weight @ loss,
                        'brier': weight @ ((p-y)**2), 'roc_auc': roc_auc_score(y, p, sample_weight=weight),
                        'pr_auc': average_precision_score(y, p, sample_weight=weight)}
            for key, value in computed.items():
                close(value, actual_metrics[model, family][key], f'{model}/{family}/{key}')
            loss_deltas[model][family] = pd.DataFrame({'task_id': scored.task_id, 'log_loss': loss, 'brier': (p-y)**2}).groupby('task_id').mean()
            bins = np.minimum((p*10).astype(int), 9)
            for bin_id in range(10):
                mask = bins == bin_id
                calibration.append({'model': model, 'family': family, 'bin': bin_id, 'count': int(mask.sum()),
                                    'mean_probability': float(np.average(p[mask], weights=weight[mask])) if mask.any() else np.nan,
                                    'fraction_correct': float(np.average(y[mask], weights=weight[mask])) if mask.any() else np.nan})
        tasks = sorted(all_auc[model]['B_MLP'])
        if any(sorted(v) != tasks for v in all_auc[model].values()):
            raise ValueError('Probe families do not share mixed tasks')
        contrasts[model] = {}
        for family in ('linear', 'MLP'):
            delta = [all_auc[model][f'HB_{family}'][t] - all_auc[model][f'B_{family}'][t] for t in tasks]
            contrasts[model][family] = paired_interval(delta)
            verify_effect(contrasts[model][family], final['contrasts'][model][family], f'{model}/{family}')
            differences = loss_deltas[model][f'HB_{family}'] - loss_deltas[model][f'B_{family}']
            for metric in ('log_loss', 'brier'):
                effect = paired_interval(differences[metric])
                verify_effect(effect, final['contrasts'][model][family]['delta_' + metric], f'{model}/{family}/{metric}')
                contrasts[model][family]['delta_' + metric] = effect
        gamma = [(all_auc[model]['HB_MLP'][t] - all_auc[model]['B_MLP'][t]) -
                 (all_auc[model]['HB_linear'][t] - all_auc[model]['B_linear'][t]) for t in tasks]
        contrasts[model]['Gamma'] = paired_interval(gamma)
        verify_effect(contrasts[model]['Gamma'], final['contrasts'][model]['Gamma'], model + ' Gamma')
        for task in tasks:
            per_task_rows.append({'model': model, 'task_id': task, **{f: all_auc[model][f][task] for f in FAMILIES}})
        populations[model] = {'generated': len(part), 'active': len(active), 'active_tasks': active.task_id.nunique(),
                              'mixed_tasks': len(tasks), 'zero_active_tasks': 100-active.task_id.nunique()}
    pd.DataFrame(per_task_rows).to_csv(OUT / 'reproduction/task_auc_recomputed.csv', index=False)
    saved_calibration = pd.read_csv(ROOT / 'reports/final_extension/calibration.csv').sort_values(['model', 'family', 'bin']).reset_index(drop=True)
    computed_calibration = pd.DataFrame(calibration).sort_values(['model', 'family', 'bin']).reset_index(drop=True)
    pd.testing.assert_frame_equal(computed_calibration, saved_calibration, check_exact=False, rtol=0, atol=1e-11)
    common = sorted(set(all_auc['deepseek']['B_MLP']) & set(all_auc['qwen']['B_MLP']))
    common_effect = paired_interval([(all_auc['deepseek']['HB_MLP'][t] - all_auc['deepseek']['B_MLP'][t]) -
                                    (all_auc['qwen']['HB_MLP'][t] - all_auc['qwen']['B_MLP'][t]) for t in common])
    verify_effect(common_effect, final['secondary_cross_model_common_mixed'], 'Common-mixed contrast')
    # Tie and homogeneous conventions checked independently of the real score values.
    example = pd.DataFrame({'task_id': ['mixed', 'mixed', 'homogeneous'], 'passed': [1, 0, 1], 'probability': [.5, .5, .9]})
    if pair_auc(example) != ({'mixed': .5}, 1):
        raise ValueError('Tie/homogeneous convention failed')
    print('Predictive contrasts, losses, Gamma and task denominators reproduced', flush=True)
    rebuilt = replay(bank, predictions, design)
    summaries = []
    for saved in final['operational']:
        model, budget, policy = (saved[k] for k in ('model', 'budget', 'policy'))
        part = rebuilt[(rebuilt.model == model) & (rebuilt.budget == budget) & (rebuilt.policy == policy)].sort_values('task_id')
        base = rebuilt[(rebuilt.model == model) & (rebuilt.budget == budget) & (rebuilt.policy == 'verifier')].sort_values('task_id')
        observed = {'model': model, 'budget': budget, 'policy': policy,
                    'resolved_tasks': int(part.resolved.sum()), 'mean_completion_tokens': float(part.tokens.mean()),
                    'aggregate_tokens_per_resolved_task': float(part.tokens.sum() / part.resolved.sum()) if part.resolved.sum() else None,
                    'false_aborts_started': int(part.false_aborts.sum()),
                    'aborts_started': int(part.aborts_started.sum()),
                    'attempts_started': int(part.started.sum()),
                    'retention_all_active_correct': float(part.retained_correct_in_budget.sum() / part.active_correct_in_budget.sum()) if part.active_correct_in_budget.sum() else None,
                    'lost_tasks': int(((base.resolved.to_numpy() == 1) & (part.resolved.to_numpy() == 0)).sum())}
        for key in observed:
            if key in saved and key not in ('model', 'policy'):
                close(observed[key], saved[key], f'Replay {model}/{budget}/{policy}/{key}')
        for column, result_name in [('tokens', 'delta_mean_tokens_vs_verifier'), ('resolved', 'delta_coverage_vs_verifier')]:
            verify_effect(paired_interval(part[column].to_numpy() - base[column].to_numpy()), saved[result_name], 'Replay ' + result_name)
        for top, bottom, key in [('retained_correct_in_budget', 'active_correct_in_budget', 'retention_interval'),
                                 ('tokens', 'resolved', 'tokens_per_resolved_interval')]:
            interval = ratio_interval(part[top], part[bottom])
            close(interval['estimate'], saved[key]['estimate'], key)
            close(interval['ci95'], saved[key]['ci95'], key + ' CI')
            if interval['undefined_draws'] != saved[key]['undefined_draws']:
                raise ValueError(key + ' undefined resamples differ')
        summaries.append(observed)
    print('All 5,600 task replay rows and 56 operational summaries reproduced', flush=True)
    power = None if args.skip_power else verify_power()
    bellman = verify_bellman()
    for path, expected in {**protected, **input_hashes}.items():
        if sha(ROOT / path) != expected:
            raise ValueError('Input changed during read-only audit: ' + path)
    result = {'audited_at_utc': datetime.now(timezone.utc).isoformat(), 'elapsed_seconds': time.monotonic()-started,
              'audit_script_sha256': sha(Path(__file__)),
              'execution': 'CPU; no model loading, downloads, inference, training or label reexecution',
              'bank_scope': bank_scope, 'input_hashes': input_hashes, 'protected_files_verified': len(protected), 'historical': historical,
              'populations': populations, 'contrasts': contrasts, 'common_mixed_contrast': common_effect,
              'tied_positive_negative_pairs': tie_counts, 'operational': summaries,
              'replay_task_rows': len(rebuilt), 'power': power, 'bellman': bellman,
              'unreproduced': ['Probe training, generation and functional labels not rerun by design',
                               'Full-procedure training uncertainty not estimated', 'Online latency not established']}
    (OUT / 'statistical_verification.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    write_report(result)
    print(f'Finalization audit passed in {result["elapsed_seconds"]:.1f} seconds', flush=True)


if __name__ == '__main__':
    main()
