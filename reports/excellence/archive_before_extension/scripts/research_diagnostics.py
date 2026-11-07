"""Post-hoc diagnostics of frozen predictions; never fit or select a policy here."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from slm_steering.research.protocol import file_hash, now, write_json
from slm_steering.research.statistics import paired_task_bootstrap

FAMILIES = ('B', 'H', 'HB')
IDENTITY = ['task_id', 'attempt_id']
SEED = 20260921
REPLICATIONS = 2000


def validate_predictions(frame: pd.DataFrame) -> pd.DataFrame:
    required = {*IDENTITY, 'checkpoint', 'passed', *FAMILIES}
    if not required.issubset(frame.columns) or frame.empty:
        raise ValueError('Predictions require nonempty identified attempts, labels, checkpoints and B/H/HB probabilities')
    frame = frame.copy()
    if frame[list(required)].isna().any().any():
        raise ValueError('Missing prediction identity, label or probability')
    frame['checkpoint'] = frame.checkpoint.astype(str)
    if frame.duplicated([*IDENTITY, 'checkpoint']).any():
        raise ValueError('Duplicate attempt/checkpoint identity')
    if not frame.passed.isin([0, 1]).all():
        raise ValueError('Labels must be binary')
    probabilities = frame[list(FAMILIES)].to_numpy(dtype=float)
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError('Probabilities must be finite and in [0, 1]')
    for column in ['passed', 'fold', 'generated_tokens']:
        if column in frame and frame.groupby(IDENTITY)[column].nunique().gt(1).any():
            raise ValueError(f'Attempt {column} changes between checkpoints')
    if 'fold' in frame and frame.groupby('task_id').fold.nunique().gt(1).any():
        raise ValueError('A task appears in multiple outer folds')
    if 'generated_tokens' in frame:
        for checkpoint in ('8', '16', '32', '64'):
            if frame.loc[frame.checkpoint == checkpoint, 'generated_tokens'].le(int(checkpoint)).any():
                raise ValueError('Prediction violates strict T > t eligibility')
    return frame.sort_values(['checkpoint', *IDENTITY]).reset_index(drop=True)


def task_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    """Exactly one equally weighted contribution per task and predictor."""
    rows = []
    eps = np.finfo(np.float64).eps  # sklearn 1.5 log_loss convention
    for task, group in frame.groupby('task_id', sort=True):
        y = group.passed.to_numpy(dtype=float)
        for family in FAMILIES:
            raw = group[family].to_numpy(dtype=float)
            p = np.clip(raw, eps, 1 - eps)
            rows.append({'task_id': task, 'predictor': family, 'attempts': len(group),
                         'positives': int(y.sum()), 'fold': int(group.fold.iloc[0]) if 'fold' in group else -1,
                         'log_loss': float(-(y * np.log(p) + (1 - y) * np.log1p(-p)).mean()),
                         'brier': float(np.mean((raw - y) ** 2)),
                         'within_task_auc': float(roc_auc_score(y, raw)) if len(np.unique(y)) == 2 else None})
    return pd.DataFrame(rows)


def contrasts(metrics: pd.DataFrame, *, replications=REPLICATIONS, seed=SEED) -> list[dict]:
    rows = []
    for family in ('H', 'HB'):
        for metric in ('within_task_auc', 'log_loss', 'brier'):
            wide = metrics.pivot(index='task_id', columns='predictor', values=metric)
            delta = (wide[family] - wide.B).dropna()
            rows.append({'comparison': f'{family} minus B', 'metric': metric,
                         'effect': paired_task_bootstrap(delta.to_numpy(), replications=replications, seed=seed)
                         if len(delta) else None})
    return rows


def common_cohort(frame: pd.DataFrame, checkpoints=('8', '16', '32', '64')) -> pd.DataFrame:
    """Intersect complete attempt identities, retaining scores from each original checkpoint."""
    selected = frame[frame.checkpoint.isin(checkpoints)]
    ids = selected.groupby(IDENTITY).checkpoint.nunique()
    keep = ids[ids == len(checkpoints)].index
    return selected.set_index(IDENTITY).loc[lambda x: x.index.isin(keep)].reset_index()


def calibration(frame: pd.DataFrame, family: str, *, bins=10, replications=REPLICATIONS, seed=SEED) -> dict:
    """Equal task weight; fixed-width positive-class probability bins (not top-label confidence)."""
    if bins < 1 or frame.empty:
        raise ValueError('Calibration requires observations and positive bin count')
    tasks, inverse, counts = np.unique(frame.task_id, return_inverse=True, return_counts=True)
    p, y = frame[family].to_numpy(), frame.passed.to_numpy()
    idx = np.minimum((p * bins).astype(int), bins - 1)
    mass = np.zeros((len(tasks), bins))
    outcomes, probabilities = mass.copy(), mass.copy()
    weights = 1 / counts[inverse]
    for array, value in ((mass, weights), (outcomes, weights * y), (probabilities, weights * p)):
        np.add.at(array, (inverse, idx), value)
    mass_mean, y_mean, p_mean = (a.mean(axis=0) for a in (mass, outcomes, probabilities))
    occupied = mass_mean > 0
    gaps = np.divide(y_mean - p_mean, mass_mean, out=np.zeros(bins), where=occupied)
    ece = float(np.abs(y_mean - p_mean).sum())
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(tasks), size=(replications, len(tasks)))
    differences = outcomes - probabilities
    boot = np.abs(differences[draws].mean(axis=1)).sum(axis=1)
    reliability = []
    for b in range(bins):
        reliability.append({'bin': b, 'lower': b / bins, 'upper': (b + 1) / bins,
                            'attempts': int((idx == b).sum()), 'task_weight_mass': float(mass_mean[b]),
                            'mean_probability': float(p_mean[b] / mass_mean[b]) if occupied[b] else None,
                            'observed_rate': float(y_mean[b] / mass_mean[b]) if occupied[b] else None})
    return {'predictor': family, 'tasks': len(tasks), 'attempts': len(frame), 'bins': bins,
            'ece': ece, 'ece_ci95': np.quantile(boot, [.025, .975]).tolist(),
            'rmsce': float(np.sqrt(np.sum(mass_mean * gaps ** 2))),
            'maximum_gap': float(np.max(np.abs(gaps[occupied]))), 'reliability': reliability}


def recall_evaluation(frame: pd.DataFrame, policies: list[dict], *, replications=REPLICATIONS, seed=SEED):
    """All active correct candidates at the frozen gate, including candidates never started in replay."""
    counts = []
    fold_rows = []
    for policy in policies:
        family = policy['policy']
        group = frame[frame.checkpoint == str(policy['checkpoint'])]
        if 'fold' in policy:
            group = group[group.fold == policy['fold']]
        if group.empty:
            raise ValueError('Frozen policy has no matching evaluation predictions')
        threshold = float(policy['threshold'])
        if not np.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError('Invalid frozen policy threshold')
        positive = group[group.passed == 1]
        kept = positive[family].ge(threshold)
        record = {'policy': family, 'fold': policy.get('fold', 'external'), 'checkpoint': policy['checkpoint'],
                  'threshold': threshold, 'development_recall': policy['validation_correct_recall'],
                  'active_correct': len(positive), 'retained_correct': int(kept.sum()),
                  'held_out_recall': float(kept.mean()) if len(positive) else None}
        fold_rows.append(record)
        for task, task_group in group.groupby('task_id', sort=True):
            correct = task_group.passed.eq(1)
            counts.append({'policy': family, 'task_id': task, 'positive': int(correct.sum()),
                           'retained': int((correct & task_group[family].ge(threshold)).sum())})
    count_frame = pd.DataFrame(counts)
    if count_frame.duplicated(['policy', 'task_id']).any():
        raise ValueError('More than one selected gate per policy/task')
    aggregate = []
    for family, group in count_frame.groupby('policy', sort=True):
        values = group.sort_values('task_id')[['retained', 'positive']].to_numpy()
        denominator = values[:, 1].sum()
        rng = np.random.default_rng(seed)
        draws = values[rng.integers(0, len(values), (replications, len(values)))].sum(axis=1)
        defined = draws[:, 1] > 0
        boot = draws[defined, 0] / draws[defined, 1]
        aggregate.append({'policy': family, 'tasks': len(values), 'active_correct': int(denominator),
                          'retained_correct': int(values[:, 0].sum()),
                          'recall': float(values[:, 0].sum() / denominator) if denominator else None,
                          'ci95': np.quantile(boot, [.025, .975]).tolist() if len(boot) else None,
                          'undefined_bootstrap_draws': int((~defined).sum())})
    return {'by_fold': fold_rows, 'aggregate': aggregate, 'target': .97,
            'denominator': 'All active correct candidates at the frozen policy checkpoint; no conditioning on replay initiation',
            'estimand': 'Candidate-level recall (ratio of summed counts); tasks are resampled as clusters'}


def influence(metrics: pd.DataFrame) -> list[dict]:
    rows = []
    for metric in ('within_task_auc', 'log_loss', 'brier'):
        wide = metrics.pivot(index='task_id', columns='predictor', values=metric)
        delta = (wide.HB - wide.B).dropna()
        if len(delta) < 2:
            continue
        mean = delta.mean()
        leave_one_out = (delta.sum() - delta) / (len(delta) - 1)
        for task, value in delta.items():
            rows.append({'task_id': task, 'metric': metric, 'task_delta': float(value),
                         'overall_delta': float(mean), 'leave_one_task_out_delta': float(leave_one_out[task]),
                         'change_from_overall': float(leave_one_out[task] - mean),
                         'models_refitted': False})
    return rows


def decision_ledger(main: dict, online: dict) -> list[dict]:
    h2 = main['signal']['H2']
    rows = []
    for key, label, direction in [('delta_hb_minus_b', 'Primary WT-AUC gain', 'positive'),
                                   ('delta_log_loss_hb_minus_b', 'Primary log-loss reduction', 'negative')]:
        effect = h2[key]
        passed = effect is not None and (effect['ci95'][0] > 0 if direction == 'positive' else effect['ci95'][1] < 0)
        rows.append({'criterion': label, 'estimate': effect['estimate'] if effect else None,
                     'ci95': effect['ci95'] if effect else None, 'boundary': 0, 'direction': direction,
                     'satisfied': bool(passed)})
    for comparison in main['policies']['comparisons']:
        if comparison['budget'] != 20 or comparison['comparison'] not in ('HB minus verifier', 'HB minus B'):
            continue
        for metric, boundary in [('resolution_delta', -.03), ('token_delta', 0)]:
            effect = comparison[metric]
            positive = metric == 'resolution_delta'
            rows.append({'criterion': f"N=20 {comparison['comparison']}: {metric}",
                         'estimate': effect['estimate'], 'ci95': effect['ci95'], 'boundary': boundary,
                         'direction': 'noninferiority' if positive else 'negative',
                         'satisfied': bool(effect['ci95'][0] > boundary if positive else effect['ci95'][1] < 0)})
    rows.extend([{'criterion': 'H2 joint criterion', 'satisfied': h2['supported']},
                 {'criterion': 'Online identity replay with actual abort',
                  'satisfied': bool(online.get('passed') and online.get('has_actual_abort'))},
                 {'criterion': 'Joint operational benefit',
                  'satisfied': main['policies']['H3']['operational_benefit_claim_supported']}])
    return rows


def build(root: Path, output: Path, evidence: dict, *, replications=REPLICATIONS, seed=SEED) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    paths = ['runs/probes/protocol_v0.2/outer_predictions.csv',
             'runs/probes/protocol_v0.2/selection_audit.json', 'reports/protocol_v0.2/results.json',
             'reports/research/legacy_task_presence.json']
    has_transfer = 'transfer' in evidence
    if has_transfer:
        paths += ['reports/transfer_v0.2/predictions.csv', 'runs/transfer/protocol_v0.2/models.json',
                  'reports/transfer_v0.2/results.json']
    paths += ['scripts/research_diagnostics.py', 'scripts/analyze_research_robustness.py']
    before = {p: file_hash(root / p) for p in paths}
    source = validate_predictions(pd.read_csv(root / paths[0]))
    datasets = {'HumanEval+ OOF': source}
    if has_transfer:
        datasets['MBPP+ frozen'] = validate_predictions(pd.read_csv(root / 'reports/transfer_v0.2/predictions.csv'))
    all_contrasts, calibrations, all_task_metrics, folds = [], [], [], []
    primary_metrics = task_metrics(source[source.checkpoint == '32'])
    exposure = json.loads((root / paths[3]).read_text(encoding='utf-8'))
    unexposed = source[(source.checkpoint == '32') & ~source.task_id.isin(exposure['reserved_overlap_ids'])]
    exposure_sensitivity = {'excluded_task_ids': exposure['reserved_overlap_ids'],
                            'attempts': len(unexposed), 'tasks': unexposed.task_id.nunique(),
                            'contrasts': contrasts(task_metrics(unexposed), replications=replications, seed=seed)}
    for dataset, frame in datasets.items():
        common = common_cohort(frame)
        for cohort, selected in [('original_risk_set', frame), ('common_T_gt_64', common)]:
            for checkpoint, group in selected.groupby('checkpoint', sort=True):
                metrics = task_metrics(group)
                for row in contrasts(metrics, replications=replications, seed=seed):
                    all_contrasts.append({'dataset': dataset, 'cohort': cohort, 'checkpoint': checkpoint,
                                          'attempts': len(group), 'tasks': group.task_id.nunique(), **row})
                if cohort == 'original_risk_set':
                    metrics['dataset'], metrics['checkpoint'] = dataset, checkpoint
                    all_task_metrics.append(metrics)
        for family in FAMILIES:
            calibrations.append({'dataset': dataset, 'checkpoint': '32',
                                 **calibration(frame[frame.checkpoint == '32'], family,
                                               replications=replications, seed=seed)})
    for fold, group in source[source.checkpoint == '32'].groupby('fold', sort=True):
        for row in contrasts(task_metrics(group), replications=replications, seed=seed):
            folds.append({'fold': int(fold), 'attempts': len(group), 'tasks': group.task_id.nunique(), **row})
    primary = {r['metric']: r['effect'] for r in all_contrasts if r['dataset'] == 'HumanEval+ OOF'
               and r['cohort'] == 'original_risk_set' and r['checkpoint'] == '32' and r['comparison'] == 'HB minus B'}
    h2 = evidence['main']['signal']['H2']
    reproduction = {}
    for metric, key in [('within_task_auc', 'delta_hb_minus_b'), ('log_loss', 'delta_log_loss_hb_minus_b')]:
        actual, expected = primary[metric], h2[key]
        matches = actual is None and expected is None
        if actual is not None and expected is not None:
            matches = (np.isclose(actual['estimate'], expected['estimate'], atol=1e-12, rtol=0)
                       and (replications != REPLICATIONS or seed != SEED
                            or np.allclose(actual['ci95'], expected['ci95'], atol=1e-12, rtol=0)))
        if not matches:
            raise ValueError(f'Supplemental analysis does not reproduce the frozen primary contrast: {metric}')
        reproduction[metric] = {'matches': bool(matches), 'effect': actual}
    selected = [r for r in json.loads((root / paths[1]).read_text(encoding='utf-8')) if 'policy' in r]
    retention = {'HumanEval+ OOF': recall_evaluation(source, selected, replications=replications, seed=seed)}
    if has_transfer:
        bundle = json.loads((root / 'runs/transfer/protocol_v0.2/models.json').read_text(encoding='utf-8'))
        policies = [dict(p, policy=family) for family, p in bundle['policies'].items()]
        retention['MBPP+ frozen'] = recall_evaluation(datasets['MBPP+ frozen'], policies,
                                                     replications=replications, seed=seed)
    influences = influence(primary_metrics)
    influence_summary = []
    for metric, group in pd.DataFrame(influences).groupby('metric'):
        v = group.leave_one_task_out_delta
        influence_summary.append({'metric': metric, 'overall_delta': float(group.overall_delta.iloc[0]),
                                  'leave_one_out_range': [float(v.min()), float(v.max())],
                                  'sign_preserved_all_deletions': bool(((v > 0) == (group.overall_delta.iloc[0] > 0)).all()),
                                  'largest_absolute_influence_task': group.loc[group.change_from_overall.abs().idxmax(), 'task_id']})
    result = {'status': 'post_hoc_exploratory_frozen_predictions', 'created_at': now(),
              'timing': 'Requested after completion and inspection of the fixed-design results',
              'settings': {'seed': seed, 'task_bootstrap_replications': replications, 'calibration_equal_width_bins': 10,
                           'primary_checkpoint': '32', 'common_cohort_checkpoints': ['8', '16', '32', '64']},
              'limitations': ['Conditional on fixed fitted scores; models are not refitted during bootstrap or deletions',
                             'Pointwise exploratory intervals; no simultaneous coverage or new confirmatory claims',
                             'Outer folds share training data and are not independent replications',
                             'Common cohort conditions on trajectory length T>64 and is not a causal time comparison',
                             'Calibration error depends on binning and sample size; use with proper losses',
                             'Percentile ECE bootstrap can be biased upward and need not contain the empirical ECE',
                             'Excluding historically generated tasks only changes evaluation; training is not redone',
                             'OOF source and full-source-fitted target scorers use different evaluation procedures',
                             'Development recall 0.97 is an empirical selection target, not a held-out guarantee'],
              'inputs': before, 'primary_reproduction': reproduction, 'contrasts': all_contrasts,
              'fold_contrasts': folds, 'influence_summary': influence_summary,
              'historical_exposure_sensitivity': exposure_sensitivity,
              'calibration': calibrations, 'correct_retention': retention,
              'transfer_policy_diagnostics': {
                  key: [r for r in evidence['transfer']['policies'][key] if r['budget'] == 20]
                  for key in ('summaries', 'comparisons')} if has_transfer else None,
              'decision_ledger': decision_ledger(evidence['main'], evidence.get('online', {}))}
    pd.concat(all_task_metrics, ignore_index=True).to_csv(output / 'task_metrics.csv', index=False)
    pd.DataFrame(influences).to_csv(output / 'task_influence.csv', index=False)
    pd.json_normalize(all_contrasts).to_csv(output / 'paired_contrasts.csv', index=False)
    pd.json_normalize(folds).to_csv(output / 'fold_contrasts.csv', index=False)
    pd.DataFrame([dict(row, dataset=d) for d, item in retention.items() for row in item['by_fold']]).to_csv(output / 'correct_retention.csv', index=False)
    pd.DataFrame([dict(row, dataset=c['dataset'], predictor=c['predictor']) for c in calibrations for row in c['reliability']]).to_csv(output / 'calibration_bins.csv', index=False)
    plot(output, result)
    write_summary(output, result)
    write_json(output / 'results.json', result)
    if any(file_hash(root / p) != expected for p, expected in before.items()):
        raise ValueError('An immutable scientific input changed during supplemental diagnostics')
    result['artifacts'] = {p.relative_to(root).as_posix(): file_hash(p) for p in sorted(output.rglob('*')) if p.is_file()}
    return result


def write_summary(output: Path, result: dict):
    """A directly readable companion to the machine-readable evidence."""
    def effect(e):
        return f"{e['estimate']:.5f} [{e['ci95'][0]:.5f}; {e['ci95'][1]:.5f}]" if e else 'não identificável'

    lines = ['# Diagnósticos complementares', '',
             'Análises exploratórias definidas após a conclusão e inspeção dos resultados principais.',
             'Nenhum modelo, calibrador, limiar, rótulo, semente ou critério principal foi alterado.', '',
             '## Precisão e estabilidade', '',
             '| Contraste principal H+B menos B | Estimativa [IC 95%] |', '|---|---|']
    for metric, row in result['primary_reproduction'].items():
        lines.append(f"| {metric} | {effect(row['effect'])} |")
    lines += ['', 'Os valores e intervalos principais foram reproduzidos a partir das probabilidades congeladas.',
              'Os limites acima quantificam a precisão sob as suposições utilizadas. Zero no intervalo não demonstra equivalência;',
              'log loss positivo representa maior perda de H+B. Nenhum resultado prova ausência universal de sinal.', '',
              '| Métrica | Faixa da média ao excluir uma tarefa por vez |', '|---|---|']
    for row in result['influence_summary']:
        low, high = row['leave_one_out_range']
        lines.append(f"| {row['metric']} | [{low:.5f}; {high:.5f}] |")
    lines += ['', 'A exclusão mantém os modelos fixos. A estabilidade do sinal da estimativa não garante significância em cada exclusão.', '',
              '## População constante e histórico de exposição', '',
              '| Coorte em 32 tokens | Tentativas | Tarefas | ΔWT-AUC [IC 95%] | Δlog loss [IC 95%] |', '|---|---:|---:|---|---|']
    for dataset in result['correct_retention']:
        rows = [r for r in result['contrasts'] if r['dataset'] == dataset and r['cohort'] == 'common_T_gt_64'
                and r['checkpoint'] == '32' and r['comparison'] == 'HB minus B']
        by_metric = {r['metric']: r['effect'] for r in rows}
        if rows:
            lines.append(f"| {dataset}, T>64 | {rows[0]['attempts']} | {rows[0]['tasks']} | {effect(by_metric['within_task_auc'])} | {effect(by_metric['log_loss'])} |")
    sensitivity = result['historical_exposure_sensitivity']
    by_metric = {r['metric']: r['effect'] for r in sensitivity['contrasts'] if r['comparison'] == 'HB minus B'}
    lines += [f"| HumanEval+, sem as {len(sensitivity['excluded_task_ids'])} tarefas históricas | {sensitivity['attempts']} | {sensitivity['tasks']} | {effect(by_metric['within_task_auc'])} | {effect(by_metric['log_loss'])} |", '',
              'A coorte T>64 condiciona no comprimento final. A exclusão histórica só altera a avaliação, sem retreino;',
              'nenhuma delas substitui o endpoint ou cria uma amostra confirmatória independente.', '',
              '## Retenção dos corretos', '',
              '| Avaliação | Política | Corretos retidos/ativos | Retenção [IC 95%] |', '|---|---|---:|---|']
    for dataset, item in result['correct_retention'].items():
        for row in item['aggregate']:
            value = effect({'estimate': row['recall'], 'ci95': row['ci95']}) if row['ci95'] else 'não identificável'
            lines.append(f"| {dataset} | {row['policy']} | {row['retained_correct']}/{row['active_correct']} | {value} |")
    lines += ['', 'O denominador inclui todos os candidatos corretos ativos no gate congelado, inclusive os nunca iniciados no replay.',
              'As políticas podem usar checkpoints diferentes; a tabela não é uma comparação pareada de retenção entre políticas.',
              'O alvo 0,97 se refere ao desenvolvimento. Os intervalos reamostram tarefas, usando a razão das contagens somadas.', '']
    if result['transfer_policy_diagnostics']:
        lines += ['## Consequência operacional no alvo (replay offline, N=20)', '',
                  '| Política | Resolução | Tokens por tarefa |', '|---|---:|---:|']
        for row in result['transfer_policy_diagnostics']['summaries']:
            if row['policy'] in ('B', 'HB', 'verifier'):
                lines.append(f"| {row['policy']} | {row['resolution']:.5f} | {row['completion_tokens_per_task']:.3f} |")
        lines += ['', '| Contraste | Δresolução [IC 95%] | Δtokens [IC 95%] |', '|---|---|---|']
        for row in result['transfer_policy_diagnostics']['comparisons']:
            lines.append(f"| {row['comparison']} | {effect(row['resolution_delta'])} | {effect(row['token_delta'])} |")
        lines += ['', 'Dados originais da transferência congelada, sem ajuste local e sem validação online MBPP+.',
                  'O sinal do contraste de tokens indica a direção do custo: valores positivos representam custo maior de H+B.', '']
    lines += [
              '## Calibração em 32 tokens', '',
              '| Avaliação | Preditor | ECE [percentis 2,5%; 97,5%] | RMSCE |', '|---|---|---|---:|']
    for row in result['calibration']:
        lines.append(f"| {row['dataset']} | {row['predictor']} | {effect({'estimate': row['ece'], 'ci95': row['ece_ci95']})} | {row['rmsce']:.5f} |")
    lines += ['', 'Dez bins fixos de probabilidade da classe positiva, cada tarefa com o mesmo peso total.',
              'ECE é diagnóstico dependente de discretização e tamanho amostral; não substitui log loss ou Brier.',
              'ECE menor isoladamente não implica melhor predição ou política operacional.',
              'O bootstrap percentil de ECE pode ter viés positivo e não conter a estimativa empírica.',
              'A origem usa escores OOF e o alvo um modelo ajustado em toda a origem; diferenças entre bancos não isolam um efeito causal de domínio.', '',
              '## Critérios de decisão preservados', '',
              '| Critério | Estimativa [IC 95%] | Condição | Atendido |', '|---|---|---|---|']
    for row in result['decision_ledger']:
        value = effect(row) if row.get('ci95') else '—'
        rule = ('limite inferior > ' if row.get('direction') in ('positive', 'noninferiority') else 'limite superior < ') + str(row['boundary']) if 'boundary' in row else 'regra conjunta'
        lines.append(f"| {row['criterion']} | {value} | {rule} | {'sim' if row['satisfied'] else 'não'} |")
    lines += ['', '## Reprodução e limites', '',
              '```powershell', '.venv/Scripts/python.exe scripts/analyze_research_robustness.py', '```', '',
              f"Bootstrap: {result['settings']['task_bootstrap_replications']} reamostragens de tarefas, seed {result['settings']['seed']}.",
              'Todos os intervalos adicionais são pontuais e exploratórios, condicionais aos modelos já ajustados.',
              'Folds compartilham dados de treinamento e não são replicações independentes.',
              'Identidades, hashes dos insumos e parâmetros estão em [results.json](results.json).',
              'Tabelas por tarefa, fold e bin estão nos CSV; figuras são exportadas em PNG, SVG e PDF.', '',
              'Referências metodológicas: [Guo et al. (2017)](https://proceedings.mlr.press/v70/guo17a.html),',
              '[Greenland et al. (2016)](https://doi.org/10.1007/s10654-016-0149-3).', '']
    (output / 'README.md').write_text('\n'.join(lines), encoding='utf-8')


def plot(output: Path, result: dict):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    colors = {'B': '#3274A1', 'H': '#E1812C', 'HB': '#3A923A'}
    directory = output / 'figures'
    directory.mkdir(exist_ok=True)

    def save(fig, name):
        fig.tight_layout()
        for extension in ('png', 'svg', 'pdf'):
            fig.savefig(directory / f'{name}.{extension}', dpi=180, bbox_inches='tight')
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, metric, title in zip(axes, ('within_task_auc', 'log_loss'), ('Δ WT-AUC (H+B − B)', 'Δ log loss (H+B − B)')):
        for cohort, color, offset, label in [('original_risk_set', '#3274A1', -.08, 'População ativa T > t'),
                                            ('common_T_gt_64', '#C44E52', .08, 'Mesmas tentativas T > 64')]:
            rows = [r for r in result['contrasts'] if r['dataset'] == 'HumanEval+ OOF' and r['cohort'] == cohort
                    and r['comparison'] == 'HB minus B' and r['metric'] == metric and r['checkpoint'] in ('8', '16', '32', '64') and r['effect']]
            rows.sort(key=lambda x: int(x['checkpoint']))
            y = np.array([r['effect']['estimate'] for r in rows])
            ci = np.array([r['effect']['ci95'] for r in rows]).T
            ax.errorbar(np.arange(len(rows)) + offset, y, yerr=np.maximum(np.array([y - ci[0], ci[1] - y]), 0),
                        marker='o', color=color, capsize=3, label=label)
        ax.axhline(0, color='gray', lw=.8)
        ax.set(xticks=range(4), xticklabels=['8', '16', '32', '64'], xlabel='Checkpoint (tokens)', title=title)
    axes[0].legend(fontsize=8)
    save(fig, 'common_cohort')

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, metric in zip(axes, ('within_task_auc', 'log_loss')):
        rows = [r for r in result['fold_contrasts'] if r['comparison'] == 'HB minus B' and r['metric'] == metric and r['effect']]
        for r in rows:
            effect = r['effect']
            ax.plot(effect['ci95'], [r['fold']] * 2, color='#3274A1')
            ax.scatter(effect['estimate'], r['fold'], color='#3274A1')
        ax.axvline(0, color='gray', lw=.8)
        ax.set(xlabel=f'Δ {metric} (H+B − B)', ylabel='Fold externo', yticks=range(5))
    save(fig, 'fold_stability')

    datasets = list(result['correct_retention'])
    fig, axes = plt.subplots(1, len(datasets), figsize=(6 * len(datasets), 4), squeeze=False)
    for ax, dataset in zip(axes[0], datasets):
        for row in result['calibration']:
            if row['dataset'] != dataset:
                continue
            bins = [r for r in row['reliability'] if r['attempts']]
            ax.plot([r['mean_probability'] for r in bins], [r['observed_rate'] for r in bins], 'o-',
                    color=colors[row['predictor']], label=f"{row['predictor']} ECE={row['ece']:.3f}")
        ax.plot([0, 1], [0, 1], '--', color='gray', lw=.8)
        ax.set(xlim=(0, 1), ylim=(0, 1), xlabel='Probabilidade média', ylabel='Frequência ponderada de sucesso', title=f'{dataset} — 32 tokens')
        ax.legend(fontsize=8)
    save(fig, 'calibration_transfer')

    fig, ax = plt.subplots(figsize=(10, 4))
    labels = []
    position = 0
    for dataset, item in result['correct_retention'].items():
        for row in item['by_fold']:
            if row['held_out_recall'] is None:
                continue
            ax.scatter(position, row['development_recall'], marker='x', color='gray', label='Desenvolvimento' if position == 0 else None)
            ax.scatter(position, row['held_out_recall'], color=colors[row['policy']], label='Avaliação externa' if position == 0 else None)
            labels.append(f"{row['policy']} f{row['fold']}" if row['fold'] != 'external' else f"MBPP+ {row['policy']}")
            position += 1
    ax.axhline(.97, color='black', linestyle='--', lw=.8, label='Alvo de seleção 0,97')
    ax.set(xticks=range(position), xticklabels=labels, ylim=(0, 1.02), ylabel='Retenção de candidatos corretos ativos')
    ax.tick_params(axis='x', rotation=35)
    ax.legend(fontsize=8, loc='lower left')
    save(fig, 'correct_retention')
