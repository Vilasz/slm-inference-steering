"""Prospective precision planning using task clusters, with explicit approximations."""
from __future__ import annotations

import json
import numpy as np
import pandas as pd
from scipy.stats import norm

from slm_steering.final_extension.common import ROOT, REPORT, atomic_json, read_json
from slm_steering.research.statistics import task_auc


def simulate(deltas, mixed_fraction, sizes, *, simulations=4000, seed=20261104, training_sds=(0., .01, .02)):
    deltas = np.asarray(deltas, dtype=float)
    residual = deltas - deltas.mean()
    rng = np.random.default_rng(seed)
    rows = []
    for effect in (0., .02, .03, .05, .075):
        # Affine centering fixes the population effect exactly; shrink only if
        # needed to keep hypothetical paired AUC differences within [-1,1].
        scale = min(1., (1 - effect) / max(float(residual.max()), 1e-12),
                    (1 + effect) / max(float(-residual.min()), 1e-12))
        population = effect + scale * residual
        assert np.isclose(population.mean(), effect)
        for n in sizes:
            mixed = rng.binomial(n, mixed_fraction, size=simulations)
            draws = population[rng.integers(0, len(population), (simulations, n))]
            mask = np.arange(n)[None, :] < mixed[:, None]
            mean = (draws * mask).sum(1) / np.maximum(mixed, 1)
            variance = (((draws - mean[:, None]) ** 2) * mask).sum(1) / np.maximum(mixed - 1, 1)
            se = np.sqrt(variance / np.maximum(mixed, 1))
            valid = mixed >= 3
            for training_sd in training_sds:
                # A shared training error does not disappear as external n grows.
                # These are sensitivity assumptions, not estimated uncertainty.
                estimate = mean + rng.normal(0., training_sd, simulations)
                total_se = np.sqrt(se**2 + training_sd**2)
                half = norm.ppf(.975) * total_se
                rows.append({'effect': effect, 'tasks': n, 'training_sd_assumed': training_sd,
                             'power_two_sided': float(np.mean(valid & ((estimate - half > 0) | (estimate + half < 0)))),
                             'positive_detection': float(np.mean(valid & (estimate - half > 0))),
                             'mean_interval_width': float((2 * half[valid]).mean()),
                             'mean_mixed_tasks': float(mixed.mean()),
                             'mixed_q025': float(np.quantile(mixed, .025)), 'mixed_q975': float(np.quantile(mixed, .975)),
                             'valid_fraction': float(valid.mean()), 'residual_scale': scale})
    return pd.DataFrame(rows)


def run(external_tasks):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    p = pd.read_csv(ROOT / 'runs/probes/protocol_v0.2/outer_predictions.csv')
    p = p[p.checkpoint.astype(str) == '32']
    b, hb = (task_auc(p.passed, p[c], p.task_id) for c in ('B', 'HB'))
    deltas = [hb[t] - b[t] for t in b]
    per_task = p.groupby('task_id').agg(attempts=('passed', 'size'), successes=('passed', 'sum'))
    per_task['mixed'] = (per_task.successes > 0) & (per_task.successes < per_task.attempts)
    per_task['success_rate_active'] = per_task.successes / per_task.attempts
    per_task['positive_negative_pairs'] = per_task.successes * (per_task.attempts - per_task.successes)
    for task, group in p.groupby('task_id'):
        per_task.loc[task, 'outcome_variance_active'] = group.passed.var(ddof=1)
        per_task.loc[task, 'B_score_sd'] = group.B.std(ddof=1)
        per_task.loc[task, 'HB_score_sd'] = group.HB.std(ddof=1)
        per_task.loc[task, 'within_task_predictor_correlation'] = group.B.corr(group.HB) if group.B.nunique() > 1 and group.HB.nunique() > 1 else np.nan
        per_task.loc[task, 'B_auc'] = b.get(task, np.nan)
        per_task.loc[task, 'HB_auc'] = hb.get(task, np.nan)
    per_task['delta'] = [hb[t] - b[t] if t in b else np.nan for t in per_task.index]
    per_task.to_csv(REPORT / 'historical_task_structure.csv')
    full = []
    with (ROOT / 'runs/candidate_bank/definitive_v0.2/trajectories.jsonl').open(encoding='utf-8') as stream:
        for line in stream:
            r = json.loads(line)
            full.append({'task_id': r['task_id'], 'passed': int(r['passed']), 'active': r['generated_tokens'] > 32})
    pd.DataFrame(full).groupby('task_id').agg(generations=('passed', 'size'), successes=('passed', 'sum'),
                                            active_t32=('active', 'sum')).to_csv(REPORT / 'historical_full_task_counts.csv')
    sizes = sorted({50, 80, 100, 120, 140, 150, 200, 300, 500, external_tasks})
    frames = []
    for rho in (float(per_task.mixed.mean()), .4, .55, .85):
        f = simulate(deltas, rho, sizes)
        f['mixed_fraction_assumed'] = rho
        frames.append(f)
    frame = pd.concat(frames, ignore_index=True)
    training = None
    sensitivity_path = REPORT / 'qwen_training_sensitivity.json'
    if sensitivity_path.exists():
        training = read_json(sensitivity_path)
        empirical = []
        for family, sd in training['training_sensitivity_sd'].items():
            f = simulate(deltas, float(per_task.mixed.mean()), sizes, training_sds=(sd,))
            f['family'] = family
            f['sd_scope'] = 'Measured source task-subsampling sensitivity; not a full-procedure variance estimate'
            empirical.append(f)
        pd.concat(empirical).to_csv(REPORT / 'power_training_sensitivity.csv', index=False)
    frame.to_csv(REPORT / 'power_simulation.csv', index=False)
    reference = frame[(frame.training_sd_assumed == 0) & np.isclose(frame.mixed_fraction_assumed, per_task.mixed.mean())]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for effect, g in reference[reference.effect > 0].groupby('effect'):
        axes[0].plot(g.tasks, g.positive_detection, marker='o', label=f'+{effect:.3f}')
    axes[0].axhline(.8, color='gray', ls='--')
    axes[0].set(xlabel='Tarefas independentes', ylabel='Probabilidade de detectar ganho', ylim=(0, 1))
    axes[0].legend(title='Δ WT-AUC')
    for rho in sorted(frame.mixed_fraction_assumed.unique()):
        g = frame[(frame.training_sd_assumed == 0) & (frame.effect == .05) & (frame.mixed_fraction_assumed == rho)]
        axes[1].plot(g.tasks, g.mean_interval_width, label=f'ρ={rho:.2f}')
        axes[2].plot(g.tasks, g.mean_mixed_tasks, label=f'ρ={rho:.2f}')
    axes[1].set(xlabel='Tarefas independentes', ylabel='Largura esperada IC 95%')
    axes[2].set(xlabel='Tarefas independentes', ylabel='Número esperado de tarefas mistas')
    axes[1].legend()
    fig.tight_layout()
    for suffix in ('pdf', 'png', 'svg'):
        fig.savefig(REPORT / f'power_precision.{suffix}', dpi=180)
    plt.close(fig)
    selected = reference[(reference.tasks == external_tasks) & (reference.effect > 0)]
    analytic_n = {str(effect): int(np.ceil(((norm.ppf(.975) + norm.ppf(.8)) * np.std(deltas, ddof=1) / effect)**2 / per_task.mixed.mean()))
                  for effect in (.02, .03, .05, .075)}
    result = {'historical_tasks': len(per_task), 'mixed_tasks': len(deltas), 'mixed_fraction': float(per_task.mixed.mean()),
              'paired_delta_sd': float(np.std(deltas, ddof=1)), 'predictor_correlation': float(p.B.corr(p.HB)),
              'recommendation_tasks': external_tasks, 'planning_rows': selected.to_dict('records'),
              'normal_approx_tasks_for_80pct_power': analytic_n,
              'repeated_source_fitting': training,
              'simulations_per_cell': 4000, 'seed': 20261104,
              'approximation': 'Empirical centered task effects with binomial mixed counts; normal two-sided planning intervals, not the final percentile bootstrap.',
              'training_uncertainty': 'SD 0/.01/.02 are sensitivity assumptions. No claim of full-procedure frequentist coverage.',
              'transport_limit': 'Historical Qwen HumanEval heterogeneity is a planning proxy, not a measured DeepSeek MBPP effect.'}
    atomic_json(REPORT / 'power_results.json', result)
    table = '\n'.join(f"| +{r.effect:.3f} | {r.positive_detection:.1%} | {r.mean_interval_width:.3f} | {r.mean_mixed_tasks:.1f} |" for r in selected.itertuples())
    report = f'''# Poder e precisão prospectivos

A unidade independente é a tarefa. O histórico t=32 contém {len(per_task)} tarefas,
{len(deltas)} mistas ({per_task.mixed.mean():.1%}); o desvio-padrão das diferenças
pareadas é {np.std(deltas, ddof=1):.4f}. A correlação dos scores B/H+B é {p.B.corr(p.HB):.4f}.
As distribuições de acertos, tentativas ativas e diferenças estão em
`historical_task_structure.csv`, incluindo variância dos labels/scores,
correlação intratarefa e números de pares positivo-negativo. O arquivo
`historical_full_task_counts.csv` distingue vinte gerações totais de tentativas
ativas em 32 tokens. Tarefas homogêneas não recebem AUC artificial de 0,5.

## Simulação e recomendação

Recomendam-se **{external_tasks} tarefas externas**, vinte tentativas por tarefa e modelo,
condicionalmente à viabilidade medida no piloto. Cada célula simula 4.000 bancos
de tarefas com seed 20261104. Os efeitos populacionais fixados são 0, +0,02,
+0,03, +0,05 e +0,075. As diferenças históricas são centradas, deslocadas para
o efeito conhecido e, apenas quando necessário, contraídas para permanecer em [-1,1].
Não se usa poder pós-hoc do efeito observado como validação do histórico.

| Efeito verdadeiro | Detecção positiva | Largura IC95% | Mistas esperadas |
|---|---:|---:|---:|
{table}

![Curvas de planejamento](power_precision.png)

Os intervalos da simulação usam aproximação normal studentizada por tarefa;
os intervalos finais usarão bootstrap pareado de tarefas com 10.000 réplicas.
A célula de efeito zero permite verificar o erro de tipo I aproximado no CSV.
As curvas cobrem 50–500 tarefas e frações mistas 0,40, 0,55, 0,85 e a histórica.
A disponibilidade local e o orçamento limitam a recomendação; 500 é cenário de
sensibilidade, não autorização para ampliar o banco buscando significância.
Uma aproximação normal para poder de 80% requer cerca de {analytic_n['0.02']}
tarefas para +0,02, {analytic_n['0.03']} para +0,03, {analytic_n['0.05']} para +0,05
e {analytic_n['0.075']} para +0,075, sob a fração mista e dispersão históricas.
Os primeiros cenários ultrapassam o inventário MBPP+ inteiro. Com 4.000 simulações,
o erro Monte Carlo da estimativa de poder é no máximo aproximadamente 0,8 ponto
percentual de desvio-padrão; a incerteza nas hipóteses de transporte é maior.

## Limites

Ganhos de +0,02 e +0,03 poderão continuar difíceis de distinguir. A amostra não
promete equivalência nem conclusividade para efeitos pequenos. Vinte tentativas
melhoram a observação intratarefa; não equivalem a vinte tarefas independentes.
O modelo de planejamento preserva a dispersão empírica dos efeitos pareados,
não toda a distribuição conjunta dos tokens e do ajuste dos probes.

O cenário principal é condicional a preditores fixos. As sensibilidades com
desvio-padrão de treinamento 0,01 e 0,02 acrescentam um erro comum por banco;
esses números são hipóteses, não variabilidade estimada por reajustes. O CSV
separa explicitamente esses cenários. Uma análise que não repete o ajuste e a
seleção não caracteriza integralmente a incerteza do procedimento. O transporte
de Qwen/HumanEval para DeepSeek/MBPP permanece uma aproximação de planejamento.
'''
    if training:
        report += f'''
## Repetição efetiva do procedimento na origem

Além das sensibilidades hipotéticas, foram executados **{training['repetitions']}
reajustes completos** das quatro famílias Qwen, incluindo seleção de
hiperparâmetros, PCA, early stopping, três seeds, refit e calibração.
Reservaram-se {len(training['audit_task_ids'])} tarefas exclusivamente para
auditar essa estabilidade. Cada repetição amostrou sem reposição
{training['tasks_per_perturbation']} das {training['training_pool_tasks']} tarefas
restantes. Nenhum rótulo MBPP novo foi lido. A subamostragem evita duplicar uma
mesma tarefa entre folds, mas não é bootstrap de treinamento.

O desvio-padrão das diferenças de WT-AUC entre repetições foi
{training['training_sensitivity_sd']['linear']:.4f} para a classe linear e
{training['training_sensitivity_sd']['MLP']:.4f} para MLP, no mesmo conjunto de
auditoria de origem. Valores e frequências de seleção estão em
`qwen_training_sensitivity.csv` e `.json`; curvas ilustrativas com esses desvios
estão em `power_training_sensitivity.csv`.

Oito subamostras sobrepostas, um conjunto de auditoria pequeno e um tamanho de
treino reduzido não estimam sem viés a variância do procedimento final. As curvas
adicionais são sensibilidades empíricas, não correção formal dos IC externos nem
evidência de superioridade de uma família. O IC principal continua condicional
aos probes congelados. A estabilidade entre as três seeds é publicada
separadamente nas losses dos folds e nenhum membro é selecionado por desempenho.
'''
    (REPORT / 'power_analysis.md').write_text(report, encoding='utf-8')
    return result
