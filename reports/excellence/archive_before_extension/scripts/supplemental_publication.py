"""Render explicitly exploratory findings in the scientific manuscript."""
from __future__ import annotations


def render_section(extra: dict, number, interval, table) -> str:
    def effect(row):
        return f"{number(row['estimate'], 4)} {interval(row['ci95'])}" if row else 'não identificável'

    text = '''
== Diagnósticos exploratórios posteriores ao resultado principal

As análises desta seção foram acrescentadas após a conclusão e inspeção dos
resultados do desenho fixo. Mantêm probabilidades, calibradores e políticas
congelados, sem retreino, ajuste no alvo ou mudança do critério de decisão.
Usam 2.000 reamostragens de tarefas com seed 20260921. Os intervalos adicionais
são pontuais: não têm cobertura simultânea nem status confirmatório.

=== Precisão, estabilidade e composição da população

Os contrastes e intervalos principais foram reproduzidos numericamente a
partir das previsões OOF. O intervalo quantifica os tamanhos de efeito
compatíveis sob as suposições utilizadas; incluir zero não demonstra
equivalência ou ausência universal de sinal @greenland2016.

'''
    text += table(['Métrica H+B menos B', 'Estimativa principal', 'Faixa ao excluir uma tarefa'],
                  [[r['metric'].replace('within_task_auc', 'WT-AUC'), number(r['overall_delta'], 4),
                    interval(r['leave_one_out_range'])] for r in extra['influence_summary']],
                  'tab-supp-influence', 'Influência da exclusão de uma tarefa por vez; os modelos permanecem fixos. A faixa não é um intervalo de confiança.')
    text += '''
A estabilidade do sinal da estimativa sob exclusão não estabelece significância
em cada exclusão. Os cinco folds têm conjuntos de treinamento sobrepostos e
não são replicações independentes.

#figure(image("../../reports/final_review/supplemental/figures/fold_stability.svg", width: 100%),
 caption: [Variação do contraste H+B menos B entre folds externos, em 32 tokens. Intervalos exploratórios por tarefa.]) <fig-supp-folds>

Uma coorte comum retém as mesmas identidades de tentativa nos checkpoints
8, 16, 32 e 64. Ela condiciona em T > 64: separa descritivamente a mudança de
população, mas não identifica efeito causal do tempo. A sensibilidade ao
histórico exclui da avaliação as tarefas geradas em artefatos anteriores.
Essa exclusão não refaz o treinamento nem remove a exposição histórica da
pesquisa; não constitui nova amostra confirmatória.

'''
    rows = []
    for dataset in extra['correct_retention']:
        selected = [r for r in extra['contrasts'] if r['dataset'] == dataset and r['cohort'] == 'common_T_gt_64'
                    and r['checkpoint'] == '32' and r['comparison'] == 'HB minus B']
        by_metric = {r['metric']: r['effect'] for r in selected}
        if selected:
            rows.append([dataset + ', T > 64', selected[0]['tasks'], effect(by_metric['within_task_auc']), effect(by_metric['log_loss'])])
    exposure = extra['historical_exposure_sensitivity']
    by_metric = {r['metric']: r['effect'] for r in exposure['contrasts'] if r['comparison'] == 'HB minus B'}
    rows.append(['HumanEval+, sem histórico', exposure['tasks'], effect(by_metric['within_task_auc']), effect(by_metric['log_loss'])])
    text += table(['População em 32 tokens', 'Tarefas', 'ΔWT-AUC e IC 95%', 'Δlog loss e IC 95%'], rows,
                  'tab-supp-cohorts', 'Análises de sensibilidade com escores fixos. WT-AUC utiliza somente tarefas mistas em cada coorte.')
    text += '''
#figure(image("../../reports/final_review/supplemental/figures/common_cohort.svg", width: 100%),
 caption: [Contrastes na população ativa original e na coorte comum T > 64. Os demais checkpoints são exploratórios.]) <fig-supp-common>

=== Calibração e transporte do limiar

A calibração é descrita em dez bins fixos de largura 0,1 da probabilidade
de sucesso, com peso total igual por tarefa. O ECE agrega os desvios absolutos
entre probabilidade e frequência observada; a curva de confiabilidade
complementa a discriminação @guo2017. ECE depende da discretização e do tamanho
amostral e não substitui log loss ou Brier. Seu bootstrap percentil pode ter
viés positivo e não conter o ECE empírico. Bins pouco ocupados são instáveis;
contagens e massas ponderadas são exportadas junto às figuras.

'''
    text += table(['Avaliação', 'Preditor', 'ECE', 'RMSCE'],
                  [[r['dataset'], r['predictor'].replace('HB', 'H+B'), number(r['ece'], 4), number(r['rmsce'], 4)]
                   for r in extra['calibration']], 'tab-supp-calibration',
                  'Calibração da probabilidade da classe positiva em 32 tokens; os intervalos exploratórios estão nos artefatos complementares.')
    text += '''
#figure(image("../../reports/final_review/supplemental/figures/calibration_transfer.svg", width: 100%),
 caption: [Confiabilidade em origem e alvo. Os escores são OOF em HumanEval+ e de um modelo ajustado em toda a origem para MBPP+; a diferença entre painéis não isola mudança de domínio.]) <fig-supp-calibration>

A retenção abaixo considera todos os candidatos corretos ativos no checkpoint
selecionado, inclusive candidatos nunca iniciados no replay. Ela tem o mesmo
denominador conceitual da restrição de desenvolvimento. Difere da taxa de
aborto falso operacional, que considera apenas tentativas iniciadas.
Cada política pode selecionar um checkpoint diferente; a comparação de
retenções entre políticas não usa necessariamente a mesma população.

'''
    text += table(['Avaliação', 'Política', 'Retidos/ativos corretos', 'Retenção e IC 95%'],
                  [[dataset, r['policy'].replace('HB', 'H+B'), f"{r['retained_correct']}/{r['active_correct']}",
                    effect({'estimate': r['recall'], 'ci95': r['ci95']}) if r['ci95'] else 'não identificável']
                   for dataset, item in extra['correct_retention'].items() for r in item['aggregate']],
                  'tab-supp-retention', 'Retenção de candidatos corretos sob gates congelados. Bootstrap por tarefa da razão de contagens somadas; alvo de desenvolvimento 0,97.')
    text += '''
#figure(image("../../reports/final_review/supplemental/figures/correct_retention.svg", width: 100%),
 caption: [Alvo de seleção, retenção de desenvolvimento e retenção externa nos gates congelados. O alvo empírico não é garantia de generalização.]) <fig-supp-retention>

=== Condições necessárias para a conclusão operacional

'''
    translations = {'Primary WT-AUC gain': 'Ganho primário de WT-AUC',
                    'Primary log-loss reduction': 'Redução primária de log loss',
                    'H2 joint criterion': 'Critério conjunto Q2',
                    'Online identity replay with actual abort': 'Replay online com aborto efetivo',
                    'Joint operational benefit': 'Benefício operacional conjunto'}
    rows = []
    for r in extra['decision_ledger']:
        label = translations.get(r['criterion'], r['criterion'].replace('resolution_delta', 'Δresolução').replace('token_delta', 'Δtokens').replace('minus', 'menos'))
        rule = (('LI > ' if r.get('direction') in ('positive', 'noninferiority') else 'LS < ') + number(r['boundary'])) if 'boundary' in r else 'Regra conjunta'
        rows.append([label, effect(r) if r.get('ci95') else '—', rule, 'sim' if r['satisfied'] else 'não'])
    text += table(['Condição', 'Estimativa e IC 95%', 'Regra', 'Atendida'], rows,
                  'tab-supp-decision', 'Aplicação dos critérios originais em N = 20. LI e LS são os limites inferior e superior do intervalo; nenhuma margem foi escolhida após observar os resultados.')
    return text


def discussion(extra: dict, number, interval) -> str:
    wt = extra['primary_reproduction']['within_task_auc']['effect']
    loss = extra['primary_reproduction']['log_loss']['effect']
    text = '\n== Síntese dos diagnósticos exploratórios\n\n'
    if wt is not None and loss is not None:
        text += (f"O intervalo principal de ΔWT-AUC vai de {number(wt['ci95'][0], 4)} a {number(wt['ci95'][1], 4)}. "
                 f"O Δlog loss é {number(loss['estimate'], 4)}, IC 95% {interval(loss['ci95'])}. "
                 'Esses limites descrevem a precisão alcançada sob modelos fixos; não são uma prova de equivalência nem de ausência geral de sinal.\n\n')
    if 'MBPP+ frozen' in extra['correct_retention']:
        row = next(r for r in extra['correct_retention']['MBPP+ frozen']['aggregate'] if r['policy'] == 'HB')
        text += (f"Em MBPP+, o gate H+B retém {row['retained_correct']} dos {row['active_correct']} candidatos corretos ativos "
                 f"({number(row['recall'], 4)}, IC 95% {interval(row['ci95'])}). "
                 'O alvo empírico de 0,97 no desenvolvimento não constitui garantia externa. '
                 'A retenção observada descreve o transporte do gate congelado, sem identificar isoladamente as causas de diferenças entre origem e alvo.\n\n')
    return text


def conclusion(extra: dict, number, interval) -> str:
    text = '\nOs diagnósticos exploratórios posteriores examinam a estabilidade entre tarefas, '
    text += 'a composição temporal e a calibração sem alterar as decisões principais. '
    if 'MBPP+ frozen' in extra['correct_retention']:
        row = next(r for r in extra['correct_retention']['MBPP+ frozen']['aggregate'] if r['policy'] == 'HB')
        text += (f"A retenção de candidatos corretos ativos de H+B em MBPP+ foi {number(row['recall'], 4)} "
                 f"(IC 95% {interval(row['ci95'])}), frente ao alvo empírico de 0,97 no desenvolvimento. ")
    text += 'Esses resultados delimitam a transportabilidade do sistema avaliado; extensões devem ser tratadas como novos estudos.\n'
    return text
