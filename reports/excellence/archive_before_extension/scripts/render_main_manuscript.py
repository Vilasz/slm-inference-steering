"""Render the predeclared experiment only from verified empirical artifacts."""
from __future__ import annotations

from pathlib import Path


def render(root: Path, evidence: dict, number, interval, table):
    main = evidence['main']
    signal, policies = main['signal'], main['policies']
    h2, h3 = signal['H2'], policies['H3']
    metrics = [r for r in signal['metrics'] if r['checkpoint'] == '32']
    by_name = {r['predictor']: r for r in metrics}
    primary = next(r for r in signal['risk_sets'] if r['checkpoint'] == '32')
    delta, loss = h2['delta_hb_minus_b'], h2['delta_log_loss_hb_minus_b']
    if loss['ci95'][1] < 0:
        loss_finding = 'O intervalo de perda permanece abaixo de zero, compatível com menor perda de H+B.'
    elif loss['ci95'][0] > 0:
        loss_finding = 'O intervalo de perda permanece acima de zero: H+B apresenta maior perda que B neste contraste.'
    else:
        loss_finding = 'O intervalo de perda inclui zero; a diferença de perda permanece inconclusiva.'
    decision = 'satisfaz' if h2['supported'] else 'não satisfaz'
    dw = 'não identificável por ausência de tarefas mistas' if delta is None else f"{number(delta['estimate'])}, IC 95% {interval(delta['ci95'])}"
    dl = f"{number(loss['estimate'])}, IC 95% {interval(loss['ci95'])}"
    verdict = 'Os critérios operacionais pré-especificados foram satisfeitos.' if h3['operational_benefit_claim_supported'] else 'Os critérios conjuntos para afirmar benefício operacional não foram satisfeitos.'
    out = root / 'tcc/generated'
    out.mkdir(exist_ok=True)
    delta_estimate = f"{delta['estimate']:.3f}" if delta else 'not identified'
    delta_interval = f"[{delta['ci95'][0]:.3f}, {delta['ci95'][1]:.3f}]" if delta else 'not identified'
    transfer_pt = ('A transferência aplica um sistema congelado a 640 tentativas MBPP+, sem ajuste nos dados alvo.'
                   if evidence.get('transfer') else 'A transferência congelada para MBPP+ permanece pendente.')
    transfer_en = ('Frozen transfer uses 640 MBPP+ attempts without target fitting.'
                   if evidence.get('transfer') else 'Frozen MBPP+ transfer remains pending.')
    summary = f'''#heading(numbering: none, outlined: false)[Resumo]
Este estudo avalia sinais latentes de corretude e interrupção precoce na geração
de código pelo Qwen2.5-Coder-1.5B-Instruct. Um piloto de 960 trajetórias escolhe
o regime de amostragem por regra fixa. O banco principal contém 2.640 tentativas
em 132 tarefas HumanEval+, avaliadas pelo EvalPlus. A validação aninhada por
tarefa separa treinamento, seleção e teste. No endpoint primário de 32 tokens,
{primary['trajectories']} trajetórias permanecem ativas e {primary['mixed_tasks']}
tarefas contêm ambos os resultados. A diferença de WT-AUC entre H+B e B foi
{dw}; a diferença de log loss foi {dl}. O resultado {decision} o critério de
ganho incremental definido no protocolo. Políticas são comparadas com as mesmas
sementes e orçamentos contra parada por verificador e por observáveis.
{verdict} {transfer_pt} A execução respeita uma GPU de 6 GB. Os intervalos
são condicionais aos modelos OOF ajustados. O histórico de exposição a tarefas
é divulgado e não se reivindica pré-registro independente nem aprovação do
orientador por inferência a partir da execução.

#par(first-line-indent: 0pt)[*Palavras-chave:* geração de código; probing; corretude funcional; validação agrupada; inferência adaptativa.]

#heading(numbering: none, outlined: false)[Abstract]
We evaluate hidden correctness signals and early termination with
Qwen2.5-Coder-1.5B-Instruct. A 960-trajectory pilot selects sampling by a fixed
rule. The main bank comprises 2,640 attempts across 132 HumanEval+ tasks with
official EvalPlus labels and task-grouped nested validation. At the primary
32-token endpoint, {primary['trajectories']} trajectories remain active and
{primary['mixed_tasks']} tasks contain both outcomes. The H+B versus B within-task
AUC difference is {delta_estimate} with a 95% interval
{delta_interval}. The log-loss difference is
{loss['estimate']:.3f} [{loss['ci95'][0]:.3f}, {loss['ci95'][1]:.3f}].
The predeclared incremental-gain criterion is {'met' if h2['supported'] else 'not met'}.
Paired policies are compared against verifier stopping and observable-feature
aborts. {transfer_en}
Intervals are conditional on fitted out-of-fold scorers; prior task exposure
is disclosed. No independent preregistration or advisor approval is claimed.
'''
    if delta is None:
        # Avoid reporting a numeric WT-AUC contrast when it is not identified.
        summary = summary[:summary.index('#heading(numbering: none, outlined: false)[Abstract]')]
        summary += '#heading(numbering: none, outlined: false)[Abstract]\nThe primary within-task contrast is not identified because no eligible tasks contain both outcomes.\n'
    (out / 'resumo.typ').write_text(summary, encoding='utf-8')
    results = f'''#import "../lib.typ": *
= Resultados do desenho fixo

== Amostragem, elegibilidade e proveniência

O piloto e o banco principal são separados por tarefa. A coleta mantém todas
as 20 tentativas de cada tarefa principal, inclusive após um sucesso, para
permitir replay pareado. No endpoint primário há {primary['trajectories']}
tentativas elegíveis, {primary['tasks']} tarefas e {primary['mixed_tasks']}
tarefas mistas. Tarefas homogêneas contribuem para perdas e resolução, mas
não identificam AUC intratarefa. A população ativa muda entre checkpoints:
as curvas temporais não acompanham uma amostra invariável.

#figure(image("../../reports/protocol_v0.2/figures/risk_set_attrition.svg", width: 100%),
 caption: [Elegibilidade T > t e composição das tarefas. Checkpoint zero usa todos os prompts.]) <fig-risk-sets>

== Predição e ganho incremental no endpoint primário

'''
    results += table(['Preditor', 'AUC', 'WT-AUC', 'IC 95% WT-AUC', 'Log loss'],
        [[name.replace('HB', 'H+B'), number(by_name[name]['roc_auc']), number(by_name[name]['within_task_auc']),
          interval(by_name[name]['within_task_auc_ci95']) if by_name[name]['within_task_auc_ci95'] else 'não identificável',
          number(by_name[name]['log_loss'])] for name in ('B', 'H', 'HB')],
        'tab-main-primary', 'Predições externas em 32 tokens; cada tarefa recebe o mesmo peso total.')
    results += f'''
A diferença H+B menos B em WT-AUC é {dw}, e em log loss é {dl}.
Esse resultado {decision} o critério conjunto de Q2. {loss_finding}
O teste de permutação por fold avalia associação intratarefa; ele não constitui
um teste isolado de informação adicional sobre B. Os controles e as métricas
secundárias estão apresentados para diagnóstico, sem substituir o endpoint.

#figure(image("../../reports/protocol_v0.2/figures/primary_predictor_comparison.svg", width: 100%),
 caption: [Desempenho primário e controles. Intervalos por bootstrap de tarefas, condicionais aos modelos ajustados.]) <fig-primary-controls>

#figure(image("../../reports/protocol_v0.2/figures/primary_task_deltas.svg", width: 100%),
 caption: [Heterogeneidade do incremento por tarefa mista e distribuição ponderada das probabilidades OOF.]) <fig-task-deltas>

#figure(image("../../reports/protocol_v0.2/figures/primary_roc_pr.svg", width: 100%),
 caption: [Curvas agregadas ROC e precisão–recall ponderadas por tarefa em 32 tokens.]) <fig-roc-pr>

#figure(image("../../reports/protocol_v0.2/figures/calibration_32.png", width: 75%),
 caption: [Probabilidade calibrada e frequência observada no teste externo.]) <fig-calibration-main>

#figure(image("../../reports/protocol_v0.2/figures/within_task_discrimination.png", width: 90%),
 caption: [Discriminação por checkpoint. O checkpoint de 32 tokens é primário; demais pontos são secundários.]) <fig-temporal-main>

== Replay pareado e validação em execução real

{verdict} O critério exige não inferioridade de resolução com margem de
três pontos percentuais e economia de tokens, ambos contra verificador e B,
além de suporte a Q2 e validação online com aborto efetivo. Uma falha em
qualquer condição impede afirmar benefício do sistema.

'''
    results += table(['N', 'Política', 'Resolução', 'Tokens/tarefa', 'Aborto falso ativo'],
        [[r['budget'], r['policy'].replace('HB', 'H+B'), number(r['resolution']), number(r['completion_tokens_per_task'], 1),
          number(r['active_correct_false_abort_rate']) if r['active_correct_false_abort_rate'] is not None else 'não identificável']
         for r in policies['summaries'] if r['policy'] in ('verifier', 'B', 'HB')],
        'tab-main-policy', 'Contabilidade de continuação: tentativas abortadas consomem orçamento; parada no primeiro sucesso verificado.')
    results += '''
#figure(image("../../reports/protocol_v0.2/figures/compute_resolution.png", width: 90%),
 caption: [Custo e resolução nos mesmos fluxos de candidatos. Tokens não equivalem a latência.]) <fig-main-frontier>

#figure(image("../../reports/protocol_v0.2/figures/paired_policy_effects.svg", width: 100%),
 caption: [Diferenças pareadas H+B menos comparador por orçamento, com IC 95%. Linha tracejada: margem de resolução −0,03.]) <fig-policy-effects>

#figure(image("../../reports/protocol_v0.2/figures/policy_cost_and_false_aborts.svg", width: 100%),
 caption: [Custo e descarte de tentativas corretas ativas iniciadas. O denominador exclui tentativas nunca iniciadas.]) <fig-policy-risk>
'''
    online = evidence.get('online')
    if online:
        results += f"\nA validação online incluiu {online['tasks']} tarefas escolhidas por ordem lexicográfica, sem seleção por resultado. A reprodução de tokens, identidades e decisões {'passou' if online['passed'] else 'falhou'}. {'Houve aborto efetivo.' if online['has_actual_abort'] else 'Não houve aborto efetivo; a validação não estabelece o caminho de aborto.'} Os tempos são descritivos, sem repetições suficientes para inferir economia de latência.\n"
    transfer = evidence.get('transfer')
    if transfer:
        results += '\n== Transferência congelada para MBPP+\n\nO sistema foi ajustado exclusivamente no banco principal HumanEval+ e aplicado sem reajuste, recalibração ou escolha de limiar nos rótulos alvo. O resultado é secundário e offline.\n\n'
        results += table(['Checkpoint', 'Preditor', 'AUC', 'WT-AUC', 'Log loss'],
            [[r['checkpoint'], r['family'].replace('HB', 'H+B'), number(r['roc_auc']),
              number(r['within_task_auc']), number(r['log_loss'])] for r in transfer['metrics']],
            'tab-mbpp-transfer', 'Desempenho externo de transferência. Uma população homogênea não identifica WT-AUC.')
        results += '''
#figure(image("../../reports/transfer_v0.2/figures/frozen_transfer_signal.svg", width: 100%),
 caption: [Transferência externa sem ajuste nos rótulos alvo. WT-AUC e contraste pareado; checkpoints são secundários.]) <fig-transfer-signal>
'''
        results += table(['Política N=20', 'Resolução MBPP+', 'Tokens/tarefa'],
                         [[r['policy'].replace('HB', 'H+B'), number(r['resolution']), number(r['completion_tokens_per_task'], 1)]
                          for r in transfer['policies']['summaries'] if r['budget'] == 20 and r['policy'] in ('B', 'HB', 'verifier')],
                         'tab-transfer-policy', 'Replay offline em 32 tarefas alvo. Não há validação online MBPP+ nem ajuste com rótulos alvo.')
        results += table(['Contraste N=20', 'Δresolução e IC 95%', 'Δtokens e IC 95%'],
                         [[r['comparison'].replace('HB', 'H+B').replace('minus', 'menos'),
                           f"{number(r['resolution_delta']['estimate'])} {interval(r['resolution_delta']['ci95'])}",
                           f"{number(r['token_delta']['estimate'], 2)} {interval(r['token_delta']['ci95'])}"]
                          for r in transfer['policies']['comparisons'] if r['budget'] == 20],
                         'tab-transfer-effects', 'Diferenças pareadas por tarefa em transferência. Tokens positivos indicam maior custo de H+B.')
    if evidence.get('supplemental'):
        from supplemental_publication import render_section
        results += render_section(evidence['supplemental'], number, interval, table)
    (out / 'resultados.typ').write_text(results, encoding='utf-8')
    discussion = f'''== Interpretação dos resultados do desenho fixo

O incremento primário {decision} os critérios estabelecidos antes do banco
principal. A comparação por tarefa evita confundir facilidade entre problemas
com ordenação de trajetórias de um mesmo problema. O valor observado é {dw}.
Mesmo um efeito positivo não demonstra mecanismo causal: o estado é uma
representação do prefixo produzido pelo próprio modelo.

{verdict} A conclusão operacional depende das duas comparações pareadas,
do risco de perder sucessos e da execução real. O custo em tokens não mede
automaticamente tempo, energia ou custo de prefill. A transferência constitui
um teste de transportabilidade limitado a outro benchmark de código.

Os intervalos de bootstrap mantêm os modelos OOF fixos; não incluem toda a
variabilidade do procedimento de treino. As tarefas HumanEval não são
inteiramente inéditas ao histórico da pesquisa: 26 tarefas principais estão
presentes em artefatos anteriores. O protocolo é de desenho fixo com execução
autorizada pelo estudante; a aprovação do orientador permanece pendente e não
se reivindica pré-registro independente. Resultados secundários e históricos
preservam a regra principal e seus critérios de confirmação.
'''
    if evidence.get('supplemental'):
        from supplemental_publication import discussion as supplemental_discussion
        discussion += supplemental_discussion(evidence['supplemental'], number, interval)
    (out / 'discussion_findings.typ').write_text(discussion, encoding='utf-8')
    conclusion_supplement = ''
    if evidence.get('supplemental'):
        from supplemental_publication import conclusion
        conclusion_supplement = conclusion(evidence['supplemental'], number, interval)
    (out / 'conclusao.typ').write_text(f'''= Considerações finais

A avaliação principal foi executada com 2.640 tentativas em 132 tarefas
HumanEval+, precedida por piloto separado de 960 trajetórias. A validação
aninhada agrupa tarefas e compara representações internas com observáveis
disponíveis no mesmo prefixo. Em 32 tokens, ΔWT-AUC(H+B − B) foi {dw} e
Δlog loss foi {dl}. O resultado {decision} o critério de ganho incremental.

{verdict} A comparação com verificador e observáveis, o replay pareado e a
validação real impedem atribuir ao probe economias disponíveis apenas pela
parada após sucesso. A transferência para MBPP+ utiliza modelos congelados.
As conclusões estão limitadas ao modelo, aos benchmarks e à população ativa
avaliados. Exposição histórica, incerteza condicional e ausência de inferência
robusta sobre latência são explicitadas. Replicações com novos modelos,
benchmarks e tarefas externas independentes são extensões naturais.
{conclusion_supplement}
''', encoding='utf-8')
    path = root / 'tcc/main.typ'
    text = path.read_text(encoding='utf-8')
    text = text.replace('Evidência exploratória e um protocolo de avaliação intratarefa para inferência adaptativa',
                        'Avaliação intratarefa, custo de inferência e transferência de sinais precoces')
    text = text.replace('versão empírica exploratória', 'avaliação empírica do desenho fixo')
    path.write_text(text, encoding='utf-8')
    chapters = root / 'tcc/chapters'
    changes = {
        '01_introducao.typ': [(
            'A entrega reúne uma reanálise exploratória dos dados históricos, a validação técnica da coleta revisada e uma implementação do protocolo v0.2. Os resultados precoces confirmatórios dependem de coleta completa e congelamento científico. Cada capítulo distingue o que foi observado do que permanece como desenho experimental.',
            'A avaliação principal usa o desenho fixo v0.2, com piloto separado, banco completo e validação por tarefa. A reanálise histórica permanece exploratória e é apresentada separadamente. As decisões empíricas seguem os critérios definidos antes do banco principal; os capítulos divulgam a exposição histórica e a autorização de execução.')],
        '06_discussao.typ': [(
            'A discussão será preenchida a partir dos resultados, seguindo Q1–Q3.',
            'A discussão segue os resultados medidos para Q1–Q3, distinguindo associação, ganho incremental e utilidade operacional.')],
    }
    if evidence.get('transfer'):
        changes['04_metodologia.typ'] = [(
            'Esta entrega valida o isolamento em testes,\nmas ainda não reporta resultados empíricos de transferência.',
            'Os resultados da transferência congelada são reportados no capítulo empírico,\nsem retreino, recalibração ou seleção nos rótulos alvo.')]
    for name, substitutions in changes.items():
        source = chapters / name
        if source.exists():
            contents = source.read_text(encoding='utf-8')
            for old, new in substitutions:
                contents = contents.replace(old, new)
            source.write_text(contents, encoding='utf-8')
