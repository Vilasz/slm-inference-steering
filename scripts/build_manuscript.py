"""Render observed results into editable Typst sources and optionally compile."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def number(value, digits=3):
    if value is None or not math.isfinite(value):
        return 'não identificável'
    return f'{value:.{digits}f}'.replace('.', ',')


def interval(values):
    return f'[{number(values[0])}; {number(values[1])}]'


def table(headers, rows, label, caption):
    cells = ',\n'.join(', '.join('[' + str(cell) + ']' for cell in row) for row in rows)
    return ('#figure(\n { set text(size: 10pt); set par(leading: 0.34em, first-line-indent: 0pt); table(columns: ' + str(len(headers)) + ', align: left,\n table.header(' +
            ', '.join('[*' + text + '*]' for text in headers) + '),\n' + cells + '\n ) },\n kind: table,\n caption: [' +
            caption + ']\n) <' + label + '>\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--compile', action='store_true')
    args = parser.parse_args()
    completed_extension = ROOT / 'reports/final_extension/results.json'
    if completed_extension.exists():
        extension = json.loads(completed_extension.read_text(encoding='utf-8'))
        if extension.get('status') == 'completed':
            from render_integrated_manuscript import render
            render(ROOT, compile_pdf=args.compile)
            print('Integrated manuscript rendered from both completed experimental stages')
            return
    path = ROOT / 'reports/final_review/evidence.json'
    evidence = json.loads(path.read_text(encoding='utf-8'))
    if evidence['confirmatory_results_available']:
        from render_main_manuscript import render
        render(ROOT, evidence, number, interval, table)
        if evidence.get('excellence'):
            import sys
            sys.path.insert(0, str(ROOT / 'src'))
            from slm_steering.excellence.report import render_typst
            render_typst(ROOT, evidence['excellence'], number, interval, table)
        import sys
        sys.path.insert(0, str(ROOT / 'src'))
        from slm_steering.excellence.print_figures import route_generated_chapters
        route_generated_chapters(ROOT)
        if args.compile:
            import typst
            output = ROOT / 'tcc/build/manuscrito.pdf'
            output.parent.mkdir(exist_ok=True)
            typst.compile(str(ROOT / 'tcc/main.typ'), output=str(output), root=str(ROOT))
            print(f'Main experiment manuscript compiled: {output}')
        return
    late = evidence['legacy_reanalysis']
    first, last = evidence['legacy_sampling'][0], evidence['legacy_sampling'][-1]
    metrics = {row['family']: row for row in late['metrics']}
    dw, dl = late['delta_wt_auc'], late['delta_log_loss']
    values = {'TASKS': str(late['tasks']), 'ATTEMPTS': str(late['attempts']), 'MIXED': str(late['mixed_tasks']),
              'FIRST': str(first['solved']), 'LAST': str(last['solved']), 'FULL': str(last['fixed_tokens']),
              'STOP': str(last['verifier_tokens']), 'SAVING': number(100 * last['token_saving_fraction'], 2),
              'HAUC': number(metrics['H_final']['roc_auc']), 'HWT': number(metrics['H_final']['within_task_auc']),
              'DW': number(dw['estimate']), 'DWCI': interval(dw['ci95']), 'DL': number(dl['estimate']),
              'DLCI': interval(dl['ci95']), 'DATE': evidence['as_of']}
    def fill(text):
        for key, value in values.items():
            text = text.replace('@@' + key + '@@', value)
        return text
    generated = ROOT / 'tcc/generated'
    generated.mkdir(exist_ok=True)
    summary = fill('''#heading(numbering: none, outlined: false)[Resumo]
Este trabalho investiga se estados internos de um modelo de linguagem especializado
em código distinguem trajetórias corretas e incorretas da mesma tarefa e podem apoiar
interrupção precoce. A contribuição metodológica separa previsibilidade agregada,
discriminação intratarefa, ganho sobre observáveis e utilidade operacional. Uma
reanálise retrospectiva de @@ATTEMPTS@@ tentativas de Qwen2.5-Coder-1.5B-Instruct
em @@TASKS@@ tarefas HumanEval utiliza representações do último token, regressão
logística e validação aninhada por tarefa. Das tarefas, @@MIXED@@ apresentaram
resultados mistos. O preditor latente final obteve AUC agregada de @@HAUC@@ e
WT-AUC de @@HWT@@. A combinação com observáveis tardios apresentou diferença
WT-AUC de @@DW@@ em relação a esses observáveis, com intervalo de 95% @@DWCI@@,
e diferença de log loss de @@DL@@, com intervalo @@DLCI@@. A melhora probabilística
não foi acompanhada de evidência precisa de ganho de ordenação intratarefa.
No mesmo banco, parar após aprovação pelo verificador reduz o custo de @@FULL@@
para @@STOP@@ tokens para N=5, preservando as @@LAST@@ tarefas resolvidas; essa
economia não é atribuída a aborto precoce. A coleta revisada com EvalPlus foi
validada tecnicamente em GPU de 6 GB. As hipóteses precoces em 32 tokens e a
utilidade operacional do probe permanecem pendentes de avaliação confirmatória.

#par(first-line-indent: 0pt)[*Palavras-chave:* geração de código; probing; corretude funcional; validação agrupada; inferência adaptativa.]

#heading(numbering: none, outlined: false)[Abstract]
This work studies whether hidden states of a code language model distinguish
successful and unsuccessful trajectories for the same task and can support early
termination. The methodology separates aggregate predictability, within-task
discrimination, incremental value over observables, and operational utility.
A retrospective analysis of 150 Qwen2.5-Coder-1.5B-Instruct completions across
30 HumanEval tasks uses final-token representations, logistic regression and
task-grouped nested validation. Sixteen tasks contain both outcomes. Final-state
prediction reaches aggregate AUC 0.781 and within-task AUC 0.844. Adding hidden
states to late observable features yields a within-task AUC difference of 0.021
(95% interval −0.104 to 0.157) and a log-loss difference of −0.143 (−0.232 to −0.043).
Improved probability prediction therefore does not establish an incremental
within-task ranking gain. On the same candidate bank, verifier stopping reduces
N=5 completion cost from 8,149 to 4,029 tokens while preserving 25 solved tasks.
This is not evidence of prefix-based early-abort savings. A revised EvalPlus
collection pipeline was technically validated on a 6 GB GPU. Early prediction
at 32 tokens and operational probe utility remain open confirmatory questions.
''')
    # Derive the English quantitative paragraph from the same evidence as Portuguese.
    summary = summary.replace('150 Qwen', f"{late['attempts']} Qwen").replace('30 HumanEval', f"{late['tasks']} HumanEval")
    summary = summary.replace('Sixteen tasks', f"{late['mixed_tasks']} tasks")
    summary = summary.replace('AUC 0.781', f"AUC {metrics['H_final']['roc_auc']:.3f}")
    summary = summary.replace('AUC 0.844', f"AUC {metrics['H_final']['within_task_auc']:.3f}")
    summary = summary.replace('0.021\n(95% interval −0.104 to 0.157)',
                              f"{dw['estimate']:.3f}\n(95% interval {dw['ci95'][0]:.3f} to {dw['ci95'][1]:.3f})")
    summary = summary.replace('−0.143 (−0.232 to −0.043)', f"{dl['estimate']:.3f} ({dl['ci95'][0]:.3f} to {dl['ci95'][1]:.3f})")
    summary = summary.replace('8,149 to 4,029', f"{last['fixed_tokens']:,} to {last['verifier_tokens']:,}")
    summary = summary.replace('25 solved', f"{last['solved']} solved")
    (generated / 'resumo.typ').write_text(summary, encoding='utf-8')
    results = fill('''#import "../lib.typ": *
= Resultados observados

== Escopo da evidência

Os resultados deste capítulo foram consolidados em @@DATE@@. Há dois conjuntos:
um banco histórico com @@ATTEMPTS@@ tentativas em @@TASKS@@ tarefas HumanEval,
avaliadas pelo verificador local então utilizado, e bancos parciais de validação
técnica com EvalPlus. A reanálise dos estados finais é retrospectiva. Não se
combinam rótulos históricos e HumanEval+, nem se apresentam esses resultados
como evidência de previsão em 32 tokens.

== Composição e amostragem repetida

No banco histórico, @@MIXED@@ tarefas contêm acertos e erros. Elas sustentam
a estimativa intratarefa; todas as @@TASKS@@ entram nas perdas e nos resultados
operacionais. A @fig-composicao apresenta a composição observada. Categorias
com apenas uma classe não estabelecem probabilidades verdadeiras iguais a zero ou um.

#figure(image("../../reports/final_review/figures/legacy_task_composition.svg", width: 85%),
 caption: [Composição histórica com cinco tentativas por tarefa; verificador local original.]) <fig-composicao>

A resolução passa de @@FIRST@@/@@TASKS@@ em N=1 para @@LAST@@/@@TASKS@@ em N=5.
O ganho descreve cobertura no prefixo de candidatos armazenados. Em N=5, gerar
todos os candidatos custa @@FULL@@ tokens, enquanto parar após a primeira
aprovação custa @@STOP@@, uma redução de @@SAVING@@%. A resolução é igual por
construção nesse replay pareado: o verificador encerra a busca apenas após um
sucesso. Essa comparação isola tentativas dispensadas depois da solução, sem
creditar ao probe qualquer interrupção de um candidato ainda ativo.

#figure(image("../../reports/final_review/figures/legacy_sampling.svg", width: 100%),
 caption: [Resolução e custo no mesmo banco histórico. A faixa mostra IC percentílico por tarefa; custo em tokens de continuação.]) <fig-amostragem>

''')
    results += table(['N', 'Resolvidas', 'Gerar todos', 'Parar pelo verificador'],
                     [[row['budget'], f"{row['solved']}/{row['tasks']}", row['fixed_tokens'], row['verifier_tokens']]
                      for row in evidence['legacy_sampling']], 'tab-orcamentos',
                     'Orçamentos históricos. Custos são totais de tokens, calculados a partir das mesmas trajetórias.')
    results += fill('''
== Representações finais: reanálise com seleção interna

As ativações disponíveis correspondem ao último token da resposta, com sete
índices históricos de hidden states e dimensão 1.536. A reanálise usa cinco
folds externos agrupados e três internos, ajustando camada e regularização
exclusivamente no treino externo. A grade é C ∈ {0,001; 0,01; 0,1; 1; 10}.
O baseline tardio contém log(1 + tokens do prompt) e log(1 + tokens finais).
O comprimento final é válido nessa análise tardia, mas seria informação futura
na análise precoce. Não havia estatísticas de confiança por token compatíveis
com a nova especificação; por isso B tardio não é o B do endpoint confirmatório.

Os classificadores fornecem probabilidades logísticas sem calibração adicional
na reanálise histórica. A @tab-probes e a @fig-probes mostram as predições externas.
Os intervalos mantêm os ajustes fixos e reamostram tarefas inteiras, com 2.000
replicações e seed 20261012. AUC agregada e WT-AUC usam populações/ordenações
diferentes e não são versões intercambiáveis da mesma medida.

''')
    labels = {'B_late': 'B tardio', 'H_final': 'H final', 'HB_late': 'H+B tardio'}
    results += table(['Preditor', 'AUC', 'WT-AUC', 'IC 95% WT-AUC', 'Log loss'],
                     [[labels[row['family']], number(row['roc_auc']), number(row['within_task_auc']),
                       interval(row['wt_auc_interval']['ci95']), number(row['log_loss'])] for row in late['metrics']],
                     'tab-probes', 'Predição externa de corretude final; WT-AUC estimada nas tarefas mistas.')
    results += fill('''
#figure(image("../../reports/final_review/figures/legacy_late_probes.svg", width: 100%),
 caption: [Resultados exploratórios tardios. Barras de incerteza da WT-AUC condicionais aos modelos OOF ajustados.]) <fig-probes>

A diferença H+B tardio menos B tardio na WT-AUC é @@DW@@, com IC de 95%
@@DWCI@@. O intervalo inclui ausência de ganho e efeitos de ambas as direções.
A diferença de log loss é @@DL@@, com IC @@DLCI@@, nas @@TASKS@@ tarefas.
Assim, há melhoria probabilística nessa comparação histórica, mas a evidência
de ganho incremental de ordenação intratarefa é imprecisa. As duas observações
podem coexistir porque as métricas avaliam propriedades distintas.

A AUC máxima de 0,953 registrada na exploração anterior era uma estatística de
configurações em um split específico. Ela não é diretamente comparável ao
procedimento aninhado aqui avaliado e não é adotada como estimativa principal.
A presente reanálise também permanece retrospectiva: seleção interna não
transforma decisões tomadas depois da exploração em pré-registro.

== Validação técnica da coleta revisada

''')
    results += table(['Banco', 'Trajetórias', 'Tarefas', 'Pico reservado', 'Causal / seed'],
                     [[f"v{p['version']}", f"{p['attempts']}/{p['planned']}", p['tasks'],
                       number(p['peak_reserved_mib'] / 1024, 2) + ' GiB',
                       'passou / passou' if p['causal_validation'] and p['seed_validation'] else 'ver manifesto']
                      for p in evidence['pilot']], 'tab-pilotos',
                     'Bancos técnicos separados por versão. A v0.2 usa rótulo funcional e margem anterior ao top-p.')
    results += '''
O pico reservado cabe na GPU de 6 GB. As execuções técnicas demonstram que a
extração, a avaliação oficial e a retomada podem ser realizadas localmente;
não constituem uma estimativa de latência de uma política já treinada. Tempos
de desenvolvimento podem variar com aquecimento, comprimentos e concorrência
de outros processos. Nenhum dos pilotos parciais permite selecionar temperatura.

== Situação das perguntas de pesquisa

Q1 precoce e Q2 em 32 tokens ainda não foram estimadas no banco principal.
A existência de sinal ao final da geração não determina quando ele se torna
disponível. Q3 também permanece aberta: a economia observada acima pertence
à parada por verificador. A política precoce revisada foi verificada com testes
de término, seleção, isolamento e contabilidade, mas seus efeitos reais ainda
dependem de treinamento e avaliação externa. MBPP+ possui fluxo de transferência
congelada implementado, sem resultados empíricos nesta entrega.
'''
    if evidence.get('pilot_temperatures'):
        current = next(p for p in evidence['pilot'] if p['version'] == '0.2')
        results += f'''\n== Snapshot da execução integral solicitada

A coleta revisada está em execução. Este snapshot, preparado em
{evidence['built_at']}, contém {current['attempts']}/{current['planned']}
trajetórias do piloto v0.2. Os dados usados nesta figura estão copiados
separadamente em reports/final_review/snapshots quando o banco está ativo,
permitindo que a coleta continue sem alterar os dados da publicação.
Nenhuma taxa parcial seleciona temperatura ou testa a hipótese principal.

#figure(image("../../reports/final_review/figures/pilot_sampling_progress.svg", width: 100%),
 caption: [Cobertura do piloto e comprimentos observados. Partes cinzas ainda não foram geradas; comparação descritiva de desenvolvimento.]) <fig-pilot-progress>
'''
        if (ROOT / 'reports/final_review/figures/pilot_lengths_and_verifier.svg').exists():
            results += '''
#figure(image("../../reports/final_review/figures/pilot_lengths_and_verifier.svg", width: 100%),
 caption: [Comprimentos por temperatura e resultados das suítes no snapshot do piloto. Estatísticas descritivas, sem teste de ganho precoce.]) <fig-pilot-distributions>

#figure(image("../../reports/final_review/figures/pilot_task_temperature_grid.svg", width: 64%),
 caption: [Acertos sobre tentativas por tarefa e temperatura. Células parciais não estimam o resultado do experimento principal.]) <fig-pilot-grid>
'''
        if (ROOT / 'reports/final_review/figures/legacy_incremental_by_task.svg').exists():
            results += '''
#figure(image("../../reports/final_review/figures/legacy_incremental_by_task.svg", width: 100%),
 caption: [Incremento retrospectivo de WT-AUC por tarefa mista. Observações discretas de cinco tentativas, sem inferência individual por tarefa.]) <fig-legacy-task-effects>
'''
    (generated / 'resultados.typ').write_text(results, encoding='utf-8')
    discussion = fill('''== O que os resultados acrescentam

A comparação histórica evidencia por que a avaliação precisa distinguir as
perguntas do estudo. Estados finais permitem predição agregada, mas B tardio
já ordena tentativas da mesma tarefa. O incremento de WT-AUC de @@DW@@ é
impreciso, enquanto a perda probabilística melhora. Uma política de alta
retenção depende especialmente dos erros cometidos perto do limiar e de quais
sucessos são descartados; nenhuma dessas propriedades decorre de AUC isolada.

O replay mostra uma economia substancial disponível apenas com parada após
verificação. Esse comparador é essencial: atribuir toda a redução relativa à
geração de N candidatos a um probe confundiria dois mecanismos de economia.
A contribuição empírica atual é delimitar o que os artefatos históricos
sustentam e estabelecer uma avaliação mais exigente para a hipótese precoce.

''')
    (generated / 'discussion_findings.typ').write_text(discussion, encoding='utf-8')
    conclusion = fill('''= Considerações finais

Este trabalho formulou uma avaliação de sinais internos de corretude que
distingue dificuldade da tarefa, discriminação entre trajetórias e utilidade
operacional. A reanálise de @@ATTEMPTS@@ tentativas em @@TASKS@@ tarefas mostra
que representações finais podem apoiar predição probabilística de corretude,
mas o ganho de ordenação intratarefa sobre observáveis tardios permanece
inconclusivo: ΔWT-AUC = @@DW@@, IC de 95% @@DWCI@@.

A parada por verificador reduz o custo histórico de @@FULL@@ para @@STOP@@
tokens em N=5, preservando @@LAST@@ tarefas resolvidas. Esse resultado estabelece
um comparador operacional necessário e não demonstra benefício de aborto
precoce. A infraestrutura revisada passou em verificações técnicas locais,
com custo de memória compatível com uma GPU de 6 GB.

As respostas confirmatórias a Q1 precoce, Q2 em 32 tokens e Q3 exigem completar
o piloto, congelar o protocolo, gerar o banco principal e avaliar as políticas
em tarefas externas. A entrega atual fornece fundamentação, implementação,
evidência exploratória auditável e critérios para essa continuidade, sem
converter resultados tardios ou técnicos em confirmação de hipóteses precoces.
''')
    (generated / 'conclusao.typ').write_text(conclusion, encoding='utf-8')
    if args.compile:
        import typst
        output = ROOT / 'tcc/build/manuscrito.pdf'
        output.parent.mkdir(exist_ok=True)
        typst.compile(str(ROOT / 'tcc/main.typ'), output=str(output), root=str(ROOT))
        print(f'Manuscript compiled: {output}')
    else:
        print('Empirical sections generated from evidence.json')


if __name__ == '__main__':
    main()
