#import "../lib.typ": *
= Primeira etapa: Qwen, transferência e diagnósticos <chap-primeira>

== Composição do banco e comparação intratarefa

A primeira etapa estabelece a referência empírica do estudo. O piloto separado produziu 960 trajetórias e selecionou a configuração de geração. O banco principal contém 2.640 tentativas em 132 tarefas HumanEval+. Como os probes precoces só podem atuar enquanto uma resposta continua sendo gerada, o tamanho de sua população diminui com o checkpoint.

#figure(
 { set text(size: 10pt); set par(first-line-indent: 0pt, justify: false, leading: 0.4em);
 table(columns: 4, align: left, inset: 5pt,
 table.header([*Tokens*], [*Ativas*], [*Tarefas ativas*], [*Mistas*]),
[8], [2640], [132], [100],
[16], [2608], [132], [97],
[32], [2158], [132], [92],
[64], [1228], [122], [69]
 ) },
 kind: table, caption: [Qwen em HumanEval+: população ativa e tarefas que identificam WT-AUC.]
) <tab-original-risk>


Em 32 tokens, 2.158 tentativas permanecem ativas e 92 tarefas apresentam ambos os resultados. Essa é a população do contraste principal de WT-AUC. Tarefas homogêneas continuam contribuindo para perdas e replay, mas não recebem uma AUC intratarefa fictícia. O conjunto de trajetórias longas em 64 tokens é diferente; a curva temporal não deve ser interpretada como evolução de desempenho sobre uma amostra invariável.

#figure(image("../../reports/excellence/print_ready/core/figures/risk_set_attrition.svg", width: 100%), caption: [Composição do banco Qwen por checkpoint. A condição T > t muda a população e o número de tarefas mistas.]) <fig-original-risk>

#figure(
 { set text(size: 10pt); set par(first-line-indent: 0pt, justify: false, leading: 0.4em);
 table(columns: (0.6fr, 1fr, 1.8fr, 1fr, 1fr), align: left, inset: 5pt,
 table.header([*Probe*], [*AUC agregada*], [*WT-AUC e IC95%*], [*Log loss*], [*Brier*]),
[B], [0,701], [0,628 [0,583; 0,675]], [0,630], [0,220],
[H], [0,592], [0,568 [0,521; 0,614]], [0,687], [0,246],
[H+B], [0,646], [0,581 [0,534; 0,629]], [0,668], [0,236]
 ) },
 kind: table, caption: [Probes lineares da primeira etapa, fora da amostra por tarefa em HumanEval+, t=32.]
) <tab-original-metrics>


O baseline observável B alcança WT-AUC de 0,628, enquanto H+B alcança 0,581. A diferença é −0,047, IC95% [−0,099; 0,006]. Na perda probabilística, H+B−B é 0,038, IC95% [0,007; 0,071]. O critério incremental original não foi satisfeito: a ordenação não apresenta ganho demonstrado, e o intervalo da diferença de log loss indica maior perda de H+B.

Esse resultado não equivale a provar ausência de qualquer associação entre estados e corretude. O probe H pode apresentar variação preditiva sem superar B. Além disso, o teste de permutação por fold examina associação intratarefa, e não a hipótese mais forte de ausência de informação condicional adicional. O contraste pareado e seus intervalos respondem à pergunta operacional sobre o procedimento ajustado.

#figure(image("../../reports/excellence/print_ready/core/figures/within_task_discrimination.svg", width: 100%), caption: [WT-AUC Qwen por checkpoint. Intervalos por tarefa condicionais aos scorers OOF; 32 tokens é o endpoint primário, e os demais pontos são secundários.]) <fig-original-temporal>

== O resultado de classificação e a política original

Uma diferença de AUC não determina o custo de uma política. Por isso, a avaliação original também reproduziu a ordem dos candidatos e a parada ao primeiro sucesso mantido. Em N=20, os resultados são:

#figure(
 { set text(size: 10pt); set par(first-line-indent: 0pt, justify: false, leading: 0.4em);
 table(columns: 3, align: left, inset: 5pt,
 table.header([*Política*], [*Resolvidas / 132*], [*Tokens / tarefa*]),
[Vinte completas], [112], [1534,0],
[Verificador], [112], [467,3],
[B], [107], [447,2],
[H+B], [110], [460,2]
 ) },
 kind: table, caption: [Primeira etapa: replay com N=20 e toda a população de tarefas.]
) <tab-original-policy>


Contra o verificador, H+B apresenta diferença de resolução de −0,015, IC95% [−0,038; 0,000] e diferença média de tokens de −7,18, IC95% [−18,73; 0,80]. Contra B, os respectivos contrastes são 0,023, IC95% [−0,015; 0,061] e 12,96, IC95% [−26,34; 77,05]. Os critérios conjuntos de resolução, economia e incremento preditivo não foram satisfeitos. Economias obtidas apenas por parar após um sucesso pertencem ao baseline com verificador e não podem ser creditadas ao probe.

A validação online em doze tarefas conferiu a reprodução de tokens, identidades, decisões e contagem de custo, incluindo aborto efetivo. Seu escopo foi técnico: não forneceu repetição suficiente para afirmar redução de latência. O replay dos demais orçamentos e seus intervalos permanece nos artefatos e é distinguido da execução online.

#figure(image("../../reports/excellence/print_ready/core/figures/paired_policy_effects.svg", width: 100%), caption: [Diferenças pareadas de custo e resolução do procedimento Qwen original. A margem de três pontos percentuais pertence ao critério operacional desse protocolo.]) <fig-original-policy>

== Transporte para MBPP+ e validação independente dos gates

Na primeira transferência, os modelos, calibradores e limiares ajustados em HumanEval+ foram aplicados a 32 tarefas MBPP+, sem reajuste no alvo. Em 32 tokens, a WT-AUC pontual de H+B ficou próxima de B, mas o comportamento de uma política depende do limiar transportado e do momento da decisão. A retenção pode se deteriorar mesmo quando uma métrica de ordenação permanece semelhante.

A comparação posterior em um checkpoint comum de oito tokens torna essa diferença mais clara. Nas 32 tarefas inicialmente examinadas, H+B manteve 276 dos 366 corretos ativos, enquanto B manteve 364. A observação motivou uma hipótese prospectiva delimitada: os dois gates congelados perderiam diferentes proporções de corretos em tarefas locais ainda não examinadas?

A validação independente foi concluída em outras 120 tarefas MBPP+, com vinte tentativas cada. Seu denominador comum contém 1.192 candidatos corretos ativos. Os resultados são:

#figure(
 { set text(size: 10pt); set par(first-line-indent: 0pt, justify: false, leading: 0.4em);
 table(columns: (0.8fr, 1fr, 1fr, 1.6fr), align: left, inset: 5pt,
 table.header([*Gate em t=8*], [*Corretos mantidos*], [*Corretos ativos*], [*Retenção e IC95%*]),
[H+B], [986], [1192], [82,72% [75,00; 89,72]],
[B], [1181], [1192], [99,08% [98,30; 99,71]]
 ) },
 kind: table, caption: [Validação independente anterior, 120 tarefas Qwen/MBPP+. Intervalos por tarefas, em percentuais.]
) <tab-independent-gates>


A diferença de retenção H+B−B é −16,36 pontos percentuais, IC95% [−24,02; −9,43]. O intervalo inteiramente negativo satisfaz o critério direcional daquela avaliação. Essa evidência confirma uma limitação dos gates transportados em oito tokens; não é uma nova avaliação do contraste original em 32 tokens e não prova impossibilidade universal de informação latente.

Retenção de candidatos também não é resolução. Perder o único sucesso potencial de uma tarefa pode impedir sua solução; perder um entre vários sucessos pode não modificar a cobertura. Os denominadores dos candidatos iniciados e de todos os corretos potenciais são, por isso, mantidos separados.

== O que as ablações exploratórias acrescentam

Os resultados motivaram diagnósticos de leitura: regularização mais forte, PCA e um corretor residual que parte da previsão de B. Como foram escolhidos após a inspeção do resultado principal, são exploratórios, ainda que seus ajustes internos respeitem o isolamento de tarefas.

#figure(
 { set text(size: 10pt); set par(first-line-indent: 0pt, justify: false, leading: 0.4em);
 table(columns: (1.4fr, 0.8fr, 2.2fr), align: left, inset: 5pt,
 table.header([*Leitura*], [*WT-AUC*], [*ΔWT-AUC contra B e IC95%*]),
[H+B original], [0,581], [−0,047, IC95% [−0,099; 0,006]],
[H+B com PCA64], [0,630], [0,002, IC95% [−0,030; 0,034]],
[Correção residual de B], [0,635], [0,006, IC95% [−0,009; 0,024]]
 ) },
 kind: table, caption: [Diagnósticos posteriores no mesmo banco Qwen de origem; intervalos exploratórios condicionais.]
) <tab-exploratory-readouts>


PCA64 e correção residual aproximam o desempenho do baseline e reduzem a desvantagem da combinação original. Seus intervalos não demonstram superioridade sobre B. Essa observação altera a interpretação do resultado: uma leitura latente de alta dimensão pode ser sensível à parametrização. Ela justifica fixar uma representação reduzida e testar outra classe de decodificador em uma nova avaliação, sem escolher retrospectivamente o vencedor desta tabela como evidência confirmatória.

Diagnósticos de prefixos idênticos fornecem um controle complementar. Com o mesmo prompt e os mesmos tokens já processados, um scorer determinístico deve empatar as tentativas, mesmo que suas continuações tenham resultados diferentes. A consistência dessas previsões verifica a instrumentação; não revela, por si só, por que uma determinada família perde informação ou calibração.

== O espaço econômico disponível

O oracle contrafactual mantém todos os corretos e interrompe incorretos ativos, conhecendo seus labels finais. A comparação mostra que há trabalho dispensável no banco, mas não que um probe saiba identificá-lo. O espaço idealizado de economia e a capacidade empírica de aproveitá-lo são objetos distintos.

#figure(
 { set text(size: 10pt); set par(first-line-indent: 0pt, justify: false, leading: 0.4em);
 table(columns: 3, align: left, inset: 5pt,
 table.header([*Política*], [*Resolução*], [*Tokens / tarefa*]),
[oracle], [0,848], [212,75],
[verifier], [0,848], [467,34]
 ) },
 kind: table, caption: [Limite contrafactual na primeira etapa, N=20 e checkpoint de 32 tokens.]
) <tab-original-oracle>


Esses diagnósticos conectam a primeira à segunda etapa: há oportunidade econômica sob informação perfeita, sensibilidade à parametrização do leitor e problemas de transporte de decisões. A avaliação seguinte pergunta se uma comparação correspondente entre classes de probes, em outra família de gerador e em tarefas externas novas, modifica o incremento acessível de H.
