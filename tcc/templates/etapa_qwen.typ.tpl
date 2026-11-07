#import "../lib.typ": *
= Primeira etapa: Qwen, transferência e diagnósticos <chap-primeira>

== Composição do banco e comparação intratarefa

A primeira etapa estabelece a referência empírica do estudo. O piloto separado produziu 960 trajetórias e selecionou a configuração de geração. O banco principal contém 2.640 tentativas em 132 tarefas HumanEval+. Como os probes precoces só podem atuar enquanto uma resposta continua sendo gerada, o tamanho de sua população diminui com o checkpoint.

@@OLD_RISK_TABLE@@

Em 32 tokens, 2.158 tentativas permanecem ativas e 92 tarefas apresentam ambos os resultados. Essa é a população do contraste principal de WT-AUC. Tarefas homogêneas continuam contribuindo para perdas e replay, mas não recebem uma AUC intratarefa fictícia. O conjunto de trajetórias longas em 64 tokens é diferente; a curva temporal não deve ser interpretada como evolução de desempenho sobre uma amostra invariável.

#figure(image("../../reports/excellence/print_ready/core/figures/risk_set_attrition.svg", width: 100%), caption: [Composição do banco Qwen por checkpoint. A condição T > t muda a população e o número de tarefas mistas.]) <fig-original-risk>

@@OLD_METRICS_TABLE@@

O baseline observável B alcança WT-AUC de @@OLD_B_WT@@, enquanto H+B alcança @@OLD_HB_WT@@. A diferença é @@OLD_DELTA@@, IC95% @@OLD_CI@@. Na perda probabilística, H+B−B é @@OLD_LOSS_DELTA@@, IC95% @@OLD_LOSS_CI@@. O critério incremental original não foi satisfeito: a ordenação não apresenta ganho demonstrado, e o intervalo da diferença de log loss indica maior perda de H+B.

Esse resultado não equivale a provar ausência de qualquer associação entre estados e corretude. O probe H pode apresentar variação preditiva sem superar B. Além disso, o teste de permutação por fold examina associação intratarefa, e não a hipótese mais forte de ausência de informação condicional adicional. O contraste pareado e seus intervalos respondem à pergunta operacional sobre o procedimento ajustado.

#figure(image("../../reports/excellence/print_ready/core/figures/within_task_discrimination.svg", width: 100%), caption: [WT-AUC Qwen por checkpoint. Intervalos por tarefa condicionais aos scorers OOF; 32 tokens é o endpoint primário, e os demais pontos são secundários.]) <fig-original-temporal>

== O resultado de classificação e a política original

Uma diferença de AUC não determina o custo de uma política. Por isso, a avaliação original também reproduziu a ordem dos candidatos e a parada ao primeiro sucesso mantido. Em N=20, os resultados são:

@@OLD_POLICY_TABLE@@

Contra o verificador, H+B apresenta diferença de resolução de @@OLD_POLICY_COVERAGE@@ e diferença média de tokens de @@OLD_POLICY_TOKENS@@. Contra B, os respectivos contrastes são @@OLD_POLICY_VS_B_COVERAGE@@ e @@OLD_POLICY_VS_B_TOKENS@@. Os critérios conjuntos de resolução, economia e incremento preditivo não foram satisfeitos. Economias obtidas apenas por parar após um sucesso pertencem ao baseline com verificador e não podem ser creditadas ao probe.

A validação online em doze tarefas conferiu a reprodução de tokens, identidades, decisões e contagem de custo, incluindo aborto efetivo. Seu escopo foi técnico: não forneceu repetição suficiente para afirmar redução de latência. O replay dos demais orçamentos e seus intervalos permanece nos artefatos e é distinguido da execução online.

#figure(image("../../reports/excellence/print_ready/core/figures/paired_policy_effects.svg", width: 100%), caption: [Diferenças pareadas de custo e resolução do procedimento Qwen original. A margem de três pontos percentuais pertence ao critério operacional desse protocolo.]) <fig-original-policy>

== Transporte para MBPP+ e validação independente dos gates

Na primeira transferência, os modelos, calibradores e limiares ajustados em HumanEval+ foram aplicados a 32 tarefas MBPP+, sem reajuste no alvo. Em 32 tokens, a WT-AUC pontual de H+B ficou próxima de B, mas o comportamento de uma política depende do limiar transportado e do momento da decisão. A retenção pode se deteriorar mesmo quando uma métrica de ordenação permanece semelhante.

A comparação posterior em um checkpoint comum de oito tokens torna essa diferença mais clara. Nas 32 tarefas inicialmente examinadas, H+B manteve 276 dos 366 corretos ativos, enquanto B manteve 364. A observação motivou uma hipótese prospectiva delimitada: os dois gates congelados perderiam diferentes proporções de corretos em tarefas locais ainda não examinadas?

A validação independente foi concluída em outras 120 tarefas MBPP+, com vinte tentativas cada. Seu denominador comum contém 1.192 candidatos corretos ativos. Os resultados são:

@@INDEPENDENT_TABLE@@

A diferença de retenção H+B−B é @@INDEPENDENT_PP@@ pontos percentuais, IC95% @@INDEPENDENT_PP_CI@@. O intervalo inteiramente negativo satisfaz o critério direcional daquela avaliação. Essa evidência confirma uma limitação dos gates transportados em oito tokens; não é uma nova avaliação do contraste original em 32 tokens e não prova impossibilidade universal de informação latente.

Retenção de candidatos também não é resolução. Perder o único sucesso potencial de uma tarefa pode impedir sua solução; perder um entre vários sucessos pode não modificar a cobertura. Os denominadores dos candidatos iniciados e de todos os corretos potenciais são, por isso, mantidos separados.

== O que as ablações exploratórias acrescentam

Os resultados motivaram diagnósticos de leitura: regularização mais forte, PCA e um corretor residual que parte da previsão de B. Como foram escolhidos após a inspeção do resultado principal, são exploratórios, ainda que seus ajustes internos respeitem o isolamento de tarefas.

@@EXPLORATORY_TABLE@@

PCA64 e correção residual aproximam o desempenho do baseline e reduzem a desvantagem da combinação original. Seus intervalos não demonstram superioridade sobre B. Essa observação altera a interpretação do resultado: uma leitura latente de alta dimensão pode ser sensível à parametrização. Ela justifica fixar uma representação reduzida e testar outra classe de decodificador em uma nova avaliação, sem escolher retrospectivamente o vencedor desta tabela como evidência confirmatória.

Diagnósticos de prefixos idênticos fornecem um controle complementar. Com o mesmo prompt e os mesmos tokens já processados, um scorer determinístico deve empatar as tentativas, mesmo que suas continuações tenham resultados diferentes. A consistência dessas previsões verifica a instrumentação; não revela, por si só, por que uma determinada família perde informação ou calibração.

== O espaço econômico disponível

O oracle contrafactual mantém todos os corretos e interrompe incorretos ativos, conhecendo seus labels finais. A comparação mostra que há trabalho dispensável no banco, mas não que um probe saiba identificá-lo. O espaço idealizado de economia e a capacidade empírica de aproveitá-lo são objetos distintos.

@@OLD_ORACLE_TABLE@@

Esses diagnósticos conectam a primeira à segunda etapa: há oportunidade econômica sob informação perfeita, sensibilidade à parametrização do leitor e problemas de transporte de decisões. A avaliação seguinte pergunta se uma comparação correspondente entre classes de probes, em outra família de gerador e em tarefas externas novas, modifica o incremento acessível de H.
