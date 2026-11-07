#import "../lib.typ": *
= Comparação entre modelos e decodificadores <chap-comparacao>

== Desenvolvimento, precisão e população externa

A segunda etapa examina a limitação deixada pela primeira: o resultado incremental poderia depender da família do gerador e da capacidade do probe? O piloto DeepSeek foi concluído nas doze tarefas de desenvolvimento, com 240 tentativas, oito tarefas mistas, 41,67% de acertos, 0,42% de truncamento e 0,83% de erro de parsing. Os critérios de viabilidade foram satisfeitos. Esses números descrevem desenvolvimento e não são resultados do teste externo.

O planejamento adotou cem tarefas por gerador, a partir da hierarquia técnica prevista, preservando vinte tentativas em cada.

Antes de utilizar o conjunto externo, as escolhas de hiperparâmetros produziram os procedimentos abaixo. Ambos os conjuntos de features selecionaram largura 32 para o MLP, mas os números de épocas e os limiares foram próprios de cada ajuste.

@@FINAL_PROBE_TABLE@@

Com a dispersão histórica e a frequência mista usadas como aproximação, a análise prospectiva de precisão forneceu os seguintes cenários:

@@POWER_TABLE@@

Os valores mostram que a amostra não prometia detecção confiável de ganhos pequenos. Não são poder pós-hoc calculado a partir do efeito externo. Também não garantem que DeepSeek/MBPP reproduza a heterogeneidade de Qwen/HumanEval. As oito repetições de seleção e ajuste completos na origem Qwen apresentaram desvios das diferenças de WT-AUC de @@TRAIN_SD_LINEAR@@ para logística e @@TRAIN_SD_MLP@@ para MLP. Essa sensibilidade reforça o limite dos IC condicionais, sem estimar sem viés a variância do procedimento final.

As quatro mil gerações externas foram concluídas nos mesmos cem IDs. Apesar disso, o conjunto ativo aos 32 tokens é bastante diferente entre os modelos:

@@FINAL_POPULATION_TABLE@@

No Qwen, 1.137 respostas terminam até esse checkpoint; dezenove tarefas não possuem nenhuma tentativa ativa. No DeepSeek, somente 57 respostas terminam até ele. A comparação intratarefa se apoia em 35 tarefas mistas no Qwen e 58 no DeepSeek. Portanto, diferenças de AUC entre geradores não podem ser atribuídas apenas à arquitetura: a população condicionada ao comprimento também muda.

== Hipótese primária: incremento no DeepSeek com MLP

O contraste primário compara H+B/MLP e B/MLP em 32 tokens, com os procedimentos ajustados em HumanEval+ e congelados antes do teste. B/MLP alcança WT-AUC de @@DS_B_MLP_WT@@; H+B/MLP, @@DS_HB_MLP_WT@@. A diferença é @@PRIMARY_DELTA@@, IC95% @@PRIMARY_CI@@, nas @@PRIMARY_MIXED@@ tarefas mistas.

O resultado é inconclusivo. O intervalo inclui zero, ganhos próximos de 0,05 e perdas próximas de 0,06. Não se demonstrou ganho incremental, deterioração ou equivalência prática para esse procedimento. Essa é uma limitação de precisão e da evidência observada, e não uma prova de que nenhum decodificador possa acessar sinal latente.

@@FINAL_CONTRAST_TABLE@@

#figure(image("../../reports/manuscript_revision_20261107/figures/contrastes.svg", width: 100%), caption: [ΔWT-AUC H+B−B no teste MBPP+ aos 32 tokens: 58 tarefas mistas DeepSeek e 35 Qwen. O ponto destacado é primário; os demais são secundários. IC95% pareados por tarefa, condicionais aos procedimentos congelados.]) <fig-cross-model-contrasts>

== Capacidade do probe e diferenças entre geradores

Nenhum dos outros três contrastes de WT-AUC possui IC95% inteiramente positivo. No DeepSeek, as diferenças de logística e MLP são próximas de zero. No Qwen, a desvantagem pontual de H+B é maior na logística e menor no MLP. Para avaliar essa mudança sem confundir incremento absoluto e interação, reporta-se $Gamma$:

@@INTERACTION_TABLE@@

No Qwen, o intervalo secundário de $Gamma$ fica acima de zero. Isso significa que o contraste H+B−B se torna mais favorável ao trocar a classe do probe; não significa que H+B/MLP supere B/MLP. Seu incremento absoluto permanece @@QW_MLP_DELTA@@, IC95% @@QW_MLP_CI@@. A distinção é necessária para não transformar a redução de uma desvantagem em uma demonstração de superioridade.

O contraste entre os incrementos MLP de DeepSeek e Qwen, restrito às 28 tarefas mistas comuns, foi @@COMMON_DELTA@@, IC95% @@COMMON_CI@@. Essa análise mantém a tarefa como cluster compartilhado, mas muda a população em relação aos contrastes de cada modelo. É secundária e não estabelece uma diferença geral entre famílias de SLMs.

== Probabilidades, ranking e calibração

As métricas pontuais dos oito procedimentos evidenciam por que a ordenação intratarefa não esgota a avaliação:

@@FINAL_METRICS_TABLE@@

Os contrastes de perdas usam todas as tarefas com candidatos ativos, inclusive homogêneas: cem no DeepSeek e 81 no Qwen. Eles não utilizam o mesmo denominador das AUCs intratarefa.

@@FINAL_LOSS_TABLE@@

No DeepSeek, as diferenças secundárias de log loss e Brier favorecem H+B nas duas classes. É possível melhorar probabilidades sem melhorar de forma demonstrada a ordenação entre tentativas. No Qwen, H+B/MLP apresenta maior perda probabilística que B/MLP. A inclusão de H, portanto, não produz um comportamento uniforme entre modelos ou métricas. Esses intervalos são pontuais, sem controle de multiplicidade, e não confirmam uma hipótese primária que teve resultado inconclusivo.

A calibração é examinada com os bins fixados antes do teste. Cada ponto resume um bin não vazio; a linha diagonal indica correspondência perfeita entre probabilidade e frequência. Os tamanhos dos bins diferem, e pontos ligados não constituem uma curva populacional estimada com precisão uniforme. Os valores por bin estão preservados nos artefatos. Probabilidade alta não constitui uma garantia certificada de retenção.

#figure(image("../../reports/manuscript_revision_20261107/figures/calibracao.svg", width: 100%), caption: [Calibração descritiva no MBPP+, T>32: 1.943 candidatos em cem tarefas DeepSeek e 863 em 81 Qwen. Dez bins fixos, com ponderação por tarefa; eixos em proporções e pontos sem IC. Calibradores ajustados exclusivamente na origem.]) <fig-cross-calibration>

== Replay secundário e consequência das decisões

O replay utiliza as mesmas tentativas, na ordem original, com limiares da origem. Em N=20, os resultados por política foram:

@@DS_OPERATIONAL_TABLE@@

@@QW_OPERATIONAL_TABLE@@

“Retenção” é a fração mantida de todos os corretos ativos do banco no orçamento. “Ab. corretos” conta os corretos efetivamente interrompidos entre as tentativas iniciadas. As duas colunas não são complementares porque a parada após sucesso impede que parte do banco seja iniciada. Nenhuma política perdeu tarefas em relação ao verificador neste conjunto, mas essa ausência observada não certifica ausência de perdas futuras.

@@OPERATIONAL_EFFECT_TABLE@@

No DeepSeek, H+B/MLP reduz o custo médio de 1.260,38 para 1.243,65 tokens por tarefa, mantendo 71 resolvidas no banco. A economia pontual é aproximadamente 1,33%: são @@DS_HB_MLP_ABORTS@@ abortos entre @@DS_HB_MLP_STARTED@@ tentativas iniciadas, um deles correto. B/MLP não interrompe nenhuma das @@DS_B_MLP_STARTED@@ tentativas iniciadas e reproduz a parada por verificador. A pequena economia secundária de H+B precisa ser comparada aos custos de adquirir e executar o sinal.

No Qwen, B/MLP e H+B/MLP preservam as 75 tarefas resolvidas, mas têm custos médios de 271,76 e 300,05 tokens. H+B/MLP mantém maior fração dos corretos ativos potenciais, enquanto B/MLP gera menos tokens. Esse compromisso não é resumido por AUC, e as metas de retenção da origem não escolhem automaticamente a preferência econômica no alvo.

Incluindo os gastos das tarefas não resolvidas, o custo agregado por tarefa resolvida no DeepSeek é @@DS_VERIFIER_COST_PER_SOLUTION@@ tokens com verificador e @@DS_HB_MLP_COST_PER_SOLUTION@@ com H+B/MLP. No Qwen, os valores são @@QW_VERIFIER_COST_PER_SOLUTION@@ com verificador, @@QW_B_MLP_COST_PER_SOLUTION@@ com B/MLP e @@QW_HB_MLP_COST_PER_SOLUTION@@ com H+B/MLP. Trata-se de uma razão agregada, e não do custo médio apenas das tarefas que tiveram sucesso.

O oracle resolve as mesmas tarefas que o verificador com 304,37 tokens médios no DeepSeek e 200,37 no Qwen. A distância aos gates é um limite contrafactual sob conhecimento dos labels, não uma promessa de ganho alcançável. O replay completo de N=1, 5, 10 e 20 está preservado nos artefatos; nenhuma leitura secundária modifica a hipótese primária.

== Recursos medidos e síntese da etapa

@@RESOURCE_TABLE@@

A etapa consumiu @@ACTIVE_HOURS@@ horas de sessão ativa, dentro do orçamento de sessenta. O tempo de parede bruto foi @@RAW_HOURS@@ horas, incluindo @@SLEEP_HOURS@@ horas de suspensão do sistema, excluídas por evidência do Windows e preservadas nos registros originais. Sessão ativa inclui carregamento e verificação; não é tempo de kernels CUDA. Medições de score em lote na CPU também não estimam, por si só, o overhead de uma política online.

A síntese é delimitada: a mudança de família e o MLP não produziram ganho de WT-AUC demonstrado nos contrastes testados; a parametrização e a métrica escolhida modificam outros aspectos do desempenho. O estudo não estabelece impossibilidade de sinal interno, equivalência entre features ou superioridade econômica geral. Esses limites orientam a discussão conjunta das duas etapas.
