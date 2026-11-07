#import "../lib.typ": *
= Comparação entre modelos e decodificadores <chap-comparacao>

== Desenvolvimento, precisão e população externa

A segunda etapa examina a limitação deixada pela primeira: o resultado incremental poderia depender da família do gerador e da capacidade do probe? O piloto DeepSeek foi concluído nas doze tarefas de desenvolvimento, com 240 tentativas, oito tarefas mistas, 41,67% de acertos, 0,42% de truncamento e 0,83% de erro de parsing. Os critérios de viabilidade foram satisfeitos. Esses números descrevem desenvolvimento e não são resultados do teste externo.

O planejamento adotou cem tarefas por gerador, a partir da hierarquia técnica prevista, preservando vinte tentativas em cada.

Antes de utilizar o conjunto externo, as escolhas de hiperparâmetros produziram os procedimentos abaixo. Ambos os conjuntos de features selecionaram largura 32 para o MLP, mas os números de épocas e os limiares foram próprios de cada ajuste.

#figure(
 { set text(size: 10pt); set par(first-line-indent: 0pt, justify: false, leading: 0.4em);
 table(columns: (0.8fr, 1.3fr, 1fr, 0.6fr, 0.8fr), align: left, inset: 5pt,
 table.header([*Gerador*], [*Procedimento*], [*Seleção*], [*Épocas*], [*Limiar*]),
[DeepSeek], [B / logística], [C=0,01], [—], [0,4964],
[DeepSeek], [H+B / logística], [C=0,01], [—], [0,5368],
[DeepSeek], [B / MLP], [Largura 32], [167], [0,4783],
[DeepSeek], [H+B / MLP], [Largura 32], [65], [0,5598],
[Qwen], [B / logística], [C=0,01], [—], [0,3764],
[Qwen], [H+B / logística], [C=0,01], [—], [0,3913],
[Qwen], [B / MLP], [Largura 32], [99], [0,3725],
[Qwen], [H+B / MLP], [Largura 32], [76], [0,3962]
 ) },
 kind: table, caption: [Configurações escolhidas exclusivamente em HumanEval+. Cada procedimento usa 106 tarefas para ajuste/seleção e 26 para calibração e limiar.]
) <tab-source-probes>


Com a dispersão histórica e a frequência mista usadas como aproximação, a análise prospectiva de precisão forneceu os seguintes cenários:

#figure(
 { set text(size: 10pt); set par(first-line-indent: 0pt, justify: false, leading: 0.4em);
 table(columns: 3, align: left, inset: 5pt,
 table.header([*Ganho verdadeiro simulado*], [*Detecção positiva*], [*Largura esperada do IC95%*]),
[0,020], [10,3%], [0,124],
[0,030], [15,6%], [0,124],
[0,050], [36,1%], [0,124],
[0,075], [64,9%], [0,124]
 ) },
 kind: table, caption: [Planejamento para cem tarefas, condicional aos scores e à heterogeneidade histórica usada como aproximação.]
) <tab-cross-power>


Os valores mostram que a amostra não prometia detecção confiável de ganhos pequenos. Não são poder pós-hoc calculado a partir do efeito externo. Também não garantem que DeepSeek/MBPP reproduza a heterogeneidade de Qwen/HumanEval. As oito repetições de seleção e ajuste completos na origem Qwen apresentaram desvios das diferenças de WT-AUC de 0,036 para logística e 0,045 para MLP. Essa sensibilidade reforça o limite dos IC condicionais, sem estimar sem viés a variância do procedimento final.

As quatro mil gerações externas foram concluídas nos mesmos cem IDs. Apesar disso, o conjunto ativo aos 32 tokens é bastante diferente entre os modelos:

#figure(
 { set text(size: 10pt); set par(first-line-indent: 0pt, justify: false, leading: 0.4em);
 table(columns: 5, align: left, inset: 5pt,
 table.header([*Gerador*], [*Geradas*], [*Ativas*], [*Tarefas ativas*], [*Tarefas mistas*]),
[DeepSeek], [2000], [1943], [100], [58],
[Qwen], [2000], [863], [81], [35]
 ) },
 kind: table, caption: [Teste externo completo: cem tarefas e vinte tentativas por modelo, com elegibilidade T>32.]
) <tab-cross-population>


No Qwen, 1.137 respostas terminam até esse checkpoint; dezenove tarefas não possuem nenhuma tentativa ativa. No DeepSeek, somente 57 respostas terminam até ele. A comparação intratarefa se apoia em 35 tarefas mistas no Qwen e 58 no DeepSeek. Portanto, diferenças de AUC entre geradores não podem ser atribuídas apenas à arquitetura: a população condicionada ao comprimento também muda.

== Hipótese primária: incremento no DeepSeek com MLP

O contraste primário compara H+B/MLP e B/MLP em 32 tokens, com os procedimentos ajustados em HumanEval+ e congelados antes do teste. B/MLP alcança WT-AUC de 0,5225; H+B/MLP, 0,5183. A diferença é −0,0042, IC95% [−0,0648; 0,0548], nas 58 tarefas mistas.

O resultado é inconclusivo. O intervalo inclui zero, ganhos próximos de 0,05 e perdas próximas de 0,06. Não se demonstrou ganho incremental, deterioração ou equivalência prática para esse procedimento. Essa é uma limitação de precisão e da evidência observada, e não uma prova de que nenhum decodificador possa acessar sinal latente.

#figure(
 { set text(size: 10pt); set par(first-line-indent: 0pt, justify: false, leading: 0.4em);
 table(columns: (1.3fr, 2.5fr, 0.5fr, 0.8fr), align: left, inset: 5pt,
 table.header([*Gerador / classe*], [*ΔWT-AUC e IC95%*], [*Mistas*], [*Estatuto*]),
[DeepSeek / MLP], [−0,0042, IC95% [−0,0648; 0,0548]], [58], [Primário],
[DeepSeek / logística], [−0,0038, IC95% [−0,0668; 0,0554]], [58], [Secundário],
[Qwen / MLP], [−0,0074, IC95% [−0,0944; 0,0766]], [35], [Secundário],
[Qwen / logística], [−0,0552, IC95% [−0,1347; 0,0206]], [35], [Secundário]
 ) },
 kind: table, caption: [H+B menos B dentro da mesma classe. Bootstrap pareado de 10.000 réplicas.]
) <tab-cross-contrasts>


#figure(image("../../reports/manuscript_revision_20261107/figures/contrastes.svg", width: 100%), caption: [ΔWT-AUC H+B−B no teste MBPP+ aos 32 tokens: 58 tarefas mistas DeepSeek e 35 Qwen. O ponto destacado é primário; os demais são secundários. IC95% pareados por tarefa, condicionais aos procedimentos congelados.]) <fig-cross-model-contrasts>

== Capacidade do probe e diferenças entre geradores

Nenhum dos outros três contrastes de WT-AUC possui IC95% inteiramente positivo. No DeepSeek, as diferenças de logística e MLP são próximas de zero. No Qwen, a desvantagem pontual de H+B é maior na logística e menor no MLP. Para avaliar essa mudança sem confundir incremento absoluto e interação, reporta-se $Gamma$:

#figure(
 { set text(size: 10pt); set par(first-line-indent: 0pt, justify: false, leading: 0.4em);
 table(columns: 3, align: left, inset: 5pt,
 table.header([*Gerador*], [*Γ e IC95%*], [*Tarefas mistas*]),
[DeepSeek], [−0,0003, IC95% [−0,0313; 0,0323]], [58],
[Qwen], [0,0478, IC95% [0,0178; 0,0810]], [35]
 ) },
 kind: table, caption: [Mudança secundária do incremento de H ao passar da logística para MLP.]
) <tab-cross-interaction>


No Qwen, o intervalo secundário de $Gamma$ fica acima de zero. Isso significa que o contraste H+B−B se torna mais favorável ao trocar a classe do probe; não significa que H+B/MLP supere B/MLP. Seu incremento absoluto permanece −0,0074, IC95% [−0,0944; 0,0766]. A distinção é necessária para não transformar a redução de uma desvantagem em uma demonstração de superioridade.

O contraste entre os incrementos MLP de DeepSeek e Qwen, restrito às 28 tarefas mistas comuns, foi −0,0130, IC95% [−0,1355; 0,1125]. Essa análise mantém a tarefa como cluster compartilhado, mas muda a população em relação aos contrastes de cada modelo. É secundária e não estabelece uma diferença geral entre famílias de SLMs.

== Probabilidades, ranking e calibração

As métricas pontuais dos oito procedimentos evidenciam por que a ordenação intratarefa não esgota a avaliação:

#figure(
 { set text(size: 10pt); set par(first-line-indent: 0pt, justify: false, leading: 0.4em);
 table(columns: (0.9fr, 1.3fr, 0.9fr, 0.8fr, 0.8fr, 0.8fr), align: left, inset: 5pt,
 table.header([*Gerador*], [*Features / probe*], [*AUC agregada*], [*WT-AUC*], [*Log loss*], [*Brier*]),
[DeepSeek], [B / logística], [0,532], [0,516], [0,728], [0,267],
[DeepSeek], [H+B / logística], [0,542], [0,512], [0,723], [0,265],
[DeepSeek], [B / MLP], [0,568], [0,522], [0,726], [0,266],
[DeepSeek], [H+B / MLP], [0,574], [0,518], [0,718], [0,262],
[Qwen], [B / logística], [0,735], [0,660], [0,638], [0,223],
[Qwen], [H+B / logística], [0,721], [0,604], [0,641], [0,225],
[Qwen], [B / MLP], [0,746], [0,640], [0,624], [0,217],
[Qwen], [H+B / MLP], [0,711], [0,632], [0,652], [0,230]
 ) },
 kind: table, caption: [Métricas pontuais do teste externo aos 32 tokens. Populações ativas próprias de cada modelo.]
) <tab-cross-metrics>


Os contrastes de perdas usam todas as tarefas com candidatos ativos, inclusive homogêneas: cem no DeepSeek e 81 no Qwen. Eles não utilizam o mesmo denominador das AUCs intratarefa.

#figure(
 { set text(size: 10pt); set par(first-line-indent: 0pt, justify: false, leading: 0.4em);
 table(columns: (1.1fr, 2.1fr, 2.1fr), align: left, inset: 5pt,
 table.header([*Gerador / classe*], [*Δlog loss e IC95%*], [*ΔBrier e IC95%*]),
[DeepSeek / logística], [−0,0047, IC95% [−0,0080; −0,0016]], [−0,0023, IC95% [−0,0038; −0,0008]],
[DeepSeek / MLP], [−0,0075, IC95% [−0,0138; −0,0014]], [−0,0036, IC95% [−0,0066; −0,0007]],
[Qwen / logística], [0,0037, IC95% [−0,0084; 0,0156]], [0,0019, IC95% [−0,0041; 0,0077]],
[Qwen / MLP], [0,0278, IC95% [0,0085; 0,0468]], [0,0132, IC95% [0,0038; 0,0225]]
 ) },
 kind: table, caption: [Contrastes secundários H+B−B. Valores negativos indicam menor perda; IC pontuais sem ajuste de multiplicidade.]
) <tab-cross-losses>


No DeepSeek, as diferenças secundárias de log loss e Brier favorecem H+B nas duas classes. É possível melhorar probabilidades sem melhorar de forma demonstrada a ordenação entre tentativas. No Qwen, H+B/MLP apresenta maior perda probabilística que B/MLP. A inclusão de H, portanto, não produz um comportamento uniforme entre modelos ou métricas. Esses intervalos são pontuais, sem controle de multiplicidade, e não confirmam uma hipótese primária que teve resultado inconclusivo.

A calibração é examinada com os bins fixados antes do teste. Cada ponto resume um bin não vazio; a linha diagonal indica correspondência perfeita entre probabilidade e frequência. Os tamanhos dos bins diferem, e pontos ligados não constituem uma curva populacional estimada com precisão uniforme. Os valores por bin estão preservados nos artefatos. Probabilidade alta não constitui uma garantia certificada de retenção.

#figure(image("../../reports/manuscript_revision_20261107/figures/calibracao.svg", width: 100%), caption: [Calibração descritiva no MBPP+, T>32: 1.943 candidatos em cem tarefas DeepSeek e 863 em 81 Qwen. Dez bins fixos, com ponderação por tarefa; eixos em proporções e pontos sem IC. Calibradores ajustados exclusivamente na origem.]) <fig-cross-calibration>

== Replay secundário e consequência das decisões

O replay utiliza as mesmas tentativas, na ordem original, com limiares da origem. Em N=20, os resultados por política foram:

#figure(
 { set text(size: 10pt); set par(first-line-indent: 0pt, justify: false, leading: 0.4em);
 table(columns: (1.4fr, 0.7fr, 1fr, 0.9fr, 0.7fr, 0.7fr), align: left, inset: 5pt,
 table.header([*Política*], [*Resolvidas*], [*Tokens / tarefa*], [*Retenção*], [*Ab. corretos*], [*Perdidas*]),
[Vinte completas], [71], [2709,31], [100,00%], [0], [0],
[Verificador], [71], [1260,38], [100,00%], [0], [0],
[B / logística], [71], [1260,38], [100,00%], [0], [0],
[H+B / logística], [71], [1256,94], [100,00%], [0], [0],
[B / MLP], [71], [1260,38], [100,00%], [0], [0],
[H+B / MLP], [71], [1243,65], [99,45%], [1], [0],
[Oracle], [71], [304,37], [100,00%], [0], [0]
 ) },
 kind: table, caption: [DeepSeek: replay N=20 nas cem tarefas externas. Retenção entre corretos ativos potenciais.]
) <tab-deepseek-operational>


#figure(
 { set text(size: 10pt); set par(first-line-indent: 0pt, justify: false, leading: 0.4em);
 table(columns: (1.4fr, 0.7fr, 1fr, 0.9fr, 0.7fr, 0.7fr), align: left, inset: 5pt,
 table.header([*Política*], [*Resolvidas*], [*Tokens / tarefa*], [*Retenção*], [*Ab. corretos*], [*Perdidas*]),
[Vinte completas], [75], [791,88], [100,00%], [0], [0],
[Verificador], [75], [318,71], [100,00%], [0], [0],
[B / logística], [75], [282,80], [97,80%], [0], [0],
[H+B / logística], [75], [282,09], [97,48%], [0], [0],
[B / MLP], [75], [271,76], [96,23%], [0], [0],
[H+B / MLP], [75], [300,05], [99,06%], [0], [0],
[Oracle], [75], [200,37], [100,00%], [0], [0]
 ) },
 kind: table, caption: [Qwen: replay N=20 nas cem tarefas externas. Retenção entre corretos ativos potenciais.]
) <tab-qwen-operational>


“Retenção” é a fração mantida de todos os corretos ativos do banco no orçamento. “Ab. corretos” conta os corretos efetivamente interrompidos entre as tentativas iniciadas. As duas colunas não são complementares porque a parada após sucesso impede que parte do banco seja iniciada. Nenhuma política perdeu tarefas em relação ao verificador neste conjunto, mas essa ausência observada não certifica ausência de perdas futuras.

#figure(
 { set text(size: 10pt); set par(first-line-indent: 0pt, justify: false, leading: 0.4em);
 table(columns: (1.3fr, 2fr, 2fr), align: left, inset: 5pt,
 table.header([*Gerador / gate*], [*Δtokens e IC95%*], [*Δresolução e IC95%*]),
[DeepSeek / B / MLP], [0,00, IC95% [0,00; 0,00]], [0,000, IC95% [0,000; 0,000]],
[DeepSeek / H+B / MLP], [−16,73, IC95% [−35,47; −3,34]], [0,000, IC95% [0,000; 0,000]],
[Qwen / B / MLP], [−46,95, IC95% [−93,10; −16,33]], [0,000, IC95% [0,000; 0,000]],
[Qwen / H+B / MLP], [−18,66, IC95% [−40,58; −4,53]], [0,000, IC95% [0,000; 0,000]]
 ) },
 kind: table, caption: [Diferenças pareadas contra parada por verificador. Resultados operacionais secundários.]
) <tab-cross-operational-effects>


No DeepSeek, H+B/MLP reduz o custo médio de 1.260,38 para 1.243,65 tokens por tarefa, mantendo 71 resolvidas no banco. A economia pontual é aproximadamente 1,33%: são 7 abortos entre 754 tentativas iniciadas, um deles correto. B/MLP não interrompe nenhuma das 752 tentativas iniciadas e reproduz a parada por verificador. A pequena economia secundária de H+B precisa ser comparada aos custos de adquirir e executar o sinal.

No Qwen, B/MLP e H+B/MLP preservam as 75 tarefas resolvidas, mas têm custos médios de 271,76 e 300,05 tokens. H+B/MLP mantém maior fração dos corretos ativos potenciais, enquanto B/MLP gera menos tokens. Esse compromisso não é resumido por AUC, e as metas de retenção da origem não escolhem automaticamente a preferência econômica no alvo.

Incluindo os gastos das tarefas não resolvidas, o custo agregado por tarefa resolvida no DeepSeek é 1775,18 tokens com verificador e 1751,62 com H+B/MLP. No Qwen, os valores são 424,95 com verificador, 362,35 com B/MLP e 400,07 com H+B/MLP. Trata-se de uma razão agregada, e não do custo médio apenas das tarefas que tiveram sucesso.

O oracle resolve as mesmas tarefas que o verificador com 304,37 tokens médios no DeepSeek e 200,37 no Qwen. A distância aos gates é um limite contrafactual sob conhecimento dos labels, não uma promessa de ganho alcançável. O replay completo de N=1, 5, 10 e 20 está preservado nos artefatos; nenhuma leitura secundária modifica a hipótese primária.

== Recursos medidos e síntese da etapa

#figure(
 { set text(size: 10pt); set par(first-line-indent: 0pt, justify: false, leading: 0.4em);
 table(columns: 4, align: left, inset: 5pt,
 table.header([*Coleta*], [*Tentativas*], [*Horas ativas*], [*Pico reservado / GiB*]),
[Piloto DeepSeek], [240], [1,46], [3,63],
[Técnico Qwen], [12], [0,05], [4,63],
[Origem DeepSeek], [2640], [18,83], [3,63],
[Externo DeepSeek], [2000], [9,54], [3,38],
[Externo Qwen], [2000], [3,61], [4,64]
 ) },
 kind: table, caption: [Sessões da extensão final na RTX 3050 Laptop. Valores de parede ativa e memória do PyTorch; treino dos probes em CPU.]
) <tab-cross-resources>


A etapa consumiu 33,49 horas de sessão ativa, dentro do orçamento de sessenta. O tempo de parede bruto foi 43,17 horas, incluindo 9,68 horas de suspensão do sistema, excluídas por evidência do Windows e preservadas nos registros originais. Sessão ativa inclui carregamento e verificação; não é tempo de kernels CUDA. Medições de score em lote na CPU também não estimam, por si só, o overhead de uma política online.

A síntese é delimitada: a mudança de família e o MLP não produziram ganho de WT-AUC demonstrado nos contrastes testados; a parametrização e a métrica escolhida modificam outros aspectos do desempenho. O estudo não estabelece impossibilidade de sinal interno, equivalência entre features ou superioridade econômica geral. Esses limites orientam a discussão conjunta das duas etapas.
