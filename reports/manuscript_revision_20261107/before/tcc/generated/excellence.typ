#import "../lib.typ": *
= Diagnósticos, limite econômico e validação independente

== Perguntas explicativas e desenho das ablações

Esta extensão foi definida em 17/10/2026, depois da conclusão e inspeção dos
resultados anteriores. Os diagnósticos não substituem o endpoint original. A avaliação
independente subsequente tem hipótese e congelamento próprios. Seu objetivo é distinguir explicações compatíveis com a desvantagem
de H+B, examinar decisões do gate transferido e quantificar a oportunidade
econômica de uma interrupção idealizada.

Mantiveram-se os cinco folds externos originais, com tarefas inteiras. Cada
ablação de observáveis seleciona C nos três folds internos. A penalização
conjunta forte usa C em 0,00001, 0,0001 e 0,001. PCA compara 16 e 64 componentes,
com camada e C escolhidos no treino; imputation, padronização e PCA são refeitos
dentro de cada split. A padronização e o classificador ponderam igualmente as
tarefas; PCA utiliza as observações de treino, sem rótulos, e whitening.

A calibração Platt recebe scores produzidos por cross-fitting de toda a seleção.
O residual usa uma probabilidade de B cujo ajuste, seleção e calibrador excluem
o rótulo do candidato pontuado. Dentro da seleção residual, esse primeiro estágio
é novamente refeito apenas no respectivo treino. Seu preditor é

$ hat(p)_"res"=sigma("logit"(hat(p)_B)+w_H^top tilde(H)). $

O offset fica fixo durante o ajuste L2 de w; não há segundo Platt. Esse método
ancora B e restringe a correção, mas a comparação não identifica separadamente
todos os efeitos de parametrização e calibração. Todos os intervalos adicionais
são bootstrap pontual de 2.000 tarefas com seed 20260921, condicionais aos modelos
ajustados; não há cobertura simultânea nem promoção de vencedores à hipótese principal.

== Componentes do baseline e leituras latentes alternativas

#figure(
 { set text(size: 10pt); set par(leading: 0.34em, first-line-indent: 0pt); table(columns: 4, align: left,
 table.header([*Família*], [*WT-AUC*], [*Log loss*], [*Δlog loss contra B e IC95%*]),
[B_original], [0,6282], [0,6302], [0,0000 [0,000; 0,000]],
[HB_original], [0,5811], [0,6681], [0,0379 [0,007; 0,071]],
[HB_pca16], [0,6240], [0,6334], [0,0032 [-0,010; 0,017]],
[HB_pca64], [0,6298], [0,6318], [0,0016 [-0,012; 0,016]],
[HB_stronger], [0,5476], [0,6764], [0,0462 [0,013; 0,081]],
[H_original], [0,5681], [0,6869], [0,0566 [0,024; 0,090]],
[H_residual_B], [0,6345], [0,6289], [-0,0013 [-0,006; 0,003]],
[confidence_only], [0,6242], [0,6316], [0,0013 [-0,017; 0,019]],
[entropy_pair], [0,6036], [0,6307], [0,0005 [-0,020; 0,020]],
[last_nll], [0,5023], [0,6997], [0,0695 [0,035; 0,104]],
[logit_margin], [0,6311], [0,6564], [0,0261 [0,002; 0,049]],
[max_entropy], [0,5146], [0,6562], [0,0260 [-0,004; 0,060]],
[mean_entropy], [0,6128], [0,6384], [0,0082 [-0,010; 0,026]],
[mean_nll], [0,6378], [0,6564], [0,0262 [0,001; 0,050]],
[nll_pair], [0,6404], [0,6565], [0,0263 [0,001; 0,050]],
[prompt_only], [0,5000], [0,6741], [0,0439 [0,018; 0,069]]
 ) },
 kind: table,
 caption: [Predições em tarefas externas no checkpoint de 32 tokens. Comparações exploratórias; sinais positivos de perda representam piora.]
) <tab-excellence-models>

O ranking obtido com NLL média ou margem isoladas é próximo ao de B na
estimativa pontual, enquanto sua perda probabilística é maior. Portanto,
identificar componentes com bom ranking não basta para justificar um limiar
absoluto de interrupção. A combinação das estatísticas beneficia a qualidade
probabilística neste regime. Prompt-only tem WT-AUC igual a 0,5 por construção.

PCA e o residual reduzem a desvantagem do H+B original; os intervalos contra B
continuam incluindo zero. Penalização conjunta mais forte, sozinha, não resolve
o problema na família testada. Esses resultados são compatíveis com sensibilidade
à parametrização e ao custo de estimação, sem demonstrar uma causa única ou
ausência universal de informação latente incremental.

#figure(image("../../reports/excellence/print_ready/models/figures/model_ablation.svg", width: 100%),
 caption: [Ablações e leituras alternativas com seleção agrupada. À direita, diferenças de perda e intervalos pontuais; todas as famílias são mostradas.]) <fig-excellence-models>

== Geometria e sensibilidade ao número de tarefas

Os espectros descritivos usam vetores brutos, distintos dos padronizados na
ablação preditiva, e são ajustados exclusivamente no treino externo. Fração de variância
explicada não mede informação sobre corretude. Direções só são comparadas entre
folds que escolheram a mesma camada, depois de desfazer a escala das coordenadas.
Sobreposição dos treinos impede tratar seus ângulos como replicações independentes.

Na camada comum, os cossenos das direções variam de 0,716 a 0,745. A correlação entre escores H e B centrados por tarefa é 0,170. Correlação pequena não demonstra independência condicional, e direções
distintas não garantem informação útil adicional para o rótulo. A predição
linear do componente H+B latente a partir de B apresenta R² baixo ou negativo
em alguns folds, sem estabelecer redundância completa.

A curva de tarefas usa algoritmos explicitamente fixos: C=0,01, camada 13 e
PCA16 para a combinação. Subconjuntos de tarefas são definidos por hash dentro
do treino externo, com aproximadamente 32, 63 e 105 tarefas. Ela não representa
a curva do procedimento original completo nem estima um tamanho amostral ótimo.
WT-AUC não cresce monotonamente neste diagnóstico; aumentar a amostra não é
demonstrado como solução suficiente para a desvantagem original.

#figure(image("../../reports/excellence/print_ready/geometry/geometry_learning.svg", width: 100%),
 caption: [Variância de treino por camada e sensibilidade de algoritmos fixos à quantidade de tarefas. Não são efeitos causais da dimensão ou do tamanho amostral.]) <fig-excellence-geometry>

=== Prefixos compartilhados e divergência

O teste de prefixos reagrupa as tentativas pelo prompt e IDs de tokens
observados, e confirma constância dos scores em cada grupo idêntico. Em t=8,
15 das 100 tarefas mistas HumanEval+ e três das 24 mistas MBPP+ têm um único
prefixo; nelas, leitores determinísticos não distinguem os resultados.
Em t=32, nenhuma das 92 tarefas mistas principais está nessa condição.
Portanto, o colapso integral de cada tarefa em um único prefixo não explica
o contraste negativo principal. Colisões parciais limitam alguns pares, sem
identificar sozinhas a causa da deterioração no alvo. As contagens por tarefa
e checkpoint estão exportadas, sem escolher casos pelos resultados do probe.

== Anatomia da deterioração do gate em MBPP+

A análise distingue origem OOF, desenvolvimento de origem utilizado na escolha
do gate e aplicação do bundle completo ao alvo. Reconstituir o desenvolvimento
reproduz sua retenção aparente de 0,9715 para H+B em t=8; o calibrador e o limiar
foram ajustados nesses rótulos, portanto esse valor não é validação externa.

No alvo, H+B em t=8 mantém 276 dos 366 corretos potenciais e descarta 90,
distribuídos por 12 tarefas; cinco concentram 60 descartes. Os corretos têm
mediana de score próxima de 0,373 e quartil inferior próximo de 0,286, com limiar
0,2695. O ranking intratarefa nesse ponto é 0,494, com intervalo amplo. Mudança
da prevalência e dos comprimentos acompanha a diferença de scores, mas os
procedimentos de ajuste também diferem entre origem e alvo: nenhuma comparação
isola por si só uma causa de domínio.

Mbpp/166 perde seus oito sucessos e Mbpp/259 perde seu único sucesso no banco,
explicando duas tarefas a menos resolvidas. Entre corretos ativos iniciados,
23 de 48 são descartados; entre todos os corretos potenciais, são 90 de 366.
Essa seleção pelo caminho de execução explica por que as duas taxas respondem
perguntas diferentes. As curvas no alvo são diagnósticos retrospectivos; não
fornecem um limiar reajustado ou uma nova política congelada.

#figure(image("../../reports/excellence/print_ready/transfer/retention_curves.svg", width: 100%),
 caption: [Retenção versus limiar nas três avaliações distintas. Linhas verticais indicam gates escolhidos na origem, sem seleção sobre MBPP+.]) <fig-excellence-retention>

#figure(image("../../reports/excellence/print_ready/transfer/conditional_scores.svg", width: 100%),
 caption: [Distribuições por resultado funcional no gate H+B de 8 tokens. São densidades por candidato; o desenvolvimento não é avaliação externa.]) <fig-excellence-score>

== Potencial econômico e distância ao oracle

O oracle conserva a mesma ordem, N e checkpoint e conhece os rótulos finais.
Preserva todos os corretos, interrompe os incorretos ativos e para no primeiro
sucesso. As condições matemáticas e a derivação estão na fundamentação. A
implementação independente do replay também reproduziu os custos, resolução,
tentativas iniciadas e descartes das políticas originais, tarefa por tarefa.

#figure(
 { set text(size: 10pt); set par(leading: 0.34em, first-line-indent: 0pt); table(columns: 4, align: left,
 table.header([*Banco*], [*t*], [*Tokens oracle/tarefa*], [*Economia contra verificador e IC95%*]),
[HumanEval+], [8], [104,0], [0,7775 [0,705; 0,823]],
[HumanEval+], [16], [141,6], [0,6970 [0,617; 0,750]],
[HumanEval+], [32], [212,8], [0,5448 [0,446; 0,617]],
[HumanEval+], [64], [307,9], [0,3411 [0,234; 0,428]],
[MBPP+], [8], [66,5], [0,7625 [0,574; 0,836]],
[MBPP+], [16], [98,0], [0,6500 [0,460; 0,739]],
[MBPP+], [32], [156,0], [0,4426 [0,239; 0,562]],
[MBPP+], [64], [227,4], [0,1876 [0,057; 0,302]]
 ) },
 kind: table,
 caption: [Oracle retrospectivo, N=20. Sua resolução coincide exatamente com a do verificador; economia é razão de médias por tarefa.]
) <tab-excellence-oracle>
Em HumanEval+, 71,4% da economia oracle em t=8 vem de tarefas não resolvidas no banco. Em MBPP+, 86,1% da economia oracle em t=8 vem de tarefas não resolvidas no banco. Isso delimita onde está grande parte da oportunidade: evitar prolongar
tentativas que, retrospectivamente, não produzem solução. Acesso a esse rótulo
não existe online. Uma política que descarta sucessos pode até custar menos que
este oracle em um caminho, ao custo de perder cobertura; ele não é limite global
de todas as políticas possíveis.

#figure(image("../../reports/excellence/print_ready/economics/oracle_frontier.svg", width: 100%),
 caption: [Custo e resolução para N=1,5,10,20 no checkpoint 8. B e H+B usam limiares específicos desse checkpoint escolhidos somente na origem; o oracle é idealizado.]) <fig-excellence-oracle>

== Medições técnicas e avaliação independente

O benchmark contém 50 pares, 100 gerações e 26 tarefas conhecidas. Tokens e estados coincidiram exatamente. A GPU conectada é NVIDIA GeForce RTX 3050 6GB Laptop GPU; pico reservado 4978 MiB. Os tempos médios de geração instrumentada foram 8,764 s na captura original e 8,475 s na seletiva. São tempos descritivos, parcialmente concorrentes com análises CPU;
não constituem inferência de aceleração da política. Os microbenchmarks separados
de gate, prefill e verificador estão nos artefatos. A RTX 4060 mencionada na
solicitação não está disponível; seu desempenho não é extrapolado.


=== Latência completa em tarefas conhecidas

Foram medidas 72 execuções em seis tarefas do piloto já examinado, três repetições e até três tentativas por tarefa, com a ordem dos quatro modos alternada. O plano precedeu as medições; não usa a nova reserva. Warmup dos caminhos e dos oracles do verificador e carregamento dos pesos foram excluídos. Tokens, estados capturados e rótulos funcionais foram comparados ao banco conhecido.

O somatório dos tempos completos de tarefa foi 33,6 minutos. O orçamento temporal de 30 minutos é verificado antes de cada bloco de quatro modos; o bloco final é concluído mesmo quando ultrapassa esse valor. Não se trata de um limite rígido por processo.

#figure(
 { set text(size: 10pt); set par(leading: 0.34em, first-line-indent: 0pt); table(columns: 5, align: left,
 table.header([*Modo*], [*Tempo total médio (s)*], [*Mín–máx (s)*], [*Verificador RPC (s)*], [*Resoluções/execuções*]),
[verifier plain], [32,570], [2,714–112,960], [0,281], [9/18],
[verifier selective], [32,595], [2,733–112,842], [0,281], [9/18],
[B], [14,441], [2,721–29,622], [0,100], [9/18],
[HB], [32,549], [2,731–112,963], [0,280], [9/18]
 ) },
 kind: table,
 caption: [Tempos de ponta a ponta na GPU efetivamente conectada, N=3. Variabilidade descritiva, sem teste confirmatório de aceleração.]
) <tab-operational-latency>
Os arquivos por tarefa decompõem geração, prefill, cópia seletiva, classificação, parsing e RPC do verificador. Prefill, cópia e classificação são partes da geração medida; não se somam novamente a ela. Mesmo o modo sem probe utiliza o sampler instrumentado da pesquisa. Orçamento, tarefas conhecidas e cache aquecido diferem do replay N=20: não se extrapola uma economia de latência, energia ou desempenho na RTX 4060.


== Validação independente aprovada

Depois dos diagnósticos, o estudante aprovou explicitamente a proposta e informou
que o orientador já a havia revisado. Esse relato, sem assinatura criada em nome
do orientador, foi vinculado aos hashes do protocolo, desenho e implementação.
Um snapshot prospectivo de 46 arquivos precede a primeira trajetória nova.
O cabeçalho da proposta registra seu estado histórico de elaboração; o arquivo
de aprovação e o manifesto de congelamento registram a autorização subsequente.
Não se reivindica pré-registro externo nem se modifica a autorização histórica v0.2.

A hipótese única compara retenção de candidatos corretos por H+B e B no mesmo
checkpoint de oito tokens. Os limiares de origem são 0,26953905544150975 e
0,3166734153299228, respectivamente, sem ajuste nos novos dados. A reserva por
hash exclui tarefas com resultados locais anteriores e o controle Mbpp/2.
Vinte tarefas técnicas disjuntas verificam a implementação; suas 400 tentativas
não entram na inferência principal. A amostra principal contém 120 novas tarefas
MBPP+ e 2.400 tentativas completas, mesmo após sucessos. A população é T>8;
o denominador comum contém todos os candidatos corretos ativos, inclusive os
que não seriam iniciados no replay. Não se usa a média das retenções por tarefa.

O estimando é a diferença das razões das contagens esperadas por tarefa:
$ theta = r_(H+B)-r_B. $
Reamostram-se tarefas inteiras, mantendo o pareamento, 10.000 vezes com seed
20261017. O único critério de suporte é limite superior do IC bilateral de
95% estritamente negativo. Mais de 1% de replicações sem denominador positivo
tornaria o endpoint não identificável. Não houve troca de tarefa, seleção de
limiar, expansão por resultado ou parada por significância.

Em T>8 restam 2400 candidatos, 74 tarefas mistas e 1192 corretos ativos. A estimativa de θ é -0,1636, IC95% [-0,240; -0,094]. Houve 0 replicações indefinidas. O critério direcional foi satisfeito: o limite superior do intervalo é negativo.

#figure(
 { set text(size: 10pt); set par(leading: 0.34em, first-line-indent: 0pt); table(columns: 4, align: left,
 table.header([*Gate*], [*Corretos retidos/ativos*], [*Retenção*], [*IC95%*]),
[H+B], [986/1192], [0,8272], [[0,750; 0,897]],
[B], [1181/1192], [0,9908], [[0,983; 0,997]]
 ) },
 kind: table,
 caption: [Retenções na nova amostra. Seus intervalos absolutos são descritivos; a confirmação usa somente o contraste pareado.]
) <tab-independent-recall>

#figure(image("../../reports/excellence/independent_v1/independent_retention.svg", width: 100%),
 caption: [Nova reserva de 120 tarefas MBPP+, gates de origem em t=8: retenções de corretos ativos e contraste pareado H+B−B, IC95% por tarefa. A linha em 0,97 é meta empírica do desenvolvimento, sem garantia de transporte.]) <fig-independent-recall>

=== Resultados secundários e limites da confirmação

#figure(
 { set text(size: 10pt); set par(leading: 0.34em, first-line-indent: 0pt); table(columns: 5, align: left,
 table.header([*Preditor*], [*AUC*], [*WT-AUC*], [*Log loss*], [*Brier*]),
[B], [0,5087], [0,5368], [0,7002], [0,2534],
[H+B], [0,4577], [0,4605], [0,7883], [0,2899]
 ) },
 kind: table,
 caption: [Métricas secundárias na população ativa. Discriminação não equivale a calibração ou utilidade operacional.]
) <tab-independent-metrics>
#figure(
 { set text(size: 10pt); set par(leading: 0.34em, first-line-indent: 0pt); table(columns: 4, align: left,
 table.header([*Política*], [*Tarefas resolvidas*], [*Tokens médios*], [*Falsos abortos iniciados*]),
[fixed], [94/120], [873,26], [0],
[verifier], [94/120], [342,99], [0],
[B], [94/120], [342,38], [1],
[H+B], [88/120], [334,85], [64],
[oracle], [94/120], [77,36], [0]
 ) },
 kind: table,
 caption: [Replay secundário N=20; apenas tentativas iniciadas geram custo. Oracle conhece os rótulos e preserva todos os corretos.]
) <tab-independent-replay>

#figure(image("../../reports/excellence/independent_v1/independent_secondary.svg", width: 100%),
 caption: [Nova reserva MBPP+, gates de origem em t=8: custo e resolução para N=1,5,10,20 e calibração com dez bins de largura 0,1, ponderada por tarefa. Resultados secundários descritivos, sem seleção de política no alvo.]) <fig-independent-secondary>

Os CSVs publicam todos os orçamentos N=1,5,10,20, custos, perdas de cobertura,
retenções, calibração e denominadores dos abortos iniciados. A auditoria reabriu
as ativações de ambas as fases, conferiu todas as identidades e seeds e reaplicou
os scorers congelados. A hipótese nova versa sobre transporte relativo de decisões;
seu suporte não confirma ganho incremental de representações, os critérios Q2/H3
originais, uma causa de domain shift ou superioridade universal de um método.

São novas tarefas sem resultados locais anteriores, não tarefas certificadamente
ausentes do pré-treinamento. O bootstrap pressupõe intercambiabilidade entre
tarefas da população; a reserva determinística por hash não é um censo nem
valida outros modelos ou benchmarks. Os intervalos secundários são pontuais.
O replay mede tokens e resolução, sem demonstrar economia de latência, energia
ou FLOPs. Nenhum resultado retrospectivo substituiu o critério aprovado.
Há 94 tarefas com algum correto ativo e 26 sem correto ativo. 33 tarefas contêm decisões discordantes dos gates. A faixa das estimativas com uma tarefa retirada é [-0,168; -0,150]; são diagnósticos condicionais aos scorers, sem alterar a amostra ou testar significância de cada exclusão.

Uma sensibilidade posterior agrupa os pares de prompts sinalizados pela auditoria textual: 118 clusters, estimativa -0,1636 e IC95% [-0,240; -0,095]. Os pares são variantes de máximo/mínimo sobre tuplas ou listas heterogêneas, com exemplos compartilhados. Esse agrupamento heurístico não substitui o bootstrap por tarefa aprovado nem certifica independência semântica.
