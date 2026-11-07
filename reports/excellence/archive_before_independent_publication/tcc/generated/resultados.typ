#import "../lib.typ": *
= Resultados do desenho fixo

== Amostragem, elegibilidade e proveniência

O piloto e o banco principal são separados por tarefa. A coleta mantém todas
as 20 tentativas de cada tarefa principal, inclusive após um sucesso, para
permitir replay pareado. No endpoint primário há 2158
tentativas elegíveis, 132 tarefas e 92
tarefas mistas. Tarefas homogêneas contribuem para perdas e resolução, mas
não identificam AUC intratarefa. A população ativa muda entre checkpoints:
as curvas temporais não acompanham uma amostra invariável.

#figure(image("../../reports/protocol_v0.2/figures/risk_set_attrition.svg", width: 100%),
 caption: [Elegibilidade T > t e composição das tarefas. Checkpoint zero usa todos os prompts.]) <fig-risk-sets>

== Predição e ganho incremental no endpoint primário

#figure(
 table(columns: 5, align: left,
 table.header([*Preditor*], [*AUC*], [*WT-AUC*], [*IC 95% WT-AUC*], [*Log loss*]),
[B], [0,701], [0,628], [[0,583; 0,675]], [0,630],
[H], [0,592], [0,568], [[0,521; 0,614]], [0,687],
[H+B], [0,646], [0,581], [[0,534; 0,629]], [0,668]
 ),
 kind: table,
 caption: [Predições externas em 32 tokens; cada tarefa recebe o mesmo peso total.]
) <tab-main-primary>

A diferença H+B menos B em WT-AUC é -0,047, IC 95% [-0,099; 0,006], e em log loss é 0,038, IC 95% [0,007; 0,071].
Esse resultado não satisfaz o critério conjunto de Q2. O intervalo de perda permanece acima de zero: H+B apresenta maior perda que B neste contraste.
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

Os critérios conjuntos para afirmar benefício operacional não foram satisfeitos. O critério exige não inferioridade de resolução com margem de
três pontos percentuais e economia de tokens, ambos contra verificador e B,
além de suporte a Q2 e validação online com aborto efetivo. Uma falha em
qualquer condição impede afirmar benefício do sistema.

#figure(
 table(columns: 5, align: left,
 table.header([*N*], [*Política*], [*Resolução*], [*Tokens/tarefa*], [*Aborto falso ativo*]),
[1], [B], [0,462], [68,7], [0,033],
[1], [H+B], [0,470], [70,2], [0,019],
[1], [verifier], [0,477], [72,2], [0,000],
[5], [B], [0,659], [190,0], [0,146],
[5], [H+B], [0,674], [192,0], [0,096],
[5], [verifier], [0,697], [199,9], [0,000],
[10], [B], [0,750], [296,0], [0,197],
[10], [H+B], [0,773], [300,2], [0,103],
[10], [verifier], [0,795], [310,5], [0,000],
[20], [B], [0,811], [447,2], [0,297],
[20], [H+B], [0,833], [460,2], [0,120],
[20], [verifier], [0,848], [467,3], [0,000]
 ),
 kind: table,
 caption: [Contabilidade de continuação: tentativas abortadas consomem orçamento; parada no primeiro sucesso verificado.]
) <tab-main-policy>

#figure(image("../../reports/protocol_v0.2/figures/compute_resolution.png", width: 90%),
 caption: [Custo e resolução nos mesmos fluxos de candidatos. Tokens não equivalem a latência.]) <fig-main-frontier>

#figure(image("../../reports/protocol_v0.2/figures/paired_policy_effects.svg", width: 100%),
 caption: [Diferenças pareadas H+B menos comparador por orçamento, com IC 95%. Linha tracejada: margem de resolução −0,03.]) <fig-policy-effects>

#figure(image("../../reports/protocol_v0.2/figures/policy_cost_and_false_aborts.svg", width: 100%),
 caption: [Custo e descarte de tentativas corretas ativas iniciadas. O denominador exclui tentativas nunca iniciadas.]) <fig-policy-risk>

A validação online incluiu 12 tarefas escolhidas por ordem lexicográfica, sem seleção por resultado. A reprodução de tokens, identidades e decisões passou. Houve aborto efetivo. Os tempos são descritivos, sem repetições suficientes para inferir economia de latência.

== Transferência congelada para MBPP+

O sistema foi ajustado exclusivamente no banco principal HumanEval+ e aplicado sem reajuste, recalibração ou escolha de limiar nos rótulos alvo. O resultado é secundário e offline.

#figure(
 table(columns: 5, align: left,
 table.header([*Checkpoint*], [*Preditor*], [*AUC*], [*WT-AUC*], [*Log loss*]),
[8], [B], [0,470], [0,545], [0,718],
[8], [H], [0,372], [0,502], [0,880],
[8], [H+B], [0,364], [0,494], [0,883],
[16], [B], [0,725], [0,715], [0,659],
[16], [H], [0,610], [0,538], [0,673],
[16], [H+B], [0,644], [0,618], [0,663],
[32], [B], [0,756], [0,642], [0,648],
[32], [H], [0,706], [0,626], [0,658],
[32], [H+B], [0,733], [0,643], [0,640],
[64], [B], [0,915], [0,847], [0,590],
[64], [H], [0,614], [0,529], [0,689],
[64], [H+B], [0,741], [0,542], [0,644]
 ),
 kind: table,
 caption: [Desempenho externo de transferência. Uma população homogênea não identifica WT-AUC.]
) <tab-mbpp-transfer>

#figure(image("../../reports/transfer_v0.2/figures/frozen_transfer_signal.svg", width: 100%),
 caption: [Transferência externa sem ajuste nos rótulos alvo. WT-AUC e contraste pareado; checkpoints são secundários.]) <fig-transfer-signal>
#figure(
 table(columns: 3, align: left,
 table.header([*Política N=20*], [*Resolução MBPP+*], [*Tokens/tarefa*]),
[B], [0,844], [257,0],
[H+B], [0,781], [292,6],
[verifier], [0,844], [279,9]
 ),
 kind: table,
 caption: [Replay offline em 32 tarefas alvo. Não há validação online MBPP+ nem ajuste com rótulos alvo.]
) <tab-transfer-policy>
#figure(
 table(columns: 3, align: left,
 table.header([*Contraste N=20*], [*Δresolução e IC 95%*], [*Δtokens e IC 95%*]),
[H+B menos verifier], [-0,062 [-0,156; 0,000]], [12,72 [-0,595; 34,531]],
[H+B menos B], [-0,062 [-0,156; 0,000]], [35,66 [13,092; 64,438]]
 ),
 kind: table,
 caption: [Diferenças pareadas por tarefa em transferência. Tokens positivos indicam maior custo de H+B.]
) <tab-transfer-effects>

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

#figure(
 table(columns: 3, align: left,
 table.header([*Métrica H+B menos B*], [*Estimativa principal*], [*Faixa ao excluir uma tarefa*]),
[brier], [0,0163], [[0,014; 0,018]],
[log_loss], [0,0379], [[0,032; 0,042]],
[WT-AUC], [-0,0470], [[-0,055; -0,039]]
 ),
 kind: table,
 caption: [Influência da exclusão de uma tarefa por vez; os modelos permanecem fixos. A faixa não é um intervalo de confiança.]
) <tab-supp-influence>

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

#figure(
 table(columns: 4, align: left,
 table.header([*População em 32 tokens*], [*Tarefas*], [*ΔWT-AUC e IC 95%*], [*Δlog loss e IC 95%*]),
[HumanEval+ OOF, T > 64], [122], [-0,0212 [-0,089; 0,048]], [0,0287 [-0,017; 0,075]],
[MBPP+ frozen, T > 64], [17], [-0,0208 [-0,312; 0,251]], [0,0025 [-0,065; 0,067]],
[HumanEval+, sem histórico], [106], [-0,0341 [-0,093; 0,025]], [0,0192 [-0,016; 0,054]]
 ),
 kind: table,
 caption: [Análises de sensibilidade com escores fixos. WT-AUC utiliza somente tarefas mistas em cada coorte.]
) <tab-supp-cohorts>

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

#figure(
 table(columns: 4, align: left,
 table.header([*Avaliação*], [*Preditor*], [*ECE*], [*RMSCE*]),
[HumanEval+ OOF], [B], [0,0445], [0,0620],
[HumanEval+ OOF], [H], [0,0313], [0,0754],
[HumanEval+ OOF], [H+B], [0,0261], [0,0642],
[MBPP+ frozen], [B], [0,1521], [0,1685],
[MBPP+ frozen], [H], [0,1266], [0,1387],
[MBPP+ frozen], [H+B], [0,1208], [0,1341]
 ),
 kind: table,
 caption: [Calibração da probabilidade da classe positiva em 32 tokens; os intervalos exploratórios estão nos artefatos complementares.]
) <tab-supp-calibration>

#figure(image("../../reports/final_review/supplemental/figures/calibration_transfer.svg", width: 100%),
 caption: [Confiabilidade em origem e alvo. Os escores são OOF em HumanEval+ e de um modelo ajustado em toda a origem para MBPP+; a diferença entre painéis não isola mudança de domínio.]) <fig-supp-calibration>

A retenção abaixo considera todos os candidatos corretos ativos no checkpoint
selecionado, inclusive candidatos nunca iniciados no replay. Ela tem o mesmo
denominador conceitual da restrição de desenvolvimento. Difere da taxa de
aborto falso operacional, que considera apenas tentativas iniciadas.
Cada política pode selecionar um checkpoint diferente; a comparação de
retenções entre políticas não usa necessariamente a mesma população.

#figure(
 table(columns: 4, align: left,
 table.header([*Avaliação*], [*Política*], [*Retidos/ativos corretos*], [*Retenção e IC 95%*]),
[HumanEval+ OOF], [B], [1158/1219], [0,9500 [0,907; 0,983]],
[HumanEval+ OOF], [H+B], [1056/1079], [0,9787 [0,961; 0,993]],
[MBPP+ frozen], [B], [356/366], [0,9727 [0,955; 0,989]],
[MBPP+ frozen], [H+B], [276/366], [0,7541 [0,619; 0,877]]
 ),
 kind: table,
 caption: [Retenção de candidatos corretos sob gates congelados. Bootstrap por tarefa da razão de contagens somadas; alvo de desenvolvimento 0,97.]
) <tab-supp-retention>

#figure(image("../../reports/final_review/supplemental/figures/correct_retention.svg", width: 100%),
 caption: [Alvo de seleção, retenção de desenvolvimento e retenção externa nos gates congelados. O alvo empírico não é garantia de generalização.]) <fig-supp-retention>

=== Condições necessárias para a conclusão operacional

#figure(
 table(columns: 4, align: left,
 table.header([*Condição*], [*Estimativa e IC 95%*], [*Regra*], [*Atendida*]),
[Ganho primário de WT-AUC], [-0,0470 [-0,099; 0,006]], [LI > 0,000], [não],
[Redução primária de log loss], [0,0379 [0,007; 0,071]], [Estimativa < 0,000], [não],
[N=20 HB menos verifier: Δresolução], [-0,0152 [-0,038; 0,000]], [LI > -0,030], [não],
[N=20 HB menos verifier: Δtokens], [-7,1818 [-18,735; 0,804]], [LS < 0,000], [não],
[N=20 HB menos B: Δresolução], [0,0227 [-0,015; 0,061]], [LI > -0,030], [sim],
[N=20 HB menos B: Δtokens], [12,9621 [-26,335; 77,053]], [LS < 0,000], [não],
[Critério conjunto Q2], [—], [Regra conjunta], [não],
[Replay online com aborto efetivo], [—], [Regra conjunta], [sim],
[Benefício operacional conjunto], [—], [Regra conjunta], [não]
 ),
 kind: table,
 caption: [Critérios originais em N = 20. Q2 exige redução pontual de log loss; seu IC qualifica a incerteza. LI e LS são limites do intervalo; nenhuma margem foi escolhida após observar os resultados.]
) <tab-supp-decision>
