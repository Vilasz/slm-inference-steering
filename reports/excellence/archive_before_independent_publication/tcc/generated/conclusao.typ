= Considerações finais

A avaliação principal foi executada com 2.640 tentativas em 132 tarefas
HumanEval+, precedida por piloto separado de 960 trajetórias. A validação
aninhada agrupa tarefas e compara representações internas com observáveis
disponíveis no mesmo prefixo. Em 32 tokens, ΔWT-AUC(H+B − B) foi -0,047, IC 95% [-0,099; 0,006] e
Δlog loss foi 0,038, IC 95% [0,007; 0,071]. O resultado não satisfaz o critério de ganho incremental.

Os critérios conjuntos para afirmar benefício operacional não foram satisfeitos. A comparação com verificador e observáveis, o replay pareado e a
validação real impedem atribuir ao probe economias disponíveis apenas pela
parada após sucesso. A transferência para MBPP+ utiliza modelos congelados.
As conclusões estão limitadas ao modelo, aos benchmarks e à população ativa
avaliados. Exposição histórica, incerteza condicional e ausência de inferência
robusta sobre latência são explicitadas. Replicações com novos modelos,
benchmarks e tarefas externas independentes são extensões naturais.

Os diagnósticos exploratórios posteriores examinam a estabilidade entre tarefas, a composição temporal e a calibração sem alterar as decisões principais. A retenção de candidatos corretos ativos de H+B em MBPP+ foi 0,7541 (IC 95% [0,619; 0,877]), frente ao alvo empírico de 0,97 no desenvolvimento. Esses resultados delimitam a transportabilidade do sistema avaliado; extensões devem ser tratadas como novos estudos.

O aprofundamento exploratório mostra que a leitura e sua parametrização
afetam o resultado: PCA e uma correção residual estritamente cross-fitted
reduzem a desvantagem original sem estabelecer ganho sobre B. A anatomia dos
descartes distingue perda de candidatos e de tarefas. O oracle retrospectivo
quantifica oportunidade de economia e a derivação econômica explicita como
retenção, overhead, orçamento e heterogeneidade limitam essa oportunidade.
Essas contribuições delimitam o resultado negativo e fornecem hipóteses
testáveis, sem convertê-lo em superioridade ou mecanismo causal. A avaliação
independente proposta permanece pendente de aprovação do estudante e revisão
do orientador.

