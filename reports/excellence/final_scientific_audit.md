# Auditoria científica final

Verificação: 2026-10-18T16:22:21.592116+00:00.

## Verificações concluídas

Os valores abaixo foram recalculados dos CSVs e bancos, sem copiar os números do texto. A comparação usou tolerância absoluta de 2×10⁻¹².
AUC foi recontada por pares, incluindo pesos de tarefa na AUC agregada; perdas são médias por tarefa.

ΔWT-AUC principal: {'estimate': -0.04702865751183576, 'ci95': [-0.09937495818447421, 0.005583498622638404]}; Δlog loss: {'estimate': 0.03785060488521885, 'ci95': [0.006657260077646494, 0.0706158375881566]}.
Foram conferidas 39 linhas de preditores e 14432 linhas de replay, incluindo todas as parametrizações do oracle.
3280 rótulos dos bancos principal e alvo conferem com os status oficiais salvos; o piloto também integra a reconstrução de seeds e scorer arquivada.

| Avaliação | Política | Corretos retidos/ativos | Retenção e IC95% |
|---|---|---:|---|
| HumanEval+ OOF | B | 1158/1219 | 0.949959 [0.9074776873520638, 0.9825192220714608] |
| HumanEval+ OOF | HB | 1056/1079 | 0.978684 [0.9605702925607335, 0.9925324855117886] |
| MBPP+ frozen | B | 356/366 | 0.972678 [0.9550554366565602, 0.9887656142112862] |
| MBPP+ frozen | HB | 276/366 | 0.754098 [0.6186639175257732, 0.8773604614495191] |

Tabela de preditores principal, tabela de retenção, ablações/leitura e oracle: todas reconstruídas. A numeração pode mudar após integrar a extensão; os identificadores semânticos dos artefatos permanecem.
Os critérios Q2/Q3 são os originais. Q2 exige IC de incremento WT-AUC inteiramente positivo e redução pontual de perda; não exige IC da perda inteiramente negativo. O resultado negativo não mudou.
O replay direto confirmou custos, ordem, orçamento, primeira solução, tentativas iniciadas e falsos abortos, sem chamar a implementação de simulate.
Os casos manuais adicionais cobrem nenhum sucesso, sucesso inicial/final, término exatamente no checkpoint, aborto de corretos, informação futura e enumeração da fórmula de custo.

Precisão de serialização: o parser padrão do CSV altera alguns empates de H em t=8 no alvo (até 8,33×10⁻¹⁷ nas probabilidades, com diferença de 0,002193 na WT-AUC). A leitura round_trip recupera todas as métricas da avaliação original em precisão completa. O leitor complementar foi corrigido; não houve retreino, alteração de rótulo ou redefinição do endpoint. A reexecução integral da transferência em pasta separada reproduziu exatamente métricas, deltas e políticas.

## Controles e alcance

A reconstrução de scorers e a auditoria das partições foram reexecutadas para arquivo novo, preservando o recibo anterior byte a byte: [final_primary_reconstruction.json](final_primary_reconstruction.json).
As 270 partições de calibração e as escolhas one-SE foram verificadas. O teste de residual exclui o rótulo pontuado da seleção e calibração do primeiro estágio; a seleção residual repete essa separação internamente.
PCA é ajustado apenas em treinos. A reprodução não reexecuta toda a busca histórica em todos os checkpoints; isso é distinto da reconstrução dos scorers principais e das duas execuções já arquivadas das 70 unidades exploratórias.
As rotinas congeladas de estatística foram também reexecutadas em pastas separadas, incluindo permutações e bootstrap: {'main_statistics': {'exact_json_equality': True, 'original_sha256': '0cc3d457fe59a656418dae2df02e69a38ce424703ee78e036b1a31d9561d2949', 'rerun_sha256': '0cc3d457fe59a656418dae2df02e69a38ce424703ee78e036b1a31d9561d2949', 'rerun': 'reports/excellence/final_main_reconstruction/results.json'}, 'transfer_statistics': {'exact_json_equality': True, 'original_sha256': '841bec2240af6d56f120ed72ad7ae9d34561c6694e5d6937f911b1f31c860910', 'rerun_sha256': '841bec2240af6d56f120ed72ad7ae9d34561c6694e5d6937f911b1f31c860910', 'rerun': 'reports/excellence/final_transfer_reconstruction/results.json'}, 'transfer_policy_replay': {'exact_json_equality': True, 'original_sha256': 'a212b53f1b94074a325f343175fb3bfc99ada5d95d4b3f5365d397d8ecfc37f2', 'rerun_sha256': 'a212b53f1b94074a325f343175fb3bfc99ada5d95d4b3f5365d397d8ecfc37f2', 'rerun': 'reports/excellence/final_transfer_reconstruction/policy_replay.json'}}. Não se confundem com novo ajuste de todas as famílias em todos os checkpoints.
Os três bancos antigos tiveram todos os arquivos de ativação reabertos e seus hashes/finitude verificados na auditoria anterior, preservada. A nova avaliação reabre suas próprias ativações ao concluir.

## Problemas reais corrigidos

Corrigidos o critério de perda na tabela complementar, a perda de entradas na regeneração da bibliografia, a alegação não demonstrada de sandbox completo, o teste legado que contradizia o padrão de normalização e o estado desatualizado da seção 1.4.
As correções de apresentação não reescreveram protocolos, dados, seeds, fontes congelados ou decisões originais.

## Avaliação independente e verificações posteriores

Aprovação explícita e revisão do orientador informada pelo estudante já foram registradas, antes da primeira trajetória. Não houve nova solicitação da mesma aprovação.
A auditoria de IDs, prompts, estrutura canônica e hashes dos testes está em [reserved_task_audit.json](reserved_task_audit.json). Foi feita após o início sob aprovação anterior, sem usar rótulos, excluir tarefas ou alterar o endpoint.
Os helpers de parsing e dataset não eram entradas separadas no snapshot de 46 arquivos. Seus bytes anteriores à aprovação foram reconstruídos do Git HEAD e patch preservados antes da coleta, e coincidem com os fontes atuais: [parser_provenance_reconstruction.json](parser_provenance_reconstruction.json). Essa recuperação documenta a omissão; não cria retroativamente uma entrada no congelamento.
O estado e os resultados completos estão em [independent_v1/](independent_v1/README.md). Dados incompletos não foram usados para calcular o endpoint.

## Incertezas e não verificável

Os intervalos antigos condicionam nos scorers ajustados; não incluem toda a variabilidade de retreino. Há exposição histórica a 26 tarefas. Ausência de significância não demonstra equivalência.
A ausência de exposição no pré-treinamento e de toda semelhança semântica não pode ser certificada por estes arquivos. Aprovação nos testes não prova corretude universal.
Não há assinatura do orientador nem aprovação formal da banca criada pelo agente. A revisão da extensão foi informada explicitamente pelo estudante e está descrita com esse alcance.
Tempos técnicos e replay em tokens não demonstram economia de energia/FLOPs; a medição operacional adicional tem a escala efetivamente executada registrada em seus próprios artefatos.

Dados numéricos, comandos e hashes: [final_table_reconstruction.json](final_table_reconstruction.json).

## Resultado independente concluído

120 tarefas principais e 2400 tentativas; piloto disjunto de 20 tarefas.
Endpoint primário: {'estimate': -0.16359060402684564, 'ci95': [-0.2402298061170319, -0.0942741120201171], 'tasks': 120, 'replications': 10000, 'seed': 20261017, 'undefined_draws': 0, 'identified': True, 'directional_hypothesis_supported': True}.
Reconstrução aritmética separada do endpoint e de todas as linhas do replay novo: {'estimate': -0.16359060402684564, 'ci95': [-0.2402298061170319, -0.0942741120201171], 'counts_HB_B_active': [986, 1181, 1192], 'replay_rows_verified': 2400, 'undefined_bootstrap_draws': 0, 'method': 'Direct candidate-count and replay loops; independent NumPy task bootstrap; no simulate call'}.
O endpoint novo não substitui Q2/H3 originais. Influência e agrupamento textual são sensibilidades posteriores, sem mudança da amostra ou novo critério de sucesso.
