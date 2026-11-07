# Auditoria estatística final

Execução: 2026-11-07T22:13:05.368517+00:00. Duração: 47.7 segundos.
Reconstrução em CPU, sem carregar geradores, ajustar probes ou executar código gerado.

## Insumos e integridade

`external_predictions.csv`, `replay_per_task.csv`, manifests dos probes, desenho externo,
simulação prospectiva e metadados de candidatos já existentes. Os hashes estão em
`statistical_verification.json`. Os registros originais permanecem intactos.
O CSV compacto `reproduction/external_candidates.csv` conserva somente modelo, tarefa,
tentativa, comprimento e label; permite a mesma reconstrução em uma cópia pública.

Foram conferidos 107 arquivos científicos e históricos protegidos.

## AUC, pareamento e população

A implementação independente conta pares positivo-negativo dentro de cada tarefa,
atribui meio ponto aos empates e exclui homogêneas somente da WT-AUC. A média dá o
mesmo peso a cada tarefa mista. Cada família possui exatamente os mesmos candidatos
e labels de sua contraparte; foram verificados IDs, duplicações e elegibilidade T>32.

| Modelo | Geradas | Ativas | Tarefas ativas | Mistas | Sem tentativa ativa |
|---|---:|---:|---:|---:|---:|
| deepseek | 2000 | 1943 | 100 | 58 | 0 |
| qwen | 2000 | 863 | 81 | 35 | 19 |

Também foram reconstruídos os dois contrastes H2 originais (2.000 réplicas,
seed 20260921) e a diferença de retenção independente (10.000 réplicas, seed
20261017), respeitando a ordem e os denominadores dos registros históricos:

- Qwen original: ΔWT-AUC −0,0470, IC95% [−0,0994; 0,0056].
- Qwen original: Δlog loss 0,0379, IC95% [0,0067; 0,0706].
- Gates anteriores: Δretenção −0,1636, IC95% [−0,2402; −0,0943], 120 tarefas.

Os scores apresentam 28 pares positivo-negativo empatados por família no DeepSeek
e 13 no Qwen; a convenção foi também verificada em um exemplo com AUC exatamente 0,5.

| Modelo | Classe | ΔWT-AUC H+B−B | IC95% | Estatuto |
|---|---|---:|---|---|
| deepseek | linear | −0,0038 | [−0,0668; 0,0554] | Secundário |
| deepseek | MLP | −0,0042 | [−0,0648; 0,0548] | Primário |
| qwen | linear | −0,0552 | [−0,1347; 0,0206] | Secundário |
| qwen | MLP | −0,0074 | [−0,0944; 0,0766] | Secundário |

Os quatro pontos e IC reproduzem os resultados salvos com tolerância absoluta
de 1e−11. Foram reamostradas tarefas inteiras, com 10.000 réplicas, seed 20261104
e percentis 2,5% e 97,5%, após formar as diferenças pareadas. Não se reamostram
tentativas ou pares como unidades independentes.

## Interação Γ

- deepseek: Γ = −0,0003, IC95% [−0,0313; 0,0323], 58 tarefas mistas.
- qwen: Γ = 0,0478, IC95% [0,0178; 0,0810], 35 tarefas mistas.

Cada tarefa contribui com (AUC_HB_MLP−AUC_B_MLP)−(AUC_HB_linear−AUC_B_linear).
O IC reamostra esse vetor conjunto; não subtrai limites de dois IC individuais.
No Qwen, a interação positiva reduz uma desvantagem linear; o incremento absoluto
MLP permanece negativo e inconclusivo. Γ é secundário, com IC pontual sem ajuste
de multiplicidade. A comparação entre modelos foi igualmente reproduzida nas
28 tarefas mistas comuns, preservando o cluster compartilhado.

## Probabilidades e calibração

Foram reproduzidas as oito linhas de AUC agregada, PR-AUC, WT-AUC, log loss e Brier,
os contrastes pareados de perdas e os 80 bins descritivos de calibração. As perdas
usam 100 tarefas ativas no DeepSeek e 81 no Qwen, incluindo homogêneas, com peso
total igual por tarefa. O DeepSeek favorece H+B nas perdas secundárias, enquanto
Qwen/MLP o desfavorece; isso não substitui o endpoint primário de ranking.

## Poder prospectivo

As 540 células salvas foram reproduzidas; 135 do cenário histórico também foram
conferidas por uma implementação independente do cálculo. A simulação usa 4.000
bancos por célula, seed 20261104, efeitos históricos centrados e deslocados,
contagem mista binomial e IC normais de planejamento. Não usa o efeito externo
observado nem substitui o bootstrap percentil final.

| Efeito fixado | Detecção positiva, 100 tarefas | Largura média do IC |
|---:|---:|---:|
| 0,020 | 10,32% | 0,124 |
| 0,030 | 15,60% | 0,124 |
| 0,050 | 36,12% | 0,124 |
| 0,075 | 64,88% | 0,124 |

No cenário nulo, a rejeição bilateral em 100 tarefas foi 4,575%.
O erro Monte Carlo máximo de um desvio-padrão é aproximadamente 0,79 ponto
percentual. A aproximação para 80% de poder exige 1.998/888/320/143 tarefas
para efeitos 0,02/0,03/0,05/0,075. Esses números dependem da heterogeneidade
Qwen/HumanEval e da fração mista 92/132; não garantem transporte ao DeepSeek.
Os cenários de SD de treinamento são sensibilidades, não correções formais
dos IC condicionais. Pouco poder não demonstra equivalência nem inexistência
de sinal; o IC primário ainda admite ganho próximo de 0,05 e perda próxima de 0,06.

## Replay operacional

Foram reconstruídas independentemente todas as 5.600 linhas tarefa/política/orçamento
e as 56 sínteses, incluindo IC pareados de custo/cobertura e IC das razões de
retenção e tokens por tarefa resolvida. A ordem é attempt_id, a condição é T>32,
e o aborto ocorre se score < limiar congelado. N=1,5,10,20 usa a mesma regra.

Resumo em N=20; custo agregado por resolvida inclui os tokens das não resolvidas.

| Modelo | Política | Resolvidas | Tokens/tarefa | Tokens/resolvida | Retenção ativa | Abortos/iniciadas | Corretos abortados | Perdidas |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| deepseek | B_MLP | 71 | 1260,38 | 1775,18 | 100,00% | 0/752 | 0 | 0 |
| deepseek | B_linear | 71 | 1260,38 | 1775,18 | 100,00% | 0/752 | 0 | 0 |
| deepseek | HB_MLP | 71 | 1243,65 | 1751,62 | 99,45% | 7/754 | 1 | 0 |
| deepseek | HB_linear | 71 | 1256,94 | 1770,34 | 100,00% | 1/752 | 0 | 0 |
| deepseek | fixed | 71 | 2709,31 | 3815,93 | 100,00% | 0/2000 | 0 | 0 |
| deepseek | oracle | 71 | 304,37 | 428,69 | 100,00% | 669/752 | 0 | 0 |
| deepseek | verifier | 71 | 1260,38 | 1775,18 | 100,00% | 0/752 | 0 | 0 |
| qwen | B_MLP | 75 | 271,76 | 362,35 | 96,23% | 102/658 | 0 | 0 |
| qwen | B_linear | 75 | 282,80 | 377,07 | 97,80% | 73/658 | 0 | 0 |
| qwen | HB_MLP | 75 | 300,05 | 400,07 | 99,06% | 41/658 | 0 | 0 |
| qwen | HB_linear | 75 | 282,09 | 376,12 | 97,48% | 80/658 | 0 | 0 |
| qwen | fixed | 75 | 791,88 | 1055,84 | 100,00% | 0/2000 | 0 | 0 |
| qwen | oracle | 75 | 200,37 | 267,16 | 100,00% | 339/658 | 0 | 0 |
| qwen | verifier | 75 | 318,71 | 424,95 | 100,00% | 0/658 | 0 | 0 |

Retenção abrange todos os corretos ativos potenciais no orçamento; falsos abortos
contam apenas candidatos iniciados. Não são complementares. Alguns gates DeepSeek
não interrompem nenhuma tentativa iniciada; custos iguais ao verificador representam
ausência de economia adicional. A economia de H+B/MLP nesse modelo é pequena.
Nenhuma tarefa perdida no banco produz IC empírico degenerado de cobertura; isso
não certifica ausência de perdas raras futuras. Tokens excluem prompt, captura,
score, verificação, energia e preço. Não se reproduziu aceleração online.

## Validação matemática e limites da reprodução

Bellman conferido em 216 casos finitos e 27 comparações de limiar.
A enumeração usa p conhecido, custos inteiros e tentativas iid. Ela não demonstra
otimalidade dos gates empíricos. Geração, treinamento, seleção e labels funcionais
não foram reexecutados; o escopo desta auditoria é reconstruir os cálculos a partir
das evidências existentes, preservando seu histórico. A incerteza completa do
treinamento e a ausência de contaminação no pré-treinamento permanecem abertas.

Reprodução: `python scripts/finalization_audit.py`.
