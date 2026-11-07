# Diagnósticos complementares

Análises exploratórias definidas após a conclusão e inspeção dos resultados principais.
Nenhum modelo, calibrador, limiar, rótulo, semente ou critério principal foi alterado.

## Precisão e estabilidade

| Contraste principal H+B menos B | Estimativa [IC 95%] |
|---|---|
| within_task_auc | -0.04703 [-0.09937; 0.00558] |
| log_loss | 0.03785 [0.00666; 0.07062] |

Os valores e intervalos principais foram reproduzidos a partir das probabilidades congeladas.
Os limites acima quantificam a precisão sob as suposições utilizadas. Zero no intervalo não demonstra equivalência;
log loss positivo representa maior perda de H+B. Nenhum resultado prova ausência universal de sinal.

| Métrica | Faixa da média ao excluir uma tarefa por vez |
|---|---|
| brier | [0.01449; 0.01811] |
| log_loss | [0.03247; 0.04204] |
| within_task_auc | [-0.05506; -0.03875] |

A exclusão mantém os modelos fixos. A estabilidade do sinal da estimativa não garante significância em cada exclusão.

## População constante e histórico de exposição

| Coorte em 32 tokens | Tentativas | Tarefas | ΔWT-AUC [IC 95%] | Δlog loss [IC 95%] |
|---|---:|---:|---|---|
| HumanEval+ OOF, T>64 | 1228 | 122 | -0.02121 [-0.08867; 0.04755] | 0.02875 [-0.01664; 0.07530] |
| MBPP+ frozen, T>64 | 125 | 17 | -0.02083 [-0.31250; 0.25052] | 0.00247 [-0.06472; 0.06703] |
| HumanEval+, sem as 26 tarefas históricas | 1755 | 106 | -0.03413 [-0.09255; 0.02468] | 0.01921 [-0.01571; 0.05359] |

A coorte T>64 condiciona no comprimento final. A exclusão histórica só altera a avaliação, sem retreino;
nenhuma delas substitui o endpoint ou cria uma amostra confirmatória independente.

## Retenção dos corretos

| Avaliação | Política | Corretos retidos/ativos | Retenção [IC 95%] |
|---|---|---:|---|
| HumanEval+ OOF | B | 1158/1219 | 0.94996 [0.90748; 0.98252] |
| HumanEval+ OOF | HB | 1056/1079 | 0.97868 [0.96057; 0.99253] |
| MBPP+ frozen | B | 356/366 | 0.97268 [0.95506; 0.98877] |
| MBPP+ frozen | HB | 276/366 | 0.75410 [0.61866; 0.87736] |

O denominador inclui todos os candidatos corretos ativos no gate congelado, inclusive os nunca iniciados no replay.
As políticas podem usar checkpoints diferentes; a tabela não é uma comparação pareada de retenção entre políticas.
O alvo 0,97 se refere ao desenvolvimento. Os intervalos reamostram tarefas, usando a razão das contagens somadas.

## Consequência operacional no alvo (replay offline, N=20)

| Política | Resolução | Tokens por tarefa |
|---|---:|---:|
| B | 0.84375 | 256.969 |
| HB | 0.78125 | 292.625 |
| verifier | 0.84375 | 279.906 |

| Contraste | Δresolução [IC 95%] | Δtokens [IC 95%] |
|---|---|---|
| HB minus verifier | -0.06250 [-0.15625; 0.00000] | 12.71875 [-0.59453; 34.53125] |
| HB minus B | -0.06250 [-0.15625; 0.00000] | 35.65625 [13.09219; 64.43828] |

Dados originais da transferência congelada, sem ajuste local e sem validação online MBPP+.
O sinal do contraste de tokens indica a direção do custo: valores positivos representam custo maior de H+B.

## Calibração em 32 tokens

| Avaliação | Preditor | ECE [percentis 2,5%; 97,5%] | RMSCE |
|---|---|---|---:|
| HumanEval+ OOF | B | 0.04447 [0.03840; 0.09749] | 0.06203 |
| HumanEval+ OOF | H | 0.03130 [0.02460; 0.08743] | 0.07540 |
| HumanEval+ OOF | HB | 0.02606 [0.02760; 0.08414] | 0.06420 |
| MBPP+ frozen | B | 0.15214 [0.08457; 0.24494] | 0.16853 |
| MBPP+ frozen | H | 0.12660 [0.05626; 0.22084] | 0.13874 |
| MBPP+ frozen | HB | 0.12082 [0.06483; 0.21834] | 0.13406 |

Dez bins fixos de probabilidade da classe positiva, cada tarefa com o mesmo peso total.
ECE é diagnóstico dependente de discretização e tamanho amostral; não substitui log loss ou Brier.
ECE menor isoladamente não implica melhor predição ou política operacional.
O bootstrap percentil de ECE pode ter viés positivo e não conter a estimativa empírica.
A origem usa escores OOF e o alvo um modelo ajustado em toda a origem; diferenças entre bancos não isolam um efeito causal de domínio.

## Critérios de decisão preservados

| Critério | Estimativa [IC 95%] | Condição | Atendido |
|---|---|---|---|
| Primary WT-AUC gain | -0.04703 [-0.09937; 0.00558] | limite inferior > 0 | não |
| Primary log-loss reduction | 0.03785 [0.00666; 0.07062] | estimativa < 0 | não |
| N=20 HB minus verifier: resolution_delta | -0.01515 [-0.03788; 0.00000] | limite inferior > -0.03 | não |
| N=20 HB minus verifier: token_delta | -7.18182 [-18.73485; 0.80379] | limite superior < 0 | não |
| N=20 HB minus B: resolution_delta | 0.02273 [-0.01515; 0.06061] | limite inferior > -0.03 | sim |
| N=20 HB minus B: token_delta | 12.96212 [-26.33542; 77.05265] | limite superior < 0 | não |
| H2 joint criterion | — | regra conjunta | não |
| Online identity replay with actual abort | — | regra conjunta | sim |
| Joint operational benefit | — | regra conjunta | não |

## Reprodução e limites

```powershell
.venv/Scripts/python.exe scripts/analyze_research_robustness.py
```

Bootstrap: 2000 reamostragens de tarefas, seed 20260921.
Todos os intervalos adicionais são pontuais e exploratórios, condicionais aos modelos já ajustados.
Folds compartilham dados de treinamento e não são replicações independentes.
Identidades, hashes dos insumos e parâmetros estão em [results.json](results.json).
Tabelas por tarefa, fold e bin estão nos CSV; figuras são exportadas em PNG, SVG e PDF.

Referências metodológicas: [Guo et al. (2017)](https://proceedings.mlr.press/v70/guo17a.html),
[Greenland et al. (2016)](https://doi.org/10.1007/s10654-016-0149-3).
