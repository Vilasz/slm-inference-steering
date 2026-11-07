# Estado da evidência

Snapshot: 2026-10-16T03:19:21.625357+00:00.

A execução integral foi autorizada pelo estudante. O desenho e os fontes foram
copiados antes do banco principal. Aprovação do orientador pendente;
não se reivindica pré-registro independente. Consulte a
[emenda administrativa](docs/execution_amendment_v0.2.md).

O status atual, sem carregar a GPU, é mostrado por:

```powershell
.venv/Scripts/python.exe scripts/research_status.py
```

## Coleta

| Banco | Trajetórias | Tarefas observadas | Estado |
|---|---:|---:|---|
| Piloto v0.1 | 39/960 | 13 | partial |
| Piloto v0.2 | 960/960 | 32 | completed |
| definitive v0.2 | 2640/2640 | 132 | completed |
| transfer v0.2 | 640/640 | 32 | completed |

O piloto escolhe temperatura somente após as 960 trajetórias.
O banco principal mantém todas as tentativas, mesmo após um sucesso.

## Resultados principais

| Preditor em 32 tokens | AUC | WT-AUC | Log loss |
|---|---:|---:|---:|
| B | 0.7005 | 0.6282 | 0.6302 |
| H | 0.5924 | 0.5681 | 0.6869 |
| HB | 0.6456 | 0.5811 | 0.6681 |

Decisão de Q2 e intervalos pareados:

```json
{
  "supported": false,
  "checkpoint": 32,
  "delta_hb_minus_b": {
    "estimate": -0.04702865751183576,
    "ci95": [
      -0.09937495818447424,
      0.005583498622638421
    ],
    "tasks": 92,
    "replications": 2000,
    "seed": 20260921,
    "unit": "task_id"
  },
  "descriptive_delta_h_minus_b": {
    "estimate": -0.06004633395733728,
    "ci95": [
      -0.1143251150204028,
      -0.003948685130266649
    ],
    "tasks": 92,
    "replications": 2000,
    "seed": 20260921,
    "unit": "task_id"
  },
  "permutation": {
    "p_value": 0.030996900309969003,
    "folds": [
      {
        "fold": 0,
        "p_value": 0.1100889911008899,
        "observed_wt_auc": 0.558784776511577,
        "mixed_tasks": 19,
        "seed": 20260921
      },
      {
        "fold": 1,
        "p_value": 0.006199380061993801,
        "observed_wt_auc": 0.594928908515974,
        "mixed_tasks": 22,
        "seed": 20260922
      },
      {
        "fold": 2,
        "p_value": 0.017598240175982403,
        "observed_wt_auc": 0.5929434742363664,
        "mixed_tasks": 18,
        "seed": 20260923
      },
      {
        "fold": 3,
        "p_value": 0.07879212078792121,
        "observed_wt_auc": 0.5686118166243599,
        "mixed_tasks": 17,
        "seed": 20260924
      },
      {
        "fold": 4,
        "p_value": 0.051294870512948704,
        "observed_wt_auc": 0.5886816171587843,
        "mixed_tasks": 16,
        "seed": 20260925
      }
    ],
    "combination": "Bonferroni: min(1, K * min(p_k))",
    "null": "No within-task score/outcome association in every outer fold",
    "not_a_test_of": "Incremental information beyond observable features",
    "replications_per_fold": 10000,
    "alternative": "greater",
    "assumption": "Within-task label exchangeability conditional on the held-out fold scorer"
  },
  "delta_log_loss_hb_minus_b": {
    "estimate": 0.03785060488521885,
    "ci95": [
      0.006657260077646495,
      0.07061583758815658
    ],
    "tasks": 132,
    "replications": 2000,
    "seed": 20260921,
    "unit": "task_id"
  },
  "probabilistic_gain_uncertain": true,
  "observable_log_loss": 0.6302167257514909,
  "combined_log_loss": 0.6680673306367098,
  "conditional_population": "Held-out tasks with both correct and incorrect eligible attempts"
}
```

Decisão operacional:

```json
{
  "offline_statistical_support": false,
  "online_replay_validated": true,
  "incremental_over_observables": false,
  "system_result_finalized": false,
  "primary_budget": 20,
  "h2_supported": false,
  "operational_benefit_claim_supported": false,
  "interpretation": "Predeclared diagnostic only; no operational benefit claim when H2 is unsupported"
}
```

Bootstrap por tarefa condicional aos modelos OOF ajustados; não refaz o treino.
Tempos online são descritivos. Economia de tokens não implica menor latência.

## Diagnósticos complementares posteriores ao resultado

Robustez por fold e exclusão de tarefa, coorte temporal constante, sensibilidade ao
histórico, calibração e retenção externa foram acrescentadas como análises exploratórias.
Modelos, calibradores, limiares e critérios principais permanecem congelados.

[Relatório completo dos diagnósticos](reports/final_review/supplemental/README.md).

| Avaliação | Política | Corretos retidos/ativos | Retenção [IC 95%] |
|---|---|---:|---|
| HumanEval+ OOF | B | 1158/1219 | 0.9500 [0.9075; 0.9825] |
| HumanEval+ OOF | HB | 1056/1079 | 0.9787 [0.9606; 0.9925] |
| MBPP+ frozen | B | 356/366 | 0.9727 [0.9551; 0.9888] |
| MBPP+ frozen | HB | 276/366 | 0.7541 [0.6187; 0.8774] |

O denominador usa todos os corretos ativos no gate congelado, inclusive candidatos
não iniciados no replay. As políticas podem usar checkpoints diferentes.
O alvo empírico de 97% no desenvolvimento não constitui garantia externa.
A retirada de qualquer tarefa isolada preserva o sinal das estimativas principais
(com os escores fixos); isso não garante significância em cada exclusão.

## Histórico exploratório separado

| Preditor tardio | AUC | WT-AUC | Log loss |
|---|---:|---:|---:|
| B_late | 0.556 | 0.823 | 0.657 |
| H_final | 0.781 | 0.844 | 0.514 |
| HB_late | 0.782 | 0.844 | 0.514 |

Estados finais, 150 tentativas/30 tarefas, verificador histórico local.
Não são evidência de predição precoce ou de EvalPlus.
O incremento de WT-AUC permanece inconclusivo; a perda melhora nesta reanálise.

## Verificação e apresentação

Suíte: 135 testes passaram, 4 ignorados, 0 falhas/erros.

[Notebook executado](notebooks/final_research_summary.ipynb),
[HTML](reports/final_review/notebook.html), [PDF](tcc/build/manuscrito.pdf),
[evidência e hashes](reports/final_review/evidence.json),
[auditoria de publicação](reports/final_review/delivery_manifest.json).

O runner gera figuras temporais, controles, ROC/PR, calibração, deltas por tarefa,
efeitos pareados das políticas, risco de aborto e transferência congelada.
Resultados negativos e intervalos que incluem zero são preservados.
