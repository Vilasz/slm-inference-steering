# Piloto DeepSeek e viabilidade

Doze tarefas de desenvolvimento fixadas antes da coleta; vinte tentativas por
tarefa, sem seleção pelo resultado. A amostra final permanece sem rótulos novos.

```json
{
  "tasks": 12,
  "attempts": 240,
  "all_success_tasks": 1,
  "all_failure_tasks": 3,
  "mixed_tasks": 8,
  "success_rate": 0.4166666666666667,
  "mean_tokens": 160.33333333333334,
  "truncation_rate": 0.004166666666666667,
  "parsing_error_rate": 0.008333333333333333,
  "mean_seconds_deepseek": 20.559622466666664,
  "mean_seconds_qwen": 11.344481008333332,
  "tokens_per_second_deepseek": 7.798457077374934,
  "mean_activation_copy_seconds": 0.2631521575000486,
  "feasible": true
}
```

O benchmark Qwen usa uma tentativa nas mesmas doze tarefas. Tempo por tentativa
para dimensionamento usa o maior entre geração média e parede média da sessão,
incluindo carga/verificação. A contingência multiplica o custo futuro por 1,5.
Os manifests registram pico de memória e o benchmark pareado de captura.

| Tarefas externas por modelo | Horas previstas, incluindo contingência |
|---:|---:|
| 150 | 72.25 |
| 140 | 69.14 |
| 120 | 62.91 |
| 100 | 56.68 |
| 80 | 50.45 |

Decisão viável: True. Motivos de inviabilidade: [].
O tamanho escolhido usa custo e inventário, sem examinar labels externos.
Em caso de inviabilidade: If formatting/performance fails, audit prompt/extraction on development only and repeat a versioned pilot. If only budget fails, review a smaller external sample below 80 with a new precision analysis before any external generation; retain both models, matched classifier comparisons and 20 attempts. No automatic scientific change is authorized by this rule..
Não se altera automaticamente temperatura, modelo, tarefas ou limiares.
