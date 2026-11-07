# Latência operacional descritiva

Six known pilot tasks, three repeated timings per mode, fixed N=3. Descriptive variability, no robust speedup inference, no RTX4060 extrapolation.

Total includes prefill, decoding, gate, parsing and warm-checker RPC; excludes model loading and warmup

Prefill/capture/classification/remainder partition the generation wall span; generation+parsing+verification+other partition total. Copy can wait for preceding GPU work; spans are not isolated kernel costs.

| Modo | Execuções | Tempo total médio (s) | Mín–máx (s) | Geração (s) | Verificador RPC (s) | Tokens médios |
|---|---:|---:|---:|---:|---:|---:|
| verifier_plain | 18 | 32.570 | 2.714–112.960 | 32.284 | 0.281 | 286.5 |
| verifier_selective | 18 | 32.595 | 2.733–112.842 | 32.310 | 0.281 | 286.5 |
| B | 18 | 14.441 | 2.721–29.622 | 14.336 | 0.100 | 124.3 |
| HB | 18 | 32.549 | 2.731–112.963 | 32.265 | 0.280 | 286.5 |

GPU: NVIDIA GeForce RTX 3050 6GB Laptop GPU. Dados por tarefa, repetição e decisão estão nos JSONL/CSV; o plano precedeu as medições.
Resolver menos tarefas pode reduzir latência. Compare cobertura e tentativas junto com tempo; estas médias não demonstram ganho operacional robusto.

Nota de execução: o somatório dos 72 tempos completos de tarefa foi 2.018,815 s (33,65 min), excluindo carregamento e aquecimento. O orçamento de 1.800 s é verificado antes de cada bloco completo de quatro modos; o último bloco foi concluído após esse valor. Não era um limite rígido por processo. A variabilidade entre repetições está em `within_task_timing_variability.csv`, com 24 combinações tarefa/modo e três repetições cada.
