# Auditoria científica da extensão final

Atualizado: 2026-11-06T08:50:52.054825+00:00. Estado e contagens reproduzíveis em `status.json`.

Contabilidade de recursos: {"raw_session_wall_hours": 43.169389166666654, "excluded_os_suspension_hours": 9.676916720277777, "charged_active_session_hours": 33.492472446388874, "scope": "Active session wall time including loading/verification; not CUDA kernel profiling"}.
Suspensões comprovadas pelo log do sistema são excluídas do orçamento de GPU,
mantendo manifests e tempos brutos intactos. A emenda de recursos e os eventos
originais ficam versionados antes da avaliação externa. Esse valor continua
sendo tempo de parede ativo da sessão, incluindo carga e verificação, não um
perfil de tempo de kernels CUDA. Os IDs, seeds, labels e a amostra não mudam.

## Evidência histórica preservada

O banco original contém 2640 gerações, 132
tarefas e 92 tarefas mistas em t=32. O contraste
original foi reconstruído exatamente a partir dos scores por candidato, com
empates conferidos por contagem independente de pares. Seeds, labels gravados
versus status base+plus, identidades e agrupamento externo foram verificados.
Hashes em `historical_hashes.json` protegem os experimentos anteriores, inclusive
a validação independente de 120 tarefas. Seus resultados e conclusões não mudam.
A retenção 82,72% versus 99,08% (−16,36 pp) continua uma conclusão sobre os gates
congelados anteriores, não uma impossibilidade universal de sinal interno.

O inventário local encontra 173 IDs MBPP expostos,
189 elegíveis após a triagem lexical, entre
378 tarefas. IDs, hashes de fontes e pares
semelhantes ficam em `task_inventory.json`; não há garantia de independência
semântica completa ou ausência de contaminação no pré-treinamento.

## Correções delimitadas à nova extensão

O worker histórico não fornecia isolamento de rede/arquivos pelo sistema
operacional. A extensão executa EvalPlus em bubblewrap/WSL, com namespaces,
ambiente limpo e mounts mínimos read-only. `sandbox_preflight.json` documenta
rede inacessível, host oculto e limites de recursos. Não se modificaram labels
antigos. A reavaliação adicional dos 2640 programas Qwen de origem no sandbox, com limite de 3 GiB em vez do padrão histórico de 4 GiB, encontrou 0 discrepâncias. O artefato source_label_audit.json registra a comparação; labels anteriores permanecem intactos.

O template DeepSeek já inclui BOS; o novo adapter usa add_special_tokens=False.
Estados são capturados depois de consumir o token do checkpoint, somente em
posições necessárias. A validação com o modelo compara cache, prefixo isolado,
sequência completa e probabilidades teacher-forced. Esses checks só contam
como executados quando aparecem no manifest da coleta; testes unitários não
substituem validação do modelo real.

PCA64, imputação e escalas são ajustados dentro dos treinos; GroupKFold separa
tarefas, early stopping usa tarefas internas e calibração/limiar usam reserva
da origem. Quatro famílias com grids correspondentes impedem confundir H com
uma mudança isolada de classe do classificador. Três seeds são retidas sem
escolher a melhor. Intervalos finais permanecem condicionais aos probes;
cenários de variabilidade de treinamento no planejamento são aproximações.
Foram executadas 8 repetições completas de seleção, refit e calibração em subamostras de tarefas de origem, com desvio de ΔWT-AUC 0.0360 (linear) e 0.0450 (MLP) no conjunto fixo de auditoria. São sensibilidades em subamostras sobrepostas, não estimativas sem viés da variância do procedimento final.

## Verificações e execução

Testes específicos: {'tests': 399, 'errors': 0, 'failures': 0, 'skipped': 0}. Conferem observáveis sem futuro, PCA sem dados
retidos, seeds de MLP, probabilidades, AUC/empates/homogêneas, pareamento,
caso T=32, replay/custos e Bellman contra enumeração exata de políticas iid.
Estado do piloto: {"tasks": 12, "attempts": 240, "all_success_tasks": 1, "all_failure_tasks": 3, "mixed_tasks": 8, "success_rate": 0.4166666666666667, "mean_tokens": 160.33333333333334, "truncation_rate": 0.004166666666666667, "parsing_error_rate": 0.008333333333333333, "mean_seconds_deepseek": 20.559622466666664, "mean_seconds_qwen": 11.344481008333332, "tokens_per_second_deepseek": 7.798457077374934, "mean_activation_copy_seconds": 0.2631521575000486, "feasible": true}.
Protocolo final travado: True. Resultados externos completos:
True. Fontes e bundles somente ficam travados
por `freeze.json` após treino, piloto e dimensionamento técnico.

Não publicar inferência externa a partir de um banco parcial. Tabelas finais
são reconstruídas dos JSON por candidato/NPZ e escritas por `evaluation.py`.
Tarefas homogêneas entram nos custos, Brier/log loss e denominadores gerais;
AUC intratarefa exige as mistas. As trajetórias dos dois modelos na mesma
tarefa compartilham o cluster. Tokens economizados não medem latência.

## Pendências explicitamente delimitadas

Enquanto `results.json` não existir, não há conclusão empírica nova sobre a
hipótese DeepSeek/MLP nem justificativa para reinterpretar os achados anteriores.
A extensão matemática em `docs/final_extension_decision.md` é conceitual,
com antecedentes identificados e testes sintéticos; não é teorema inédito.
