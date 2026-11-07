# Relatório final de qualidade

Verificação em 2026-10-18T16:30:55.556886+00:00.

## PASSOU

- Ciência original preservada: dezesseis artefatos protegidos, fontes congelados e os quinze registros científicos originais idênticos.
- Tabelas reconstruídas de artefatos: 39 linhas de preditores, 14.432 linhas de replay histórico e 2.400 linhas do replay independente, retenção, critérios conjuntos, ablações e oracle. O novo endpoint e seu bootstrap foram recontados por implementação aritmética separada. Reexecuções estatísticas integrais original/transferência coincidem com os JSONs históricos.
- Resultado original negativo mantido: ΔWT-AUC −0,0470, IC95% [−0,0994; 0,0056]; Δlog loss +0,0379, IC95% [0,0067; 0,0706]. Benefício operacional conjunto não demonstrado.
- Reserva aprovada concluída: 120 tarefas principais/2.400 tentativas e vinte tarefas técnicas/400 tentativas disjuntas; endpoint único {'estimate': -0.16359060402684564, 'ci95': [-0.2402298061170319, -0.0942741120201171], 'tasks': 120, 'replications': 10000, 'seed': 20261017, 'undefined_draws': 0, 'identified': True, 'directional_hypothesis_supported': True}.
- Gates e limiares originais em t=8; denominador comum; bootstrap pareado por tarefa; nenhum ajuste no alvo, troca de tarefa ou interrupção por resultado.
- Auditoria de prompts/testes e parser; dois pares textualmente semelhantes qualificados e agrupados somente em sensibilidade posterior. Omissão dos helpers no snapshot explicitada com recuperação pré-aprovação por HEAD+patch.
- Casos sintéticos de replay/oracle e custo finito verificados; nenhum sucesso perdido pelo oracle no mesmo banco/orçamento. Oracle apresentado como contrafactual e tokens separados de latência.
- Regressão: 171 testes sem falhas nem ignorados, Ruff e mypy aprovados. Notebook com 21 células executadas, sem erros.
- Bibliografia primária revisada, matriz final e fonte recente incorporadas; nenhuma prioridade universal ou causalidade atribuída às ablações.
- Manuscrito, resumo de duas páginas e doze slides compilados; 44 perguntas respondidas. Todas as páginas dos três PDFs inspecionadas visualmente, com hashes e notas em `visual_review/manifest.json`.
- Formato ajustado ao manual geral FGV consultado, com escopo institucional explicitado; comandos de análise separados de coleta longa em `docs/FINAL_REPRODUCIBILITY.md`.
- Latência operacional medida em 72 execuções, seis tarefas conhecidas, três repetições e N=3, quatro modos; GPU real NVIDIA GeForce RTX 3050 6GB Laptop GPU. Resultado completed_descriptive, descritivo, com componentes de tempo e cobertura publicados.

## PENDENTE

- Revisão final e anuência do orientador sobre a versão integrada; ensaio pelo estudante para defender pessoalmente cada decisão. Isso é distinto da revisão da proposta já informada antes da coleta. Risco: dúvidas de autoria/argumentação na defesa. Ação: enviar a versão final e realizar defesa simulada.
- Confirmação pelo curso/biblioteca do modelo institucional exigido, ficha catalográfica quando aplicável e folha de aprovação após defesa real. Risco: devolução administrativa. Ação: validar o arquivo com EMAP/biblioteca e inserir somente documentos autênticos.
- Preparação de entrega/depósito e conferência final de autoria. Reservar 1–14/12/2026 prioritariamente para redação, orientação e defesa; não iniciar coleta de alto risco nesse período.

## NÃO VERIFICÁVEL

- Nota futura, aceitação em mestrado, aprovação formal da banca e manifestação direta/assinatura do orientador: dependem de avaliação externa. A revisão da proposta foi informada pelo estudante, sem assinatura fabricada.
- Ausência de contaminação no pré-treinamento, de toda dependência semântica e de adaptação humana histórica: os artefatos só sustentam os controles e limitações documentados.
- Corretude universal além da suíte, desempenho em outros modelos/benchmarks, causalidade de dimensão/domain shift e aceleração robusta em produção ou RTX4060 não acessível.
- Energia e FLOPs não foram medidos; nem tokens nem tempos descritivos substituem essas medidas.
