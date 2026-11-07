# Inspeção dos pares de prompts semelhantes

Inspeção pública e sem novos rótulos, posterior ao início da coleta aprovada,
em 17/10/2026. Fonte: `reserved_task_contents.json`; diagnóstico quantitativo:
`reserved_task_audit.json`.

| Par da reserva principal | Relação observável nos enunciados | Implicação |
|---|---|---|
| Mbpp/120 e Mbpp/409 | Produto de pares de tuplas de uma lista; máximo absoluto versus mínimo, com o mesmo exemplo de lista | Variantes próximas do problema, apesar de prompts e ASTs normalizadas não serem idênticos |
| Mbpp/294 e Mbpp/410 | Máximo versus mínimo em lista heterogênea, com o mesmo exemplo | Variantes próximas; igualdade do ID não é a única possível unidade de dependência |

Portanto, “nenhuma duplicata exata” não significa “tarefas semanticamente
independentes”. O agrupamento posterior desses dois pares em 118 clusters é
uma sensibilidade motivada por conteúdo público, sem seleção por desfecho.
Não altera os 120 IDs, as tentativas, os gates, o único endpoint nem a regra
de sucesso aprovados. A análise primária continua condicionada à hipótese de
intercambiabilidade por tarefa; a sensibilidade não recebe status confirmatório.

A inspeção não certifica ausência de outros parentescos algorítmicos ou de
exposição no pré-treinamento. A reserva é independente dos resultados locais
anteriores e do ajuste dos gates no sentido documentado, sem garantir ausência
de toda dependência entre seus problemas.
