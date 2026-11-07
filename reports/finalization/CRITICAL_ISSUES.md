# Triagem crítica da versão recebida

Auditoria iniciada em 07/11/2026, 21:41:15 UTC, com limite de 180 minutos.
A autorização posterior de GPU permite usá-la se necessário; a verificação
planejada continua baseada em scores existentes, sem novas gerações ou ajustes.

A versão recebida já é a edição integrada de 58 páginas, e não a antiga edição
de 75 páginas descrita na missão. A triagem considera os fontes efetivamente
incluídos por `tcc/main.typ`. Protocolos e relatórios históricos não serão
reescritos para aparentar terem sido redigidos depois dos resultados.

| ID | Classe | Constatação na entrada | Prioridade / ação |
|---|---|---|---|
| C01 | Científica ou estatística | O texto já distingue interação positiva, incremento absoluto, equivalência e métricas secundárias. Os números ainda precisam de reconstrução independente nesta rodada. | Alta: reconstruir AUC por pares, contrastes, bootstrap, Γ, poder e replay a partir dos arquivos salvos. |
| C02 | Entre capítulos | Resumo/abstract não identificam explicitamente a validação independente de 120 tarefas e suas retenções, embora o corpo a discuta. | Alta: incorporar o achado histórico sem alongar excessivamente o resumo. |
| C03 | Proveniência | README principal descreve o trabalho como restrito ao Qwen e oferece uma cadeia antiga de publicação; pode induzir o leitor a reconstruir outra apresentação. | Alta: substituir a entrada pública por um mapa da edição final e comandos verificados. |
| C04 | Proveniência | A pasta de trabalho mistura fontes usados, bancos extensos, arquivos temporários e apresentações históricas. Copiar tudo para o GitHub não representa uma entrega final curada. | Alta: preparar cópia de publicação com seleção explícita, manifesto de hashes e instruções, preservando o workspace e o índice do Git. |
| C05 | Bibliográfica | Russell e Wefald (1991) estão no documento conceitual congelado, mas não constam da bibliografia ativa. Callaway, BEACON e DeepSeek já constam. | Média: conferir as quatro fontes primárias, acrescentar a referência ausente e citar no texto. |
| C06 | Editorial | A conclusão integra os dois experimentos, mas não organiza explicitamente a resposta às cinco perguntas formuladas na introdução. | Média: tornar o vínculo identificável, mantendo os resultados negativos e inconclusivos. |
| C07 | Editorial | As antigas quebras da tabela de probes e a afirmação de apenas um gerador já foram corrigidas na edição recebida. | Verificar novamente no PDF final; não registrar como erro novo. |
| C08 | Matemática | A redução pV ≥ c já explicita a hipótese de fallback comum. A recursão ainda pode tornar mais explícitas as atualizações de orçamento na aquisição de sinal e verificação. | Média: conferir equações, fronteiras e solver sintético; esclarecer somente onde necessário. |
| C09 | Proveniência | Conversões de fim de linha pelo Git podem invalidar os hashes congelados em um clone, mesmo sem mudar o conteúdo científico. As figuras ativas também precisam integrar a proveniência editorial. | Alta: incluir os SVG usados no manifesto e testar a cópia pública através de um Git isolado, com conversão automática habilitada. |
| C10 | Bibliográfica / editorial | A inspeção visual final encontrou referências de anais com “In:” vazio: o estilo ABNT suprime o booktitle quando falta eventtitle. | Corrigido: eventtitle acrescentado a quatro entradas de anais, sem alterar seu tipo; compilação confirma ausência do prefixo vazio. |

Não foi identificado, na triagem, um erro numérico que justifique alterar os
resultados originais. A verificação efetiva e as limitações ficarão em
`STATISTICAL_AUDIT.md`; a resolução de cada item, em `FINAL_REVIEW.md`.
