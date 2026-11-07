# Auditoria científica inicial — 17/10/2026

## Veredito e reprodução

O resultado negativo principal é reproduzível a partir dos bancos e das ativações originais. A auditoria não encontrou vazamento entre tarefas nos registros das partições examinadas. Isso não constitui prova de ausência de toda forma de adaptação humana ou contaminação de pré-treinamento.

`python scripts/run_excellence.py audit` reconstruiu as métricas, contou pares correto/incorreto diretamente para WT-AUC, aplicou os estimadores B/H+B salvos aos vetores originais e reconstituiu H e seus calibradores a partir dos ajustes de cross-fitting registrados. O maior erro absoluto de probabilidade foi inferior a 5×10⁻¹⁶. Os 270 registros de partições de calibração foram verificados como disjuntos e completos dentro do respectivo treino externo. As escolhas finais satisfazem a regra one-SE nas grades arquivadas.

| Verificação | Valor reconstruído |
|---|---:|
| Piloto | 960 tentativas, 32 tarefas |
| Principal | 2.640 tentativas, 132 tarefas |
| Transferência | 640 tentativas, 32 tarefas |
| Risco estrito T>32 | 2.158 tentativas, 132 tarefas, 92 mistas |
| B: AUC / WT-AUC / log loss | 0,700539 / 0,628155 / 0,630217 |
| H: AUC / WT-AUC / log loss | 0,592359 / 0,568109 / 0,686866 |
| H+B: AUC / WT-AUC / log loss | 0,645650 / 0,581126 / 0,668067 |
| ΔWT-AUC(H+B−B), IC 95% | −0,047029 [−0,099375; 0,005583] |
| Δlog loss(H+B−B), IC 95% | +0,037851 [+0,006657; +0,070616] |

Dados detalhados e erros numéricos por fold: [reproduction.json](reproduction.json). Fontes e publicação recebidas: [baseline.json](baseline.json), `archive_before_extension/`. As seeds de todas as 4.240 trajetórias foram reconstituídas de suas identidades; não houve colisões dentro dos bancos.

## Problemas e limitações por gravidade

| Gravidade | Constatação | Consequência e tratamento |
|---|---|---|
| Crítica se ocultada; já divulgada | Aprovação do orientador e tag científica não existem; há autorização administrativa prospectiva do estudante | Usar “desenho fixo executado”, divulgar a emenda e não alegar pré-registro independente. Nova confirmação requer aprovação explícita. |
| Alta, limitação | Há 26 tarefas principais em artefatos locais anteriores | O teste não é inteiramente intocado pelo pesquisador. Sensibilidade sem essas tarefas é exploratória e não apaga exposição ou refaz o treino. |
| Alta, limitação | Bootstrap mantém estimadores OOF fixos; folds compartilham treinamento | Intervalos condicionais, não incerteza integral de uma nova execução do procedimento. Analisar estabilidade e propor validação independente; não declarar equivalência pelo resultado negativo. |
| Média, erro de apresentação confirmado | O complemento anterior tratava IC de log loss inteiramente negativo como condição necessária de Q2 | O protocolo v0.2 exige redução pontual de log loss e divulgação/qualificação do intervalo, além do IC positivo de ΔWT-AUC. Corrigir a tabela complementar e testar fronteiras. Não altera o resultado original, pois a diferença de perda é positiva. |
| Média, documentação desatualizada | `docs/TCC_GUIDELINE.md` ainda informa 21/960 e análise principal pendente | Atualizar o guia e revisar tempos verbais no manuscrito sem reescrever protocolos históricos. |
| Média, desenho exploratório | C=0,001 foi escolhido para H e H+B nos cinco folds principais, na borda mais regularizada da grade | A grade pode limitar a investigação explicativa. Testar C ainda menor e PCA com seleção exclusivamente interna; não substituir o endpoint original. |
| Média, validade externa | Retenção OOF na origem e transferência de modelo ajustado em toda a origem são procedimentos distintos | Comparar distribuições condicionais e gates por checkpoint; não atribuir automaticamente a diferença a domain shift. |
| Média, recursos | A GPU conectada é RTX 3050 Laptop 6 GB, não RTX 4060 | Dimensionar a extensão com medição real, preservando o registro histórico. A RTX 4060 precisará ser validada se usada em outro ambiente. |
| Baixa, interpretação de diagnóstico | Bootstrap percentil de ECE pode não conter o ECE empírico, por viés e não suavidade | Conservar a ressalva, bins fixos e contagens; não usar ECE isoladamente para selecionar ou declarar superioridade. |

## Checklist metodológico

| Item | Evidência examinada | Resultado / limite |
|---|---|---|
| 1. Vazamento treino–teste | IDs de seleção, metadados dos modelos, recomputação no teste | Partições externas disjuntas; não equivale a ausência de contaminação de pré-treino. |
| 2. Agrupamento por tarefa | `GroupKFold`, IDs por candidato e auditoria | Todas as tentativas da mesma tarefa permanecem no mesmo fold externo. |
| 3. Seeds | Hash de versão, benchmark, tarefa e tentativa | Todas reconstruídas; identidades não se repetem. Determinismo numérico é condicionado ao ambiente. |
| 4. Validação aninhada | Grades internas e código de seleção | Seleção no treino externo; one-SE é heurística, não intervalo de confiança. Não foi reexecutada toda a busca histórica em todos os checkpoints. |
| 5. Camadas e C | Grades completas arquivadas e escolhas | Regra final reproduzida; escores H principais reconstituídos. |
| 6. Cross-fitting e calibração | 270 partições e reconstrução de calibradores H | Seleção é cross-fitted; calibrador e gate são ajustes de desenvolvimento, sem certificação independente de recall. |
| 7. WT-AUC | Contagem explícita de pares e empates | Média de AUC por tarefa mista reproduzida sem chamar sklearn para a contagem de pares. |
| 8. Tarefas mistas | Rótulos reais no risco T>32 | 92 mistas; homogêneas permanecem nas perdas e resolução. |
| 9. Bootstrap | Contribuições pareadas e seed 20260921 | IC de WT-AUC e perda reproduzidos com 2.000 reamostragens por tarefa. |
| 10. Término precoce | Filtro estrito T>t e ordem das decisões | Candidato terminado em t é verificado antes do gate. |
| 11. Alinhamento dos estados | Código e relatórios de invariância prefixo/full/cached | O token amostrado é alimentado antes da captura; testes arquivados passam, com tolerância de armazenamento. Cobertura numérica não é prova para toda configuração futura. |
| 12. Features sem futuro | Slices `[:checkpoint]` e indicadores de prompt | Comprimento futuro não entra nos preditores precoces. Teste de invariância à alteração do sufixo aprovado. |
| 13. Verificador | EvalPlus 0.3.1, suites base e plus, controles | Rótulos concordam com os status oficiais; passar em testes não prova corretude universal. |
| 14. Erros e truncamento | Protocolo, worker, registros | Infraestrutura interrompe coleta; truncamento é atributo, não negativo automático. Piloto tem 1 truncamento; bancos principal e alvo têm zero. |
| 15. MBPP+ congelado | Bundle, hashes, momento da coleta e função de scoring | Não há fitting com os rótulos alvo; nova investigação também não ajustará o gate usando MBPP+ explorado. |
| 16. Replay | Implementação, decisões salvas e validação online original | Contabiliza apenas tentativas iniciadas; candidato abortado não resolve tarefa. Nova implementação independente reproduziu as políticas tarefa a tarefa; oracle e casos sintéticos verificaram a contabilidade. |
| 17. Inferência e hipóteses | Código H2/H3 e protocolo v0.2 | Associação por permutação não testa incremento condicional; Q2 exige o critério exato do protocolo. |
| 18. Consistência documental | JSON, tabelas, PDF, notebook e fontes | Números principais conferem; há os reparos de apresentação e documentação listados acima. |

## Próximas análises autorizadas

Ablação de observáveis; regularização forte/PCA; offset residual com primeiro estágio integralmente cross-fitted; diagnósticos do gate transferido; oracle retrospectivo e formalização econômica; benchmark técnico pareado em tarefas já examinadas. Os resultados serão exploratórios e terão identidades, fontes, comandos, seeds e hashes. Nenhuma nova avaliação independente foi iniciada.

## Complemento após as execuções

As análises acima foram executadas; a síntese está em [README.md](README.md), com tabelas e insumos por estágio. A auditoria atual dos três bancos reabriu todos os arquivos de ativação e verificou sua integridade e finitude. O replay independente reproduziu os resultados originais tarefa a tarefa. A repetição integral das 70 unidades exploratórias apresentou erro de probabilidade igual a zero.

Foi confirmado outro erro de apresentação de gravidade média: o método afirmava execução sem rede e credenciais em isolamento completo. O fonte instalado de EvalPlus0.3.1, `evalplus/eval/utils.py`, declara que reliability_guard não é sandbox de segurança, e o cliente utilizado lança subprocesso WSL sem demonstrar aquelas restrições. A redação foi corrigida para o ambiente e os limites efetivos. [Fonte primária do mecanismo](https://github.com/evalplus/evalplus/blob/master/evalplus/eval/utils.py). Isso não altera os rótulos já obtidos; cinquenta programas conhecidos tiveram seus rótulos reproduzidos pelo checker na nova medição.

O gerador bibliográfico também descartava entradas adicionadas quando reexecutado. A atualização passou a preservar essas referências, com regressão de idempotência. São 19 entradas após atualização pelas páginas primárias. Não se fabricou confirmação independente, aprovação, publicação ou aceite.

Na inspeção do PDF, a classe bibliográfica genérica dos 13 preprints introduzia uma vírgula sem mês após o ano. Essas entradas foram classificadas como manuscritos não publicados, preservando títulos, autores, anos, URLs e chaves. Uma compilação mínima e a regeneração do manuscrito verificaram a pontuação corrigida; não se atribuiu publicação em periódico a um preprint.

A suíte integral com PyTorch habilitado encerrou com 171 testes aprovados, sem falhas nem testes ignorados. Ruff passou e mypy não encontrou problemas em 119 fontes. O benchmark de 50 pares/100 gerações confirmou igualdade exata de tokens e estados. Os testes do novo coletor usam backends simulados para persistência e falhas; a integração de geração seletiva foi medida nas tarefas antigas, enquanto a nova coleta científica aguarda a revisão exigida.

O diagnóstico adicional de prefixos preservou o filtro estrito T>t e confirmou scores constantes para prompts e prefixos idênticos. Em t=8, 15 das 100 tarefas mistas de origem e 3 das 24 do alvo tinham somente um prefixo observado; em t=32, nenhuma das 92 tarefas mistas de origem nem das 18 do alvo estava nessa situação. Portanto, essa forma extrema de prefixo indistinguível não explica o resultado negativo principal. Colisões parciais e informação funcional ainda indisponível no prefixo continuam possíveis; não se identificou uma causa única. Dados completos: [prefixes/results.json](prefixes/results.json).

Ao habilitar os quatro testes opcionais de PyTorch, um teste legado esperava uma direção bruta embora a função use normalização L2 por padrão. A execução com essa falha foi arquivada em `torch_test_failure_before_repair.xml`. O teste agora verifica explicitamente ambos os modos, preservação das posições não selecionadas e ausência de mutação da entrada. A implementação das intervenções e os fontes científicos congelados não foram alterados. O README documenta a suíte com PyTorch habilitado e inicializado antes das outras bibliotecas científicas.

<!-- independent-completed -->
## Validação independente após aprovação

O relato explícito do estudante autorizou o congelamento prospectivo da proposta
e informou revisão do orientador. O protocolo manteve seu hash e foi copiado com
os fontes para um snapshot de 46 arquivos antes da primeira trajetória nova.
Foram concluídas 400 tentativas técnicas e 2.400 principais. A auditoria conferiu
identidades, seeds, hashes de todas as ativações, controles e probabilidades
reaplicadas: erro máximo 4.440892098500626e-16.

Endpoint único H+B−B em t=8: -0.16359060402684564; IC95% [-0.2402298061170319, -0.0942741120201171]; critério **satisfeito**. Os critérios
originais e os resultados exploratórios foram preservados. A revisão informada
pelo estudante não foi transformada em assinatura ou pré-registro externo.
Detalhes e limitações: [independent_v1/README.md](independent_v1/README.md).
