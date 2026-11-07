# Revisão integrada do manuscrito — 7 de novembro de 2026

O [manuscrito revisado](../../tcc/build/manuscrito.pdf) apresenta as duas etapas
como partes da mesma investigação. A segunda etapa, com DeepSeek e comparação
entre regressão logística e MLP, participa da pergunta central desde o resumo
e a introdução. A cronologia dos experimentos continua explícita.

## Redação e estrutura

- Resumo e abstract descrevem os dois geradores, os procedimentos comparados,
  o teste externo e a conclusão primária inconclusiva.
- A introdução explica a diferença entre reconhecer tarefas fáceis, ordenar
  tentativas da mesma tarefa, obter incremento sobre observáveis e melhorar
  uma decisão de interrupção.
- A fundamentação conecta extração causal, métricas, calibração, capacidade
  dos probes e custo sequencial. A recursão de Bellman explicita estados,
  ações, fronteiras e as condições necessárias à regra simplificada de limiar.
- A metodologia reúne as duas etapas, identificando diferenças de ajuste,
  separação de tarefas, tokenização, conjuntos ativos e congelamento.
- Os resultados mantêm capítulos próprios para a primeira etapa e para a
  comparação prospectiva. Tabelas numéricas são preenchidas a partir dos
  resultados existentes; duas figuras sintetizam os contrastes e a calibração.
- Discussão e conclusão respondem às mesmas perguntas. Não descrevem MLP ou
  um segundo gerador como experimentos ainda ausentes.
- Referências e apêndice de reprodutibilidade foram integrados à estrutura.

## Preservação científica

Não houve nova coleta, seleção de hiperparâmetros, ajuste de probes ou
reestimação de intervalos nesta revisão. Os scripts científicos congelados,
resultados e artefatos históricos protegidos passaram pela verificação de
integridade.

Foram preservados o resultado Qwen original, os diagnósticos exploratórios,
as transferências e a validação independente de 120 tarefas. O contraste
primário DeepSeek/MLP continua sendo ΔWT-AUC = −0,0042, IC95%
[−0,0648; 0,0548]. A interação secundária positiva no Qwen não é apresentada
como ganho absoluto de H+B. Melhorias nas perdas não substituem o teste de
ranking, e ausência observada de tarefas perdidas não certifica segurança
populacional.

A versão anterior de 75 páginas e seus fontes estão preservados em
[before/](before/), com hashes no manifesto do arquivo. A nova organização
reduz repetições e substitui a apresentação antiga por uma discussão conjunta.
Os arquivos anteriores continuam disponíveis.

## Validação da edição

- PDF final de 58 páginas, compilado sem avisos, inclusive de citações e
  referências cruzadas.
- Hashes dos insumos, fontes editoriais, arquivos gerados e PDF conferidos.
- Quatro contrastes externos de WT-AUC e seus IC encontrados no PDF;
  resultados históricos centrais também conferidos.
- Busca por declarações antigas de escopo no conjunto de fontes ativas.
- Inspeção visual do resumo, fórmulas, tabelas, figuras e conclusão.
- Verificação geométrica sem conteúdo fora da largura e sem blocos abaixo
  da margem inferior verificada.
- Ruff aprovado para os scripts de construção, renderização e auditoria.

O registro verificável está em [validation.json](validation.json), e a
proveniência da construção está em [manifest.json](manifest.json).

Para reconstruir e conferir a edição, a partir da raiz do projeto:

```powershell
.venv/Scripts/python.exe scripts/build_manuscript.py --compile
.venv/Scripts/python.exe scripts/audit_integrated_manuscript.py
```

Essas verificações documentam a consistência editorial local. A revisão não
altera o estatuto das análises nem substitui a apreciação acadêmica do texto.
