# Guia atual do TCC

Referência: 2026-10-14. As correções identificadas na revisão anterior foram
implementadas no protocolo v0.2 e no código. A revisão histórica foi arquivada
em [TCC_GUIDELINE_v0.1.md](legacy/TCC_GUIDELINE_v0.1.md).

## Materiais para revisar

- [Notebook executado](../notebooks/final_research_summary.ipynb) e
  [HTML](../reports/final_review/notebook.html).
- [Manuscrito PDF](../tcc/build/manuscrito.pdf), fonte em `tcc/main.typ`.
- [Estado das execuções](../RESULTS.md), [protocolo](../RESEARCH_PROTOCOL.md) e
  [comandos atuais](research_workflow.md).

O manuscrito contém fundamentação, metodologia, resultados observados,
discussão e bibliografia reconstruída por fontes primárias. A análise principal
precoce permanece pendente; a redação distingue esse fato dos resultados tardios
já obtidos. Confira as referências e as decisões antes da submissão.

## O que já foi executado

Há reanálise exploratória de 150 tentativas em 30 tarefas, usando estados finais
e seleção interna com avaliação agrupada. O ganho de WT-AUC de H+B sobre B tardio
é 0,021, com intervalo incluindo zero; a log loss melhora. A parada pelo
verificador economiza tokens no mesmo banco. Esses resultados não demonstram
aborto precoce.

O piloto revisado possui 21/960 trajetórias, com teste causal, reprodução de
seeds, rótulos oficiais, contabilidade e retomada auditados. O verificador MBPP+
passou nos controles positivo/negativo. O notebook executa e o PDF compila.

## Sequência ainda necessária

1. Completar as 939 trajetórias restantes do piloto, sem escolher temperatura
   pela AUC dos probes.
2. Selecionar o regime e registrar aprovação/congelamento científico reais.
3. Gerar e auditar 2.640 trajetórias HumanEval+ nas 132 tarefas principais.
4. Produzir predições externas por checkpoint e concluir Q1/Q2.
5. Avaliar as políticas B/H+B por replay e validação online, incluindo tempos.
6. Executar MBPP+ com sistema congelado, se o orçamento permitir a extensão.

A aprovação do orientador é exigida pelo protocolo; nenhuma aprovação foi
fabricada. As tarefas historicamente expostas permanecem declaradas.

## O que olhar e como escrever

Em Q1, compare AUC agregada e WT-AUC e reporte a população ativa em cada posição.
Em Q2, preserve 32 tokens e examine os deltas H+B−B e seus intervalos. Permutação
apoia associação, não equivale ao teste do ganho sobre B. Em Q3, compare contra
verificador e B, com margem de resolução de 3 pp e evidência de redução de tokens;
economia de tempo exige medições próprias. A retenção de 97% no desenvolvimento
é uma meta empírica, sujeita a falha em tarefas novas.

Mantenha os sete capítulos do manuscrito: introdução, relacionados, fundamentação,
metodologia, resultados, discussão e conclusão. Capítulos autorais são editáveis;
seções empíricas derivam de `reports/final_review/evidence.json`. Preserve os
denominadores, intervalos, limitações e proveniência. Um resultado inconclusivo
preciso e bem interpretado é cientificamente útil.

A versão atual do renderizador descreve evidência exploratória. Após executar
os experimentos principais, revise sua interpretação, resumo e conclusão
explicitamente. Evite tratar atualização de números como atualização automática
da validade de uma reivindicação científica.
