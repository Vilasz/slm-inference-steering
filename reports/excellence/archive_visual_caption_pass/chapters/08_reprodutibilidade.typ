#import "../lib.typ": *
= Reprodutibilidade e histórico metodológico

== Artefatos e comandos

O protocolo vigente está em `RESEARCH_PROTOCOL.md`; o anterior foi preservado
em `docs/protocols/RESEARCH_PROTOCOL_v0.1.md`. O texto originalmente enviado
foi preservado em `tcc/archive/`. O notebook executado é
`notebooks/final_research_summary.ipynb`; sua exportação HTML permite leitura
sem iniciar um kernel. O manifesto `reports/final_review/evidence.json` contém
os hashes dos arquivos de entrada e o escopo de cada resultado.

```powershell
$env:SLM_STEERING_TEST_TORCH = "1"
.venv/Scripts/python.exe -c `
 "import torch,pytest; raise SystemExit(pytest.main(['-q']))"
.venv/Scripts/python.exe scripts/build_research_delivery.py
.venv/Scripts/python.exe scripts/build_final_notebook.py --execute
.venv/Scripts/python.exe scripts/build_manuscript.py --compile
```

A geração de evidência precisa dos arquivos históricos em `runs/`; o notebook
apresenta os resultados já exportados. Os capítulos de fundamentação e método
são editáveis. As seções empíricas são geradas a partir de um único manifesto,
reduzindo divergência entre números no texto, tabelas e notebook.

== Correções da versão 0.2

#grade(
 [*Dimensão*], [*Implementação revisada*],
 [Desfecho], [Aprovação funcional; truncamento registrado separadamente],
 [Temporalidade], [Término antes do gate; conjunto ativo estrito],
 [Margem de logits], [Após temperatura e antes do top-p],
 [Seleção e calibração], [Cross-fitting da seleção; avaliação externa por tarefa],
 [Inferência], [Associação por fold; incremento por deltas externos pareados],
 [Política], [Busca por custo em todos os limiares admissíveis, incluindo ausência de aborto],
 [Transferência], [Bundle treinado na origem, congelado antes da coleta alvo],
)

Nenhuma alteração é apresentada como anterior a resultados já conhecidos.
Há arquivos históricos de 30 tarefas HumanEval, incluindo 26 do conjunto hoje
reservado para a avaliação principal. Essa exposição é diferente de vazamento
entre os folds de treinamento, mas limita alegações de um teste totalmente
intocado pelo pesquisador. A aprovação científica e o congelamento continuam
sendo decisões reais do aluno e do orientador.

Em 14 de outubro de 2026, o estudante autorizou a execução integral. A emenda
administrativa e o snapshot verificável dos fontes registram essa autorização
antes do banco principal. O registro histórico mantém pendente a aprovação do orientador;
não se infere aprovação a partir da execução e não se reivindica pré-registro
independente. O arquivo normativo original foi preservado para manter a
proveniência do piloto, então em andamento. A validação online usa as primeiras 12
tarefas por ID lexicográfico, sem escolha por economia observada.

A extensão EXCELLENCE_EXTENSION_v1 tem aprovação posterior própria: o estudante
aprovou a proposta e informou revisão do orientador antes do snapshot de 46
arquivos e das novas trajetórias. Esse relato não foi convertido em assinatura
nem aprovação retroativa do experimento original. Os helpers de parsing e dataset,
omitidos como entradas separadas do snapshot, coincidem com os bytes recuperados
do HEAD e patch preservados antes da aprovação; o recibo explicita essa limitação.
Comandos de análise estatística, publicação e coleta longa estão separados em
`docs/FINAL_REPRODUCIBILITY.md`. Análise e compilação não iniciam nova geração.

== Integridade e assistência computacional

Em 15 de outubro de 2026, uma sessão de coleta foi interrompida por bloqueio
temporário do Windows durante a substituição atômica do manifesto JSON. A
auditoria confirmou 2.433 registros completos, que foram preservados na
retomada. O launcher `scripts/run_research_safely.py` acrescenta tentativas
limitadas a dez segundos para essa operação de filesystem. Os fontes
científicos congelados, os bytes publicados, as sementes e os procedimentos
numéricos permanecem preservados. O registro administrativo, os hashes da
camada de compatibilidade e o teste com bloqueio real estão documentados em
`docs/windows_execution_recovery.md` e nos artefatos de recuperação em
`reports/research/`.
O manifesto preservado da sessão interrompida registra pico reservado de
5.582 MiB, superior ao pico de aproximadamente 4,49 GiB do piloto. O valor
da última sessão concluída, isoladamente, não resume o pico de toda a coleta.

Os testes verificam rótulos, sementes, posições, término exatamente no checkpoint,
contabilidade do replay, isolamento de tarefas e ausência de ajuste com rótulos
do benchmark de transferência. Nenhum teste demonstra, por si só, ausência de
adaptação humana ao conjunto de dados. O histórico de decisões complementa as
verificações automatizadas.

Esta versão foi preparada com assistência de um agente de programação para
implementação, auditoria, organização dos artefatos e revisão textual. A autoria
científica requer que o pesquisador confira a argumentação, as referências e
as decisões experimentais e assuma responsabilidade pela versão submetida.

== Diagnósticos posteriores à análise principal

Após a conclusão das 19 etapas e a inspeção de seus resultados, a solicitação
de análises mais completas motivou uma extensão exploratória de publicação.
`scripts/analyze_research_robustness.py` reproduz as análises sem ajuste de
modelos, calibradores ou limiares. Os fontes científicos congelados e todos
os bancos permanecem idênticos por hash. A publicação anterior e os hashes
protegidos estão em `reports/research/publication_before_supplement_20261016/`.

Os diagnósticos incluem contrastes pareados por tarefa, folds externos,
influência da exclusão de uma tarefa, coorte comum T > 64, sensibilidade à
exclusão das tarefas presentes no histórico, calibração da probabilidade
positiva em dez bins fixos e transporte da retenção dos corretos. Cada tarefa
tem peso total igual nas perdas e na calibração; a retenção usa razão das
contagens de candidatos, reamostrando tarefas como clusters. São 2.000
reamostragens com seed 20260921, sem refazer treinamento.

O gerador exige reprodução das estimativas e intervalos principais. As
extensões não introduzem margens de equivalência nem substituem os critérios
originais. Os intervalos são pontuais e condicionais aos escores; comparações
exploratórias múltiplas não recebem interpretação confirmatória. A coorte de
trajetórias longas condiciona no comprimento final; a exclusão histórica
não refaz modelos nem apaga decisões anteriores. O relatório complementar
exporta tabelas por tarefa e bin, figuras vetoriais, definições e proveniência.
