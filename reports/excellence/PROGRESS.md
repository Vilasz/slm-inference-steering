# Programa de aprimoramento científico

Solicitação: 17/10/2026 (data informada pelo estudante). Execuções registram timestamps UTC reais.

## Escopo autorizado e preservação

- Auditoria, reprodução, análises exploratórias, correções técnicas reversíveis, benchmark técnico em tarefas já examinadas, escrita e materiais de defesa estão autorizados.
- Novo protocolo independente será apresentado para aprovação do estudante e revisão do orientador **antes** do congelamento e de qualquer coleta confirmatória adicional.
- O desenho v0.2, bancos, seeds, modelos e resultados originais são protegidos por hashes em `baseline.json`. A publicação recebida foi copiada para `archive_before_extension/`.
- O estado inicial do Git e o patch das alterações preexistentes estão arquivados. Nenhuma alteração anterior será revertida, commitada ou apagada.

## Registro inicial da missão

1. Inspeção dos protocolos, manuscrito, geração, extração temporal, avaliação, validação aninhada, calibração e replay: realizada; inventário auditável em preparação.
2. Suíte de regressão inicial: em execução; relatório em `.tmp/excellence_baseline_tests.xml`.
3. Reprodução independente das métricas e escores do endpoint principal: em execução com os vetores originais, contagem de pares e reconstituição dos calibradores H.
4. Hardware: `nvidia-smi` detectou **RTX 3050 Laptop, 6.144 MiB, compute capability 8.6, driver 576.52**. A RTX 4060 mencionada na missão não está conectada neste ambiente. O protocolo novo será dimensionado com medições do dispositivo real.
5. Novas análises: ablação de observáveis, regularização além da grade original, PCA e correção residual com offset e cross-fitting completo do primeiro estágio; diagnóstico do gate transferido e oracle econômico.

## Limitações já identificadas

- As tarefas e resultados existentes já foram examinados; todas as novas análises sobre esses dados são exploratórias.
- Folds externos compartilham treinamento; os intervalos condicionais a escores fixos não cobrem toda a variabilidade de retreino.
- Aprovação do orientador permanece pendente no histórico original; não se presume aprovação ou pré-registro independente.
- A classificação de erros do checker não separa assertion e runtime de modo confiável; a análise preservará os status disponíveis.
- `docs/TCC_GUIDELINE.md` ainda descreve piloto e avaliação principal concluídos como pendentes: correção editorial necessária.
- A tabela complementar anterior impõe IC de log loss inteiramente negativo como condição necessária, embora o protocolo v0.2 exija redução pontual de log loss e qualificação de sua incerteza. A interpretação será corrigida; o veredito original continua negativo.

## Aprovações futuras

Nenhuma nova coleta independente foi iniciada. A proposta `docs/protocols/EXCELLENCE_EXTENSION_v1.md` será preparada após os diagnósticos e medições; aprovação será solicitada sobre um documento concreto, com amostra, hipótese, precisão e orçamento definidos.

## Entregas executadas em 17/10/2026

- **Auditoria e reprodução:** três bancos completos e todos os arquivos de ativação rechecados; 4.240 seeds reconstruídas, 270 partições de calibração auditadas. B/H/H+B no endpoint principal reconstituídos com erro máximo inferior a 5×10⁻¹⁶. Métricas e IC principais reproduzidos por contagem independente de pares e perdas por tarefa.
- **Ablações e leituras:** 14 famílias novas em cinco folds, 70 unidades. Nova execução integral após limpeza de nomes de variáveis reproduziu exatamente as probabilidades. A primeira execução e seus fontes estão arquivados.
- **Cross-fitting residual:** testes alteram os rótulos de validação e verificam que os offsets correspondentes não mudam, incluindo a seleção e calibração do primeiro estágio. PCA só vê treinos de cada split.
- **Geometria:** espectros de treino por camada, direções na mesma camada, contribuição dos blocos, previsibilidade do componente latente por B e curvas de treinamento com 32, 63–64 e 105–106 tarefas.
- **Transferência:** scores de desenvolvimento de origem reconstituídos em 8/16/32/64 tokens; nenhuma calibração ou escolha de limiar no alvo. Curvas de retenção, scores condicionais, prevalência, comprimento, concentração dos descartes e perda de tarefas exportados.
- **Economia:** implementação independente reproduziu todas as políticas originais tarefa a tarefa. Oracle em 8/16/32/64 e N=1/5/10/20, preservando os corretos. Equações e hipóteses de custo finito/ilimitado, retenção global/ativa, heterogeneidade e overhead acrescentadas à fundamentação.
- **GPU:** 50 pares/100 gerações, 26 tarefas conhecidas; tokens e estados exatamente iguais entre capturas. Cinquenta verificações oficiais repetiram os rótulos. Prefill, forward, cópia de estados, score CPU e verificador foram separados; tempos descritivos, com concorrência CPU documentada.
- **Recursos:** RTX3050 Laptop 6GB, CUDA12.6, Torch2.12.0+cu126, driver576.52, pico reservado4.978MiB. Média de geração instrumentada original8,764s versus seletiva8,475s; isso não demonstra aceleração da política.
- **Proposta independente:** vinte tarefas técnicas e 120 principais reservadas por hash entre345 IDs elegíveis, excluindo32 tarefas geradas e um controle. Comparação B/H+B no mesmo t=8; 2.800 tentativas no total. Estimativa4,44h, com margem de50%:6,66h. Protocolo, IDs, precisão, gates e hashes propostos publicados; aprovação/revisão solicitadas, sem resposta presumida.
- **Executor futuro:** implementação separada, bloqueada por aprovação e hashes. Testes com backends simulados verificam idempotência, persistência, seleção de estados e ausência de negativo fabricado após crash. A nova coleta real permanece não executada; esses testes não a substituem.
- **Materiais acadêmicos:** research summary em inglês com exatamente duas páginas; defesa com 12 slides; roteiro e 33 perguntas respondidas. Manuscrito de 48 páginas e notebook com 18 células de código executadas, sem erros, reconstruídos e auditados.
- **Prefixos compartilhados:** diagnóstico em 8/16/32/64 tokens, com scores constantes para prompts e prefixos idênticos. Em t=32 nenhuma das 92 tarefas mistas da origem nem das 18 do alvo tem um único prefixo; esse colapso integral não explica o resultado negativo principal. Em t=8 são 15/100 e 3/24, respectivamente. Colisões parciais permanecem documentadas.

## Resultados e hipóteses examinadas

| Hipótese explicativa | Resultado | Alcance |
|---|---|---|
| Basta aumentar a regularização conjunta | Não sustentada na família testada; WT-AUC0,548 e log loss0,676 | Não testa penalização separada de blocos nem todos os algoritmos |
| A parametrização da leitura contribui para a desvantagem | PCA64 recupera WT-AUC0,630/perda0,632; residual0,635/0,629 | Evidência de sensibilidade; IC contra B incluem zero, sem causa única identificada |
| H é apenas cópia linear de B | Não estabelecida; correlação centrada H/B≈0,170 e R² baixo | Distinção geométrica não implica utilidade preditiva incremental |
| A retenção de97% de desenvolvimento se transporta | Não se manteve no alvo examinado:276/366=75,41% | Não separa causalmente todos os efeitos de ajuste, população e domínio |
| A perda de candidatos equivale à perda proporcional de tarefas | Não;90 descartes e duas tarefas perdidas | Denominadores e multiplicidade de sucessos precisam ser separados |
| Falta espaço econômico para interrupção | Oracle t8 poupa77,75% dos tokens HumanEval+ e76,25% MBPP+ | Contrafactual com rótulos finais, preserva corretos; não é online nem limite universal |

## Correções técnicas/editoriais

1. Critério complementar de Q2 corrigido para a redução **pontual** de log loss prevista no protocolo; intervalo continua exposto. Teste de regressão cobre IC que cruza zero com estimativa negativa. Decisão original não mudou.
2. Atualização bibliográfica agora preserva entradas adicionadas, com teste de idempotência;19 referências mantidas. Ruan2026 foi explicitamente vinculado à versão2 consultada.
3. Documentação e método deixam de descrever bancos concluídos como pendentes. Uma afirmação de isolamento completo do checker foi removida após inspeção do fonte: reliability_guard não é sandbox de segurança.
4. Inicialização CUDA anterior a SciPy/sklearn resolveu erro DLL1114 no launcher técnico Windows. Fontes congelados e ambiente numérico original não foram alterados.
5. Escrevemos documentos UTF-8 diretamente para evitar perda de acentos no pipe do Windows PowerShell. Artefatos finais passam por revisão textual e visual.
6. Os 13 preprints foram classificados como manuscritos não publicados na bibliografia, corrigindo uma vírgula vazia após o ano no estilo APA. Metadados e chaves foram preservados, sem atribuição de publicação em periódico. Compilação mínima, teste bibliográfico e PDF final verificados.
7. Os quatro testes opcionais de PyTorch foram habilitados. Um teste legado incompatível com o padrão de normalização passou a verificar os modos bruto e normalizado explicitamente, sem mudar a implementação. Falha anterior arquivada em `torch_test_failure_before_repair.xml`; suíte completa repetida com sucesso. O README inclui o comando de execução integral.

## Validação atual

Suíte integral com PyTorch habilitado: **157 passaram, nenhum ignorado**, sem falhas. Ruff passou; mypy não encontrou problemas em **119 fontes**. O teste bibliográfico foi repetido após o último ajuste de apresentação. Os avisos são de deprecação de Pyparsing/Matplotlib. A publicação final passou na auditoria de hashes, execução das células, estrutura dos PDFs e preservação dos resultados históricos.

As **19 etapas** da execução original estão concluídas e verificadas; os registros das **15 etapas científicas** coincidem integralmente com o estado arquivado antes desta extensão. Os 16 artefatos científicos protegidos e o snapshot dos fontes congelados conservam seus hashes. As 4.240 trajetórias originais permanecem intactas. Foram acrescentadas 100 gerações técnicas em tarefas conhecidas; nenhuma trajetória independente nova foi coletada.

Estado verificável em [final_verification.json](final_verification.json), [delivery_manifest.json](../final_review/delivery_manifest.json) e [progress.json](../research/full_execution_v0.2/progress.json). O PDF, o notebook e os materiais acadêmicos incorporam os diagnósticos finais. A inspeção visual conferiu legibilidade das figuras e da discussão; o resumo e os slides também tiveram páginas renderizadas e examinadas.

## Organização das mudanças

- `src/slm_steering/excellence/`: auditoria, modelos, offset, geometria, transferência, economia, benchmark, orçamento auxiliar, aprovação e executor independente.
- `scripts/run_excellence*.py`, `finalize_excellence_proposal.py`, `build_excellence_materials.py`: reprodução, preparação e entrega.
- `tests/test_excellence_*.py`: casos matemáticos, isolamento de partições, aprovação, persistência e bibliografia.
- `scripts/build_*`, `research_diagnostics.py`, `supplemental_publication.py`: integração aos artefatos anteriores, sem editar os fontes científicos congelados.
- `tcc/chapters/`, bibliografia, guias, proposta e defesa: interpretação e apresentação científica.

## Continuidade e decisões pendentes

A nova coleta exige aprovação do estudante e revisão do orientador, explicitamente solicitadas na Parte V da missão. Não se criou aprovação, tag ou resultado novo em nome deles. O executor só aceita o contrato revisto por hash.

Planejamento até meados de dezembro: concluir revisão/aprovações e piloto técnico em outubro; executar a avaliação independente aprovada e sua análise em novembro; reservar o início de dezembro para correções, ensaios e revisão do orientador, com margem antes da entrega. As datas são planejamento, sem execução agendada ou submissão automática.

## Fechamento da execução autorizada

Verificação final em 17/10/2026 às 07:08 UTC: 19/19 etapas verificadas; 16 artefatos históricos e 29 entradas do congelamento preservados; 157 testes aprovados sem ignorados; 18 células executadas sem erro; manuscrito com 48 páginas, research summary com duas e defesa com 12 slides/33 perguntas. O registro está em `final_verification.json`. Os processos de publicação encerraram. A nova confirmação permanece não executada, aguardando as aprovações solicitadas sobre a proposta concreta.

## Aprovação recebida e início da avaliação independente

Após o fechamento acima, o estudante respondeu à pergunta sobre esta proposta: **“Aprovo como estudante; orientador já revisou esta proposta”**. A aprovação e a revisão do orientador informada pelo estudante foram registradas em `docs/protocols/EXCELLENCE_EXTENSION_v1.approval.json`, vinculadas aos seis hashes apresentados. Não se criou assinatura do orientador nem se alegou verificação direta de sua manifestação.

O protocolo aprovado conservou os bytes e o hash da proposta. Um congelamento prospectivo com 46 arquivos foi criado em `reports/excellence/independent_v1/freeze/`, antes da primeira trajetória nova. Inclui amostra, configuração, gates, fontes, bundle, orçamento e registro da aprovação. O cabeçalho histórico da proposta continua descrevendo a situação da data em que foi redigida; o novo registro determina seu estado aprovado.

Piloto técnico de 400 tentativas iniciado na RTX 3050 Laptop de 6 GB. O supervisor `scripts/run_approved_excellence.py` acompanha a sessão inicial, retoma sessões duráveis de 600 segundos e só libera as 2.400 tentativas principais após a conclusão dos controles técnicos. Erros técnicos interrompem o fluxo; resultados não alteram tarefas, gates ou critério de confirmação. Progresso persistido em `reports/excellence/independent_v1/execution/status.json`.

Os primeiros controles reais passaram: identidade de tokens e estados entre as duas capturas, invariância de prefixo, verificador positivo/negativo e conferência do dataset. A primeira sessão terminou normalmente e o supervisor iniciou a seguinte sem duplicar identidades.

A integração dos resultados completos foi implementada em `independent_report.py` e `scripts/publish_excellence_independent.py`: auditoria de todas as ativações/identidades/seeds, reaplicação dos scorers, retenções com bootstrap por tarefa, métricas secundárias, calibração, replay e figuras. O endpoint original continua no coletor congelado; o relatório não altera sua decisão. Após as 2.800 tentativas, o supervisor executará a análise, a suíte integral e a reconstrução do PDF, notebook, resumo e defesa automaticamente. Não há resultados principais novos completos neste momento.

Validação adicional: **160 testes passaram sem ignorados**, incluindo razões de contagens versus médias de razões, ausência de corretos e calibração ponderada por tarefa com score igual a um. Ruff passou; mypy passou em 123 fontes. Fixtures sintéticas de diagramação, identificadas e restritas a `.tmp/independent_publication_fixture/`, verificaram resumo de duas páginas e 12 slides nos cenários negativo, cruzamento de zero e endpoint indefinido. Nenhuma fixture foi publicada como resultado experimental.

## Missão final: auditoria, apresentação e execução preservada

A solicitação final foi lida integralmente e o estado preexistente foi preservado em `final_pass_baseline/`, incluindo 69 figuras anteriores. A aprovação da proposta já recebida continua válida; não foi solicitada novamente.

Reexecuções estatísticas completas em pastas separadas reproduziram exatamente os JSONs de resultados principal, transferência e replay da transferência. A auditoria aritmética independente conferiu 39 linhas de preditores, 14.432 linhas de replay e 3.280 status funcionais. Casos manuais adicionais verificaram custo finito, oracle, término no checkpoint e ausência de informação futura. A suíte integral passou com 169 testes; um teste posterior do contrato funcional da nova namespace também passou (cinco testes específicos do relatório).

Uma divergência real de serialização foi explicada: o leitor CSV padrão perturba alguns empates de H em t=8 no alvo, alterando WT-AUC em cerca de 0,002193. A leitura round_trip reproduz todas as métricas originais; o leitor complementar foi corrigido sem mudar o endpoint, os scores gerados ou dados históricos. Os helpers parser/dataset não estavam separados no snapshot de 46 entradas; seus bytes foram recuperados do HEAD+patch preservados antes da aprovação e coincidem com os fontes atuais, sem fingir congelamento retroativo.

A auditoria detalhada dos prompts/testes foi feita após o início da coleta, sem novos rótulos. Não há duplicatas exatas nem estruturas canônicas normalizadas idênticas na reserva. Dois pares principais têm semelhança textual elevada; nenhum foi excluído. O relatório prevê sensibilidade posterior de bootstrap com esses pares agrupados, mantendo o estimando e o critério primário por tarefa.

A cronologia da seção 1.4 e as qualificações de aprovação foram corrigidas. Uma referência primária de 08/10/2026 foi incorporada, com matriz comparativa final. O guia chegou a 42 perguntas, antes das duas questões específicas que serão preenchidas com o resultado independente. O formato foi aproximado ao manual geral FGV efetivamente consultado; regras específicas do curso, ficha e aprovação formal continuam externas. Cópias vetoriais de impressão ampliam rótulos sem alterar dados ou fontes científicos congelados. A revisão visual já encontrou e corrigiu problemas de tamanho e espaçamento; a inspeção de todas as páginas finais será registrada após a publicação definitiva.

A implementação da medição operacional usa somente seis tarefas antigas do piloto, três repetições e até três tentativas, comparando gerador sem probe, captura seletiva, B e H+B. Ela só pode carregar o modelo após a nova coleta terminar. Seus tempos serão descritivos e uma falha técnica não redefine nem bloqueia o endpoint independente.

Em 17/10/2026 às 17:14 UTC: controles técnicos 400/400 concluídos; principal 1.441/2.400, controles numéricos aprovados. Não foram calculados endpoints parciais. O supervisor preserva o mesmo hash e os 46 inputs continuam idênticos.

## Retomada após conclusão da coleta

As duas fases terminaram: 400 tentativas técnicas e 2.400 principais. O endpoint congelado encontrou diferença de retenção H+B−B de −0,1635906040, IC95% [−0,2402298061; −0,0942741120], com 10.000 reamostragens e nenhuma indefinida. As contagens são 986/1.192 e 1.181/1.192, respectivamente. O resultado satisfaz o critério direcional pré-especificado de retenção inferior de H+B, sem alterar Q2/H3 originais.

A primeira publicação parou porque uma variável de configuração sobrescrevia a chave da fase no auditor complementar. O erro foi corrigido somente no relatório não congelado. O status e o log da falha foram arquivados em `independent_v1/execution/publication_failure_*/`; a publicação foi retomada sem regenerar trajetórias ou modificar fontes científicos congelados. Medições operacionais e inspeção visual final ainda dependem desta retomada.

## Fechamento técnico e visual em 18/10/2026

A retomada terminou sem nova coleta ou alteração dos gates. Foram concluídas as 72 medições operacionais, em seis tarefas conhecidas, quatro modos e três repetições, com N=3. Todos os tokens, estados comparados e rótulos coincidiram com o banco conhecido. O somatório dos tempos completos foi 2.018,815 s (33,65 min); o orçamento de 30 minutos era checado entre blocos, com conclusão do bloco final. Os tempos e sua variabilidade são descritivos e não certificam aceleração geral.

A auditoria recontou também as 2.400 linhas do replay novo e reproduziu separadamente o endpoint e seu bootstrap. A última suíte integral passou com 171 testes, nenhum ignorado, Ruff e mypy em 134 fontes. A publicação verificou 19 etapas, 21 células de notebook executadas sem erro, resumo em inglês de duas páginas, 12 slides e 44 perguntas respondidas.

A inspeção visual cobriu as 69 páginas do manuscrito e as 14 páginas dos materiais, em 22 grupos, com onze ampliações adicionais. Nove grupos conservaram exatamente o hash das imagens já inspecionadas nesta sessão; os demais foram vistos novamente. A comparação e as notas estão no manifesto de revisão visual. Foram corrigidos títulos e numeração dos seis quadros e mantidas as fontes junto das figuras/tabelas. Não há texto fora da página nem fontes isoladas no início de outra página. A página curta ao final do capítulo 6 contém a continuação completa do parágrafo, sem conteúdo ausente.

O relatório `final_quality_report.md`, com as seções PASSOU, PENDENTE e NÃO VERIFICÁVEL, foi emitido após essas verificações. Permanecem externos a anuência sobre a versão final integrada, o ensaio pessoal de defesa e a validação administrativa/depósito. A revisão da proposta já informada não foi solicitada outra vez. A falha original de publicação continua arquivada; a recuperação e os hashes finais estão em `independent_v1/execution/publication_recovery.json`.

<!-- independent-publication-completed -->
## Avaliação independente concluída e publicada

Conclusão em 2026-10-18T16:27:07.521465+00:00: 400 tentativas técnicas e 2.400 principais, sem substituir tarefas nem alterar gates. Endpoint: {'estimate': -0.16359060402684564, 'ci95': [-0.2402298061170319, -0.0942741120201171], 'tasks': 120, 'replications': 10000, 'seed': 20261017, 'undefined_draws': 0, 'identified': True, 'directional_hypothesis_supported': True}. 171 testes aprovados, sem ignorados; Ruff e mypy aprovados. Notebook com 21 células executadas sem erros; manuscrito com 69 páginas. Research summary com duas páginas, defesa com 12 slides e 44 perguntas respondidas. As 15 etapas científicas históricas permanecem idênticas. Relatório: `reports/excellence/independent_v1/README.md`; verificação: `final_verification.json`.
