# Fluxo principal: protocolo v0.2

O [protocolo](../RESEARCH_PROTOCOL.md) define o desenho; o
[RESULTS.md](../RESULTS.md) registra as execuções. O protocolo anterior foi
arquivado em `docs/protocols/`; bancos de versões distintas não são misturados.

## Estado e pré-requisitos

O piloto v0.2 concluiu as 960 trajetórias, passou pelas auditorias e selecionou
temperatura 1,0 pela regra congelada. O registro de sampling está preparado.
As 19 etapas terminaram em 15/10/2026, com 2.640 trajetórias principais e 640 de
transferência, notebook executado, PDF e auditoria. As retomadas preservaram
os registros anteriores. Em 14/10/2026, o estudante autorizou a execução integral.
A [emenda administrativa](execution_amendment_v0.2.md) registra essa autorização
com snapshot verificável, mantendo a aprovação do orientador pendente.
O caminho original de aprovação e tag Git permanece disponível. A execução
autorizada não é apresentada como pré-registro independente.

O acompanhamento atualizado está disponível com
`.venv/Scripts/python.exe scripts/research_status.py`. O runner
`scripts/run_complete_research.py` encadeia todas as etapas e a publicação final.
No Windows, inicie esse runner pelo launcher com suporte à gravação atômica:

```powershell
.venv/Scripts/python.exe scripts/run_research_safely.py --wsl-python /home/joaof/.venvs/slm-evalplus-v031/bin/python
```

O [registro de recuperação](windows_execution_recovery.md) descreve o suporte
de filesystem, seus testes e a preservação dos fontes científicos congelados.

O piloto seleciona temperatura sem probes. O banco principal contém 132 tarefas
com 20 tentativas cada, mesmo depois de sucesso. A posição de 32 tokens é
confirmatória; posições adicionais são secundárias. Retenção de 97% no
desenvolvimento não é uma garantia externa. A margem operacional é 3 pp.

## Ambiente

Instale `requirements.txt` para a pesquisa e `requirements-notebooks.txt` para
a apresentação/PDF. Use o Python da `.venv`. O worker oficial roda no WSL:

```powershell
wsl -d Ubuntu -- /home/joaof/.venvs/slm-evalplus-v031/bin/python -m pip install -r /mnt/c/Users/joaof/slm-inference-steering/requirements-evalplus.txt
```

Esse ambiente de avaliação não usa a GPU nem precisa de PyTorch. A geração
usa GPU, batch 1, BF16 nos pesos, projeções FP32 e eager. Extração e testes
causais são registrados nos manifestos. Seeds são independentes por tentativa.

## Coleta e auditoria

```powershell
$pythonPesquisa = ".\.venv\Scripts\python.exe"
$bancoPesquisa = ".\runs\candidate_bank\definitive_v0.2"
$probesPesquisa = ".\runs\probes\protocol_v0.2"

& $pythonPesquisa scripts/build_candidate_bank.py --phase pilot --dry-run

& $pythonPesquisa scripts/build_candidate_bank.py --phase pilot --output-dir runs/candidate_bank/pilot_v0.2 --wsl-python /home/joaof/.venvs/slm-evalplus-v031/bin/python --time-budget-seconds 600

& $pythonPesquisa scripts/audit_candidate_bank.py --bank-dir runs/candidate_bank/pilot_v0.2 --output reports/research/pilot_v0.2_audit.json
```

Repita a coleta para retomar. Uma sessão para entre tentativas, sem reduzir o
tamanho planejado. O orçamento pode ser ultrapassado pelo término da tentativa
em andamento. Não execute dois escritores no mesmo banco. Configuração, protocolo
e código de geração/verificador precisam corresponder à proveniência registrada.

O worker recebeu uma correção exclusiva de serialização para hashing MBPP+.
O evento de compatibilidade do piloto atual reavaliou todos os 21 rótulos e
preservou o hash do dataset HumanEval+ antes de permitir esse worker na retomada.
Não use essa exceção auditada como permissão genérica para alterar o verificador.

Depois do piloto completo:

```powershell
& $pythonPesquisa scripts/prepare_sampling_freeze.py
```

Após a liberação dos gates e o congelamento, pela aprovação científica ou pela
autorização de execução registrada na emenda administrativa:

```powershell
& $pythonPesquisa scripts/build_candidate_bank.py --phase definitive --pilot-manifest runs/candidate_bank/pilot_v0.2/manifest.json --output-dir $bancoPesquisa --wsl-python /home/joaof/.venvs/slm-evalplus-v031/bin/python --time-budget-seconds 600
& $pythonPesquisa scripts/audit_candidate_bank.py --bank-dir $bancoPesquisa --output reports/research/definitive_v0.2_audit.json
& $pythonPesquisa scripts/run_grouped_probes.py --bank-dir $bancoPesquisa --output-dir $probesPesquisa
& $pythonPesquisa scripts/validate_online_replay.py --bank-dir $bancoPesquisa --probes-dir $probesPesquisa --wsl-python /home/joaof/.venvs/slm-evalplus-v031/bin/python --max-tasks 1
& $pythonPesquisa scripts/generate_tcc_artifacts.py --probes-dir $probesPesquisa
```

Uma tarefa online é um smoke test. Defina o subset de medição antes de examinar
resultados, aumente `--max-tasks` de acordo com esse desenho e reporte cobertura
de abortos. Verificador, B e H+B usam as mesmas streams. O script mede tempos
instrumentados, exclui aquecimento e alterna a ordem por tarefa. Nenhuma
economia de tempo é inferida apenas do replay de tokens.

## Transferência MBPP+

Depois da análise principal, prepare o bundle de origem e congele-o antes da
geração alvo. A seleção de 32 tarefas MBPP+ é por hash, com 20 tentativas cada.

```powershell
& $pythonPesquisa scripts/run_frozen_transfer.py prepare --bank-dir $bancoPesquisa --probes-dir $probesPesquisa
& $pythonPesquisa scripts/build_candidate_bank.py --phase transfer --pilot-manifest runs/candidate_bank/pilot_v0.2/manifest.json --transfer-bundle runs/transfer/protocol_v0.2/models.joblib --output-dir runs/candidate_bank/transfer_v0.2 --wsl-python /home/joaof/.venvs/slm-evalplus-v031/bin/python --time-budget-seconds 600
& $pythonPesquisa scripts/audit_candidate_bank.py --bank-dir runs/candidate_bank/transfer_v0.2 --output reports/research/transfer_v0.2_audit.json
& $pythonPesquisa scripts/run_frozen_transfer.py evaluate --bank-dir runs/candidate_bank/transfer_v0.2
```

Os rótulos de MBPP+ não participam de fitting/calibração/seleção. O arquivo do
bundle é imutável por versão e seu hash acompanha a coleta alvo. Ajuste local
seria outro experimento. Essa extensão é secundária e deve ter seu orçamento
planejado antes de ver seus resultados.

## Notebook e manuscrito atuais

```powershell
& $pythonPesquisa scripts/build_research_delivery.py
& $pythonPesquisa scripts/build_final_notebook.py --execute
& $pythonPesquisa scripts/build_manuscript.py --compile
```

A reanálise histórica usa `--reanalyze-legacy` e não utiliza GPU. O notebook
lê tabelas e figuras exportadas. O gerador do manuscrito preserva capítulos
autorais e atualiza seções empíricas. Quando houver resultados confirmatórios,
revise explicitamente o escopo e a conclusão do texto; não apenas substitua
números de uma análise tardia por resultados de um estimando diferente.

## Diagnósticos posteriores ao resultado principal

`build_research_delivery.py` também gera os diagnósticos complementares em
`reports/final_review/supplemental/`, apresentados no notebook e no PDF.
Para reproduzi-los isoladamente, sem alterar os bancos ou os resultados principais:

```powershell
& $pythonPesquisa scripts/analyze_research_robustness.py
```

Essas análises foram definidas após a conclusão e inspeção do experimento.
Incluem contrastes por fold, exclusão de uma tarefa por vez, coorte comum T>64,
exclusão das 26 tarefas com geração histórica, calibração em dez bins e retenção
dos candidatos corretos nos gates congelados. Bootstrap por tarefa mantém os
escores fixos; a retenção é a razão das contagens somadas. O denominador inclui
candidatos corretos ativos nunca iniciados no replay. O alvo empírico de 97% em
desenvolvimento não é garantia externa. Nenhum limiar é escolhido com rótulos MBPP+.

As figuras têm versões PNG, SVG e PDF; os CSV contêm contribuições por tarefa,
fold e bin. O `results.json` registra definições, insumos, hashes, seed e limites.
O gerador interrompe se os contrastes principais não forem reproduzidos. A
auditoria de publicação confere os hashes e a preservação das decisões originais.
O estado e a publicação anteriores à extensão estão arquivados em
`reports/research/publication_before_supplement_20261016/`.
