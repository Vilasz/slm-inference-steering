# Reprodução final e comandos de coleta

Execute os comandos PowerShell na raiz do repositório com a `.venv` e os dados
arquivados. As dependências estão em `requirements*.txt`, `pyproject.toml` e
`uv.lock`; modelo/tokenizer e datasets têm revisões nos manifests. Não atualize
dependências durante uma réplica e não use a suíte de teste como prova de
ausência de contaminação.

## Conferência sem gerar respostas nem carregar o modelo na GPU

```powershell
.venv\Scripts\python.exe scripts/run_final_scientific_audit.py
.venv\Scripts\python.exe scripts/run_frozen_transfer.py evaluate --bank-dir runs/candidate_bank/transfer_v0.2 --output-dir reports/excellence/final_transfer_reconstruction
.venv\Scripts\python.exe scripts/generate_tcc_artifacts.py --output-dir reports/excellence/final_main_reconstruction
```

O primeiro comando reconta AUC por pares, perdas por tarefa, intervalos e replay
direto e reconstrói os três scorers principais a partir dos ajustes salvos. Os
dois seguintes reexecutam estatísticas, bootstrap e permutações pelos fontes
históricos em pastas separadas; não treinam novamente todo o grid nem alteram os
bancos. A busca exploratória já foi executada duas vezes: a auditoria compara
as probabilidades das 70 unidades e verifica seus hashes. Não é necessário
reabrir uma busca por modelos para reconstruir as tabelas.

Após o banco independente completo, estes comandos recalculam apenas análise:

```powershell
.venv\Scripts\python.exe scripts/run_excellence_extension.py --evaluate
.venv\Scripts\python.exe scripts/analyze_excellence_independent.py
```

O coletor congelado calcula o endpoint primário; o segundo comando reabre todas
as ativações e confere identidades, seeds, hashes, scorers e contagens, acrescentando
diagnósticos secundários. Um banco parcial causa erro; não se estima confirmação
antes das 2.400 tentativas principais.

## Testes e publicação dos artefatos existentes

```powershell
.venv\Scripts\python.exe -c "import os; os.environ['SLM_STEERING_TEST_TORCH']='1'; import torch,pytest; raise SystemExit(pytest.main(['-q']))"
.venv\Scripts\python.exe -m ruff check .
.venv\Scripts\python.exe -m mypy src scripts
.venv\Scripts\python.exe scripts/prepare_final_figures.py
.venv\Scripts\python.exe scripts/run_excellence.py report
.venv\Scripts\python.exe scripts/build_excellence_materials.py
.venv\Scripts\python.exe scripts/build_research_delivery.py
.venv\Scripts\python.exe scripts/build_final_notebook.py --execute
.venv\Scripts\python.exe scripts/build_manuscript.py --compile
.venv\Scripts\python.exe scripts/audit_research_delivery.py
```

As figuras de impressão usam somente resultados salvos; o relatório agregado
confere os hashes e a publicação valida dependências. Execute na ordem indicada.
Esses comandos não coletam novas trajetórias. A publicação supervisionada também
registra validação e preserva os quinze registros científicos históricos.

## Execução longa na GPU — separada da reprodução estatística

Não execute junto à coleta já ativa. Para retomada autorizada, o supervisor
`scripts/run_approved_excellence.py` verifica aprovação e os 46 arquivos do
congelamento antes de cada sessão. Os comandos elementares, mostrados apenas
para documentação da execução, são:

```powershell
.venv\Scripts\python.exe scripts/run_excellence_extension.py --phase technical --time-budget-seconds 600
.venv\Scripts\python.exe scripts/run_excellence_extension.py --phase main --time-budget-seconds 600
```

Cada sessão acrescenta somente identidades faltantes, sob lock, sem substituir
falhas por rótulos artificiais. O controle técnico completo precede a fase
principal. Não apague bancos ou locks para forçar uma nova execução. O snapshot
prospectivo, os logs e a aprovação estão em `reports/excellence/independent_v1/`
e `docs/protocols/EXCELLENCE_EXTENSION_v1.approval.json`.

O benchmark `scripts/run_operational_benchmark.py` também usa GPU, mas somente
seis tarefas já examinadas, três repetições e N=3. Ele exige coleta independente
encerrada e impede concorrência com o coletor. É descritivo; um erro técnico nele
não invalida nem redefine o endpoint independente.

## Proveniência e limitações

Os dezesseis artefatos científicos protegidos, os fontes congelados e os registros
originais são verificados por hash. `final_pass_baseline/` e os arquivos de
arquivamento preservam publicações anteriores e alterações locais preexistentes.
O parser/dataset foram recuperados do HEAD+patch pré-aprovação, conforme recibo
específico; sua omissão como entradas separadas do snapshot não foi ocultada.
O bootstrap da análise original é condicional aos scorers ajustados. Ausência
de exposição local não certifica ausência no pré-treinamento. Dados oficiais
requerem EvalPlus POSIX/WSL; o caminho Python local usado é registrado nos logs.

Os bancos completos e pesos são grandes e podem não integrar uma cópia mínima
do Git. Confira os manifests antes de reproduzir: sem JSONL, NPZ ou bundles
correspondentes, a réplica numérica deve falhar, com o arquivo ausente identificado.
Nenhum resultado deve ser reconstruído de números copiados do PDF.
