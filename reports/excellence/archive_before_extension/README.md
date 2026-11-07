# Sinais de corretude em geração de código

Pesquisa sobre Qwen2.5-Coder-1.5B-Instruct: distinguir tentativas corretas e
incorretas da mesma tarefa, medir ganho sobre observáveis e avaliar aborto precoce.
O [protocolo v0.2](RESEARCH_PROTOCOL.md) incorpora as correções metodológicas;
o orientador ainda não aprovou o congelamento científico. A
[execução integral autorizada pelo estudante](docs/execution_amendment_v0.2.md)
preserva o desenho e os fontes em um snapshot verificável anterior ao banco principal.

## Comece pelos resultados

| Artefato | Conteúdo |
|---|---|
| [Notebook executado](notebooks/final_research_summary.ipynb) | Resultados principais, figuras, escopo e hashes |
| [Notebook em HTML](reports/final_review/notebook.html) | Leitura sem iniciar Jupyter |
| [Manuscrito PDF](tcc/build/manuscrito.pdf) | Fundamentação, método revisado, resultados e discussão |
| [Fonte do manuscrito](tcc/main.typ) | Capítulos editáveis e bibliografia |
| [Estado da evidência](RESULTS.md) | Execuções concluídas e decisões principais |
| [Diagnósticos complementares](reports/final_review/supplemental/README.md) | Robustez, calibração, retenção e limites da conclusão |
| [Fluxo de execução](docs/research_workflow.md) | Comandos atuais, gates e transferência |

A reanálise histórica usa 150 tentativas em 30 tarefas e **estados finais**.
Seu ganho incremental de WT-AUC é inconclusivo. Ela não responde ao endpoint
precoce de 32 tokens. A execução integral concluiu as 19 etapas em 15/10/2026:
960 trajetórias de piloto, 2.640 principais e 640 de transferência MBPP+.
O ganho incremental e o benefício operacional conjunto não satisfizeram os
critérios fixados. Os diagnósticos exploratórios posteriores detalham estabilidade,
precisão e limites de transporte sem alterar o desenho ou ajustar o sistema no alvo.
Consulte o status com `.venv/Scripts/python.exe scripts/research_status.py`.

## Organização

`src/slm_steering/research/` contém geração, rótulos, features, probes, estatística,
política, transferência e preparação de evidência. `configs/` guarda configurações;
`runs/` guarda bancos grandes locais; `reports/` contém resultados pequenos e
auditorias; `tcc/` contém o documento; `notebooks/` contém a apresentação executada.
Arquivos exploratórios anteriores permanecem em seus caminhos; documentos
anteriores estão em `docs/legacy/` e o protocolo anterior em `docs/protocols/`.

## Ambiente e reprodução

Python 3.12, GPU RTX 3050 Laptop de 6 GB. Geração com BF16 nos pesos, projeções
FP32, batch 1 e atenção eager. EvalPlus 0.3.1 roda em Ubuntu/WSL para seus
timeouts POSIX. O piloto mediu aproximadamente 4,49 GiB de memória reservada;
uma sessão longa do banco principal atingiu 5,45 GiB (5.582 MiB).

```powershell
uv pip install --python .venv/Scripts/python.exe -r requirements.txt
uv pip install --python .venv/Scripts/python.exe -r requirements-notebooks.txt
.venv/Scripts/python.exe -m ipykernel install --prefix .venv --name slm-research --display-name "SLM Research (.venv)"
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe scripts/build_research_delivery.py
.venv/Scripts/python.exe scripts/build_final_notebook.py --execute
.venv/Scripts/python.exe scripts/build_manuscript.py --compile
.venv/Scripts/python.exe scripts/audit_research_delivery.py
```

Enquanto a coleta está ativa, acrescente `--allow-running-snapshot` ao comando
de evidências para copiar um prefixo JSONL completo e publicá-lo separadamente.
Para retomar o fluxo integral no Windows, use `scripts/run_research_safely.py --wsl-python
/home/joaof/.venvs/slm-evalplus-v031/bin/python` depois de conferir o status e as travas.
O launcher preserva os fontes congelados e tolera bloqueios temporários de arquivos
JSON; a [recuperação de execução](docs/windows_execution_recovery.md) registra a correção.

Para repetir a análise histórica, acrescente `--reanalyze-legacy` e disponibilize
os arquivos originais em `runs/`. O notebook lê as evidências exportadas e pode
ser executado sem carregar o modelo. Os capítulos autorais são preservados;
o gerador atualiza as seções empíricas a partir de um manifesto único.

O [guia do TCC](docs/TCC_GUIDELINE.md) descreve a escrita e a sequência de análises.
Não trate a implementação ou os testes como substitutos da aprovação científica,
da amostra completa ou da avaliação externa.
