# slm-inference-steering

Baseline experimental para o TCC:

```text
Prompt -> Qwen2.5-Coder-1.5B -> resposta -> verificador -> metricas
```

Nesta primeira etapa ainda nao ha activation steering. O objetivo e medir a
capacidade empirica do modelo base em problemas de codigo do HumanEval.

## Setup

Use Python 3.10-3.12 e, de preferencia, um ambiente virtual. No Windows,
PyTorch com CUDA nao deve ser instalado neste projeto com Python 3.13.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Na sua maquina, o driver NVIDIA esta instalado, mas a venv precisa de um wheel
do PyTorch com CUDA. Com `uv`, o caminho recomendado e:

```powershell
deactivate
Remove-Item -Recurse -Force .venv
uv venv --python 3.12
.\.venv\Scripts\Activate.ps1
uv pip install -r requirements.txt
```

Se preferir `pip` puro dentro da venv:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Nao instale `torchvision`/`torchaudio` para este baseline; eles nao sao usados
e podem carregar DLLs desnecessarias no Windows.
O `requirements.txt` ja fixa `torch==2.12.0+cu126` usando o indice CUDA do
PyTorch.

Valide antes de rodar o benchmark:

```powershell
python scripts/diagnose_cuda.py
```

## Rodar um smoke test

Comece com poucos problemas para confirmar download do modelo, CUDA e verificador:

```powershell
python scripts/run_baseline.py --limit 3 --n 1 --max-new-tokens 256 --require-cuda
```

Na sua RTX 3050 Laptop de 6 GB, esse e um primeiro teste mais seguro. Depois
de confirmar que esta rodando em CUDA, aumente `--max-new-tokens` para 512.
Se aparecer erro de memoria, reduza `--max-new-tokens` ou rode com `--n` menor.

## Roteiro da Etapa 1

Antes de qualquer activation steering, rode e documente estes experimentos:

1. Piloto curto com 5 problemas e `N=5`.
2. Baseline maior com `N=1` para medir `pass@1` limpo.
3. Best-of-N sem parada antecipada para estimar a curva `pass@k`.
4. Best-of-N com `--early-stop` para medir economia real de tokens e latencia.

O piloto recomendado e:

```powershell
python scripts/run_phase1_pilot.py --limit 5 --n 5 --max-new-tokens 256
```

Ele grava arquivos em `runs/phase1_pilot/`, incluindo:

- JSONL com respostas, codigo verificado e resultado de cada tentativa.
- JSON com resumo das metricas.
- Markdown com relatorio interpretavel.
- CSVs por tarefa, por tentativa e por curva Best-of-K.

## Rodar o baseline Best-of-N

Exemplo com 10 respostas por problema:

```powershell
python scripts/run_baseline.py `
  --limit 20 `
  --n 10 `
  --temperature 0.8 `
  --top-p 0.95 `
  --max-new-tokens 512 `
  --output-jsonl runs/baseline_humaneval_n10.jsonl `
  --summary-json runs/baseline_humaneval_n10_summary.json
```

Por padrao o script usa `Qwen/Qwen2.5-Coder-1.5B-Instruct`, CUDA quando
disponivel, e dtype automatico (`float16` em GPU). Esse padrao evita um probe
instavel de BF16 em algumas combinacoes Windows/driver/PyTorch.

Para avaliar a versao base nao-instruct:

```powershell
python scripts/run_baseline.py `
  --model-id Qwen/Qwen2.5-Coder-1.5B `
  --no-chat-template `
  --n 10
```

Para simular parada antecipada assim que o verificador encontra uma solucao
correta:

```powershell
python scripts/run_baseline.py `
  --limit 20 `
  --n 10 `
  --early-stop `
  --output-jsonl runs/baseline_humaneval_n10_earlystop.jsonl `
  --summary-json runs/baseline_humaneval_n10_earlystop_summary.json
```

Para gerar relatorios a partir de um JSONL existente:

```powershell
python scripts/analyze_run.py `
  runs/baseline_humaneval_n10.jsonl `
  --summary-json runs/baseline_humaneval_n10_summary.json `
  --update-summary-json
```

Para comparar duas ou mais runs:

```powershell
python scripts/compare_runs.py `
  runs/phase1_pilot/humaneval5_n5_seed1234_summary.json `
  runs/phase1_pilot/humaneval5_n5_seed1234_earlystop_summary.json `
  --labels best_of_5 early_stop `
  --output-md runs/phase1_pilot/comparison.md `
  --output-csv runs/phase1_pilot/comparison.csv
```

## Notebook de Analise Avancada

Depois de rodar o piloto, abra:

```text
notebooks/phase1_advanced_analysis.ipynb
```

O notebook transforma a etapa inicial em uma bancada experimental mais rica
para o TCC. Ele carrega os JSONL/JSON gerados em `runs/phase1_pilot/` e monta:

- tabela comparativa entre Best-of-N e early stop;
- fronteira custo-acuracia em tokens;
- curva Best-of-K observada e estimada;
- mapa de acertos por tentativa e por problema;
- classificacao empirica de dificuldade das tarefas;
- analise contrafactual de economia com parada antecipada;
- relacao entre tokens, latencia e sucesso;
- diversidade entre respostas geradas e taxa de acerto;
- intervalos de confianca por bootstrap para `pass@1` e Best-of-N;
- roteiro de slides para uma apresentacao da primeira etapa.

Essa analise ajuda a responder perguntas mais academicas antes do steering:
o ganho vem de mais amostras ou de melhores primeiras respostas? Quais tarefas
continuam dificeis mesmo com varias tentativas? O verificador reduz custo sem
perder acuracia? A diversidade das geracoes esta associada a maior chance de
encontrar uma resposta correta?

Se ainda nao tiver instalado as dependencias de visualizacao:

```powershell
uv pip install -r requirements.txt
```

## Etapa 2: Matriz de Modelos e Decoding

Antes dos hooks, a proxima evolucao natural e comparar modelos e estrategias
de inferencia sob o mesmo verificador. Isso separa quatro fatores:

- ganho por escala do modelo;
- ganho por amostragem Best-of-N;
- economia por `early-stop`;
- relacao entre diversidade e acerto.

Liste os modelos e presets disponiveis:

```powershell
python scripts/run_model_matrix.py --list
```

Rode primeiro um plano sem executar:

```powershell
python scripts/run_model_matrix.py --preset core --limit 5 --dry-run
```

Smoke test leve, bom para baixar e validar um segundo modelo:

```powershell
python scripts/run_model_matrix.py `
  --preset smoke `
  --limit 5 `
  --max-new-tokens 256 `
  --require-cuda
```

Matriz principal recomendada para a RTX 3050 de 6 GB:

```powershell
python scripts/run_model_matrix.py `
  --preset core `
  --limit 20 `
  --max-new-tokens 256 `
  --skip-existing `
  --continue-on-error `
  --require-cuda
```

Depois que os modelos ja estiverem baixados, voce pode repetir runs sem chamadas
ao Hugging Face adicionando:

```powershell
--local-files-only
```

O preset `core` compara:

- `Qwen/Qwen2.5-Coder-0.5B-Instruct`;
- `Qwen/Qwen2.5-Coder-1.5B-Instruct`;
- `deepseek-ai/deepseek-coder-1.3b-instruct`;
- `greedy_n1`, `sample_n5` e `sample_n5_early_stop`.

O script salva `manifest.json`, `matrix_summary.csv`, `matrix_report.md` e os
JSONL/summaries individuais em `runs/model_matrix/`. O notebook de analise
avancada detecta esses arquivos automaticamente quando eles existem.

## Saidas

O JSONL contem um registro por problema, incluindo cada tentativa, resposta
bruta, codigo testado, tempo, tokens gerados e resultado do verificador.

O arquivo de resumo contem as metricas principais:

- `strict_pass_at_1`: acuracia da primeira resposta.
- `observed_best_of_n`: taxa de problemas resolvidos por pelo menos uma das N respostas.
- `observed_best_of_k`: curva observada de Best-of-K.
- `pass_at_k_estimate`: estimador padrao do HumanEval para pass@k.
- `mean_generation_seconds_per_attempt`: tempo medio de geracao por tentativa.
- `mean_generated_tokens_per_attempt`: tokens medios gerados por tentativa.
- `mean_attempts_until_first_success_solved_only`: media de tentativas ate o primeiro acerto entre problemas resolvidos.
- `mean_attempts_until_success_or_budget`: media considerando problemas nao resolvidos como esgotados no orcamento de N.
- `mean_tokens_until_success_or_budget`: custo medio efetivo de uma politica com verificador.
- `token_savings_if_oracle_early_stop`: economia maxima simulada se parasse exatamente no primeiro acerto.
- `generated_tokens_per_solved_task`: custo bruto em tokens por tarefa resolvida.

## Quando Passar Para Steering

Nao implemente hooks antes de ter pelo menos:

- um piloto `limit=5, n=5` validado;
- um baseline `N=1` em uma amostra maior;
- uma curva Best-of-K com `N>=5`;
- uma medida de economia com `--early-stop`;
- relatorios Markdown/CSV salvos para comparar com a versao intervencionada.

O activation steering deve ser avaliado contra essas mesmas saidas. A pergunta
experimental passa a ser: com o mesmo verificador e o mesmo orcamento de N, o
steering aumenta o acerto nas primeiras tentativas e reduz tokens ate o primeiro
acerto?

## Nota de seguranca

O verificador executa codigo gerado pelo modelo em um subprocesso isolado com
timeout e diretorio temporario. Ainda assim, qualquer benchmark de codigo que
executa saidas de modelo deve ser rodado em uma maquina/ambiente que voce
considere descartavel.
