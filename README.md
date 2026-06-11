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

## Phase 2: Difficulty-Aware Inference Scaling

Antes de manipular ativacoes internas, precisamos entender em quais problemas
e sob quais condicoes o escalonamento de inferencia via Best-of-N realmente
gera valor. A Fase 2 mede custo marginal, dificuldade, diversidade e fronteira
de Pareto, criando a base experimental para testar se Activation Steering
desloca essa fronteira para maior acuracia com menor custo.

O script principal da fase e:

```powershell
python scripts/run_experiment_matrix.py --limit 10 --dry-run
```

Para rodar uma matriz pequena e segura:

```powershell
python scripts/run_experiment_matrix.py `
  --models qwen2.5-coder-0.5b-instruct `
  --temperatures 0.2,0.8 `
  --n-values 1,5 `
  --limit 10 `
  --skip-existing
```

Para a matriz maior proposta para a fase:

```powershell
python scripts/run_experiment_matrix.py `
  --models qwen2.5-coder-0.5b-instruct,qwen2.5-coder-1.5b-instruct `
  --temperatures 0.2,0.6,0.8,1.0 `
  --n-values 1,2,5,10 `
  --limit 50 `
  --skip-existing `
  --continue-on-error
```

Para comparar Best-of-N completo com parada antecipada, adicione:

```powershell
--early-stop-modes full,early_stop
```

Depois de baixar os modelos uma vez, use `--local-files-only` para repetir
experimentos sem depender de chamadas ao Hugging Face.

A Fase 2 grava resultados em `runs/phase2/`:

- `*.jsonl`: uma linha por problema, com todas as tentativas;
- `*_summary.json`: metricas agregadas da run;
- `manifest.json`: indice das combinacoes executadas;
- `phase2_summary.csv`: tabela comparativa;
- `phase2_report.md`: resumo em Markdown.

Abra o notebook:

```text
notebooks/phase2_difficulty_scaling.ipynb
```

Metricas principais desta fase:

- `difficulty`: `easy`, `sampling_sensitive`, `fragile` ou `hard`;
- `solved_by_sampling`: problema que falhou no pass@1, mas passou em tentativa posterior;
- `best_of_k_accuracy`: acuracia acumulada ate K tentativas;
- `marginal_gain_k`: ganho comprado pela tentativa adicional K;
- `accuracy_per_1k_tokens`: retorno em acuracia por custo medio de tokens;
- `is_pareto_efficient`: configuracao nao dominada em custo-acuracia.

## Phase 3: Activation Probing and Representation Extraction

A Fase 3 ainda nao implementa Activation Steering. Ela observa ativacoes
internas do modelo em respostas ja geradas, separando tentativas corretas e
incorretas. O objetivo e investigar se existe uma direcao latente associada a
respostas bem-sucedidas.

Fluxo recomendado:

1. Rode uma matriz da Fase 2 com `N > 1`.
2. Escolha uma run com tarefas `sampling_sensitive` ou `fragile`.
3. Extraia ativacoes por camada e posicao de token.
4. Analise separabilidade, PCA, probes lineares e direcoes candidatas.

Exemplo de planejamento sem carregar modelo:

```powershell
python scripts/extract_activations.py `
  --input-jsonl runs/phase2/qwen15b_temp08_n5.jsonl `
  --summary-json runs/phase2/qwen15b_temp08_n5_summary.json `
  --output-dir runs/phase3/qwen15b_temp08_n5_probe `
  --layers 0:28:4 `
  --token-position completion_last `
  --difficulty-filter sampling_sensitive,fragile `
  --dry-run
```

Extraindo de fato, depois que modelo/dataset ja estiverem em cache:

```powershell
python scripts/extract_activations.py `
  --input-jsonl runs/phase2/qwen15b_temp08_n5.jsonl `
  --summary-json runs/phase2/qwen15b_temp08_n5_summary.json `
  --output-dir runs/phase3/qwen15b_temp08_n5_probe `
  --layers 0:28:4 `
  --token-position completion_last `
  --difficulty-filter sampling_sensitive,fragile `
  --local-files-only `
  --require-cuda
```

Analise os artefatos:

```powershell
python scripts/analyze_activations.py runs/phase3/qwen15b_temp08_n5_probe
```

O armazenamento da Fase 3 fica em:

- `activations.npz`: tensor `[amostras, camadas, hidden_size]`;
- `metadata.jsonl`: metadados por tentativa;
- `manifest.json`: configuracao de extracao;
- `separability.csv`: distancia de centroides e Fisher ratio por camada;
- `linear_probes.csv`: acuracia de probes lineares simples;
- `latent_directions.npz`: vetores `mean_correct - mean_incorrect`.

Abra o notebook:

```text
notebooks/phase3_activation_probing.ipynb
```

Interpretacao esperada:

- `centroid_distance` alto sugere separacao entre corretas e incorretas;
- `test_accuracy` do probe indica se a separacao e linearmente exploravel;
- PCA 2D ajuda a visualizar, mas nao deve ser usado como unica evidencia;
- a melhor camada candidata vira hipotese para uma Fase 4 de steering.

## Phase 3.5: Statistical Latent Geometry

Antes de intervir causalmente nas ativacoes, o projeto testa se ha sinal
estatisticamente robusto no espaco latente associado a corretude. A Fase 3.5
combina probes lineares, controles negativos, permutation tests, regressao de
sucesso, calibracao e survival analysis para identificar camadas e direcoes
candidatas para steering.

Motivacao:

- PCA e visualmente util, mas nao prova separabilidade nem significancia;
- probes precisam usar split por `task_id`, porque varias tentativas do mesmo
  problema nao sao independentes;
- permutation tests ajudam a verificar se o sinal latente excede o que surgiria
  por acaso com labels embaralhados;
- survival analysis trata Best-of-N como tempo ate primeiro sucesso, que e a
  medida diretamente ligada a custo de inferencia;
- regressao testa se `latent_score` prediz sucesso mesmo controlando tokens,
  tentativa, dificuldade e configuracao observavel.

Comando principal:

```powershell
python scripts/analyze_latent_geometry.py `
  --activations-dir runs/phase3/qwen_phase3_probe `
  --phase2-runs-dir runs/phase2 `
  --output-dir runs/phase35/latent_geometry `
  --n-bootstrap 1000 `
  --n-permutations 500
```

Para uma validacao curta:

```powershell
python scripts/analyze_latent_geometry.py `
  --activations-dir runs/phase3/qwen_phase3_probe `
  --phase2-runs-dir runs/phase2 `
  --output-dir runs/phase35/latent_geometry `
  --n-bootstrap 100 `
  --n-permutations 100
```

A fase salva:

- `probe_results.csv`: AUC, accuracy, F1, Brier e calibracao por camada;
- `latent_score_results.csv`: projecoes nas direcoes latentes e controles;
- `permutation_results.csv`: significancia empirica dos scores;
- `regression_results.csv`: ablations com e sem `latent_score`;
- `survival_results.csv`: Kaplan-Meier, hazard e custo ate sucesso;
- `bootstrap_results.csv`: intervalos de confianca por reamostragem de tarefas;
- `summary.json`: sintese para TCC com camadas candidatas.

Abra:

```text
notebooks/phase35_statistical_latent_geometry.ipynb
```

Leitura esperada:

- `best_probe_auc` alto sugere que a corretude e linearmente acessivel;
- `permutation_p_value` baixo sugere que a separacao nao e facilmente explicada
  por acaso;
- `latent_score_delta_auc` positivo sugere poder preditivo incremental do
  espaco latente;
- `recommended_steering_layers` define hipoteses para a Fase 4;
- se a amostra for pequena, trate tudo como piloto e reporte essa limitacao.

## Phase 4: Causal Activation Steering

A Fase 4 testa causalmente as direcoes candidatas da Fase 3. Agora a pergunta
nao e apenas "existe separabilidade?", mas sim: adicionar a direcao durante a
geracao muda acuracia, custo, diversidade e tipos de erro?

O sweep minimo recomendado e:

```powershell
python scripts/run_steering_sweep.py `
  --limit 5 `
  --n 2 `
  --alphas 0,1 `
  --layers 12 `
  --skip-existing
```

Para validar sem carregar o modelo:

```powershell
python scripts/run_steering_sweep.py --limit 5 --n 2 --alphas 0,1 --layers 12 --skip-existing --dry-run
```

Controles implementados:

- `correctness_direction`: direcao `mean_correct - mean_incorrect`;
- `negative_correctness_direction`: direcao invertida;
- `random_direction`: direcao aleatoria normalizada;
- `shuffled_label_direction`: direcao estimada com labels embaralhados;
- `length_direction`: direcao associada ao comprimento da resposta.

Abra:

```text
notebooks/phase4_causal_activation_steering.ipynb
```

## Phase 5: Latent-Adaptive Best-of-N

A Fase 5 transforma as evidencias latentes em uma politica de amostragem. Em
vez de sempre gastar `N` tentativas, a politica decide continuar ou parar com
base em sucesso do verificador, orcamento, custo medio e score latente quando
`latent_directions.npz` estiver disponivel.

Comando minimo:

```powershell
python scripts/run_adaptive_bon.py `
  --limit 5 `
  --max-n 3 `
  --policy latent_adaptive
```

Validacao leve:

```powershell
python scripts/run_adaptive_bon.py --limit 5 --max-n 3 --policy latent_adaptive --dry-run
```

Politicas implementadas:

- `fixed_n_1`, `fixed_n_5`, `fixed_n_10`;
- `verifier_early_stop`;
- `difficulty_adaptive`;
- `latent_adaptive`;
- `steering_fixed_n` e `steering_latent_adaptive` como familias para comparacao futura.

Abra:

```text
notebooks/phase5_latent_adaptive_bon.ipynb
```

## Phase 6: Robustness and Stress Benchmark

A Fase 6 adiciona um pequeno benchmark local `humaneval_stress`, com casos de
borda inspirados em tarefas HumanEval. Ele serve para testar se ganhos do
baseline, steering ou politicas adaptativas sobrevivem a entradas adversariais
simples.

Exemplo:

```powershell
python scripts/run_baseline.py `
  --benchmark humaneval_stress `
  --limit 3 `
  --n 2
```

Tambem e possivel usar o benchmark de stress nas Fases 4 e 5 com:

```powershell
--benchmark humaneval_stress
```

## Phase 7: Cross-Model Transfer

A Fase 7 organiza a pergunta de transferencia: uma direcao extraida em um
modelo ajuda em outro modelo coder pequeno? A primeira analise fica em:

```text
notebooks/phase7_cross_model_transfer.ipynb
```

O desenho recomendado e extrair direcoes no `Qwen/Qwen2.5-Coder-1.5B-Instruct`
e testar sweeps no `Qwen/Qwen2.5-Coder-0.5B-Instruct`, sempre mantendo
`random_direction` e `negative_correctness_direction` como controles.

## Phase 8: Final Artifacts

Para consolidar tabelas e figuras para entrega/apresentacao:

```powershell
python scripts/generate_final_report_artifacts.py
```

O script cria:

- `reports/tables/main_results.csv`;
- `reports/tables/steering_controls.csv`;
- `reports/tables/adaptive_policy_results.csv`;
- `reports/tables/robustness_results.csv`;
- `reports/tables/cross_model_results.csv`;
- `reports/tables/statistical_tests.csv`;
- `reports/figures/accuracy_cost_scatter.png`;
- `reports/figures/phase_coverage.png`;
- `runs/final_summary.json`.

Abra:

```text
notebooks/final_research_summary.ipynb
```

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

## Criterio Para Interpretar Steering

Os hooks da Fase 4 so devem ser interpretados como evidencia experimental depois
de ter pelo menos:

- um piloto `limit=5, n=5` validado;
- um baseline `N=1` em uma amostra maior;
- uma curva Best-of-K com `N>=5`;
- uma medida de economia com `--early-stop`;
- uma analise de dificuldade por tarefa;
- uma curva de ganho marginal por K;
- uma fronteira de Pareto custo-acuracia;
- relatorios Markdown/CSV salvos para comparar com a versao intervencionada.

O activation steering deve ser avaliado contra essas mesmas saidas. A pergunta
experimental passa a ser: com o mesmo verificador e o mesmo orcamento de N, o
steering aumenta o acerto nas primeiras tentativas e reduz tokens ate o primeiro
acerto? Em termos da Fase 2: steering so sera convincente se deslocar a
fronteira de Pareto para maior acuracia, menor custo ou ambos, e se controles
negativos nao explicarem o mesmo ganho.

## Nota de seguranca

O verificador executa codigo gerado pelo modelo em um subprocesso isolado com
timeout e diretorio temporario. Ainda assim, qualquer benchmark de codigo que
executa saidas de modelo deve ser rodado em uma maquina/ambiente que voce
considere descartavel.
