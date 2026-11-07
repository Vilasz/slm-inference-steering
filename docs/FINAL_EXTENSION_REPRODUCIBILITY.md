# Reprodução da extensão final

Esta extensão é separada do protocolo v0.2 e da avaliação independente anterior.
Sua execução local foi autorizada no pedido do estudante de 04/11/2026; não
depende de reutilizar a aprovação antiga nem atribui aprovação ao orientador.

O estado executado fica em `reports/final_extension/status.json`. Os manifests
por fase são mais atuais durante uma sessão de coleta. O relatório inferencial
externo só existe quando `results.json` tem status `completed`.

## Ambiente e segurança

Use a `.venv` existente com Python 3.12.8, PyTorch 2.12.0+cu126,
Transformers 4.46.3 e EvalPlus 0.3.1. Os pacotes completos estão em
`environment_packages.json`. O sistema acessível tem RTX 3050 Laptop 6 GB,
CUDA runtime 12.6, driver 576.52 e suporte BF16. Não há quantização.

WSL/Ubuntu contém `/home/joaof/.venvs/slm-evalplus-v031`. Bubblewrap 0.9.0
foi extraído, sem instalação global, em
`/home/joaof/.local/slm-sandbox/usr/bin/bwrap`, a partir do pacote Ubuntu
0.9.0-1ubuntu0.3. Os datasets oficiais já estão no cache EvalPlus do WSL.
`sandbox.py` monta somente runtime, scripts e dataset, limpa ambiente e isola
rede, home e discos Windows. Limites e sondas ficam no protocolo/preflight.

O guard oficial usa limite padrão de memória de 4 GiB. A extensão fixa 3 GiB
por processo e passa explicitamente EVALPLUS_MAX_MEMORY_BYTES com o mesmo valor,
para não tentar elevar o hard limit dentro do sandbox. O banco Qwen de origem
é reavaliado nesse sandbox; discrepâncias suspendem o congelamento externo,
preservando os labels anteriores. Esse ajuste de recursos não é ocultado.

## Comandos

```powershell
# Fluxo completo, retomável por candidato e limitado pelo orçamento científico
.venv/Scripts/python.exe -u scripts/run_final_extension.py run

# Estado e publicação do que já existe, sem gerar dados
.venv/Scripts/python.exe scripts/run_final_extension.py report
.venv/Scripts/python.exe scripts/run_final_extension.py publish

# Fases separadas; não iniciar outro coletor enquanto houver uma fase na GPU
.venv/Scripts/python.exe scripts/run_final_extension.py deepseek-pilot
.venv/Scripts/python.exe scripts/run_final_extension.py qwen-technical
.venv/Scripts/python.exe scripts/run_final_extension.py design
.venv/Scripts/python.exe scripts/run_final_extension.py deepseek-source
.venv/Scripts/python.exe scripts/run_final_extension.py train-qwen
.venv/Scripts/python.exe scripts/run_final_extension.py train-deepseek
.venv/Scripts/python.exe scripts/run_final_extension.py audit-qwen-labels
.venv/Scripts/python.exe scripts/run_final_extension.py freeze
.venv/Scripts/python.exe scripts/run_final_extension.py deepseek-external
.venv/Scripts/python.exe scripts/run_final_extension.py qwen-external
.venv/Scripts/python.exe scripts/run_final_extension.py evaluate
.venv/Scripts/python.exe scripts/run_final_extension.py publish
```

As coletas individuais usam sessões de 600 segundos por padrão, encerradas
entre candidatos; `run` encadeia e retoma essas sessões. Aumentar a duração da
sessão com `--session-seconds` não amplia amostra nem orçamento. Não remova uma
trava de escritor ativo. Após queda, verifique o PID do lock e a ausência do
processo antes de arquivar uma trava comprovadamente obsoleta. JSONs por
candidato são commits atômicos; NPZ órfão não torna uma tentativa concluída.

O launcher oculto grava PID e logs em `reports/final_extension/`. Manter a
máquina ligada, sem suspensão e com WSL disponível é necessário para completar
o trabalho local. O código para diante de erros técnicos; não fabrica labels
negativos nem substitui tarefas. `execution_outcome.json` registra suspensão
por erro ou inviabilidade. Retomada usa o mesmo comando `run` após corrigir
somente a causa técnica e registrar qualquer emenda antes de novos rótulos.

## Travas e verificação

`pilot_plan.json` fixa antes do piloto as tarefas e fontes de coleta. As emendas
de preflight registram correções de validação de whitespace/template e limites
de memória anteriores ao primeiro candidato. `final_design.json` só pode surgir
após piloto completo; aplica a hierarquia técnica de tamanhos sem consultar
o externo. `freeze.json` exige bundles, testes, auditorias e código estáveis;
nenhuma geração externa é autorizada sem sua validação.

```powershell
.venv/Scripts/python.exe -c "import os; os.environ['SLM_STEERING_TEST_TORCH']='1'; import torch,pytest; torch.set_num_threads(4); raise SystemExit(pytest.main(['-q','--junitxml=reports/final_extension/full_tests.xml']))"
.venv/Scripts/python.exe -m ruff check src/slm_steering/final_extension scripts/run_final_extension.py scripts/final_extension_worker.py tests/test_final_extension.py
```

Não confundir testes de software com a confirmação científica. A hipótese
primária exige banco externo completo e IC pareado. Repetições de treinamento
na origem são sensibilidades de estabilidade, não novos testes confirmatórios.
O capítulo da extensão declara explicitamente quando os resultados externos
ainda estão pendentes. O manuscrito anterior e seus hashes estão arquivados em
`reports/final_extension/manuscript_before_extension/`.
