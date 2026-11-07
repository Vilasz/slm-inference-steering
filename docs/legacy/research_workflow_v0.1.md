# Fluxo principal: protocolo v0.1

O documento normativo é [RESEARCH_PROTOCOL.md](../RESEARCH_PROTOCOL.md).
O código principal está em `src/slm_steering/research/`. Os scripts e notebooks
das antigas fases continuam disponíveis para exploração e não sustentam as
hipóteses confirmatórias do protocolo.

## Pergunta e limites de interpretação

O estimando principal é o incremento de AUC **dentro da mesma tarefa** de H+B
sobre B no checkpoint de 32 tokens. As tarefas do piloto são reservadas ao
desenvolvimento. Todos os checkpoints usam os tokens efetivamente sorteados,
incluindo o token EOS na contagem, sem retokenização do texto decodificado.

Modelos: prior global, B1 (prompt observável), B2 (`prompt_last`), B3 (confiança
do prefixo), B (B1+B3), H (ativação) e H+B. Controles: direção aleatória,
probe com rótulos de treino permutados dentro da tarefa, comprimento do prefixo
e indicador de EOS observado. Como o modelo principal é um probe logístico,
a direção de diferença de médias e seu sinal invertido não integram essa
análise; o controle de sinal será obrigatório se essa extensão for adicionada.

Cada tarefa tem peso total igual no ajuste, na avaliação de perdas e na WT-AUC.
ROC-AUC e PR-AUC globais também usam ponderação por tarefa. A composição dos
conjuntos de risco é relatada separadamente. Resultados em `final` são um limite
superior tardio, nunca uma evidência de previsão precoce.

## Recursos e geração

O perfil local usa Qwen2.5-Coder-1.5B-Instruct, GPU única de 6 GB, batch 1,
pesos em bfloat16 e projeções lineares calculadas em float32
(float16 produziu logits não finitos e bfloat16 nativo apresentou deriva causal
acima da tolerância na validação local),
KV cache e apenas quatro saídas de blocos: 6, 13, 20 e 27. As ativações ficam
em arquivos NPZ por tentativa; logits de todos os tokens não são acumulados na
GPU. A extração durante a geração evita um segundo processamento completo do
banco. Na verificação causal, somente o backbone é executado, evitando a matriz
de logits de todas as posições por vocabulário.

A tolerância causal de bfloat16 é por vetor de camada: erro L2 relativo ≤1% e
erro máximo absoluto ≤0,02 + 1% do maior módulo do vetor de referência. Isso
considera arredondamento dependente da forma do GEMM, sem usar os desfechos.
O cálculo temporário das projeções em float32 preserva armazenamento de pesos
em 16 bits e evita manter uma segunda cópia completa do modelo na GPU. TF32 e
reduções de GEMM em baixa precisão ficam desativados. Essa configuração deve
ser relatada como precisão mista; não como inferência inteiramente bfloat16.
O manifesto registra todos os erros por camada e checkpoint, incluindo a
comparação da ativação capturada com KV cache contra o forward só do prefixo.

Uma sessão termina **entre tentativas**, após atingir seu orçamento de tempo.
Esse limite não altera max_new_tokens=512, o conjunto de tarefas ou as 10/20
tentativas planejadas. Uma sessão pode ultrapassar o limite pela duração da
última tentativa e dos testes iniciais. Retome com o mesmo comando e diretório.
O escritor é exclusivo e as linhas JSONL são sincronizadas em disco. Caso uma
queda abrupta deixe `.writer.lock`, confirme que o processo terminou antes de
remover esse arquivo. Linhas corrompidas exigem recuperação em uma nova versão.

Temperatura e top-p são implementados explicitamente; top-k=0. Log-probabilidade,
entropia e margem são calculadas em float32 após renormalização da distribuição
filtrada. Se apenas um token sobrevive, a margem é ausente; a imputação mediana
e seu indicador são ajustados exclusivamente no treino. EOS encerra a geração;
chegar a 512 tokens sem EOS é truncamento e falha primária, mesmo se o código
extraído passar nos testes.

## Verificador oficial no Windows

O EvalPlus usa `signal.setitimer`, disponível em POSIX. O worker
`scripts/evalplus_worker.py` roda no Ubuntu via WSL e chama
`evalplus.gen.util.trusted_exec` para os oráculos e
`evalplus.eval.untrusted_check` para **base e plus completos**. `fast_check=True`
interrompe uma suíte no primeiro erro, preservando o resultado de passagem;
não ativa o subconjunto `mini`. São mantidos os oráculos especiais e tolerâncias
da biblioteca. A geração nunca recebe a solução canônica.

Instalação mínima no Ubuntu, com o repositório montado em `/mnt/c/...`:

```bash
python3 -m venv ~/.venvs/slm-evalplus-v031
~/.venvs/slm-evalplus-v031/bin/python -m pip install --no-deps evalplus==0.3.1
~/.venvs/slm-evalplus-v031/bin/python -m pip install -r requirements-evalplus.txt
```

Se `ensurepip` não estiver disponível, instale `python3.12-venv` ou crie o
ambiente com `uv venv`. A instalação mínima deixa ausentes dependências do
EvalPlus destinadas à geração via API/modelos; o worker usa apenas avaliação.
O manifesto registra a versão do Python/Linux do worker, versão do EvalPlus,
MD5 oficial e SHA256 do conteúdo do dataset, tamanhos das suítes e resultados
dos controles positivo e negativo. Falhas de infraestrutura interrompem a
execução; não viram rótulos de incorretude.

EvalPlus não retorna a exceção interna do candidato. Erros de compilação são
identificados separadamente, mas falhas restantes ficam como `evalplus_fail`
ou `timeout`; não se inventa uma distinção entre assertion e runtime.

## Ordem de execução

1. Validar hardware, distribuição de amostragem, sementes, posições causais,
   verificador e contabilidade com testes e um piloto em sessões.
2. Completar as **960** tentativas do piloto; selecionar temperatura pelo número
   de tarefas mistas, usando somente os critérios da seção 6.3. Não executar
   probes no piloto para escolher temperatura.
3. Obter aprovação e congelar protocolo/config/ambiente no Git, com tag imutável.
4. Gerar as **2.640** tentativas definitivas nas 132 tarefas restantes.
5. Executar predições aninhadas com GroupKFold 5 externo/3 interno; produzir
   métricas Q1/Q2 antes de qualquer extensão exploratória.
6. Selecionar checkpoint/limiar B e H+B usando apenas OOF interno do treino,
   comparar replay para N=1,5,10,20 e validar uma execução online pequena.
7. Executar MBPP+ com seleção de tarefas independente dos desfechos; escrever
   limitações e interpretação compatível com H2/H3.

O checkpoint e o C de H+B são escolhidos pela perda logarítmica balanceada por
tarefa, com regra de um erro padrão. B, H e B2 têm seleção interna independente,
sem uso de resultados externos. O máximo de iterações é fixado em 2.000 e
falha de convergência interrompe a análise. Platt scaling é ajustado aos scores
OOF internos de cada treino externo. A regressão de efeitos fixos complementar
é inferencial sobre scores OOF; não é utilizada para selecionar o modelo ou a
política. A incerteza dessa regressão usa clusters de tarefa e não substitui o
bootstrap/permutação da hipótese principal.

## Artefatos e auditoria

- `runs/candidate_bank/pilot_v0.1/`: piloto, manifesto e ativações de desenvolvimento.
- `runs/candidate_bank/definitive_v0.1/`: banco definitivo, append-only.
- `runs/probes/protocol_v0.1/`: predições externas, escolhas internas, modelos e replay.
- `reports/protocol_v0.1/`: tabelas e figuras do protocolo.
- `reports/research/`: relatos pequenos e manifestos que podem ser versionados.
- `RESULTS.md`: estado da evidência e restrições, atualizado com resultados reais.

Arquivos grandes são ignorados pelo Git. Manifestos pequenos, configurações,
relatórios e comandos devem ser versionados. O protocolo ainda está em rascunho;
o código recusa a geração definitiva e os probes confirmatórios antes dos gates.
Não atualize aprovação, status ou tags automaticamente em nome do orientador.

Fontes primárias:
[EvalPlus, execução oficial](https://github.com/evalplus/evalplus/blob/v0.3.1/evalplus/eval/__init__.py),
[timeout POSIX](https://github.com/evalplus/evalplus/blob/v0.3.1/evalplus/eval/utils.py),
[modelo Qwen](https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct),
[artigo EvalPlus](https://arxiv.org/abs/2305.01210).
