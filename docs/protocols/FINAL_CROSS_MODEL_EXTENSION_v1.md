# FINAL_CROSS_MODEL_EXTENSION v1

Protocolo prospectivo local, autorizado pelo pedido do estudante em 04/11/2026.
Não constitui pré-registro independente ou aprovação atribuída ao orientador.
O instante exato de cada trava e hashes ficam em `reports/final_extension/`.
Piloto e desenvolvimento precedem a trava final; nenhum rótulo externo novo pode
ser inspecionado antes de `freeze.json`. Alterações científicas posteriores
exigem nova versão prospectiva, mantendo este documento e seus resultados.

## Questão e população

Investigar se estados internos em 32 tokens incrementam observáveis causais
das gerações de dois SLMs. A hipótese primária única é ΔDeepSeek,MLP =
WT-AUC(H+B,MLP) − WT-AUC(B,MLP), na população de tarefas MBPP+ novas que sejam
mistas no conjunto ativo T>32. Intervalo bilateral pareado de 95% por tarefa.
Limite inferior >0 sustenta ganho; limite superior <0 sustenta deterioração;
intervalo incluindo zero não demonstra diferença e não estabelece equivalência.

Secundárias descritivas: contraste linear no DeepSeek; ambos no Qwen;
Γ=ΔMLP−Δlinear; heterogeneidade; log loss, Brier, calibração em dez bins fixos;
AUC agregada com pesos iguais por tarefa; quantidade de tarefas mistas;
replay operacional. Sem ajuste de multiplicidade, nenhum resultado secundário
substitui a hipótese primária ou é promovido a confirmação independente.

## Modelos, instrumentação e geração

- Qwen/Qwen2.5-Coder-1.5B-Instruct, pesos e tokenizer:
  `2e1fd397ee46e1388853d2af2c993145b0f1098a`.
- deepseek-ai/deepseek-coder-1.3b-instruct, pesos e tokenizer:
  `e063262dac8366fc1f28a4da0ff3c50ea66259ca`.
- Classes nativas Transformers, sem trust_remote_code, sem quantização;
  BF16 nos pesos, projeções densas FP32 como no Qwen histórico; atenção eager,
  TF32 desligado. Se BF16/CUDA não estiver disponível, parar e documentar emenda.
- Templates oficiais fixados em cada revisão. Mesmas mensagens de sistema e
  especificação de função do renderer histórico. A referência textual HumanEval
  no renderer é preservada inclusive para MBPP por compatibilidade com a origem;
  essa imperfeição é documentada, não corrigida só no conjunto externo.
- DeepSeek: LlamaForCausalLM, 24 camadas, hidden 2048, 16 heads, contexto 16384,
  BOS 32013, EOS/EOT 32021. O template já inclui BOS: encode com
  add_special_tokens=False. Qwen usa seu tokenizer próprio; tokens não são
  alinhados entre modelos. Metadados e templates completos em model_metadata.json.
- Licenças: Qwen Apache-2.0; DeepSeek Model License para pesos, MIT para o código
  do projeto oficial. Fontes oficiais vinculadas em model_metadata.json e README.
- H fixado prospectivamente na saída do bloco de profundidade relativa 75%:
  índice base-zero 20/28 no Qwen e 17/24 no DeepSeek, após alimentar o 32º token.
  É uma escolha restrita de desenho; não se busca camada em rótulos externos.
- B: dez observáveis históricos (cinco de prompt, NLL médio/último, entropia
  média/máxima e margem média). Distribuição após temperatura/top-p para
  NLL/entropia; margem antes do top-p, após temperatura. Somente tokens 1–32.
- Temperatura 1,0; top-p 0,95; top-k 0; máximo 512 tokens; batch 1; vinte
  tentativas por tarefa, mesmo após sucesso. Seeds SHA256 canônico com namespace
  final-cross-model-v1-{modelo}, versão do benchmark, task_id e attempt_id 0–19.
- T<=32: candidato não é submetido ao gate; vai ao verificador normalmente.
  Truncamentos são registrados e avaliados; nunca rotulados automaticamente.

## Inventário e IDs

EvalPlus 0.3.1; HumanEval+ v0.1.10 na origem e MBPP+ v0.2.0 no externo.
`task_inventory.json` contém IDs explícitos, hashes, proveniência e exclusões:
todos os JSONL históricos em runs e IDs de tabelas CSV em reports, incluindo
transferência de 32 tarefas, piloto de 20, validação de 120 e controle Mbpp/2.
Arquivos copiados também são examinados. Reservas futuras sem resultados não
são interpretadas como exposição a rótulos.

Triagem conservadora de variantes: normalizar espaços/nome da função no prompt;
SequenceMatcher>=0,85 exclui novos vizinhos de tarefas expostas e retém apenas
um representante lexicográfico de pares novos. Não certifica equivalência
semântica completa, independência absoluta ou ausência de pré-treinamento.

Ordenação dos IDs elegíveis: SHA256 de `final-cross-model-v1:`+ID, desempate por
ID. Hierarquia predefinida: 150, 140, 120, 100, 80 tarefas. Usar o maior tamanho
disponível e compatível com custo medido mais contingência de 50%. Subamostras
são prefixos dessa ordem; a escolha usa custo e inventário, nunca rótulos finais.
`final_design.json` fixa o tamanho e IDs antes de qualquer geração externa.
Se nem 80 couberem, declarar inviabilidade do desenho no orçamento e não gerar
o teste externo. Nenhuma ampliação para obter significância.

## Piloto e origem

Doze tarefas HumanEval+ do piloto histórico, ordenadas por SHA256 de
`final-deepseek-pilot-v1:`+ID, disjuntas das 132 de origem e do externo.
DeepSeek: vinte tentativas em cada. Qwen: uma tentativa em cada, exclusivamente
para benchmark técnico de custo na GPU atual. Controles de verificador usam
tarefa de desenvolvimento ou Mbpp/2 já exposta.

Critérios fixos para seguir: >=4/12 tarefas DeepSeek mistas; >=5% de acertos
agregados; truncamento <=25%; erro de parsing <=15%; todos os testes numéricos,
reprodução das seeds e controles do sandbox/verificador aprovados. Se falhar,
publicar o piloto e propor emenda mínima; não mudar temperatura, substituir o
modelo, tarefas ou tokenizer para favorecer o teste externo.

Origem: 132 tarefas HumanEval+ idênticas às históricas. Qwen reutiliza o banco
válido; DeepSeek coleta vinte tentativas em cada uma. Testes base+plus com
checker oficial, mínimo 1s, fator 4 da referência, mesmos testes e parser.

## Quatro famílias de probes e seleção

B/logística, H+B/logística, B/MLP e H+B/MLP, por modelo. Comparações principais
sempre dentro da mesma classe. H usa PCA64 com whitening após padronização;
PCA/imputação/scalers são ajustados apenas nas tarefas de treino de cada split.
Se o rank de treino não suportar 64, parar em vez de trocar silenciosamente.

Reservar por hash 20% das tarefas de origem exclusivamente para calibração e
limiar. Nos 80% restantes, GroupKFold de três folds seleciona hiperparâmetros
pela menor média de log loss balanceada por tarefa, com desempate por menor
capacidade/C. Nenhum dado externo participa de seleção, early stopping ou
calibração. Todos os splits e tarefas são publicados nos manifests.

Logística: C em {0,001;0,01;0,1;1}, lbfgs, até 3000 iterações. MLP: uma camada
oculta ReLU, largura {16,32}, dropout 0,1, AdamW lr=0,001, weight decay=0,001,
BCE com pesos por tarefa, CPU, full batch, teto 200 épocas, patience 20 e
melhora mínima 1e−5. Early stopping usa 20% das tarefas do treino interno,
sem incluir o fold de validação. Reajustar no treino interno inteiro pelo número
mediano de épocas escolhido. Refit final usa a mediana das épocas nos folds.
Três seeds fixas 20261104/105/106; ensemble da média de probabilidades com todos
os membros, sem seleção de seed. Publicar estabilidade por seed nos folds.

Platt em tarefas reservadas de calibração da origem, C=1e6; slope deve ser
positivo ou declarar inviabilidade da calibração antes do externo. Limiar:
maior score observado preservando >=97% dos positivos ativos de calibração.
Essa retenção de desenvolvimento não é garantia externa. Congelar os quatro
bundles e hashes por modelo antes da geração externa.

## Poder, inferência e operação

Análise de planejamento baseada em heterogeneidade histórica por tarefa, efeitos
conhecidos +0,02/+0,03/+0,05/+0,075, cenários de frequência mista e tamanho.
Curvas e aproximações em power_analysis.md; limites de efeitos pequenos são
declarados. Não equiparar tentativas ou pares positivo-negativo a tarefas.

Bootstrap final pareado com 10.000 réplicas, seed 20261104, intervalo percentil.
Uma contribuição AUC por tarefa mista, empates com peso 1/2, homogêneas excluídas
apenas do estimando AUC e contadas separadamente. Intervalos condicionais aos
probes ajustados; sensibilidades de treinamento não são apresentadas como
bootstrap do procedimento inteiro. Comparações entre modelos usam clusters
de tarefa comuns; contraste de AUC entre modelos restringe às mistas comuns
e identifica expressamente esse denominador. Γ reamostra as quatro famílias
juntas. Publicar estimativas e intervalos independentemente do sinal.

Replay na ordem attempt_id, mesma população e N={1,5,10,20}: todos os candidatos,
parada por verificador, quatro gates e oracle contrafactual que conhece labels.
Interrupção custa 32 tokens; candidato naturalmente terminado custa seu tamanho;
encerrar ao primeiro sucesso retido, exceto política de gerar todos. Reportar
resolução, tokens médios, tokens totais/resolvida, corretos interrompidos entre
iniciados, retenção entre todos os corretos ativos no orçamento, tarefas perdidas
e diferenças pareadas de custo/cobertura com IC. Denominadores explícitos.
Overhead de score/cópia medido separadamente; tokens não demonstram latência.

## Segurança, orçamento, integridade e encerramento

GPU realmente medida: RTX 3050 Laptop 6 GB; hardware.json registra bibliotecas,
CUDA runtime/driver e BF16. Limite aproximado de 60h locais, incluindo piloto
e contingência; sem serviços pagos ou infraestrutura externa. Projetar custos
usando tempos reais por modelo no piloto técnico, não benchmarks de RTX 4060.
Sessões limitadas e retomáveis; verificar orçamento antes de cada candidato.

Código gerado só executa em WSL/bubblewrap, namespaces de rede/PID/usuário,
ambiente limpo, sem /home real ou /mnt/c, bibliotecas/scripts/dataset read-only,
/tmp efêmero, limites de espaço de endereços 3GiB por processo, CPU 7200s por
processo e timeout RPC 300s, processos 64 e arquivos 32MiB. Timeouts EvalPlus
permanecem ativos. Esses são limites por processo, não cgroup de memória total.
Controles negativos de acesso à rede e ao host devem passar antes de avaliar.

Estados copiados só em posições necessárias; NPZ atômico precede JSON atômico
por identidade, que marca commit. Lock exclusivo impede escritores concorrentes.
Retomada confere hashes, seeds, identidade, contrato e conjunto esperado.
Erros de infraestrutura, não finitude, divergência causal/seed, OOM, corrupção,
mudança de fontes ou falha de controle suspendem a coleta sem fabricar labels.
Final incompleto não autoriza análise inferencial. Arquivos históricos protegidos
por hashes não são alterados. Tabelas são reconstruídas diretamente dos bancos.

## Travas auditáveis

`pilot_plan.json`: instante, IDs, regras e hashes da coleta antes do piloto.
`final_design.json`: custo real, amostra definitiva e IDs antes do externo.
`freeze.json`: instante UTC, protocolo, inventário, desenho, fontes, bundles,
versões e resultados de testes. O SHA do próprio manifesto fica em arquivo
separado, evitando autorreferência. Estado inicial é planejamento; existência
destes arquivos e sua validação determinam o estado efetivo da execução.
