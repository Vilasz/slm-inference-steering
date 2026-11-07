# EXCELLENCE_EXTENSION v1 — validação independente do transporte de gates

**Data da proposta:** 17/10/2026. **Estado:** proposta, sem congelamento científico.
**Aprovação do estudante:** pendente. **Revisão do orientador:** pendente.
Nenhuma nova trajetória desta avaliação foi coletada.

## Motivação e informação já conhecida

O experimento original e todas as análises exploratórias anteriores a esta proposta
já foram examinados. No MBPP+ inicial, H+B reteve 276/366 corretos ativos em 8 tokens
(75,41%). No mesmo conjunto ativo e checkpoint, o gate B escolhido exclusivamente
no desenvolvimento HumanEval+ reteve 364/366 (99,45%); a diferença é −24,04 pontos
percentuais. Essa observação motiva a hipótese, mas não constitui sua confirmação.
PCA e correção residual reduziram a desvantagem original sem demonstrar ganho
incremental sobre B. A proposta **não escolhe o classificador exploratório vencedor**.

## Hipótese, população e estimando principal

Hipótese: ao transportar os dois gates congelados da origem para novas tarefas
MBPP+, o gate H+B perde mais candidatos corretos do que B, mesmo comparando ambos
no mesmo checkpoint de 8 tokens. O objetivo é uma comparação delimitada de
transportabilidade de decisões, sem atribuir causalmente o resultado a domain shift.

Para tarefa i e candidato j, S indica aprovação nas suites base e plus, T o
comprimento e K_g a decisão de manter do gate g. O estimando principal é

\[
\theta=r_{HB}-r_B,
\quad r_g=\frac{E_i[\sum_{j=1}^{20} S_{ij}1\{T_{ij}>8\}K_{g,ij}]}{E_i[\sum_{j=1}^{20} S_{ij}1\{T_{ij}>8\}]}.
\]

É a diferença de razões de contagens de candidatos, com tarefas como clusters.
Todas as tarefas selecionadas entram nas contagens, inclusive sem sucessos. O
denominador é comum aos gates. Retenção de candidatos não é resolução de tarefas.

## Métodos e baselines congelados

- Modelo: Qwen2.5-Coder-1.5B-Instruct, revisão
  `2e1fd397ee46e1388853d2af2c993145b0f1098a` para pesos e tokenizer.
- Bundle: `runs/transfer/protocol_v0.2/models.joblib`, já ajustado em HumanEval+.
- Endpoint único: 8 tokens, elegibilidade estrita T>8; término natural em 8 vai
  ao verificador antes da decisão. Abortaria se probabilidade < limiar.
- B: limiar **0.3166734153299228**, escolhido por menor custo dentre as opções
  admissíveis do desenvolvimento de origem em t=8, com os desempates originais.
- H+B: limiar **0.26953905544150975**, mesma regra e dados de origem em t=8.
- A política original B em t=16 pode aparecer como secundária, com o nome próprio;
  não substitui a comparação principal no checkpoint comum.
- Baselines operacionais: N completos, parada por verificador, B, H+B e oracle
  retrospectivo. N=20 principal operacional descritivo; N=1,5,10 secundários.
- Temperatura 1,0, top-p=0,95, top-k=0, máximo 512 tokens; nenhuma verificação
  interfere na geração das vinte tentativas do banco completo.
- BF16 nos pesos, projeções lineares FP32, atenção eager, batch 1, TF32 desativado.
  O benchmark técnico compara captura seletiva com original e exige tokens e estados idênticos.
- Dataset MBPP+ v0.2.0, EvalPlus 0.3.1 e hash registrado em `proposal_design.json`.
  Checker POSIX/WSL, full base+plus, mínimo 1 segundo e fator 4 do tempo ground truth.

Nenhum modelo, calibrador, C, camada ou limiar será ajustado com os novos rótulos.
O programa completo será gerado e testado; o gate será avaliado por replay pareado.

## Amostra, independência e seeds

O dataset fixado contém **378 tarefas**. A auditoria excluiu as 32 tarefas com
trajetórias anteriores e `Mbpp/2`, usado em controle de verificador: restam 345.
Arquivo de referência: `reports/excellence/target_exposure_audit.json`.

Ordenar os IDs elegíveis por SHA256 de `excellence-independent-mbpp-v1:` + ID,
com desempate lexicográfico. As primeiras 20 tarefas formam o piloto técnico
disjunto; as 120 seguintes formam a avaliação principal. IDs explícitos em
`reports/excellence/proposal_design.json`; nenhum ID é escolhido por resultado.

São **400 tentativas técnicas e 2.400 principais**, vinte por tarefa, mesmo após
um sucesso. O piloto só verifica infraestrutura, finitude, invariância e custo;
não seleciona algoritmo ou regra estatística. Seus resultados não se juntam à
amostra principal nem permitem substituir tarefas desfavoráveis. Se motivar mudança
científica, será necessária nova versão prospectiva e nova reserva ainda não gerada.

Seeds usam o mesmo hash canônico de identidade, com namespace
`excellence-extension-v1`, versão do benchmark, task_id e attempt_id 0–19.
Aborto simulado não muda a seed ou a ordem de tentativas seguintes.
“Independente” significa novas trajetórias/tarefas sem resultados locais prévios;
não certifica ausência de exposição fora deste repositório ou no pré-treinamento.

## Inferência e critério de sucesso

Estimar θ pela razão das contagens somadas. Reamostrar **tarefas inteiras**, 10.000
vezes, seed **20261017**, mantendo o pareamento dos gates e recalculando a razão
em cada amostra. Usar percentis 2,5% e 97,5%. Publicar replicações sem denominador
positivo e não substituí-las por valores convenientes. Se mais de 1% forem
indefinidas, reportar o endpoint como não identificável neste orçamento.

Suporte à hipótese direcional somente se o **limite superior do IC bilateral de
95% de θ for menor que zero**. Publicar estimativa e intervalo independentemente
do resultado. Esta é a única hipótese confirmatória proposta; métricas secundárias
têm intervalos descritivos e não podem promovê-las ao lugar do endpoint após inspeção.
Nenhuma conclusão de equivalência decorre de um intervalo que inclua zero.

Secundárias: retenções absolutas, false-abort iniciado versus potencial, WT-AUC
em tarefas mistas, AUC agregada, log loss/Brier, calibração em dez bins fixos,
resolução e custo pareados até o primeiro sucesso, perda de tarefas e oracle.
Não chamar a meta de 97% de garantia certificada ou inferir latência de tokens.

## Precisão e orçamento

Usando as 32 tarefas já exploradas apenas para planejamento, o desvio-padrão da
função de influência do contraste de razões foi **0,3863**. A meia largura normal
aproximada de um IC95% seria 0,0757 com 100 tarefas e 0,0691 com 120. Inflar esse
desvio em 25% produz meias larguras de 0,0946 e 0,0864, respectivamente. Esses
cenários não garantem precisão futura nem são “poder observado” ou nova evidência.
A escolha de 120 tarefas favorece precisão, dentro da reserva disponível.

A GPU conectada é **RTX 3050 Laptop 6 GB**, driver 576.52. A RTX 4060 mencionada
na missão não está visível. A coleta proposta só inicia no hardware revisto
em `proposal_design.json`. Usar outro dispositivo exige preflight e emenda antes
de misturar trajetórias. Os experimentos originais mantêm seu hardware histórico.

O benchmark de 50 pares/100 gerações em tarefas conhecidas mede geração, prefill,
forward por token e cópia de estados; a medição complementar separa score e
verificação. Estimativa final de horas e margem de 50% para interrupções/retries
serão registradas em `reports/excellence/resource_budget.json` antes da revisão.
Sem medição final válida, o protocolo não está pronto para congelar.

## Controle técnico, parada e retomada

O coletor isolado é `scripts/run_excellence_extension.py`; `--dry-run` não gera
candidatos. A coleta exige aprovação registrada e hashes coincidentes do protocolo,
desenho, coletor, captura, configuração e bundle. O piloto técnico precisa concluir
seus gates antes da coleta principal. O histórico v0.2 não é modificado.

Interromper antes de anexar qualquer registro em caso de não finitude, diferença
de tokens/estados entre capturas, falha da invariância de prefixo, OOM, falha de
controle positivo/negativo, erro RPC, hash divergente, identidade duplicada ou
arquivo corrompido. Timeouts de programas são negativos sob o checker fixado;
erros de infraestrutura não recebem rótulos negativos fabricados. Truncamento
é atributo e o programa extraído é avaliado normalmente.

JSONL é append-only, com lock exclusivo, flush/fsync por candidato; ativações
completadas precedem o registro. Manifests são atômicos com retries Windows
limitados. Sessões de 600 segundos param entre candidatos, sem reduzir a amostra.
Retomada confere identidades, seeds, hashes, versões e contrato. Armazenar apenas
prompt-final, t=8 e final da camada necessária; sem guardar estados de todas as posições.

Não haverá parada por significância, expansão baseada no resultado ou substituição
de tarefa por baixa taxa de acertos. Uma restrição externa de tempo antes de completar
a amostra deverá ser divulgada; uma análise parcial será limitada, sem mudar critérios.

## Revisão, congelamento e história

| Papel | Estado |
|---|---|
| Estudante | Aprovação pendente |
| Orientador | Revisão pendente |
| Coleta independente | Não iniciada |

Após aprovação e revisão reais, copiar o protocolo e os fontes relevantes para
um snapshot, registrar seus hashes e decisões e criar a aprovação vinculada à
versão revista. A existência desta proposta e de seu hash **não** é aprovação,
pré-registro externo ou congelamento científico. Não assinar em nome do orientador.

Alternativa de menor custo: 100 tarefas principais, com perda de precisão acima
quantificada. Essa mudança deve ser escolhida e versionada antes de gerar as novas
trajetórias principais; não será acionada após observar seu desempenho.

LiveCodeBench fica fora desta extensão: o checker MBPP+ já foi validado e oferece
345 tarefas elegíveis. Introduzir outro benchmark agora acrescentaria custo de
integração e outra população; nenhum resultado de LiveCodeBench é reivindicado.
