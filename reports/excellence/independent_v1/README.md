# Avaliação independente aprovada

Amostra principal: 120 tarefas e 2400 tentativas; piloto técnico disjunto: 20 tarefas.
Modelo, gates e amostra congelados antes da primeira trajetória nova. Aprovação explícita do estudante; revisão do orientador informada por ele.

## Único endpoint confirmatório

Diferença de retenções H+B−B em t=8: -0.16359060402684564; IC95%: [-0.2402298061170319, -0.0942741120201171].
Hipótese direcional sustentada pelo critério pré-especificado: True. Endpoint identificável: True.
Bootstrap: 10000 reamostragens por tarefa; seed 20261017; 0 replicações sem denominador positivo.

| Gate | Corretos retidos/ativos | Retenção | IC95% |
|---|---:|---:|---|
| HB | 986/1192 | 0.8271812080536913 | [0.7499951587916344, 0.8971813610038177] |
| B | 1181/1192 | 0.9907718120805369 | [0.9829719384391433, 0.9971182937712926] |

Todas as tarefas principais entram nas contagens, inclusive aquelas sem acertos. O denominador é comum aos gates.
Os intervalos das retenções absolutas e dos resultados abaixo são descritivos. A decisão acima não substitui Q2/H3 originais.

## Heterogeneidade e influência

Tarefas com pelo menos um correto ativo: 94; sem correto ativo: 26.
Tarefas com decisões discordantes entre gates: 33.
Faixa das estimativas ao excluir uma tarefa, sem refazer scorers: [-0.1680887372013652, -0.15004262574595056].
As exclusões são diagnósticas, não alteram a amostra principal e não testam significância após cada retirada.
Sensibilidade adicional agrupando semelhanças textuais: {'estimate': -0.16359060402684564, 'ci95': [-0.23983557923917392, -0.09531487386881299], 'tasks': 118, 'replications': 10000, 'seed': 20261017, 'undefined_draws': 0, 'identified': True, 'clusters': 118, 'scope': 'Post-hoc sensitivity: join prompt-similarity pairs flagged by the label-blind post-start audit; sole primary task-bootstrap unchanged'}.

## Replay descritivo, N=20

| Política | Tarefas resolvidas | Tokens médios | Falsos abortos iniciados |
|---|---:|---:|---:|
| fixed | 94/120 | 873.258 | 0 |
| verifier | 94/120 | 342.992 | 0 |
| B | 94/120 | 342.383 | 1 |
| HB | 88/120 | 334.850 | 64 |
| oracle | 94/120 | 77.358 | 0 |

Dados: [analysis.json](analysis.json), [predições](predictions.csv), [contagens por tarefa](retention_per_task.csv),
[replay completo](policy_per_task.csv), [calibração](calibration.csv) e [congelamento](freeze/manifest.json).

A auditoria verificou todas as identidades, seeds e ativações das duas fases e reaplicou os scorers congelados.
Erro máximo de reprodução das probabilidades: 4.44e-16.

Limites: um modelo e uma reserva MBPP+; exposição de pré-treino não certificada; bootstrap pressupõe tarefas intercambiáveis;
não há ajuste no alvo, atribuição causal, equivalência pelo cruzamento de zero ou inferência de latência a partir de tokens.
